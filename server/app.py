"""The middle layer between the browser and the detection pipeline.

WHY THIS EXISTS
---------------
The frontend is HTML/CSS/JS running in a browser. The detector is Python and
PyTorch. A browser cannot run Python, so the two cannot talk directly. This is
a small HTTP server that sits between them:

    browser  --POST video-->  this server  --calls-->  infer_pipeline.py
    browser  <--JSON result--  this server  <--dict--  infer_pipeline.py

It does three jobs and nothing else:
  1. accept an uploaded video over HTTP,
  2. run the existing pipeline on it (unmodified),
  3. translate the pipeline's dict into the shape the frontend already reads
     (see adapter.py).

Run it:
    pip install -r requirements.txt
    set PIPELINE_DIR=C:\\path\\to\\ctf_pretrained     (Windows)
    export PIPELINE_DIR=/path/to/ctf_pretrained       (macOS/Linux)
    uvicorn app:app --port 8000

The first request is slow: the model weights download from Hugging Face and
load into memory. Every request after that reuses the loaded model.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import adapter
import receipt

# --------------------------------------------------------------------------
# Locating the pipeline
# --------------------------------------------------------------------------
# The detection code lives in its own folder (ctf_pretrained). Rather than
# copying it in here — which would immediately drift out of date — we add it to
# the import path. Set PIPELINE_DIR to wherever that folder actually is.
PIPELINE_DIR = os.environ.get("PIPELINE_DIR", "").strip()

# The model now lives inside this project, one level up from server/. Defaulting
# to it means the usual case needs no environment variable at all - and, more
# importantly, there is one obvious copy rather than several on disk that can
# silently drift apart. PIPELINE_DIR still overrides, for pointing at another one.
if not PIPELINE_DIR:
    _sibling = Path(__file__).resolve().parent.parent / "ctf_pretrained"
    if (_sibling / "infer_pipeline.py").exists():
        PIPELINE_DIR = str(_sibling)

if PIPELINE_DIR and PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

# Only these extensions are accepted. The uploaded filename is NEVER used to
# build a path on disk — see _save_upload — this list is just an early reject.
ALLOWED_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024          # 200 MB

app = FastAPI(title="REEL/REAL detection API", version="1.0")

# The site runs from file:// or a static host, and the extension popup is a
# chrome-extension:// origin. Both are different origins from this server, so the
# browser needs explicit permission to read the response.
#
# ALLOWED_ORIGINS is a comma-separated list, e.g.
#   ALLOWED_ORIGINS=https://reelreal.vercel.app,chrome-extension://abc123
# Unset means "*", which is right for local development and wrong once deployed:
# it lets any website on the internet submit videos to this server.
_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

_analyzer = None          # loaded once, on the first request
_model_version = "unknown"


def _get_analyzer():
    """Load the pipeline's analyzer once and keep it in memory.

    Loading is deferred to the first request rather than done at import time so
    the server starts instantly and a missing PIPELINE_DIR produces a clear
    HTTP error instead of a crash at boot.
    """
    global _analyzer, _model_version
    if _analyzer is not None:
        return _analyzer

    try:
        import infer_pipeline
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=("Cannot import the detection pipeline. Set PIPELINE_DIR to "
                    "the folder containing infer_pipeline.py. (%s)" % exc),
        )

    _analyzer = infer_pipeline.VideoAnalyzer.load()
    calibrated = bool(getattr(_analyzer, "clip_calibrator", None))
    _model_version = "ViT-DFv2" + ("" if calibrated else " (uncalibrated)")
    return _analyzer


def _save_upload(upload: UploadFile) -> Tuple[Path, str]:
    """Stream the upload to a temp file and return (path, sha256 hex).

    The client-supplied filename is deliberately not used for the path — only
    its extension is read, and even that is validated against an allow-list.
    A filename like "../../etc/passwd" therefore cannot influence where this
    writes. The name is passed through to the report separately, as text.

    The SHA-256 is accumulated over the same chunks that are written to disk,
    so hashing costs one pass and no extra memory. It is what binds a signed
    receipt to this exact set of bytes (see receipt.py).
    """
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type '%s'. Allowed: %s" % (
                suffix or "(none)", ", ".join(sorted(ALLOWED_SUFFIXES))),
        )

    tmp_dir = Path(tempfile.gettempdir()) / "reelreal_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dest = tmp_dir / ("%s%s" % (uuid.uuid4().hex, suffix))

    written = 0
    digest = hashlib.sha256()
    with dest.open("wb") as out:
        while True:
            chunk = upload.file.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            digest.update(chunk)
            if written > MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail="File larger than %d MB." % (MAX_UPLOAD_BYTES // (1024 * 1024)),
                )
            out.write(chunk)

    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty upload.")

    return dest, digest.hexdigest()


def _probe_video(path: Path) -> Tuple[bool, str]:
    """Can OpenCV actually decode this, and how tall is it?

    Both questions are answered from one open because opening a video file is
    the expensive part.

    WHY THE DECODE CHECK EXISTS
    ---------------------------
    Without it, a file that is not a video at all - a PDF renamed to .mp4, say -
    passes the extension allow-list, decodes zero frames, and comes back out of
    the pipeline as a full report reading "not enough face visible to judge".

    That verdict is a specific claim: *we looked at your video and could not
    tell*. For a renamed PDF nothing was ever looked at, so the report states
    something untrue, and since a signed receipt is issued alongside it, the
    server would be cryptographically attesting a verdict about a document.

    A file that yields no frames is therefore rejected as input rather than
    judged as evidence. Found by scripts/check_robustness.py, which is what
    that script is for.

    Returns (decodable, resolution). Resolution is best effort and falls back
    to "unknown" without affecting the decodable answer.
    """
    try:
        import cv2
    except Exception:                                 # noqa: BLE001
        # No OpenCV: the pipeline has its own reader, so do not block the
        # upload on a probe that could not run. Nothing is claimed either way.
        return True, "unknown"

    cap = None
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False, "unknown"

        # Reading one frame is the actual test. isOpened() alone is too weak:
        # OpenCV will happily "open" a file it can decode nothing from.
        ok, _frame = cap.read()
        if not ok:
            return False, "unknown"

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        if h > 0:
            return True, "%dp" % h
        if w > 0:
            return True, "%dx?" % w
        return True, "unknown"
    except Exception:                                 # noqa: BLE001
        return False, "unknown"
    finally:
        if cap is not None:
            cap.release()


@app.get("/health")
def health():
    """Cheap check the frontend uses to decide whether a real backend is up."""
    return {
        "ok": True,
        "modelLoaded": _analyzer is not None,
        "modelVersion": _model_version,
        "pipelineDir": PIPELINE_DIR or None,
    }


@app.get("/v1/public-key")
def public_key():
    """The Ed25519 public key that signs receipts, so they can be verified.

    Published openly on purpose: a verifier needs it, and it reveals nothing.
    `ephemeral: true` means the server is running without REELREAL_SIGNING_KEY
    and signing with a throwaway key that will not survive a restart.
    """
    try:
        return receipt.public_key_info()
    except ValueError as exc:
        # A malformed REELREAL_SIGNING_KEY. Say exactly what is wrong instead of
        # letting it surface as a bare 500 with no explanation.
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/v1/verify-receipt")
def verify_receipt(body: Dict[str, Any] = Body(...)):
    """Check a receipt's signature.

    A convenience, not the authority: the point of Ed25519 is that anyone can
    verify a receipt themselves with the public key above, without asking this
    server anything. The verify page in the site does exactly that in the
    browser and only falls back to this endpoint where WebCrypto has no Ed25519.

    Accepts either the receipt object itself, or {"receipt": {...},
    "publicKey": "<base64>"} to check against a key other than this server's.
    """
    candidate = body.get("receipt") if "receipt" in body else body
    return receipt.verify(candidate, body.get("publicKey"))


@app.post("/v1/analyze")
def analyze(video: UploadFile = File(...)):
    """Analyse one uploaded video and return an AnalysisResult.

    Deliberately a plain `def`, not `async def`. Everything inside is blocking
    CPU work - decoding frames and running the model - and an `async` handler
    runs that directly on the event loop, which stops the server answering
    anything at all until it finishes. That is not a theoretical problem: the
    website probes /health to decide whether a real backend exists, and a probe
    that times out during someone else's analysis would tell the user their
    results are simulated when they are not.

    FastAPI runs a sync handler in a threadpool instead, so /health, the public
    key and receipt verification stay responsive while a video is being scored.
    """
    started = time.time()
    path, file_sha256 = _save_upload(video)

    try:
        decodable, resolution = _probe_video(path)
        if not decodable:
            raise HTTPException(
                status_code=400,
                detail=("No video frames could be decoded from this file. It may "
                        "be corrupt, or not a video despite its name."),
            )

        analyzer = _get_analyzer()
        try:
            raw = analyzer.analyze(str(path))
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Analysis failed: %s: %s" % (type(exc).__name__, exc),
            )

        result = adapter.to_analysis_result(
            raw,
            file_name=video.filename or path.name,
            file_size=path.stat().st_size,
            resolution=resolution,
            processing_ms=int((time.time() - started) * 1000),
            model_version=_model_version,
            analysed_at=datetime.now(timezone.utc).isoformat(),
        )

        # Sign the verdict. Built from `result` rather than from `raw` so the
        # receipt can only ever carry the same numbers the report displays.
        # A failure to sign must not lose the analysis the user waited for, so
        # it degrades to a result with no receipt and says why.
        try:
            result["receipt"] = receipt.build(
                analysis_id=result["id"],
                file_sha256=file_sha256,
                file_name=result["fileName"],
                file_size=result["fileSizeBytes"],
                verdict=result["verdict"],
                confidence=result["confidence"],
                model_version=result["modelVersion"],
                analysed_at=result["analysedAt"],
                decision_threshold=(result.get("pipeline") or {}).get("decisionThresh"),
            )
        except Exception as exc:                      # noqa: BLE001
            result["receipt"] = None
            result["receiptError"] = "%s: %s" % (type(exc).__name__, exc)

        return result
    finally:
        # The video is deleted as soon as the verdict is produced. Nothing is
        # retained on disk between requests.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


@app.on_event("startup")
def _check_signing_key():
    """Fail loudly at boot on a malformed signing key, not at the first upload.

    A misconfigured key that only surfaces when someone finally analyses a
    video is a much worse failure than one that appears in the startup log.
    Analysis itself is never blocked by this: a result with no receipt is still
    a result, so this warns rather than exits.
    """
    try:
        info = receipt.public_key_info()
    except ValueError as exc:
        print("[startup] Receipts DISABLED - %s" % exc, file=sys.stderr)
        return
    print("[startup] Receipts signed with key %s%s"
          % (info["keyId"], " (throwaway)" if info["ephemeral"] else ""),
          file=sys.stderr)


@app.on_event("shutdown")
def _cleanup():
    shutil.rmtree(Path(tempfile.gettempdir()) / "reelreal_uploads", ignore_errors=True)
