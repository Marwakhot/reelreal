"""Throw malformed, hostile and simply odd input at the server and see what it does.

WHY THIS EXISTS
---------------
The happy path — a clean 24-second MP4 with a face in it — is the path that
gets tested by hand a hundred times during development. Everything else is the
path a stranger takes on the first try. This script walks that second path on
purpose and prints a table of what came back.

The bar being tested is NOT "nothing fails". Several of these inputs SHOULD
fail; a server that cheerfully returns a verdict for a renamed PDF is worse
than one that rejects it. The bar is:

    every input produces a deliberate, correctly-classified HTTP response,
    and the server is still healthy afterwards.

So each case below declares which status codes are acceptable for it, and the
check passes only if the response lands in that set. A 500 is never acceptable:
it means an exception escaped rather than a case being handled.

WHAT IT SENDS
-------------
    empty file              0 bytes with a valid .mp4 name
    not a video             a PDF header renamed to .mp4
    truncated video         a real clip cut off one third of the way in
    wrong extension         a real clip renamed .txt
    no extension            a real clip with the suffix stripped
    missing field           a POST with no "video" part at all
    wrong field name        the file sent as "file" instead of "video"
    huge declared size      201 MB of zeroes, one byte over the cap
    unicode filename        a real clip named with emoji and RTL text
    path traversal          a real clip named ../../etc/passwd.mp4
    double extension        a real clip named clip.mp4.exe
    repeat of a good clip   the happy path again, AFTER all of the above,
                            to prove the server did not get wedged

USAGE
-----
    python scripts/check_robustness.py
    python scripts/check_robustness.py --api http://127.0.0.1:8000 --clip id3_0006.mp4

Exit code 0 if every case behaved as declared, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_API = "http://127.0.0.1:8000"

# A PDF magic number. Renamed to .mp4 it passes the extension allow-list and
# has to be rejected by something further down, which is the point of the case.
PDF_BYTES = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<< /Type /Catalog >>\nendobj\ntrailer\n%%EOF\n"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _multipart(field: Optional[str], filename: Optional[str],
               payload: bytes) -> Tuple[bytes, str]:
    """Build a multipart body. field=None sends a form with no file part at all."""
    boundary = uuid.uuid4().hex
    content_type = "multipart/form-data; boundary=" + boundary

    if field is None:
        body = ("--%s\r\n"
                'Content-Disposition: form-data; name="note"\r\n\r\n'
                "no file attached\r\n"
                "--%s--\r\n" % (boundary, boundary)).encode("utf-8")
        return body, content_type

    head = (
        "--%s\r\n"
        'Content-Disposition: form-data; name="%s"; filename="%s"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n" % (boundary, field, filename)
    ).encode("utf-8")
    tail = ("\r\n--%s--\r\n" % boundary).encode("utf-8")
    return head + payload + tail, content_type


def _send(api: str, field: Optional[str], filename: Optional[str],
          payload: bytes, timeout: int) -> Tuple[int, str]:
    """POST to /v1/analyze and return (status code, short response text).

    Connection-level failures are reported as status 0 — that is the signal
    that the server died or hung rather than answering, which is exactly the
    outcome this script exists to catch.
    """
    body, content_type = _multipart(field, filename, payload)
    req = urllib.request.Request(
        api.rstrip("/") + "/v1/analyze",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(400).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        text = exc.read(400).decode("utf-8", "replace")
        try:
            text = json.loads(text).get("detail", text)
        except Exception:                             # noqa: BLE001
            pass
        return exc.code, str(text)
    except Exception as exc:                          # noqa: BLE001
        return 0, "%s: %s" % (type(exc).__name__, exc)


def _health(api: str, timeout: int = 15) -> bool:
    try:
        with urllib.request.urlopen(api.rstrip("/") + "/health", timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")).get("ok") is True
    except Exception:                                 # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------
def build_cases(clip: bytes) -> List[Dict[str, Any]]:
    """Every case declares the statuses that count as correct handling.

    The `expect` set is the interesting part of this file. It is a statement
    about intended behaviour, so a change in the server that silently starts
    accepting a renamed PDF will fail this script rather than pass it quietly.
    """
    third = max(1, len(clip) // 3)
    return [
        {
            "name": "empty file",
            "field": "video", "filename": "empty.mp4", "payload": b"",
            "expect": {400},
            "why": "0 bytes is not a video; the server rejects it before the model.",
        },
        {
            "name": "not a video",
            "field": "video", "filename": "invoice.mp4", "payload": PDF_BYTES,
            "expect": {400, 415, 422},
            "why": "A PDF renamed .mp4 passes the extension check, so something "
                   "downstream must catch it. A 200 here would be a real bug.",
        },
        {
            "name": "truncated video",
            "field": "video", "filename": "cut.mp4", "payload": clip[:third],
            "expect": {200, 400, 415, 422},
            "why": "A partial file may still decode a few frames. Either a "
                   "verdict or a clean rejection is defensible; a crash is not.",
        },
        {
            "name": "wrong extension",
            "field": "video", "filename": "clip.txt", "payload": clip,
            "expect": {415},
            "why": "Extension allow-list rejects it without reading the bytes.",
        },
        {
            "name": "no extension",
            "field": "video", "filename": "clip", "payload": clip,
            "expect": {415},
            "why": "An empty suffix must not be treated as permitted.",
        },
        {
            "name": "missing field",
            "field": None, "filename": None, "payload": b"",
            "expect": {422},
            "why": "FastAPI validates the required form field before any handler runs.",
        },
        {
            "name": "wrong field name",
            "field": "file", "filename": "clip.mp4", "payload": clip,
            "expect": {422},
            "why": "The API takes 'video'; anything else is a client error, not a 500.",
        },
        {
            "name": "over size cap",
            "field": "video", "filename": "huge.mp4",
            "payload": b"\0" * (201 * 1024 * 1024),
            "expect": {413},
            "why": "One byte over the 200 MB cap; the stream must abort mid-upload "
                   "rather than buffer the whole thing first.",
        },
        {
            "name": "unicode filename",
            "field": "video", "filename": "مقطع \U0001F3AC.mp4",
            "payload": clip,
            "expect": {200},
            "why": "Arabic and an emoji in the name. The name is only echoed as "
                   "text, never used to build a path, so this must simply work.",
        },
        {
            "name": "path traversal",
            "field": "video", "filename": "../../etc/passwd.mp4", "payload": clip,
            "expect": {200},
            "why": "Must be analysed normally with nothing written outside the "
                   "temp directory. _save_upload generates its own filename.",
        },
        {
            "name": "double extension",
            "field": "video", "filename": "clip.mp4.exe", "payload": clip,
            "expect": {415},
            "why": "Only the final suffix counts, and .exe is not on the list.",
        },
        {
            "name": "good clip, after all that",
            "field": "video", "filename": "recheck.mp4", "payload": clip,
            "expect": {200},
            "why": "The happy path again at the end. This is the case that proves "
                   "the server survived everything above it.",
        },
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--api", default=DEFAULT_API, help="server base URL")
    parser.add_argument("--clip", type=Path, default=None,
                        help="a real video to use as the valid payload "
                             "(default: the first .mp4 in the repo root)")
    parser.add_argument("--timeout", type=int, default=900,
                        help="seconds per request (default 900; the first "
                             "analysis loads the model)")
    parser.add_argument("--skip-large", action="store_true",
                        help="skip the 201 MB case on a slow or metered link")
    args = parser.parse_args()

    clip_path = args.clip
    if clip_path is None:
        candidates = sorted(REPO_ROOT.glob("*.mp4"))
        clip_path = candidates[0] if candidates else None
    if clip_path is None or not clip_path.exists():
        print("Need one real video. Pass --clip PATH, or put an .mp4 in the repo root.")
        return 2

    if not _health(args.api):
        print("No healthy server at %s - start it first:" % args.api)
        print("    cd server && python -m uvicorn app:app --port 8000")
        return 2

    clip = clip_path.read_bytes()
    cases: Sequence[Dict[str, Any]] = build_cases(clip)
    if args.skip_large:
        cases = [c for c in cases if c["name"] != "over size cap"]

    print("REEL/REAL - robustness check")
    print("API:  %s" % args.api)
    print("Clip: %s (%.1f MB)\n" % (clip_path.name, len(clip) / 1024 / 1024))
    print("  %-26s %-8s %-9s %s" % ("case", "status", "verdict", "expected"))
    print("  " + "-" * 66)

    rows, failures = [], 0
    for case in cases:
        started = time.time()
        status, text = _send(args.api, case["field"], case["filename"],
                             case["payload"], args.timeout)
        ok = status in case["expect"]
        if not ok:
            failures += 1

        expected = "/".join(str(s) for s in sorted(case["expect"]))
        print("  %-26s %-8s %-9s %s%s"
              % (case["name"], status or "no reply", "PASS" if ok else "FAIL",
                 expected, "" if ok else "   <-- " + text.replace("\n", " ")[:80]))

        rows.append({
            "case": case["name"],
            "status": status,
            "expected": sorted(case["expect"]),
            "passed": ok,
            "seconds": round(time.time() - started, 2),
            "why": case["why"],
            "response": text[:200],
        })

    alive = _health(args.api)
    print("\n  server healthy afterwards: %s" % ("yes" if alive else "NO"))
    if not alive:
        failures += 1

    print()
    if failures:
        print("RESULT: %d of %d cases did not behave as declared." % (failures, len(rows)))
    else:
        print("RESULT: all %d cases handled as declared, server still healthy."
              % len(rows))
    print("Scope: these are the malformed inputs we thought of. Passing means no "
          "case\n       below produced an unhandled error, not that no such input exists.")

    out = REPO_ROOT / "reports" / "robustness.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "api": args.api,
        "clip": clip_path.name,
        "checkedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "serverHealthyAfterwards": alive,
        "cases": rows,
    }, indent=2), encoding="utf-8")
    print("Written: %s" % out.relative_to(REPO_ROOT))

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
