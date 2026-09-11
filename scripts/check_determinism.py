"""Does the same video, analysed twice, produce the same answer?

WHY THIS EXISTS
---------------
A detector that returns 0.71 on one run and 0.66 on the next is not wrong, but
it is not reproducible either, and a verdict that moves when nothing about the
input moved is impossible to defend. This script sends each clip through the
running server N times and compares the results field by field.

It tests the SERVER, not the model: it makes real HTTP requests to a running
instance, so it also covers the upload path, the adapter and anything cached
between requests. A model that is deterministic in a notebook can still be
non-deterministic here if, for example, frame sampling is re-seeded per call.

WHAT IS COMPARED
----------------
    verdict                 the word shown to the user
    confidence              the clip probability, to 4 dp as the API returns it
    pipeline.framesFlagged  how many sampled frames crossed the threshold
    pipeline.meanScore      the mean per-frame score
    timeline[].score        every per-second bar

A run is DETERMINISTIC only if all of these are identical across every repeat.
Processing time is reported but deliberately not compared — it varies with
machine load and says nothing about correctness.

WHAT THIS DOES NOT TELL YOU
---------------------------
Nothing about accuracy. A detector that is confidently wrong every single time
passes this check perfectly. Determinism is a precondition for a defensible
result, not evidence that the result is right.

USAGE
-----
    python scripts/check_determinism.py                    # every .mp4 in repo root
    python scripts/check_determinism.py clip.mp4 --runs 3
    python scripts/check_determinism.py --api http://127.0.0.1:8000

Exit code 0 if every clip was reproducible, 1 otherwise.
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
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_API = "http://127.0.0.1:8000"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------
def _post_video(api: str, path: Path, timeout: int) -> Dict[str, Any]:
    """POST one video as multipart/form-data and return the parsed JSON.

    Written against urllib rather than requests so the script runs on a clean
    Python install with nothing to pip install first — a judge should be able
    to run this immediately after cloning.
    """
    boundary = uuid.uuid4().hex
    data = path.read_bytes()

    head = (
        "--%s\r\n"
        'Content-Disposition: form-data; name="video"; filename="%s"\r\n'
        "Content-Type: video/mp4\r\n\r\n" % (boundary, path.name)
    ).encode("utf-8")
    tail = ("\r\n--%s--\r\n" % boundary).encode("utf-8")

    req = urllib.request.Request(
        api.rstrip("/") + "/v1/analyze",
        data=head + data + tail,
        headers={"Content-Type": "multipart/form-data; boundary=" + boundary},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fingerprint(result: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce a full result to only the fields that must not move between runs."""
    pipeline = result.get("pipeline") or {}
    return {
        "verdict": result.get("verdict"),
        "confidence": result.get("confidence"),
        "framesFlagged": pipeline.get("framesFlagged"),
        "meanScore": pipeline.get("meanScore"),
        "timeline": [b.get("score") for b in (result.get("timeline") or [])],
    }


def _differences(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """Human-readable list of what moved between two fingerprints."""
    out = []
    for key in ("verdict", "confidence", "framesFlagged", "meanScore"):
        if a.get(key) != b.get(key):
            out.append("%s: %r -> %r" % (key, a.get(key), b.get(key)))

    ta, tb = a.get("timeline") or [], b.get("timeline") or []
    if ta != tb:
        if len(ta) != len(tb):
            out.append("timeline length: %d -> %d" % (len(ta), len(tb)))
        else:
            moved = [i for i, (x, y) in enumerate(zip(ta, tb)) if x != y]
            worst = max((abs(ta[i] - tb[i]) for i in moved), default=0.0)
            out.append("timeline: %d of %d bars moved, largest change %.4f"
                       % (len(moved), len(ta), worst))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def check_clip(api: str, path: Path, runs: int, timeout: int
               ) -> Tuple[bool, Optional[Dict[str, Any]], List[str], List[float]]:
    """Analyse one clip `runs` times. Returns (stable, first result, diffs, times)."""
    first_full: Optional[Dict[str, Any]] = None
    first_fp: Optional[Dict[str, Any]] = None
    diffs: List[str] = []
    times: List[float] = []

    for n in range(runs):
        started = time.time()
        try:
            result = _post_video(api, path, timeout)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:200]
            return False, None, ["run %d failed: HTTP %d %s" % (n + 1, exc.code, body)], times
        except Exception as exc:                      # noqa: BLE001 - report anything
            return False, None, ["run %d failed: %s: %s" % (n + 1, type(exc).__name__, exc)], times
        times.append(time.time() - started)

        fingerprint = _fingerprint(result)
        if first_fp is None:
            first_full, first_fp = result, fingerprint
        else:
            for line in _differences(first_fp, fingerprint):
                diffs.append("run 1 vs run %d - %s" % (n + 1, line))

    return (not diffs), first_full, diffs, times


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clips", nargs="*", type=Path,
                        help="videos to test (default: every .mp4 in the repo root)")
    parser.add_argument("--api", default=DEFAULT_API, help="server base URL")
    parser.add_argument("--runs", type=int, default=2, help="analyses per clip (default 2)")
    parser.add_argument("--timeout", type=int, default=900,
                        help="seconds to wait per request; the first is slow "
                             "because the model loads (default 900)")
    args = parser.parse_args()

    if args.runs < 2:
        print("--runs must be at least 2; comparing a run to itself proves nothing.")
        return 2

    clips = args.clips or sorted(REPO_ROOT.glob("*.mp4"))
    clips = [c for c in clips if c.exists()]
    if not clips:
        print("No clips found. Pass paths explicitly, or put .mp4 files in the repo root.")
        return 2

    print("REEL/REAL - determinism check")
    print("API:   %s" % args.api)
    print("Clips: %d, analysed %d times each (%d requests total)"
          % (len(clips), args.runs, len(clips) * args.runs))
    print("The first request loads the model and is expected to be slow.\n")

    rows = []
    unstable = 0
    for path in clips:
        print("  %-16s " % path.name, end="", flush=True)
        stable, result, diffs, times = check_clip(args.api, path, args.runs, args.timeout)
        if stable and result:
            print("stable   %-13s %.4f   (%s)"
                  % (result.get("verdict"), result.get("confidence") or 0.0,
                     ", ".join("%.1fs" % t for t in times)))
        else:
            unstable += 1
            print("UNSTABLE")
            for line in diffs:
                print("      %s" % line)
        rows.append({
            "clip": path.name,
            "stable": stable,
            "verdict": (result or {}).get("verdict"),
            "confidence": (result or {}).get("confidence"),
            "differences": diffs,
            "seconds": [round(t, 2) for t in times],
        })

    print()
    if unstable:
        print("RESULT: %d of %d clips were NOT reproducible." % (unstable, len(clips)))
    else:
        print("RESULT: all %d clips reproduced exactly across %d runs each."
              % (len(clips), args.runs))
    # Say the sample size out loud. Two runs on six clips is a smoke test for
    # reproducibility, not a guarantee about every input the model may ever see.
    print("Sample: %d clips x %d runs. This shows repeatability on these clips "
          "only,\n        and says nothing about accuracy on any of them."
          % (len(clips), args.runs))

    out = REPO_ROOT / "reports" / "determinism.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({
        "api": args.api,
        "runs": args.runs,
        "checkedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "clips": rows,
    }, indent=2), encoding="utf-8")
    print("Written: %s" % out.relative_to(REPO_ROOT))

    return 1 if unstable else 0


if __name__ == "__main__":
    sys.exit(main())
