"""Build the "try a sample clip" strip from clips you already hold.

WHY THIS IS A SCRIPT AND NOT A COMMITTED FOLDER
-----------------------------------------------
Celeb-DF and FaceForensics++ are licensed for research use and must not be
redistributed, so no clip from either can live in this repository — see the
.gitignore, which excludes every video extension for exactly that reason.

That leaves a gap: a judge or a new contributor clones the repo and has nothing
to press. This script closes it from the other end. You point it at clips you
are already licensed to hold, it copies them into site/assets/samples/ and
writes the manifest the site reads. The site shows the strip when that manifest
exists and hides it completely when it does not, so a fresh clone degrades to
"upload your own" rather than to a row of broken buttons.

THE LABELS ARE GROUND TRUTH, NOT PREDICTIONS
--------------------------------------------
Each clip is labelled with what the DATASET says it is ("Real (Celeb-DF)",
"Synthetic (Celeb-DF)"), never with what the detector is expected to output.
Pressing the button runs the real analysis; announcing the answer in advance
would turn a demonstration into a performance. It also means a disagreement
between the label and the verdict is visible on screen, which is the single
most useful thing a demo can show.

CODECS: WHY THIS TRANSCODES
---------------------------
Celeb-DF ships clips encoded as MPEG-4 Part 2 (`FMP4`). OpenCV decodes those
happily, so the detector analyses them without complaint - but no browser can
play them. The symptom is confusing rather than obvious: a sample analyses
correctly, the report fills in with real numbers, and the video player beside it
stays blank.

So any clip whose codec browsers cannot decode is re-encoded to H.264 on the way
in, at near-visually-lossless quality, with the MP4 index moved to the front of
the file so it can start playing before it has finished downloading.

The re-encoded file is the one that gets analysed, because it is the one you
watch: the alternative - analysing the original while playing a converted copy -
would mean the receipt's file hash referred to a file nobody ever saw. Note that
re-encoding is exactly the operation this detector is weakest against (see
STATUS.md), so after a conversion the script re-checks nothing on your behalf:
run the clip through the site and confirm the verdict still looks right.

USAGE
-----
    python scripts/make_samples.py \\
        --clip id3_0006.mp4 "Real (Celeb-DF)" \\
        --clip id4_0000.mp4 "Real (Celeb-DF)" "Front-facing, well lit" \\
        --clip fake_0021.mp4 "Synthetic (Celeb-DF)" "Swapped face" "Pair A - fake"

    python scripts/make_samples.py --list       # what is installed now
    python scripts/make_samples.py --clear      # remove the strip entirely

Each --clip takes: PATH, GROUND TRUTH, then optionally a NOTE and a display
TITLE. Without a title the filename is used, which for a dataset clip means the
button reads "id0_id16_0000" - accurate, and meaningless to anyone who did not
build the dataset.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "site" / "assets" / "samples"
MANIFEST = SAMPLES_DIR / "samples.json"

# Matches the server's allow-list. A sample that the server would reject is not
# a sample, it is a support ticket.
ALLOWED = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}

# Roughly the size beyond which a "quick demo" stops being quick over a
# conference wifi connection. Warned about, not enforced.
BIG_FILE_MB = 25

# FourCC tags browsers can actually decode in an <video> element. Everything
# else gets converted. This is deliberately a short allow-list rather than a
# block-list of known-bad codecs: an unrecognised tag is far more likely to be
# something exotic that will not play than something that will.
BROWSER_SAFE_FOURCC = {"h264", "avc1", "vp08", "vp09", "av01", "mp4v"}

# Constant Rate Factor for the conversion. 18 is near-visually-lossless; the
# point is to make the clip playable, not to compress it.
CRF = "18"


def _fourcc(path: Path) -> str:
    """The clip's codec tag, lowercased, or "" if it cannot be read."""
    try:
        import cv2
    except ImportError:
        return ""
    cap = cv2.VideoCapture(str(path))
    try:
        raw = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    return "".join(chr((raw >> 8 * i) & 0xFF) for i in range(4)).strip().lower()


def _ffmpeg() -> Optional[str]:
    """Path to an ffmpeg binary, preferring one already on PATH."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                 # noqa: BLE001
        return None


def _transcode(source: Path, destination: Path) -> bool:
    """Re-encode to H.264 so a browser can play it. True if it worked.

    -movflags +faststart moves the MP4 index to the front of the file, which is
    what lets playback begin before the whole clip has downloaded.
    """
    binary = _ffmpeg()
    if binary is None:
        print("  ! %s uses a codec browsers cannot play, and no ffmpeg was found."
              % source.name)
        print("    Install one with:  pip install imageio-ffmpeg")
        print("    Adding it unconverted: it will analyse correctly but the "
              "player will stay blank.")
        return False

    command = [
        binary, "-y", "-loglevel", "error",
        "-i", str(source),
        "-c:v", "libx264", "-crf", CRF, "-preset", "medium",
        # yuv420p is the pixel format every browser decoder supports; some
        # source clips are in formats that encode fine and then play nowhere.
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-an",                                        # no audio: nothing here uses it
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not destination.exists():
        print("  ! could not convert %s: %s"
              % (source.name, (result.stderr or "").strip()[:200]))
        return False
    return True


def do_list() -> int:
    if not MANIFEST.exists():
        print("No samples installed. The strip is hidden on the site.")
        print("Add some with:  python scripts/make_samples.py --clip PATH \"Real (Celeb-DF)\"")
        return 0

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips = manifest.get("clips", [])
    print("%d sample(s) in %s:" % (len(clips), SAMPLES_DIR.relative_to(REPO_ROOT)))
    for clip in clips:
        path = SAMPLES_DIR / clip["file"]
        size = "%.1f MB" % (path.stat().st_size / 1024 / 1024) if path.exists() else "MISSING"
        print("  %-28s %-24s %-10s %s"
              % (clip["file"], clip.get("groundTruth", "?"), size,
                 "(re-encoded)" if clip.get("reencoded") else ""))
    return 0


def do_clear() -> int:
    if not SAMPLES_DIR.exists():
        print("Nothing to clear.")
        return 0
    shutil.rmtree(SAMPLES_DIR)
    print("Removed %s. The sample strip will not render." % SAMPLES_DIR.relative_to(REPO_ROOT))
    return 0


def do_build(entries) -> int:
    clips = []
    for entry in entries:
        if len(entry) < 2:
            print("Each --clip needs a path and a ground-truth label: "
                  "--clip PATH \"Real (Celeb-DF)\" [note]")
            return 2

        source = Path(entry[0]).expanduser()
        if not source.is_absolute():
            source = (Path.cwd() / source).resolve()

        if not source.exists():
            print("Not found: %s" % source)
            return 2
        if source.suffix.lower() not in ALLOWED:
            print("%s has extension %r, which the server will reject. Allowed: %s"
                  % (source.name, source.suffix, ", ".join(sorted(ALLOWED))))
            return 2

        size_mb = source.stat().st_size / 1024 / 1024
        if size_mb > BIG_FILE_MB:
            print("  note: %s is %.0f MB - large for a one-click demo."
                  % (source.name, size_mb))

        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        destination = SAMPLES_DIR / source.name
        codec = _fourcc(source)
        converted = False

        if codec and codec not in BROWSER_SAFE_FOURCC:
            print("  converting %s from %s to H.264 so it will play in a browser"
                  % (source.name, codec))
            # Always land on .mp4: the source extension may say .avi while the
            # output is H.264 in an MP4 container.
            destination = SAMPLES_DIR / (source.stem + ".mp4")
            converted = _transcode(source, destination)
            if not converted:
                shutil.copy2(source, SAMPLES_DIR / source.name)
                destination = SAMPLES_DIR / source.name
        else:
            shutil.copy2(source, destination)

        final_mb = destination.stat().st_size / 1024 / 1024
        clips.append({
            "file": destination.name,
            # Cosmetic label for the button. Ground truth is the claim and comes
            # from the dataset; this is only what the button is called.
            "title": entry[3] if len(entry) > 3 else source.stem,
            "groundTruth": entry[1],
            "note": entry[2] if len(entry) > 2 else "",
            "sizeBytes": destination.stat().st_size,
            # Recorded so it is never a mystery later why a sample's bytes do
            # not match the dataset original.
            "reencoded": converted,
            "sourceCodec": codec or "unknown",
        })
        print("  added %s (%.1f MB%s) - %s"
              % (destination.name, final_mb,
                 ", re-encoded from %.1f MB" % size_mb if converted else "",
                 entry[1]))

    # Clear out clips from a previous build. Each run rewrites the manifest
    # wholesale, so anything left behind is invisible on the site but still
    # sitting on disk - which reads as "these are the samples" to the next
    # person who opens the folder.
    keep = {clip["file"] for clip in clips}
    for stale in SAMPLES_DIR.iterdir():
        if stale.is_file() and stale.name not in keep and stale.name != MANIFEST.name:
            stale.unlink()
            print("  removed %s (no longer in the strip)" % stale.name)

    MANIFEST.write_text(json.dumps({
        "note": ("Generated by scripts/make_samples.py. Labels are dataset ground "
                 "truth, not detector predictions. Not committed: the clips are "
                 "licensed for research use and cannot be redistributed."),
        "clips": clips,
    }, indent=2), encoding="utf-8")

    print("\nWrote %s with %d clip(s)." % (MANIFEST.relative_to(REPO_ROOT), len(clips)))
    print("Reload the site and the sample strip will appear under the uploader.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", nargs="+", action="append", metavar=("PATH TRUTH", ""),
                        help="a video, its dataset ground truth, and optionally "
                             "a note and a display title")
    parser.add_argument("--list", action="store_true", help="show what is installed")
    parser.add_argument("--clear", action="store_true", help="remove all samples")
    args = parser.parse_args()

    if args.list:
        return do_list()
    if args.clear:
        return do_clear()
    if not args.clip:
        parser.print_help()
        return 2
    return do_build(args.clip)


if __name__ == "__main__":
    sys.exit(main())
