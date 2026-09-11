"""Prepare the two front-page clips for the web, and install them.

WHY THIS EXISTS
---------------
A clip that plays perfectly in a desktop player can still behave badly in a
browser, for reasons that are invisible until you look:

  * **The index can be at the end of the file.** An MP4 keeps its table of
    contents in a `moov` atom. If that sits after the video data, the browser
    cannot show a single frame until the *entire* file has downloaded. On an
    autoplaying hero clip this looks exactly like a broken video. Moving it to
    the front is what `-movflags +faststart` does, and it is the single most
    common cause of "it works locally but not on the site".

  * **The bitrate can be wildly oversized for the frame it plays in.** The hero
    clips are displayed inside a phone mockup a few hundred pixels wide. A
    1600 kb/s export is paying for detail nobody can see, on a file every single
    visitor downloads before they have decided to care.

So this re-encodes both clips to H.264/yuv420p - the combination every browser
decodes - at a sensible size, with the index moved to the front, and installs
them into site/assets/ and the container's mirror of it.

AUDIO IS KEPT ON PURPOSE
------------------------
The hero pair loops silently and unmutes on hover, so stripping the audio track
would quietly delete a feature. It is re-encoded at a lower bitrate rather than
dropped.

USAGE
-----
    python scripts/prepare_hero.py --fake path/to/fake.mp4 --real path/to/real.mp4

The clip given as --fake becomes site/assets/reel.mp4, which the page labels
REEL / generated; --real becomes real.mp4, labelled REAL / original. Getting
these the wrong way round would make the page state something untrue, so both
are required and neither has a default.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGETS = [
    REPO_ROOT / "site" / "assets",
    REPO_ROOT / "deploy" / "container" / "site" / "assets",
]

# Quality/size trade-off for a clip shown in a phone-sized frame. 26 is a visibly
# clean CRF at this scale; the source is usually already compressed, so anything
# lower mostly re-encodes existing artefacts at a higher bitrate.
CRF = "26"
AUDIO_BITRATE = "96k"

# Longest edge. 720 covers a phone frame on a high-density display with room to
# spare; larger is bytes every visitor downloads and nobody sees.
MAX_EDGE = 720


def _ffmpeg() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:                                 # noqa: BLE001
        return None


def _is_faststart(path: Path) -> bool:
    """True when the moov atom precedes the media data."""
    data = path.read_bytes()
    moov, mdat = data.find(b"moov"), data.find(b"mdat")
    return moov != -1 and (mdat == -1 or moov < mdat)


def encode(binary: str, source: Path, destination: Path) -> bool:
    command = [
        binary, "-y", "-loglevel", "error",
        "-i", str(source),
        # Scale the longest edge down to MAX_EDGE, never up, and keep both
        # dimensions even - H.264 with yuv420p cannot encode odd dimensions.
        "-vf", ("scale='if(gt(iw,ih),min(%d,iw),-2)':'if(gt(iw,ih),-2,min(%d,ih))'"
                ":flags=lanczos" % (MAX_EDGE, MAX_EDGE)),
        "-c:v", "libx264", "-crf", CRF, "-preset", "slow",
        "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        "-movflags", "+faststart",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not destination.exists():
        print("  ! ffmpeg failed on %s: %s"
              % (source.name, (result.stderr or "").strip()[:300]))
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fake", type=Path, required=True,
                        help="the synthetic clip -> assets/reel.mp4 (labelled 'generated')")
    parser.add_argument("--real", type=Path, required=True,
                        help="the genuine clip -> assets/real.mp4 (labelled 'original')")
    args = parser.parse_args()

    binary = _ffmpeg()
    if binary is None:
        print("No ffmpeg found. Install one with:  pip install imageio-ffmpeg")
        return 2

    jobs = [(args.fake, "reel.mp4", "generated"), (args.real, "real.mp4", "original")]
    for source, name, label in jobs:
        if not source.exists():
            print("Not found: %s" % source)
            return 2

    primary = TARGETS[0]
    primary.mkdir(parents=True, exist_ok=True)

    for source, name, label in jobs:
        destination = primary / name
        before = source.stat().st_size / 1024 / 1024
        print("  %s -> %s (%s)" % (source.name, name, label))

        if not encode(binary, source, destination):
            return 1

        after = destination.stat().st_size / 1024 / 1024
        print("     %.1f MB -> %.1f MB, faststart %s"
              % (before, after, "yes" if _is_faststart(destination) else "NO"))

        # Keep the container's copy identical; a Docker build can only see files
        # inside its own context, so the mirror is not optional.
        for target in TARGETS[1:]:
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, target / name)

    print("\nInstalled into %s and the container mirror." % primary.relative_to(REPO_ROOT))
    print("Commit both copies to publish them - the site is deployed from this repo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
