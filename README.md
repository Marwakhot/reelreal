# REEL/REAL — deepfake video detection

Upload a clip, get a verdict, and — more importantly — get a timeline showing
**which moments** were flagged and a list of the measurements behind the number.

Two front ends share one detector: a website (`site/`) and a Chrome extension
(`extension/`), both talking to a small Python server (`server/`) that wraps the
detection pipeline in `ctf_pretrained/`.

```
browser ──POST video──▶ server/app.py ──▶ ctf_pretrained/infer_pipeline.py
browser ◀──JSON result── server/app.py ◀── (per-frame scores, Grad-CAM, thresholds)
```

**What it is for.** A moderator, journalist or reviewer who has one clip in front
of them and needs a defensible second opinion within a minute. It replaces
"squint at it and guess", not a forensics lab.

---

## Run it

You need **Python 3.10+**. No GPU: everything below runs on a laptop CPU.

```bash
# 1. install
cd server
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
pip install -r ../ctf_pretrained/requirements.txt

# 2. start the detector API  (leave this running)
python -m uvicorn app:app --port 8000

# 3. in a second terminal, serve the website
cd site
python -m http.server 5500
```

Open **<http://127.0.0.1:5500/>**, drop in an MP4, press **Analyze Video**.

> **Serve the site — do not double-click `index.html`.** A `file://` page has no
> secure context, so the receipt verifier cannot hash a file, and some browsers
> block the upload outright.

### Two things that will confuse you if nobody says them

**The first analysis is slow.** It downloads and loads the model — expect around
5 minutes, and 10–45 seconds for every analysis after that.

**A page served from localhost always uses your local detector.**
[site/js/config.js](site/js/config.js) points at the deployed Azure backend for
the public site, but skips it entirely when the page itself is local — so
nothing needs editing to run the whole thing on your own machine, and a clone
can never silently talk to the deployed backend while you think you are testing
your own server.

If the server is unreachable, the site says so in an orange banner and any
result it then produces is labelled **simulated** — invented numbers for working
on the interface, never a measurement of your video.

---

## Check that it works

Two scripts make claims about the server that you can re-run rather than take on
trust. Both need the server running, and both write their output to `reports/`.

```bash
python scripts/check_determinism.py     # same clip twice -> same answer?
python scripts/check_robustness.py      # malformed and hostile input
```

**Determinism.** Every clip is analysed twice and the verdict, clip score,
flagged-frame count, mean score and every timeline bar are compared. Last run:
**8 clips × 2 runs, all identical** — including one known deepfake, which
reproduced at 0.9405 both times. That is a repeatability check on those eight
clips; it says nothing about accuracy on any of them.

**Robustness.** Twelve deliberately awkward inputs — 0 bytes, a PDF renamed
`.mp4`, a truncated file, a 201 MB file, an emoji filename, `../../etc/passwd.mp4`,
a missing form field — each declaring which HTTP statuses count as correct
handling. Passing is *not* "nothing failed": a renamed PDF **should** be
rejected, and the script fails if it is accepted. The last case re-runs the
happy path afterwards to prove the server did not get wedged.

That script earned its place on its first run: it caught the server returning a
full verdict — "not enough face visible to judge", with a confidence score — for
a 65-byte PDF renamed to `.mp4`. Nothing was ever decoded, so that verdict was a
statement about a video that did not exist. The server now rejects undecodable
files with a 400 (`_probe_video` in [server/app.py](server/app.py)).

---

## Signed verdict receipts

A screenshot of a detector's output proves nothing: anyone can edit the number
in an image editor, or pair a genuine report with a different video. So every
analysis comes back with a receipt, signed with an **Ed25519** key:

```json
{
  "payload": {
    "fileSha256": "3248d61c…a88e",
    "verdict": "authentic",
    "confidence": "0.0148",
    "modelVersion": "ViT-DFv2",
    "analysedAt": "2026-09-11T16:24:39+00:00",
    "attests": "This detector produced this verdict for the file with this SHA-256…"
  },
  "algorithm": "Ed25519",
  "keyId": "0f8117b18b430402",
  "signature": "2JH9xfD9iwJpU2nb…"
}
```

**Save signed receipt** on the report downloads it. **/verify.html** checks it —
in your browser, with WebCrypto, against the public key from `/v1/public-key`.
Load the video alongside it and the page hashes the file locally and confirms
the receipt refers to *that* clip. There is a **"alter a number, then verify"**
button that tampers with the payload and re-runs the check, so you can watch it
fail.

**What a receipt proves:** this detector produced this verdict, for this exact
file, at this time, and nobody edited the numbers afterwards.
**What it does not prove:** that the video is genuine, or that the verdict is
right. A signed wrong answer is still a wrong answer.

Set a durable key, or every restart invalidates previously issued receipts:

```bash
python server/receipt.py keygen          # prints REELREAL_SIGNING_KEY=...
set REELREAL_SIGNING_KEY=...             # Windows
export REELREAL_SIGNING_KEY=...          # macOS / Linux
```

Without one the server signs with a throwaway key, says so in the startup log,
and reports `ephemeral: true` from `/v1/public-key` — which the verify page
shows on screen. Keep the key out of git.

---

## Sample clips

The "try a sample clip" strip under the uploader is built from clips you already
hold:

```bash
python scripts/make_samples.py --clip id3_0006.mp4 "Real (Celeb-DF)" "Well lit"
python scripts/make_samples.py --list
```

Celeb-DF ships its clips as MPEG-4 Part 2, which OpenCV decodes happily and no
browser can play — the symptom is a sample that analyses correctly next to a
blank video player. Anything in a codec browsers cannot decode is therefore
re-encoded to H.264 on the way in. That needs ffmpeg; if you do not have one,
`pip install imageio-ffmpeg` supplies a bundled binary, and the script tells you
so rather than failing silently. Nothing else in the project uses it — the
server and the detector do not.

Re-encoding is the operation this detector is weakest against, so after adding a
converted clip, run it through the site once and confirm the verdict still looks
right. On the four clips installed here it cost one fake 0.91 → 0.80, still
comfortably flagged.

**The deployed site uses a different, committed pair.** `site/assets/demo/`
holds two clips cleared for publication — a deepfake the detector flags at
**0.9405 with 28 of 30 frames flagged**, and its genuine counterpart at 0.1055.
The site prefers the generated manifest when one exists and falls back to this
pair, so a local demo shows your own dataset clips while the public site and a
fresh clone still have something to press.

The front-page hero deepfake is deliberately *not* used for this: the detector
misses it (see STATUS.md), and a one-click demo whose headline button answers
"No signs of manipulation" would misrepresent the tool to the first person who
tries it.

No clip is committed to this repository. Celeb-DF and FaceForensics++ are
licensed for research use and redistributing them would breach the agreement
signed to obtain them, so `.gitignore` excludes every video extension and the
generated `site/assets/samples/` folder with them. A fresh clone simply shows no
strip.

Buttons are labelled with the clip's **dataset ground truth**, never with the
verdict the detector is expected to produce. Pressing one runs the real analysis
through the real upload path, so a disagreement between label and verdict is
visible on screen instead of hidden.

---

## How it decides

1. **Sample frames** across the clip — a fixed number, not every frame.
2. **Find the face** in each sampled frame and crop it.
3. **Score each crop** with a ViT fine-tuned on Celeb-DF, giving one probability
   per frame.
4. **Aggregate**: a sustained run of consecutive flagged frames is treated
   differently from the same count scattered at random, which is more often
   noise. A logistic calibrator maps the clip feature to a probability.
5. **Refuse to answer** when too few frames contained a usable face —
   *insufficient evidence* is a third verdict, not a quiet "clean".

The report distinguishes three states and the interface is built to keep them
apart: **measured and something was found**, **measured and nothing was found**
(green panel), and **not checked** (a dash). Signals with no detector behind
them — blink rate, lip-sync, compression history, C2PA provenance — are left out
of the website's panel entirely rather than shown as placeholders, because a
placeholder reads as a measured negative.

---

## Accuracy, and where it fails

Measured on **Celeb-DF v2**, split by identity so no person appears in both
halves. Held-out: **226 clips, 56 unseen identities**.

| | Off-the-shelf ViT | Fine-tuned |
|---|---|---|
| ROC-AUC | 0.4899 | **0.9814** |
| Accuracy | — | 0.9248 (majority baseline 0.5973) |
| Recall @ 5% FPR | — | 0.9121 |

Cross-dataset on **FaceForensics++** (265 clips, four manipulation methods the
model has never seen): ROC-AUC **0.7929** clean, falling to **0.6301** under
simulated social-media re-upload. **Fake recall under re-compression is 0.0957** —
roughly one manipulated video in ten is caught once a clip has been through a
platform's encoder.

Read that as the honest headline: the ranking generalises, the threshold does
not, and re-uploaded footage is where this tool is weakest. Full working,
including a recorded negative result where augmentation *hurt* cross-dataset
performance, is in **[STATUS.md](STATUS.md)** — read it before quoting any
number from here.

---

## Layout

| Path | What it is |
|---|---|
| [site/](site/) | The website. Plain HTML/CSS/JS, no build step. |
| [site/verify.html](site/verify.html) | Offline receipt verifier. |
| [extension/](extension/) | Chrome extension, same detector and result format. |
| [server/](server/) | FastAPI middle layer: upload, analyse, delete. |
| [server/receipt.py](server/receipt.py) | Ed25519 signing and verification. |
| [server/adapter.py](server/adapter.py) | Pipeline output → the frontend's result shape. |
| [ctf_pretrained/](ctf_pretrained/) | The detector: training, evaluation, inference. |
| [scripts/](scripts/) | Determinism, robustness and sample-building tools. |
| [deploy/container/](deploy/container/) | Docker image that serves the site and API together. |
| [STATUS.md](STATUS.md) | Every measured number and the reasoning behind each decision. |

---

## Privacy

The uploaded video is written to a temp file, analysed, and deleted in the same
request — `finally: path.unlink()` in [server/app.py](server/app.py). Nothing is
retained between requests and no video is logged. The extension's handoff to the
website passes the result in a URL **fragment**, which browsers never transmit to
a server, so an imported report stays as local as the analysis was.

Recent checks in the sidebar are stored in your browser's `localStorage` —
filenames, verdicts and receipts only, never video.
