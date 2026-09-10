# REEL/REAL — where the project stands

_Last updated: 10 September 2026_

## What this is

A deepfake video detector with two front ends — a website (`site/`) and a Chrome
extension (`extension/`) — sitting on top of a Python detection model.

## The problem that was solved

The front end is JavaScript running in a browser. The detector is Python.
**A browser cannot run Python**, so the two halves could not talk to each other.
Nobody on the team owned that middle layer.

`server/` is that middle layer. It is a small HTTP server:

```
browser  ──sends video──▶  server/app.py  ──▶  the Python detector
browser  ◀──sends result──  server/app.py  ◀──  (raw numbers)
```

Two files do the work:

- **`app.py`** — receives the upload, runs the detector, deletes the video.
- **`adapter.py`** — translates the detector's raw output into the shape the
  front end already understood. This is the only place the two vocabularies meet,
  which is what kept the UI code from having to know anything about the model.

Nothing inside the detector folder (`ctf_pretrained/`) was modified.

## Status

| Piece | State |
|---|---|
| Website UI | Working. Unchanged apart from honesty fixes listed below. |
| Chrome extension | Working. Same detector, same result format. |
| `server/` middle layer | **Built and verified end to end.** |
| Python detector runs locally | Yes — CPU only, no GPU needed. |
| Verdicts shown to the user | **Currently inverted — see the bug below.** |
| Confidence calibration | Never run. Blocked on the model owner. |

Measured speed on a normal laptop, no graphics card: **first analysis ~5 minutes**
(it downloads the model), **every one after that ~11–45 seconds** depending on
video length and resolution.

---

## ⚠️ Open bug 1 — the verdict is backwards

**This is the one that matters. It has not been fixed.**

The model returns two numbers — the chance the video is real, and the chance it is
fake. Its own configuration says:

```
id2label: {0: 'Realism', 1: 'Deepfake'}
```

So slot 0 means *real*. But `ctf_pretrained/infer_pipeline.py` reads slot 0 and
names it `fake_prob`:

```python
# Class 0 is Fake, so grab index 0 probability directly
fake_prob = torch.softmax(logits, dim=1)[0, 0].item()
```

The two slots are read the wrong way round, so **every verdict comes out as its
own opposite.** Genuine videos are reported as deepfakes; deepfakes would be
reported as genuine.

### Evidence

Three Wikitongues documentary clips (CC BY-SA) of real people talking to camera,
with a face visible in 97–98% of sampled frames:

| Clip | Reported | What the model actually meant |
|---|---|---|
| Carolin (Bavarian) | 0.865 fake | 0.865 **real** |
| Dang (Thai) | 0.899 fake | 0.899 **real** |
| Ying (Henan Chinese) | 0.924 fake | 0.924 **real** |

Read the right-hand column: the model got all three correct, confidently. It is
accurate. Only the label on the way out is wrong.

### The fix

One character, in `infer_pipeline.py`:

```python
# The checkpoint reports id2label {0: 'Realism', 1: 'Deepfake'},
# so the Deepfake probability is index 1, not index 0.
fake_prob = torch.softmax(logits, dim=1)[0, 1].item()
```

It should be fixed **there**, in the pipeline, and not worked around inside
`server/adapter.py`. If it were patched in both places the two flips would cancel
out and the bug would silently return with nothing on screen to indicate it.

### Why it went unnoticed

Nothing crashes. There is no error and no warning. The report renders cleanly and
confidently and is completely inverted. You also cannot catch it by testing a
deepfake — it will say "fake", for the wrong reason. It only shows up when you test
a video you *know* is genuine.

---

## ⚠️ Open bug 2 — the score is uncalibrated

Calibration has never been run, so `clip_calibrator` is `None`. Two consequences:

1. **The clip score is the single highest-scoring frame.** In a 106-frame video,
   one blurry frame or one half-turned face decides the whole verdict. Long videos
   are therefore biased toward being flagged.
2. **There is no confidence range.** The report prints `Uncalibrated` rather than
   a band, because no interval exists.

Until this is done, treat the number as a *ranking*, not a probability.

Fixing it needs, on the model side: a trained checkpoint (`model_best.pt`), then
`fit_clip_calibration.py --splits splits.json`, then loading that checkpoint in
`VideoAnalyzer.load` instead of the hard-coded `dummy_ckpt`. It needs the
FF++ / Celeb-DF datasets and Colab.

---

## Honesty rules built into the report

These were deliberate and should not be "tidied away":

- **Four evidence rows read "Not measured by this model"** — blink rate,
  compression trace, lip-sync alignment, and C2PA provenance. There is no detector
  behind any of them; they were invented for an early mock. They are greyed and
  italic so they can never be mistaken for findings.
- **"Face boundary blending"** will fill in by itself once `_explain()` in
  `infer_pipeline.py` is un-stubbed — it currently returns `{"region": None}`.
  The adapter already handles the populated case.
- **The confidence band prints "Uncalibrated"** rather than a made-up range.
- **A mocked result is stamped `SIMULATED RESULT`.** The mock only ever appears
  when the server cannot be reached at all. If the server answers with an error,
  the error is shown — a real failure is never replaced by an invented result.
- **"Insufficient evidence" is not styled as a pass.** If a human face is visible
  in fewer than one third of sampled frames, the tool refuses to judge. This is why
  faceless clips, screen recordings and animal videos return no verdict — the model
  only knows human faces and has genuinely learned nothing about anything else.

---

## Running it

See `server/README.md` for full setup. Short version, two terminals:

```bash
cd server
PIPELINE_DIR=/path/to/ctf_pretrained .venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

```bash
python -m http.server 8080
```

Then open <http://127.0.0.1:8080/site/index.html>.

The `server/.venv` folder is **not** included in this archive — it is about 1.3 GB
of installable libraries. Recreate it with the instructions in `server/README.md`.

## Before this is shown to anyone outside the team

- `allow_origins=["*"]` in `app.py` → replace with the real site origin.
- `host_permissions` in `extension/manifest.json` → replace the localhost entries
  with the deployed `https://` origin. Do not use `https://*/*`; Chrome will warn
  users that the extension can read data on every site.
- The site says videos are sent over an encrypted connection. That is only true
  once it is actually served over HTTPS.
- Uploads are deleted immediately and nothing is logged. If that ever changes, the
  copy on the upload panel has to change with it.
