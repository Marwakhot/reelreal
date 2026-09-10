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
| Verdicts shown to the user | **Fixed** — see below. |
| Confidence calibration | Never run. Blocked on the model owner. |

Measured speed on a normal laptop, no graphics card: **first analysis ~5 minutes**
(it downloads the model), **every one after that ~11–45 seconds** depending on
video length and resolution.

---

## ✅ Fixed — the verdict was backwards, and then stuck

**Fixed in `ctf_pretrained/infer_pipeline.py::_score()`.**

The model returns two numbers — the chance the video is real, and the chance it is
fake. Its own configuration says:

```
id2label: {0: 'Realism', 1: 'Deepfake'}
```

So slot 0 means *real*. The code read slot 0 and named it `fake_prob`, so every
verdict came out as its own opposite.

A second edit had been layered on top of that to compensate — a flat `- 0.25`
subtracted from every frame score, commented "prevent over-sensitivity on real
videos". Together the two turned a wrong answer into no answer at all:

- the offset capped any frame score at **0.75**,
- a frame is flagged only above `high_thresh`, which `VideoAnalyzer.load()` sets
  to **0.90**,
- so `n_flagged` was **always 0**, and the clip score always **0.000**.

Both routes to a `SYNTHETIC` verdict read those two numbers, which means **every
video — genuine or manipulated — returned "NO MANIPULATION DETECTED"**. The
detector could not raise a flag under any input. Nothing crashed and the report
rendered cleanly, which is why it survived testing: a real video gave the right
answer for entirely the wrong reason.

Both edits are reverted. `_score()` now returns `probs[1]` — P(deepfake) — with
no offset. Verified against the decision logic:

| Input | Verdict | Clip score | Frames flagged |
|---|---|---|---|
| Genuine video | NO MANIPULATION DETECTED | 0.000 | 0 / 30 |
| Deepfake | SYNTHETIC | 0.833 | 25 / 30 |

Do **not** re-introduce a compensating offset or flip the index back in
`server/adapter.py`. Two flips cancel out and the bug returns silently with
nothing on screen to show it.

### Evidence the model itself is sound

Three Wikitongues documentary clips (CC BY-SA) of real people talking to camera,
with a face visible in 97–98% of sampled frames, scored 0.865, 0.899 and 0.924
**real**. The model got all three right, confidently. Only the label on the way
out had been wrong.

### Also fixed — the report contradicted itself

`aggregate.apply_clip_calibrator()` computed the displayed clip score at
`config.DEFAULT_HIGH_THRESH` (0.80) while the verdict rule and the evidence rows
counted flagged frames at the analyzer's own `high_thresh` (0.90). The same
report could therefore print a non-zero confidence beside "0 frames flagged".
The caller's threshold is now passed through, which is what the function's
docstring already claimed it did.

---

## ⚠️ Open bug — the score is uncalibrated

Calibration has never been run, so `clip_calibrator` is `None`. Consequences:

1. **The clip score is a flagged-frame fraction, not a probability.** The older
   max-of-frames fallback is gone (it made long videos steadily more likely to be
   flagged, because a maximum only grows with more frames sampled). A fraction is
   bounded regardless of frame count, but it is still not calibrated.
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
