# REEL/REAL — where the project stands

_Last updated: 11 September 2026_

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

## Results

Measured on **Celeb-DF v2**, splitting by identity so no person appears in both
halves. Held-out: 226 clips across 56 identities, none seen during training.

| | Off-the-shelf ViT | Fine-tuned on Celeb-DF |
|---|---|---|
| ROC-AUC | **0.4899** | **0.9814** |
| Accuracy | — | 0.9248 (majority baseline 0.5973) |
| F1 (fake) | — | 0.9017 |
| F1 (real) | — | 0.9391 |
| Recall @ 5% FPR | — | 0.9121 |
| ECE | — | 0.0655 |

Confusion matrix, fine-tuned: `tn=131 fp=4 fn=13 tp=78`.

**The off-the-shelf number is the interesting one.** `prithivMLmods/Deep-Fake-Detector-v2-Model`
scores at chance on Celeb-DF — 0.4899, with mean, max, p90 and flagged-fraction
all landing between 0.46 and 0.53 (n=300, so the standard error is ~0.034). The
weights carry no signal for this manipulation family at all. Fine-tuning on 125
Celeb-DF identities fixed it.

### Cross-dataset: FaceForensics++

265 FF++ clips (115 fake across Deepfakes, Face2Face, FaceSwap and
NeuralTextures; 150 real), **none of which the model trained on, using four
manipulation methods it has never seen.** Degradation is applied to the crops to
simulate re-uploaded media.

| Condition | ROC-AUC | Accuracy | Simulates |
|---|---|---|---|
| clean | **0.7929** | 0.6792 | pristine dataset file |
| jpeg_q50 | 0.7602 | 0.6113 | moderate recompression |
| downscale_0.5 | 0.6843 | 0.6755 | halved resolution |
| social_recompress | 0.6301 | 0.6000 | a re-uploaded reel |
| heavy | 0.6148 | 0.5925 | worst case |

Majority baseline throughout: 0.5660.

Two things this shows.

**The ranking transfers; the threshold does not.** ROC-AUC 0.7929 on unseen
manipulation families is real generalisation. But at the 0.50 decision threshold
inherited from Celeb-DF, FF++ fake recall is only 0.2696 at precision 0.9688
(`tn=149 fp=1 fn=84 tp=31`) — the model is far too conservative on unfamiliar
data. `recall @5% FPR 0.3913` confirms the signal is there to be traded for. A
deployment against a new manipulation family needs its threshold re-tuned on
labelled data from it; reusing this one silently costs most of the recall.

**Compression hurts, and predictably.** 0.7929 clean to 0.6301 under a realistic
re-upload. **Fake recall under `social_recompress` is 0.0957** — roughly one
manipulated video in ten is caught on re-uploaded footage. The obvious fix was
tried and made things worse; see the next section.

### Recorded negative result: augmentation did not help

The compression drop above has a textbook fix. `preprocess.py` has carried
`RandomDownscale` and `RandomJPEG` since the beginning, with a docstring saying
a detector trained only on clean crops learns artifacts that recompression
destroys. Wiring them into training looked like the single largest available
win. **It was tried, and it lost on every cross-dataset condition.**

`finetune_clips.build_augment()` applies downscale-and-restore (scale 0.35–1.0)
then JPEG re-encode (quality 30–95), each at p=0.5, to training crops only —
never to held-out crops, which would measure a different question. Everything
else was held fixed: same clips, same identity split, same seed, same epochs,
same learning rate. One variable.

**In-dataset (Celeb-DF held-out identities, 226 clips):**

| | Clean-crop training | Augmented training |
|---|---|---|
| ROC-AUC | 0.9814 | **0.9832** |
| Accuracy | 0.9248 | 0.9248 |
| ECE | 0.0655 | **0.0599** |
| Confusion | `tn=131 fp=4 fn=13 tp=78` | `tn=129 fp=6 fn=11 tp=80` |

**Cross-dataset (FaceForensics++, 265 clips, four unseen manipulation methods):**

| Condition | Clean-crop training | Augmented training | Δ |
|---|---|---|---|
| clean | **0.7929** | 0.7502 | −0.043 |
| jpeg_q50 | **0.7602** | 0.7261 | −0.034 |
| downscale_0.5 | **0.6843** | 0.6582 | −0.026 |
| social_recompress | **0.6301** | 0.6030 | −0.027 |
| heavy | **0.6148** | 0.5930 | −0.022 |

**The two tables together rule out the easy explanation.** If augmentation had
merely made the task harder and left the model underfitted, the in-dataset score
would have fallen too, and more epochs would be the answer. It did not fall — it
rose slightly, on both ROC-AUC and calibration error. The model had no trouble
learning; it learned something that transfers less well.

The likely mechanism: degrading Celeb-DF crops teaches robustness to *noise*,
not robustness to *a different manipulation method*. The model got better at
recognising Celeb-DF's particular artifacts through compression, which is not
the axis FF++ varies along. Training longer would deepen that, not correct it.

**Decision: the clean-crop checkpoint stays deployed.** It is equal in-dataset
and better on all five cross-dataset conditions. The augmented run is kept as a
measurement, not as a candidate.

This is worth stating plainly because it contradicts standard advice. "Augment
for robustness" is the default recommendation, it was the recommendation made in
this project, and on this task it cost cross-dataset generalisation. Reproduce
either arm with `finetune_clips.py`; `--no-augment` selects the deployed one,
and the `augmented` flag is recorded in `finetune_metrics.json` so the two runs
cannot be confused after the fact.

What would actually address the compression drop is training on more than one
manipulation family — FF++ clips in the training set alongside Celeb-DF — rather
than degrading a single family harder. That is untested here.

### What may and may not be claimed

0.9814 is **in-dataset**: trained on Celeb-DF, tested on held-out Celeb-DF
identities. The honest summary of all three numbers is that the detector works
well on the manipulation family it was trained on (0.98), retains useful but
degraded discrimination on unseen families (0.79), and weakens further on
recompressed media (0.63). None of those is a general accuracy figure, and the
0.4899 baseline is a standing reminder of how far a model can fall on a
distribution it was not trained for.

Reproduce with `evaluate_clips.py` (baseline), `finetune_clips.py` (fine-tune and
evaluate in-dataset) and `evaluate_external.py` (cross-dataset and degradation).
All three split by identity; see below for why that matters.

---

## Grad-CAM — implemented, and it says something unexpected

`_explain()` in `infer_pipeline.py` used to return `{"region": None}` from a
stub. It now runs Grad-CAM for real, on the most suspicious crop of the clip.

**It needed a fix to work at all on this model.** Grad-CAM weights each channel
by the mean of its gradient over the two spatial axes, which assumes
convolutional activations shaped `(B, C, H, W)`. A ViT block emits
`(B, 197, 768)` — a token sequence with no spatial axes — so the same arithmetic
produces a meaningless map rather than an error. `gradcam.vit_reshape` drops the
class token and folds the 196 patch tokens back onto the 14×14 grid they came
from, giving `(B, 768, 14, 14)`. The hooked layer is `layernorm_before` of the
final encoder block, and the backward pass targets class index 1 — the same
index `_score()` reads, so the map explains the number that was actually scored.

### The dominance rule was replaced, because the old one could never fire

`region_report()` used to name a region when it held ≥40% of total CAM mass.
Measured on real crops, the three landmark discs together cover only **~14–19%
of the image each, ~41% in total**. A 40%-of-total cut is therefore
unreachable by construction: it was not a conservative threshold, it was
"never name a region" written in a way that looked like a threshold.

The rule is now **enrichment** — a region's share of CAM mass divided by the
share of image area its disc covers. Attention spread evenly scores exactly
1.0 for every region regardless of disc size, so the null is principled rather
than tuned to a backbone's feature-map resolution. The cut is 1.5×.

### What it measures on real footage

Six Celeb-DF clips, one crop each, off-the-shelf checkpoint:

| Clip | eyes | nose/cheeks | mouth/jaw |
|---|---|---|---|
| id3_0006 | 0.13× | 0.21× | 0.48× |
| id3_0007 | 0.54× | 0.72× | 0.87× |
| id3_0008 | 0.61× | 0.43× | 0.42× |
| id3_0009 | 0.42× | 0.39× | 0.40× |
| id4_0000 | 0.09× | 0.19× | 0.46× |
| id4_0001 | 0.12× | 0.27× | 0.52× |

**Every value is below 1.0.** Attention is not merely un-concentrated on the
facial landmarks — it is systematically *away* from them, out towards the crop's
periphery: hair, jaw outline, background. Not one clip comes close to naming a
region, and that is the correct output rather than a failure.

**Caveat, and it is a real one.** These numbers come from the off-the-shelf
checkpoint, which `evaluate_clips.py` measured at ROC-AUC 0.4899 — chance. The
attention of a model that discriminates nothing explains nothing, so this table
says where a *broken* detector looked. The fine-tuned checkpoint is not in the
repository (it is gitignored, and lives in Colab and in the deployed image), so
the same table has not been produced for the model actually serving traffic.
**Re-running this on the fine-tuned checkpoint is the outstanding piece**, and
the 1.5× cut is unvalidated until it is. It is one loop over
`VideoAnalyzer._explain()` on held-out clips.

Whatever it shows, the interface does not over-claim: a named region reads
"Attention concentrated on X (N× an even spread)", anything below the cut reads
"Spread out, no single area stood out", and only a failed Grad-CAM shows a dash.
All three are captioned as a description of the model's attention, never as
evidence that the video was edited — which is why the row's severity is capped
at "warn" even when a region is named.

---

## Thresholds — measured, and weaker than they look

The decision thresholds were swept on **12 Celeb-DF-v2 clips** (6 Celeb-synthesis,
6 Celeb-real) scored by this checkpoint. Ranking clips by each candidate statistic:

| Clip statistic | AUC |
|---|---|
| **Mean per-frame probability** | **0.833** |
| Flagged-frame fraction (best per-frame threshold, 0.60) | 0.806 |
| Maximum per-frame probability | **0.500** |

The maximum carries no signal whatsoever. That matters, because two earlier
designs were driven by it — first the max as the clip score, then a 0.90
per-frame threshold plus an `n_flagged >= 2` verdict shortcut. Both were reading
the one statistic that does not separate.

So the clip score is now the **mean**, with `decision_thresh = 0.40` and
`high_thresh = 0.60`. `high_thresh` now only marks frames for the evidence panel
and the timeline; it no longer decides anything.

**Read the accuracy honestly.** 10 of those 12 clips land on the correct side of
0.40, but the threshold was chosen *on those same 12 clips*. That is an in-sample
fit, not a validation score, and 12 clips is a very small sample. Do not quote it
as an accuracy figure.

Two failures are worth naming:

- `id0_id20_0001` — a deepfake with mean 0.38, just under the line. Missed.
- `id3_0007` — a **genuine** video with mean 0.63, scoring higher than five of the
  six deepfakes. No threshold fixes this one; the model is confidently wrong about
  it. Expect false positives on real footage of this kind.

## ⚠️ Still open — the score is uncalibrated

Calibration has never been run, so `clip_calibrator` is `None`. Consequences:

1. **The clip score is a mean model output, not a probability.** A clip scoring
   0.55 is not "55% likely to be fake"; it is only ranked above one scoring 0.30.
2. **There is no confidence range.** The report prints `Uncalibrated` rather than
   a band, because no interval exists.
3. **The mean dilutes partial manipulation.** A lip-sync edit that leaves most
   frames untouched will score low. The clips measured above are whole-face swaps,
   where every frame is manipulated.

### How to fix it: `evaluate_clips.py`

`fit_clip_calibration.py` assumes the *trained* pipeline — a `model_best.pt`
checkpoint and `splits.json`. This project has neither; it scores whole videos
with a pretrained ViT. `ctf_pretrained/evaluate_clips.py` is the same procedure
for that setup:

```bash
python evaluate_clips.py --root /content/celeb-df --out-dir reports --per-class 150
```

It scans Celeb-DF folders, scores every clip (caching per-frame probabilities so
re-tuning is instant), **splits by identity rather than by clip**, sweeps the
per-frame threshold on the calibration split only, fits the calibrator, and
reports accuracy, precision/recall, ROC-AUC, a confusion matrix and a reliability
diagram on held-out identities the calibrator never saw.

Splitting by identity matters. Celeb-DF names clips `id{N}_...`, and a clip-level
split puts the same face in both halves — the model then recognises the person
rather than the manipulation, and every metric is inflated.

Drop the resulting `clip_calibration.json` next to `infer_pipeline.py` and
`VideoAnalyzer.load()` picks it up automatically; no code change is needed to
deploy it. Once loaded, the clip score is a genuine calibrated probability and
the interface stops saying `Uncalibrated`.

**The loader refuses a calibrator whose held-out ROC-AUC is below 0.5.** A
previous calibration run in this project scored **0.357** because it was fitted
while `_score()` still read the wrong class index. A below-chance calibrator does
not merely underperform — it confidently inverts every verdict while the
interface reports itself as calibrated. That is strictly worse than no
calibration, so it fails loudly instead.

Until a calibrator is fitted, treat the number as a *ranking*, not a probability.

Fixing it needs, on the model side: a trained checkpoint (`model_best.pt`), then
`fit_clip_calibration.py --splits splits.json`, then loading that checkpoint in
`VideoAnalyzer.load` instead of the hard-coded `dummy_ckpt`. It needs the
FF++ / Celeb-DF datasets and Colab.

---

## Honesty rules built into the report

These were deliberate and should not be "tidied away":

- **The site report shows only rows with a measurement behind them.** Blink
  rate, compression trace, lip-sync alignment and C2PA provenance have no
  detector behind them at all — they were invented for an early mock — so the
  website's "What we measured" panel omits them rather than printing four
  placeholders. The extension still renders the full six-row array, where they
  read "Not checked yet".
- **The attention row distinguishes three states, not two.** `_explain()` runs
  Grad-CAM over the final transformer block and `region_report()` names a facial
  region only when one area holds at least 40% of the attention mass. Below that
  the report says "spread across the face, no single area" — which is a result,
  not a missing value — and only a failed or skipped Grad-CAM shows a dash. The
  row is captioned as a description of where the model looked, never as evidence
  of editing, and its severity is capped at "warn" for the same reason.
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

---

## Deploying it

Two pieces, two hosts. The site is static and goes on **Vercel**; the detector
needs Python, PyTorch and about 1 GB of memory, so it goes in a container on
**Azure Container Apps**.

Why not the obvious free options: Hugging Face moved Docker Spaces behind a PRO
subscription during 2026 and only Static Spaces remain free, which cannot run
Python. Fly.io dropped its free tier. Render and Koyeb cap free instances at
512 MB, and this needs roughly double that. Colab is not an option at any price —
its terms prohibit "web service offering not related to interactive compute" on
paid plans as well as free, and tunnelling out of it risks the Google account.

Azure Container Apps has a perpetual free monthly grant — 180,000 vCPU-seconds,
360,000 GiB-seconds, 2 million requests — which a demo fits inside.

### Backend

```bash
az login
az group create --name reelreal-rg --location eastus
az containerapp up \
  --name reelreal --resource-group reelreal-rg \
  --source deploy/container \
  --ingress external --target-port 7860 \
  --cpu 1.0 --memory 2.0Gi
```

`--source` builds the image in the cloud, so Docker is not needed locally. The
first build takes 15–25 minutes: it installs CPU-only PyTorch and bakes the
340 MB model into the image.

Then set the CORS origin, which otherwise defaults to `*`:

```bash
az containerapp update --name reelreal --resource-group reelreal-rg \
  --set-env-vars ALLOWED_ORIGINS=https://<your-site>.vercel.app
```

### Frontend

Set the backend address in `site/js/config.js` — the one line that needs editing
when the backend moves — then deploy `site/` to Vercel with **Root Directory:
`site`** and no build command.

### Cold starts

The image is ~3 GB, so a container starting from cold takes a minute or two.
Scaled to zero, that lands on the first visitor. Keep one replica warm while the
project is being demonstrated, and scale back to zero afterwards:

```bash
# before a demo
az containerapp update --name reelreal --resource-group reelreal-rg --min-replicas 1
# after
az containerapp update --name reelreal --resource-group reelreal-rg --min-replicas 0
```

A warm idle replica bills at Container Apps' idle rate, a few dollars a month.
At zero replicas it costs nothing at all.

## Before this is shown to anyone outside the team

- CORS defaults to `*` when `ALLOWED_ORIGINS` is unset, which lets any website on
  the internet submit videos to the server. Set it to the real site origin.
- `host_permissions` in `extension/manifest.json` → replace the localhost entries
  with the deployed `https://` origin. Do not use `https://*/*`; Chrome will warn
  users that the extension can read data on every site.
- The site says videos are sent over an encrypted connection. That is only true
  once it is actually served over HTTPS.
- Uploads are deleted immediately and nothing is logged. If that ever changes, the
  copy on the upload panel has to change with it.
