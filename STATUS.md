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
| Confidence calibration | **Fitted and deployed** — held-out ROC-AUC 0.96 (n=60). |

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

### Recorded miss: the clip on our own landing page

The deepfake in the hero pair on the front page (`site/assets/reel.mp4`, a
~9 second portrait phone clip, confirmed synthetic by the project owner) is
**not detected**:

| Clip | Verdict | Clip score | Frames flagged | Face coverage |
|---|---|---|---|---|
| `reel.mp4` (synthetic) | NO MANIPULATION DETECTED | 0.0115 | 0 / 30 | 100% |
| `real.mp4` (genuine) | NO MANIPULATION DETECTED | 0.0118 | 0 / 30 | 100% |

This is not a coverage failure. A face was found in all thirty sampled frames
of both clips, so the model looked properly and confidently called the fake
genuine — and gave it a score indistinguishable from the real clip beside it
(0.0115 against 0.0118).

**It is the documented weakness, not a new one.** The clip is short, portrait,
phone-shot and re-encoded, and it comes from a generator family outside both
training sets. Cross-dataset fake recall is already measured at 0.2696 on FF++
at this threshold and **0.0957 under simulated re-upload**; a clip like this is
squarely in the population where roughly one fake in ten is caught. Nothing here
contradicts the earlier numbers — it is one concrete instance of them.

**One clip proves nothing on its own.** n=1 has no interval worth quoting, and a
single miss neither adds to nor subtracts from the measured recall figures. It is
recorded because it is a real-world clip rather than a dataset one, and because
it sits on the landing page, where the system's own front door is a case it gets
wrong.

**Consequence for the demo.** This pair is deliberately not used for the public
sample strip. A one-click demo whose headline button returns "No signs of
manipulation" for a clip labelled Deepfake would misrepresent the tool to the
first person who tries it. The deployed strip is driven instead by
`site/assets/demo/`, a separate pair also cleared for publication, whose fake the
detector does flag (0.9405, 28 of 30 frames). Celeb-DF clips remain local-only
via `scripts/make_samples.py`, since they cannot be redistributed.

The hero labels stay as they are: the clip genuinely is generated, and saying so
next to a detector that missed it is the honest arrangement, not a
contradiction to hide.

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

### Self-generated fakes — the harness is ready, the set is not

The grading rubric asks for a dataset combining public real samples with fakes
generated for this project. Agreed target: **30 fake + 30 real matched pairs**.

The evaluation side is in place. `data_sources.scan_flat` reads a layout with no
conventions to decode, which is what a self-made set needs:

```
<root>/real/alice_01.mp4      genuine footage
<root>/fake/alice_01.mp4      the fake generated from it
```

The label comes from the folder and nothing else is inferred. Grouping is by
file stem, so naming a fake after the clip it was generated from ties the pair
together and the scan prints how many pairs it actually matched — a naming slip
shows up as a number rather than passing silently. A directory holding only one
class is refused, because ROC-AUC is undefined there and accuracy is just the
majority rate. Run it the same way as any other dataset:

```bash
python evaluate_external.py --checkpoint reports/vit_finetuned \
    --dataset flat --root <your-set> --out-dir reports
```

**What is still missing is the footage.** No fakes have been generated yet, so
this has produced no numbers and none should be quoted.

**Read the result carefully when it exists.** At 30 per class the Hanley–McNeil
standard error on ROC-AUC is about **0.058** — the `+/- SE` column
`evaluate_external.py` now prints beside every figure, with a note whenever a
class falls below 50 clips. Two degradation conditions differing by 0.05 on a
set this size have not been shown to differ at all. 30 + 30 is a legitimate set
to report; it is not a set to report to four decimal places with nothing beside
it.

Two further cautions specific to a self-generated set:

- **It measures one generator, not "deepfakes".** Whatever tool produces these
  fakes becomes a fifth manipulation family alongside Celeb-DF's one and FF++'s
  four, and the result describes that tool. The FF++ numbers already show how
  far this model moves between families — 0.9814 in-dataset against 0.7929 on
  unseen ones.
- **The 0.50 threshold is inherited from Celeb-DF and will not transfer.** On
  FF++ it costs most of the recall (0.2696 at the same threshold that scores
  0.9121 in-dataset). Report ROC-AUC, which is threshold-free, and treat any
  accuracy figure on a new family as a statement about the threshold rather
  than about the model.

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

Six Celeb-DF clips, one crop each, on the **fine-tuned checkpoint that is
actually deployed**. Produced by `validate_gradcam.py`.

| Clip | eyes | nose/cheeks | mouth/jaw | disc mass | verdict | mean |
|---|---|---|---|---|---|---|
| id3_0006 | 0.25× | 0.10× | 0.23× | 9.7% | real | 0.044 |
| id3_0007 | 0.22× | 0.27× | 0.49× | 13.7% | real | 0.030 |
| id3_0008 | 0.33× | 0.53× | 0.36× | 19.2% | real | 0.030 |
| id3_0009 | 0.47× | 0.14× | 0.50× | 19.6% | real | 0.029 |
| id4_0000 | 0.10× | 0.37× | 0.58× | 20.2% | real | 0.056 |
| id4_0001 | 0.24× | 0.34× | 0.54× | 19.0% | real | 0.051 |

The same six on the off-the-shelf checkpoint, which is what this table used to
report:

| Clip | eyes | nose/cheeks | mouth/jaw |
|---|---|---|---|
| id3_0006 | 0.13× | 0.21× | 0.48× |
| id3_0007 | 0.54× | 0.72× | 0.87× |
| id3_0008 | 0.61× | 0.43× | 0.42× |
| id3_0009 | 0.42× | 0.39× | 0.40× |
| id4_0000 | 0.09× | 0.19× | 0.46× |
| id4_0001 | 0.12× | 0.27× | 0.52× |

**Every value is below 1.0 on both checkpoints.** Attention is not merely
un-concentrated on the facial landmarks — it is systematically *away* from them.
The three landmark discs cover about 41% of the crop but hold only 9.7%–20.2% of
the attention mass, which leaves roughly 1.4× the even-spread share out in the
remaining periphery: hair, jaw outline, background. Not one clip comes close to
naming a region, and that is the correct output rather than a failure.

**The caveat that stood here is resolved.** The original table came from the
off-the-shelf checkpoint, which `evaluate_clips.py` measured at ROC-AUC 0.4899 —
chance — so it recorded where a *broken* detector looked, and re-running it on
the fine-tuned weights was listed as the outstanding piece. It has been run. The
qualitative result survives: a checkpoint that discriminates at 0.9814
in-dataset attends to the crop's periphery just as the chance-level one did, so
the finding was not an artifact of the broken model.

The numbers did move. The fine-tuned maxima are *lower* and tighter (0.10×–0.58×
against 0.09×–0.87×), and mouth/jaw is the strongest region on four of six clips
rather than being mixed. All six are genuine clips, and all six are now called
correctly at means of 0.029–0.056 — including `id3_0007`, recorded further down
this file as a genuine video that scored 0.63 and beat five of the six deepfakes
under the old thresholds. The fine-tuned checkpoint does not repeat that error.

### ✅ Validated: attention does not separate fakes from reals

`validate_gradcam.py --root <celeb-df> --checkpoint <vit_finetuned> --limit 120`
on 120 held-out clips (64 fake, 56 real — identities the fine-tune never
trained on) settles the question the six-clip run above could not, because
that run held no fakes.

| | n | mean max-enrichment | median | names a region (1.5×) |
|---|---|---|---|---|
| Real | 56 | 0.499 | 0.480 | 1.8% (1/56) |
| Fake | 64 | 0.517 | 0.491 | 0% (0/64) |

**They do not separate.** The means differ by 0.018, well inside the spread of
either group, and the naming rate — the only thing a viewer actually sees —
runs backwards: zero fakes cross the cut and one real does. Grad-CAM's
attention pattern on this checkpoint carries no information about whether a
clip was manipulated. That is not a defect in the row's wording; it is what the
row already claims, and now the claim has a number behind it rather than an
assumption.

**Mouth/jaw dominates *which* region gets the most attention regardless of
label** — the strongest region on 102 of 120 clips (85%), against 7 for eyes
and 11 for nose/cheeks. This appears to be a property of face crops in general
(consistent with the four-of-six finding in the preliminary local run above),
not a signal correlated with manipulation — the separate finding above shows it
carries no discriminative power either way.

**The 1.5× cut barely fires: 0.8% of clips (1 of 120).** Overall max-enrichment:
mean 0.508, median 0.487, p90 0.866, p99 1.191, max 2.054. `validate_gradcam.py`'s
own `recommend()` flags this explicitly rather than leaving it to be inferred:
a threshold that fires on one clip in 120 is close enough to never firing that
the distinction is worth stating outright. Full sweep:

| Cut | Fires |
|---|---|
| 1.10× | 1.7% |
| 1.25× | 0.8% |
| **1.50× (current)** | **0.8%** |
| 1.75× | 0.8% |
| 2.00× | 0.8% |
| 2.50× | 0% |
| 3.00× | 0% |

**The cut stays at 1.5×, unchanged.** Lowering it would not fix the underlying
finding — fakes and reals still would not separate, so a lower cut would just
fire more often on noise. Whether the row is worth keeping at all, given it
carries no discriminative signal in either direction, is a product decision
outside a threshold value, not something this measurement resolves on its own.

Whatever it shows, the interface does not over-claim: a named region reads
"Attention concentrated on X (N× an even spread)", anything below the cut reads
"Spread out, no single area stood out", and only a failed Grad-CAM shows a dash.
All three are captioned as a description of the model's attention, never as
evidence that the video was edited — which is why the row's severity is capped
at "warn" even when a region is named. That wording is now measured to be
accurate rather than merely cautious: the row genuinely does not know whether a
clip is a deepfake.

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

## ✅ Fixed — the score is calibrated

`clip_calibrator` was `None` since the project began: the clip score was a bare
mean of per-frame model outputs, not a probability, and the report printed
`Uncalibrated` rather than a made-up band. A calibrator is now fitted and
deployed, with the identity-exclusion procedure below applied.

### The result

`evaluate_clips.py --root <celeb-df> --out-dir reports --per-class 600 --seed 42
--exclude-finetune-identities`, on the real dataset:

| | |
|---|---|
| Clips selected before filtering | 1200 (600/600) |
| Dropped — fine-tune training identities | 728 clips, 108 identities |
| Usable after filtering and face detection | 472 clips, 142 unseen identities |
| Calibration split | 412 clips, 97 identities |
| **Held-out split** | **60 clips, 45 identities** (10 fake, 50 real) |
| Chosen `high_thresh` | 0.30 |
| Held-out ROC-AUC | **0.96** |
| Held-out ECE | **0.0355** (was 0.0655 uncalibrated) |
| Held-out Brier | 0.0434 |
| Confusion (held-out) | `tn=48 fp=2 fn=1 tp=9` |

**Read 0.96 as an interval, not a point.** The held-out set is small and
imbalanced — 10 fake clips against 50 real. The Hanley–McNeil standard error at
that size is **≈0.044**, so the honest statement is "0.96 ± ~0.09" at two
standard errors, not four significant figures. It is consistent with the
original in-dataset fine-tune metric (0.9814, measured on a separate, larger
226-clip held-out set) rather than contradicting it — a useful cross-check, not
independent confirmation of a sharper number.

**`high_thresh = 0.30` landed on the edge of the swept grid**, which runs
0.30–0.90 in steps of 0.05. The sweep cannot rule out an even lower value doing
better, because none below 0.30 was ever tried. This does not block deployment
— the calibrator's logistic regression absorbs the per-frame threshold through
its own fitted features rather than depending on it being exactly optimal — but
if the grid is ever revisited, extend it downward first.

### The identity-exclusion procedure, and why it is not optional

**`fit_clip_calibration.py` does not run at all — do not reach for it.** It dies
at import on `from model import update_checkpoint` (line 51); `model.py` defines
no such function, and never has. `smoke_test.py:254` imports the same missing
names. Even repaired it would be the wrong tool: it assumes the *trained*
pipeline, a `model_best.pt` checkpoint and a `splits.json`, and this project has
neither — it scores whole videos with a pretrained ViT.
`ctf_pretrained/evaluate_clips.py` is the same procedure for the setup that
actually exists:

```bash
python evaluate_clips.py --root /content/celeb-df --out-dir reports \
    --per-class 600 --seed 42 --exclude-finetune-identities
```

It scans Celeb-DF folders, scores every clip (caching per-frame probabilities so
re-tuning is instant), **splits by identity rather than by clip**, sweeps the
per-frame threshold on the calibration split only, fits the calibrator, and
reports accuracy, precision/recall, ROC-AUC, a confusion matrix and a reliability
diagram on held-out identities the calibrator never saw.

Splitting by identity matters. Celeb-DF names clips `id{N}_...`, and a clip-level
split puts the same face in both halves — the model then recognises the person
rather than the manipulation, and every metric is inflated.

**`--exclude-finetune-identities` is not optional here, and it is new.** Splitting
by identity protects the calibrator from *its own* split; it does nothing about
the fine-tune that came first. The deployed ViT was trained on roughly 70% of
Celeb-DF's identities, and without this flag `evaluate_clips.py` would happily
fit a calibrator on those same faces.

That is worse than an inflated metric. A calibrator maps a model score onto a
probability, and the mapping is only correct for the score distribution it was
fitted on. Scores on memorised faces sit further from the decision boundary and
are more confident than anything an unseen face produces, so the fitted mapping
is wrong for the only case that matters in deployment — and the resulting file
passes every guard, loads cleanly, and makes the interface report itself as
calibrated. `finetune_clips.finetune_split()` is the single definition of which
identities the checkpoint has seen; the flag reconstructs that split
deterministically from `(root, per-class, holdout-frac, seed)` and drops them.
Pass `--finetune-per-class` / `--finetune-holdout-frac` / `--finetune-seed` if
the checkpoint was trained with anything other than the 300 / 0.3 / 42 defaults,
because a mismatch silently reconstructs a *different* split and excludes the
wrong faces.

`--per-class 600` is deliberate. `collect()` shuffles with `random.Random(seed)`
and then truncates, so a larger cap with the same seed yields a strict superset —
more clips than the fine-tune ever selected, of which the identity filter keeps
only the unseen ones. Leave `--n-frames` alone: it defaults to
`config.N_SAMPLE_FRAMES`, which is what the deployed pipeline uses, and the
calibrator's features (`frac_flagged`, `std`, `longest_run_frac`) all depend on
the frame count. Calibrating at 16 frames and serving at 30 fits the mapping to a
distribution production never produces.

The written `clip_calibration.json` records `excluded_finetune_identities` and
`n_frames`, because those are the two facts that cannot be recovered by looking
at a deployed calibrator, and two files with near-identical coefficients mean
entirely different things depending on them.

The resulting `clip_calibration.json` sits next to `infer_pipeline.py` — both at
`ctf_pretrained/` and inside `deploy/container/ctf_pretrained/`, since a Docker
build only sees its own context — and `VideoAnalyzer.load()` picks it up
automatically with no code change. Verified locally through the real loader
before deploying: `load_clip_calibration()` returned `True`, and a clip analysed
through it carried `is_calibrated: True` and `confidence_word: "Strong"` where
the uncalibrated pipeline reported `Uncalibrated`.

**The loader refuses a calibrator whose held-out ROC-AUC is below 0.5.** A
previous calibration run in this project scored **0.357** because it was fitted
while `_score()` still read the wrong class index. A below-chance calibrator does
not merely underperform — it confidently inverts every verdict while the
interface reports itself as calibrated. That is strictly worse than no
calibration, so it fails loudly instead. This run passed the guard at 0.96.

### A note on `colab_pipeline.ipynb`

The notebook still documents the **superseded** pipeline — `plan_splits.py`,
`train.py`, `evaluate.py`, `fit_clip_calibration.py` — not the
`finetune_clips.py` / `evaluate_clips.py` path that produced the deployed
checkpoint. Anyone opening it today is following the wrong map, and its
calibration cell (§13) invokes the script that cannot import. Read this section
rather than the notebook until it is rewritten.

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
  region only when one area draws at least 1.5× the attention an evenly spread
  map would put there. Below that the report says "spread across the face, no
  single area" — which is a result, not a missing value — and only a failed or
  skipped Grad-CAM shows a dash. The
  row is captioned as a description of where the model looked, never as evidence
  of editing, and its severity is capped at "warn" for the same reason.
- **The confidence band reflects whether a calibrator is actually loaded.**
  With one loaded — the deployed state since the fit described above — it prints
  `Strong`/`Moderate`/`Weak` from `aggregate.confidence_word()`, which is a
  genuine calibrated probability rather than a made-up range. Without one it
  prints `Uncalibrated` rather than borrowing a word that would overstate what
  the bare mean score means.
- **A mocked result is stamped `SIMULATED RESULT`.** The mock only ever appears
  when the server cannot be reached at all. If the server answers with an error,
  the error is shown — a real failure is never replaced by an invented result.
- **"Insufficient evidence" is not styled as a pass.** If a human face is visible
  in fewer than one third of sampled frames, the tool refuses to judge. This is why
  faceless clips, screen recordings and animal videos return no verdict — the model
  only knows human faces and has genuinely learned nothing about anything else.

---

## Signed verdict receipts

Every analysis now returns a receipt: a small JSON object signed with an
**Ed25519** key held by the server (`server/receipt.py`), stating that this
detector produced this verdict for a file with this SHA-256 at this time.
`site/verify.html` checks one in the browser with WebCrypto against the public
key from `GET /v1/public-key`; `POST /v1/verify-receipt` exists only as a
fallback for browsers without Ed25519, and the page says on screen which of the
two did the checking.

**What it does and does not claim.** It binds a result to a file and to this
detector, so an edited number or a report attached to a different clip is
detectable. It says nothing about whether the video is genuine or whether the
verdict is correct — a signed wrong answer is still a wrong answer. The claim is
written into the signed payload itself, in an `attests` field, so it cannot be
stripped off in transit to make the receipt look stronger than it is.

**Two implementation details worth not rediscovering.**

*Numbers in the payload are decimal strings, not JSON floats.* Python writes the
float `0.0` as `0.0` and JavaScript writes it as `0`, so a receipt with a
confidence of exactly 0.0 or 1.0 would serialise to different canonical bytes in
the two languages: valid in Python, invalid in the browser, and only for certain
values. Strings remove the ambiguity instead of relying on both runtimes
choosing the same shortest representation. Verified across both: a receipt
signed by Python verifies under the browser's canonical form, and fails as soon
as one digit of the score is changed.

*Without `REELREAL_SIGNING_KEY` the server signs with a throwaway key* generated
at startup, prints that to the log, and reports `ephemeral: true` from
`/v1/public-key`, which the verify page shows. An ephemeral key must never be
presented as a durable attestation: the signature is real but the identity
behind it disappears on restart.

---

## Two harnesses that check the server, and what they found

Both live in `scripts/`, need the server running, and write their results to
`reports/`. Neither touches the model.

### `check_determinism.py` — the same clip twice

Analyses each clip N times through real HTTP requests and compares verdict, clip
score, flagged-frame count, mean score and every timeline bar. **Last run: 8
clips × 2 runs, every field identical** (`reports/determinism.json`).

That is a repeatability check on six clips, not an accuracy claim — a detector
that is confidently wrong every time passes it perfectly. It is worth having
because a verdict that moves when the input did not is impossible to defend,
and because it also covers the upload path and the adapter, not just the model.

### `check_robustness.py` — deliberately awkward input

Twelve cases, each declaring which HTTP statuses count as *correct handling*, so
passing is not "nothing failed": a renamed PDF **should** be rejected and the
script fails if it is accepted. The final case re-runs the happy path to prove
the server did not get wedged by the eleven before it.
**Last run: all 12 as declared, server healthy afterwards**
(`reports/robustness.json`).

#### ✅ Fixed — a 65-byte PDF got a verdict

The first run caught it. A PDF renamed to `.mp4` passed the extension
allow-list, decoded zero frames, and came back out of the pipeline as a full
report reading **"not enough face visible to judge", with a confidence of
0.0101** — and, once receipts existed, a signed attestation of that verdict.

That verdict is a specific claim: *we looked at your video and could not tell.*
Nothing had been looked at. `_probe_video()` in `server/app.py` now opens the
file and reads one frame — `isOpened()` alone is too weak, OpenCV will happily
"open" a file it can decode nothing from — and a file yielding no frames is
rejected as input with a 400 rather than judged as evidence.

#### ✅ Fixed — an analysis blocked every other request

`analyze` was `async def` while doing blocking CPU work, which runs it directly
on the event loop and stops the server answering anything until it finishes.
Found while testing the site's new backend-reachability banner: the website
probes `/health` to decide whether a real backend exists, and a probe that times
out during someone else's analysis would tell the user their results are
simulated when they are not. It is now a plain `def`, which FastAPI runs in a
threadpool.

---

## Running it

**The canonical run guide is now [README.md](README.md) at the repository root** -
that is the file someone cloning this reads first, and it carries the full
install, the two terminals, the check scripts and the receipt key. What follows
is the short version kept here for continuity; `server/README.md` has the
detailed server setup.

Short version, two terminals:

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

The Azure for Students subscription **cannot use ACR Tasks**, so `az containerapp
up --source ...` — which uploads the folder and builds it in the cloud — fails.
The image has to be built locally with Docker and pushed. This is the procedure
that is actually in use; do not reinstate the `--source` form, it does not work
on this subscription.

The live deployment: resource group `reelreal-rg`, region `switzerlandnorth`,
registry `ca1c89ee6e5cacr`, app `reelreal`.

**Before building**, refresh the copies under `deploy/container/`. A Docker build
sees only its own context, so the folder carries its own `server/`,
`ctf_pretrained/` and `site/`, and a stale copy silently ships old code however
many times the build succeeds — which is exactly how the Grad-CAM work sat
un-deployed after being committed:

```bash
cp -r ctf_pretrained/. deploy/container/ctf_pretrained/
cp -r site/.           deploy/container/site/
cp server/adapter.py   deploy/container/server/adapter.py
```

Not `server/app.py` — the copy in `deploy/container/` differs deliberately, so
one container serves the website as well as the API. The fine-tuned checkpoint
must also be at `deploy/container/ctf_pretrained/vit_finetuned/`; the build fails
without it on purpose.

Then build, push and roll, bumping the tag every time:

```bash
az acr login -n ca1c89ee6e5cacr
docker build -t ca1c89ee6e5cacr.azurecr.io/reelreal:v7 deploy/container
docker push  ca1c89ee6e5cacr.azurecr.io/reelreal:v7
az containerapp update -n reelreal -g reelreal-rg \
  --image ca1c89ee6e5cacr.azurecr.io/reelreal:v7
```

The first build takes 15–25 minutes — CPU-only PyTorch plus a 340 MB model baked
into the image. Later builds reuse the cached layers and take about a minute,
since only the `COPY` steps change.

Use a new tag rather than overwriting one. Container Apps decides whether to
create a revision by comparing the image *reference*, so re-pushing the same tag
can leave the old revision serving while every command reports success.

**Then verify the running service, not the exit codes.** This has drifted twice.
`az containerapp update` returning cleanly means the revision was accepted, not
that it is serving:

```bash
B=https://reelreal.salmonbeach-c14fa31b.switzerlandnorth.azurecontainerapps.io
az containerapp revision list -n reelreal -g reelreal-rg -o table   # 100% traffic, Running
curl -s "$B/health"
curl -s -X POST "$B/v1/analyze" -F video=@sample.mp4 | python -m json.tool
```

Check the response body actually carries the change you shipped. The v6 rollout
was confirmed by the `e1` artifact reading `Where the model looked` rather than
`Face edges`, and by `decisionThresh: 0.50`, which only the fine-tuned checkpoint
sets — the hub fallback uses 0.40.

Set the CORS origin, which otherwise defaults to `*`:

```bash
az containerapp update --name reelreal --resource-group reelreal-rg \
  --set-env-vars ALLOWED_ORIGINS=https://<your-site>.vercel.app
```

### Frontend

Set the backend address in `site/js/config.js` — the one line that needs editing
when the backend moves — then deploy `site/` to Vercel with **Root Directory:
`site`** and no build command.

Vercel builds from the GitHub repository, so **the site ships on `git push`, not
on commit.** A commit sitting unpushed on a local `main` is a commit the site
has not seen; check with `git rev-list --count origin/main..main` before assuming
a change is live, and confirm with `curl -s https://<site>.vercel.app/ | grep
<something-new>`.

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
