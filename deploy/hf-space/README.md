---
title: REEL/REAL Deepfake Detector
emoji: 🎬
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# REEL/REAL detection API

The backend for the REEL/REAL deepfake detector. It receives a video, runs the
detection pipeline over sampled frames, and returns a verdict as JSON. The
website and the Chrome extension both talk to this.

Opening the Space URL shows the website itself. The detector answers beneath
it on the same origin, so there is nothing else to configure:

    /                 the website
    /health           liveness check
    /v1/analyze       POST a video, get a verdict

Use `/health` to check the Space is awake.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | liveness, and whether the model has loaded yet |
| `POST` | `/v1/analyze` | multipart upload, field name `video` |

## Settings

Set these under **Settings → Variables and secrets**.

| Variable | Purpose |
|---|---|
| `ALLOWED_ORIGINS` | Comma-separated list of sites permitted to call this. **Set this.** Left unset it defaults to `*`, meaning any website on the internet can submit videos to it. |

Example:

```
ALLOWED_ORIGINS=https://your-site.vercel.app
```

The Chrome extension calls from a `chrome-extension://<id>` origin, which is
different again — add it once the extension has a stable id.

## Speed

Free Spaces run on CPU, so a clip takes roughly **30–45 seconds**. That is the
model genuinely working, not a hang.

A free Space also **sleeps after a period without traffic**, and waking it takes a
minute or two. Before any demo, open `/health` yourself first so the first real
visitor is not the one waiting.

## Known state of the detector

`_score()` in `infer_pipeline.py` returns P(deepfake) from class index 1, which is
what the checkpoint's `id2label {0: 'Realism', 1: 'Deepfake'}` declares. An
earlier version read index 0 and subtracted a flat 0.25 from every frame score;
between them, no frame could ever cross the 0.90 flag threshold and every video
came back "NO MANIPULATION DETECTED". Both edits are reverted. See `STATUS.md`.

One issue remains open: **clip calibration has never been fitted.** The number
shown is a flagged-frame fraction, not a probability, and the report says
`Uncalibrated` rather than inventing a confidence band. Read it as a ranking.
