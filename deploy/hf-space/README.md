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

Two open issues, both in `ctf_pretrained/`, neither introduced by the deployment:

1. **Verdicts are inverted.** `infer_pipeline._score()` reads class index 0 as
   "fake", but the checkpoint reports `id2label {0: 'Realism', 1: 'Deepfake'}` —
   index 0 is *real*. Every verdict comes out as its own opposite.
2. **The clip score is the single highest frame** when no calibrator is fitted, so
   one bad frame decides a whole video, and longer videos are steadily more likely
   to be called fake.

Fixes for both, with evidence, are in `REELREAL-COMPLETE-FIXED.zip` alongside this
project. Whichever `ctf_pretrained/` is uploaded here is what visitors get.
