# REEL/REAL detection API — container deployment

The backend for the REEL/REAL deepfake detector, packaged as a single container.
It receives a video, runs the detection pipeline over sampled frames, and returns
a verdict as JSON. The website and the Chrome extension both talk to this.

This folder is self-contained: it carries its own copies of `server/`,
`ctf_pretrained/` and `site/`, because a Docker build can only see files beneath
its build context. Those copies must be kept in step with the ones at the
repository root — see "Keeping the copies in step" below.

The container serves the website as well as the API, on one origin:

    /                 the website
    /health           liveness, and whether the model has loaded yet
    /v1/analyze       POST a video (multipart, field name `video`), get a verdict

Use `/health` to check the service is awake.

## Where this runs

Built for **Azure Container Apps**, which has a perpetual free monthly grant of
180,000 vCPU-seconds, 360,000 GiB-seconds and 2 million requests. A demo fits
inside the grant. It listens on port 7860 and runs as a non-root user, so it will
run unmodified on most container hosts — Google Cloud Run and Azure Container
Instances included.

It will **not** run on Hugging Face Spaces' free tier any more. Docker Spaces
moved behind a PRO subscription during 2026; only Static Spaces remain free, and
those cannot execute Python. The folder was originally written for that tier.

Deployment steps are in the repository's `STATUS.md`.

## Settings

| Variable | Purpose |
|---|---|
| `ALLOWED_ORIGINS` | Comma-separated list of sites permitted to call this. **Set this.** Left unset it defaults to `*`, meaning any website on the internet can submit videos to it. |

Example:

```
ALLOWED_ORIGINS=https://your-site.vercel.app
```

The Chrome extension calls from a `chrome-extension://<id>` origin, which is
different again — add it once the extension has a stable id.

## Speed and cold starts

On CPU a clip takes roughly **30–45 seconds**. That is the model genuinely
working, not a hang.

The image is around 3 GB, so a container starting from cold takes a minute or
two before it answers at all. Scaled to zero that cost lands on whoever sends the
first request. Before any demo, either keep one replica warm or open `/health`
yourself first, so the first real visitor is not the one waiting.

## Keeping the copies in step

After changing anything in the root `server/`, `ctf_pretrained/` or `site/`, copy
it here — but **not** `server/app.py`, which differs deliberately: the copy in
this folder mounts the website as static files so one container serves both, and
overwriting it with the root version silently removes the website.

```bash
cp -r ctf_pretrained/. deploy/container/ctf_pretrained/
cp -r site/.           deploy/container/site/
cp server/adapter.py   deploy/container/server/adapter.py
```

## Known state of the detector

`_score()` in `infer_pipeline.py` returns P(deepfake) from class index 1, which is
what the checkpoint's `id2label {0: 'Realism', 1: 'Deepfake'}` declares. An
earlier version read index 0 and subtracted a flat 0.25 from every frame score;
between them, no frame could ever cross the 0.90 flag threshold and every video
came back "NO MANIPULATION DETECTED". Both edits are reverted. See `STATUS.md`.

One issue remains open: **clip calibration has never been fitted.** The number
shown is a flagged-frame fraction, not a probability, and the report says
`Uncalibrated` rather than inventing a confidence band. Read it as a ranking.
