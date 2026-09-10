# REEL/REAL — Frontend → Backend Handoff

Frontend is done: a website and a Chrome extension, both working, both running on a
**mocked** detector. Your job is to replace the mock with a real one.

**The good news:** you should not need to touch any UI file. The entire frontend
talks to detection through exactly one function, in one file.

---

## 1. Run it first (5 minutes, no setup)

**Website** — double-click `site/index.html`. No server, no npm, no build step.
Scroll to "Check a video", click a sample chip, click "Analyze Video".

**Extension** — in Chrome:
1. Go to `chrome://extensions`
2. Turn on **Developer mode** (top-right)
3. Click **Load unpacked**, select the `extension` folder
4. Click the puzzle-piece icon in the toolbar and pin REEL/REAL
5. Click the icon → the popup opens

Everything you see is fake. It waits ~2s and invents a plausible answer.

> **One thing to fix on your machine.** The popup's "Open full report on the
> website →" button is hardcoded to the path this was built on. Open
> `extension/popup.js`, find `WEBSITE_URL` at the top, and point it at your own
> copy of `site/index.html`, e.g.
> `file:///C:/Users/<you>/Desktop/DEEPFAKE-PROJ/site/index.html`
> Then on `chrome://extensions` → REEL/REAL → **Details** → turn on
> **"Allow access to file URLs"**. Everything else in the popup works without this.

---

## 2. Where the mock lives

```
site/js/detector.js          ← the master copy
extension/js/detector.js     ← a byte-identical duplicate
```

Both files are the same. Chrome extensions can't read files outside their own
folder, so the extension needs its own physical copy. **Edit the `site/` one, then
copy it over `extension/js/detector.js`.** Keep them identical.

Inside it, find:

```js
async function analyzeVideo(file, options) { ... }
```

It's marked with a big `THE SEAM` comment block. That function body is the only
thing that needs to change.

---

## 3. What you need to build

One endpoint:

```
POST /v1/analyze
Content-Type: multipart/form-data
Body: video=<file>

→ 200 OK, application/json, the AnalysisResult below
```

Then replace the body of `analyzeVideo` with roughly:

```js
async function analyzeVideo(file, options) {
  options = options || {};
  var body = new FormData();
  body.append('video', file);

  var res = await fetch(API_BASE + '/v1/analyze', {
    method: 'POST',
    body: body,
    signal: options.signal
  });
  if (!res.ok) throw new Error('Analysis failed: ' + res.status);
  return await res.json();
}
```

Then delete `seededRandom`, `buildResult`, `STAGES`, and the `forceVerdict` option
(plus the sample chips in `site/index.html` and `extension/popup.html` that use it).

Keep `encodeResult`, `decodeResult`, `formatClock`, `formatSize`, `labelFor` — the
UI uses those directly.

---

## 4. The response contract

This is the exact shape the UI renders. Match it and nothing breaks. Every field
listed is used somewhere on screen — none are optional.

```jsonc
{
  "id": "an_4taw0i",                        // your job id, shown nowhere yet
  "fileName": "ministry_statement_final.mp4",
  "fileSizeBytes": 15414067,
  "durationSeconds": 24,                    // real duration, drives the timeline axis
  "resolution": "720p",                     // free text, printed as-is

  "verdict": "synthetic",                   // "synthetic" | "authentic" — ONLY these two
  "confidence": 0.94,                       // 0..1, probability the clip is SYNTHETIC
  "calibratedBand": { "low": 0.87, "high": 0.98 },   // 90% interval

  "manipulatedDurationSeconds": 5.2,        // 0 when authentic
  "flaggedSegment": { "startSecond": 6, "endSecond": 11 },   // null when authentic

  // One entry PER SECOND of video. Length must equal durationSeconds.
  "timeline": [
    { "second": 0, "score": 0.077, "label": "authentic" },
    { "second": 1, "score": 0.202, "label": "authentic" },
    { "second": 2, "score": 0.086, "label": "authentic" }
    // ... 24 total
  ],

  // Exactly 6 entries, ids e1..e6 IN THIS ORDER — the website has matching
  // hardcoded row labels, so the order is load-bearing.
  "artifacts": [
    { "id": "e1", "label": "Face boundary blending", "detail": "Soft seam along jawline",        "severity": "bad"  },
    { "id": "e2", "label": "Blink rate & frequency", "detail": "2 blinks in 24 s — unnatural",   "severity": "warn" },
    { "id": "e3", "label": "Temporal flicker",       "detail": "Texture reset at 6.1 s",         "severity": "bad"  },
    { "id": "e4", "label": "Compression trace",      "detail": "Double-encoded region, 0:06–0:11","severity": "warn" },
    { "id": "e5", "label": "Lip-sync alignment",     "detail": "Consistent",                     "severity": "ok"   },
    { "id": "e6", "label": "C2PA Provenance",        "detail": "Unsigned",                       "severity": "bad"  }
  ],

  "provenance": { "signed": false, "summary": "no C2PA signature" },

  "processingTimeMs": 1962,
  "modelVersion": "v2.4",
  "analysedAt": "2026-09-08T23:22:12.645Z"
}
```

### Enum values — these are hard requirements

| Field | Allowed values |
|---|---|
| `verdict` | `"synthetic"` \| `"authentic"` |
| `timeline[].label` | `"authentic"` \| `"uncertain"` \| `"synthetic"` |
| `artifacts[].severity` | `"ok"` \| `"warn"` \| `"bad"` |

They map straight onto CSS colours in `tokens.css` (`--true` / `--warn` / `--fake`).
Send anything else and that element renders uncoloured.

### Thresholds

`timeline[].label` must agree with `timeline[].score`, using the same cutoffs the
frontend uses (exported as constants from `detector.js`):

```
score >= 0.60  → "synthetic"
score >= 0.35  → "uncertain"
otherwise      → "authentic"
```

If you'd rather own these server-side, keep sending `label` and the UI will just
trust it — it only reads `label` for colour, never re-derives it.

### One judgement call to make deliberately

The clip-level `confidence` is currently **peak-driven, not averaged** — a 5-second
fake inside a 24-second clip still scores high. Averaging would dilute short edits
to nothing, which is the exact failure the site's "Micro-Edits" section warns about.
If you change this, tell me, because the copy on the page makes a promise about it.

---

## 5. Progress reporting

`analyzeVideo` takes an `onProgress` callback that the UI already uses to drive its
progress bar and status text:

```js
options.onProgress({ percent: 42, stage: 'Analyzing facial landmarks…' })
```

Call it as often as you like. Options, easiest first:

1. **Fake it** — tick a timer while the request is in flight. Fine for a demo.
2. **Upload progress** — `XMLHttpRequest.upload.onprogress` gives real bytes-sent.
   `fetch` can't do upload progress, so you'd swap to XHR.
3. **Job polling** — `POST /v1/analyze` returns `{jobId}` immediately, then poll
   `GET /v1/analyze/:jobId` for `{percent, stage, result}`. Best for slow models.

It also accepts an `AbortSignal` as `options.signal`. Nothing cancels yet, but pass
it to `fetch` so cancellation works when we add a Cancel button.

---

## 6. Two things that WILL bite you

### CORS

The website will call your API from a different origin. You need:

```
Access-Control-Allow-Origin: <the site's origin>
```

The **extension**'s origin is `chrome-extension://<some-id>`, which changes every
time it's loaded unpacked. For development, allowing `*` is simplest. Lock it down
before anything ships.

### The extension needs the API whitelisted in its manifest

MV3 blocks network requests to origins not declared up front. In
`extension/manifest.json`, add:

```json
"host_permissions": ["https://api.your-domain.com/*"]
```

Without this the popup's fetch fails with a CORS error that looks like a server
problem but isn't.

---

## 7. The one real structural change

**A 200 MB upload cannot happen inside the extension popup.**

Chrome destroys the popup the instant it loses focus — clicking anywhere kills the
page and aborts the request mid-upload. Right now that's fine because the mock
finishes in 2 seconds.

Once uploads are real, the request has to move into `extension/background.js` (the
service worker, which survives the popup closing):

```
popup  → chrome.runtime.sendMessage({type:'ANALYZE'})   → service worker
worker → fetch(API)                                     → your server
worker → writes progress + result to chrome.storage.session
popup  → reads storage on open, subscribes to storage.onChanged
```

The popup stops owning the analysis and becomes a view onto whatever storage says.

`background.js` is nearly empty right now and has a comment block explaining exactly
this. Two gotchas when you get there:

- Chrome shuts the service worker down after ~30s idle and restarts it on events.
  **Never keep state in a module-level variable** — use `chrome.storage`.
- All listeners must be registered synchronously at the top level, on every startup.
  A listener registered inside a callback will never fire.

The **website** needs none of this. It's a normal page; it just fetches.

---

## 8. A product question, not a technical one

`site/index.html` currently says, in the upload box:

> "Your video is processed securely in your browser."

If analysis moves to a server, **that sentence becomes false.** Either:

- change the copy, or
- run inference in-browser (ONNX Runtime Web / TF.js), or
- upload but delete immediately and say so explicitly

Worth deciding early — it affects your architecture, not just the wording.

---

## 9. What NOT to touch

| File | Why |
|---|---|
| `site/index.html` | Copy and structure are approved. Only the sample chips come out. |
| `site/css/*`, `extension/popup.css` | Pure presentation |
| `site/js/app.js`, `extension/popup.js` | UI wiring. Only `WEBSITE_URL` at the top of `popup.js` changes, once the site is deployed. |

If you find yourself editing a UI file to make the backend work, the contract has
drifted — message me instead, it probably means a field changed shape.

---

## 10. Quick reference

```
site/
├── index.html            page content (approved copy — don't rewrite)
├── css/tokens.css        SHARED colours/fonts (duplicated into extension/)
├── css/site.css          website layout
├── js/detector.js        ← SHARED. THE ONLY FILE YOU NEED TO CHANGE
└── js/app.js             website UI wiring

extension/
├── manifest.json         extension config — needs host_permissions later
├── popup.html/css/js     the 380×600 popup UI
├── background.js         service worker — nearly empty, grows later
├── css/tokens.css        copy of site/css/tokens.css
├── js/detector.js        copy of site/js/detector.js
└── icons/                icon16/48/128.png + the script that generated them
```

**Handoff between the two surfaces:** the popup's "Open full report" button opens
the website with the result encoded in the URL fragment
(`index.html#result=<base64>`). Fragments are never sent to a server, so the report
stays local. The **video does not travel** — the site shows an explanatory panel
where the preview would be. That stays true regardless of the backend.
