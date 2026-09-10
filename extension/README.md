# REEL/REAL browser extension

Loads as an unpacked extension: `chrome://extensions` → Developer mode →
**Load unpacked** → select this folder.

## host_permissions — read before deploying

The popup uploads the video to the detection server, which is a different origin
from the extension, so Chrome needs explicit permission to reach it. Only the local
dev server is listed today:

```json
"host_permissions": [
  "http://127.0.0.1:8000/*",
  "http://localhost:8000/*"
]
```

When the backend is deployed, **replace these two entries with the real `https://`
origin and nothing else.** Do not use a wide pattern such as `https://*/*` — Chrome
then warns every user that the extension can read data on every site they visit,
which is both untrue of this extension and enough to stop people installing it.

These notes live here rather than in `manifest.json` because JSON has no comment
syntax. A `"//comment"` key does not break anything, but Chrome flags unrecognised
manifest keys with a warning triangle on the extensions page.

## Where the server address comes from

`js/detector.js` is byte-identical to `site/js/detector.js` — one detector, two
surfaces. It reads:

```js
var API_BASE = global.REELREAL_API_BASE || 'http://127.0.0.1:8000';
```

So the deployed address can be set by defining `window.REELREAL_API_BASE` in
`popup.html` before `detector.js` loads, without editing the shared file. If you do
edit `detector.js`, mirror the change to `site/js/detector.js` — they are meant to
stay identical.

## Reading a result

The version line under the verdict appends **` · SIMULATED`** when the result is
mocked. The mock only appears when the server cannot be reached at all; if the
server answers with an error, the error is shown instead. A real failure is never
quietly replaced by an invented result, so if that marker is absent, the numbers on
screen came from the model.

## Gotcha when testing

Chrome closes an extension popup the moment it loses focus, and an analysis takes
30–45 seconds. Clicking anything outside the popup mid-analysis discards it. Keep
the popup focused until the result appears.
