/* Where the detection server lives.
 *
 * This is the ONLY file that needs editing when the backend moves. detector.js
 * reads window.REELREAL_API_BASE and falls back to localhost when it is unset,
 * so this file must load BEFORE detector.js.
 *
 * Local development: leave this commented out.
 * Deployed: uncomment and paste the Hugging Face Space URL, no trailing slash.
 */
// window.REELREAL_API_BASE = 'https://YOUR-USERNAME-reelreal.hf.space';

/* Deployed on a Hugging Face Space, the website and the detector are served by
   the same app on one URL, so the API is simply wherever this page came from.
   Opened from localhost or a file:// copy, detector.js keeps its 127.0.0.1:8000
   default and this does nothing. Setting REELREAL_API_BASE above still wins. */
(function () {
  if (window.REELREAL_API_BASE) return;
  var h = location.hostname;
  if (/^https?:$/.test(location.protocol) && h && h !== 'localhost' && h !== '127.0.0.1') {
    window.REELREAL_API_BASE = location.origin;
  }
})();
