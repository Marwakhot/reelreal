/* Where the detection server lives.
 *
 * This is the ONLY file that needs editing when the backend moves. detector.js
 * reads window.REELREAL_API_BASE and falls back to localhost when it is unset,
 * so this file must load BEFORE detector.js.
 *
 * Local development: leave the line below commented out.
 *
 * Deployed on Vercel: you MUST uncomment it and paste the backend's address,
 * with no trailing slash. Vercel serves the site only — the detector runs
 * elsewhere (Azure Container Apps), so the fallback below cannot find it and
 * every upload would 404.
 */
window.REELREAL_API_BASE = 'https://reelreal.salmonbeach-c14fa31b.switzerlandnorth.azurecontainerapps.io';

/* Fallback for when the site and the detector are served by the same container:
   the API is then simply wherever this page came from. Opened from localhost or
   a file:// copy, detector.js keeps its 127.0.0.1:8000 default and this does
   nothing. Setting REELREAL_API_BASE above always wins. */
(function () {
  if (window.REELREAL_API_BASE) return;
  var h = location.hostname;
  if (/^https?:$/.test(location.protocol) && h && h !== 'localhost' && h !== '127.0.0.1') {
    window.REELREAL_API_BASE = location.origin;
  }
})();
