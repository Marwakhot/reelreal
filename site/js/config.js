/* Where the detection server lives.
 *
 * This is the ONLY file that needs editing when the backend moves. detector.js
 * reads window.REELREAL_API_BASE and falls back to localhost when it is unset,
 * so this file must load BEFORE detector.js.
 *
 * Three cases, decided in this order:
 *
 *   1. The page is being served from localhost / 127.0.0.1 / file://
 *      -> use the local detector at 127.0.0.1:8000. Running the site and the
 *         server on your own machine is the documented way to run this project
 *         (see the README), and it must not require editing a file first. A
 *         clone that silently talks to the deployed backend is a clone where
 *         nobody notices their own server never started.
 *
 *   2. Deployed on Vercel, which serves the site only
 *      -> DEPLOYED_API_BASE below. The detector runs elsewhere (Azure Container
 *         Apps), so without this every upload would 404. Paste the backend's
 *         address with no trailing slash.
 *
 *   3. Site and detector served by the same container
 *      -> the API is simply wherever this page came from.
 */
var DEPLOYED_API_BASE = 'https://reelreal.salmonbeach-c14fa31b.switzerlandnorth.azurecontainerapps.io';

(function () {
  var host = location.hostname;
  var isLocal = !host || host === 'localhost' || host === '127.0.0.1' || host === '[::1]';

  if (isLocal) {
    // Leave it unset: detector.js already defaults to http://127.0.0.1:8000.
    return;
  }

  if (DEPLOYED_API_BASE) {
    window.REELREAL_API_BASE = DEPLOYED_API_BASE;
    return;
  }

  // Same-origin deployment: the API is wherever this page came from.
  if (/^https?:$/.test(location.protocol)) {
    window.REELREAL_API_BASE = location.origin;
  }
})();
