/* Where the detection server lives.
 *
 * This is the ONLY file that needs editing when the backend moves. detector.js
 * reads window.REELREAL_API_BASE and falls back to localhost when it is unset,
 * so this file must load BEFORE detector.js.
 *
 * Local development: leave this commented out.
 * Deployed: uncomment and paste the backend's address, no trailing slash. The
 * extension has no same-origin fallback — a popup is always a different origin —
 * so this must be set for any build that is not talking to localhost.
 */
// window.REELREAL_API_BASE = 'https://reelreal.<id>.<region>.azurecontainerapps.io';
