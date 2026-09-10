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
