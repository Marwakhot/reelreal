/* ============================================================================
   REEL/REAL — UI EFFECTS
   ----------------------------------------------------------------------------
   Purely additive. Reads no analysis state, writes no analysis state.
   Safe to delete — removing this file returns the site to its static look
   with no other side effects.
   ========================================================================== */

(function () {
  'use strict';

  /* --------------------------------------------------------------------------
     1. TYPEWRITER — statement paragraph
     Fires once when the element scrolls into view. Types at ~40ms per char,
     then removes the blinking cursor once done.
     ---------------------------------------------------------------------- */
  var stEl = document.getElementById('statementText');
  if (stEl) {
    var fullText = stEl.getAttribute('data-fulltext') || '';
    stEl.textContent = '';

    var cursor = document.createElement('span');
    cursor.className = 'typewriter-cursor';
    cursor.setAttribute('aria-hidden', 'true');
    stEl.appendChild(cursor);

    var typed = false;

    function typeText() {
      if (typed) return;
      typed = true;

      var i = 0;
      var interval = setInterval(function () {
        if (i < fullText.length) {
          stEl.insertBefore(document.createTextNode(fullText[i]), cursor);
          i++;
        } else {
          clearInterval(interval);
          // Remove cursor after a short pause
          setTimeout(function () {
            if (cursor.parentNode) cursor.parentNode.removeChild(cursor);
          }, 1200);
        }
      }, 28);
    }

    if ('IntersectionObserver' in window) {
      var obs = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) {
            typeText();
            obs.disconnect();
          }
        });
      }, { threshold: 0.4 });
      obs.observe(stEl);
    } else {
      // Fallback: just show the text immediately
      stEl.textContent = fullText;
    }
  }

  /* --------------------------------------------------------------------------
     2. FORENSIC SCAN LINE — drop zone
     Adds .scanning to #drop while the analysis bar is active (has class .on),
     so the green/orange sweep plays only during real analysis.
     ---------------------------------------------------------------------- */
  var sPreview = document.getElementById('sPreview');
  var bar      = document.getElementById('bar');

  if (sPreview && bar) {
    var scanObserver = new MutationObserver(function () {
      if (bar.classList.contains('on')) {
        sPreview.classList.add('scanning');
      } else {
        sPreview.classList.remove('scanning');
      }
    });
    scanObserver.observe(bar, { attributes: true, attributeFilter: ['class'] });
  }

  /* --------------------------------------------------------------------------
     3. TICKER — pause on hover for accessibility
     ---------------------------------------------------------------------- */
  var track = document.querySelector('.ticker-track');
  if (track) {
    track.addEventListener('mouseenter', function () {
      track.style.animationPlayState = 'paused';
    });
    track.addEventListener('mouseleave', function () {
      track.style.animationPlayState = 'running';
    });
  }

})();
