/* ============================================================================
   REEL/REAL — RECEIPT VERIFIER
   ----------------------------------------------------------------------------
   Checks the Ed25519 signature on a verdict receipt, and (optionally) that the
   receipt refers to the video file you have in front of you.

   WHY THIS RUNS IN THE BROWSER
   ----------------------------
   The whole value of a signed receipt is that verifying it does not require
   trusting the server that issued it. So the check is done here, with
   WebCrypto, against a public key. The server is asked for the public key as a
   convenience — you can paste a key you already trust instead, and nothing
   else in this file talks to it.

   Where WebCrypto has no Ed25519 (older browsers), the page falls back to
   POST /v1/verify-receipt and says on screen that the server did the checking,
   because that is a meaningfully weaker claim and hiding it would be dishonest.

   CANONICAL FORM
   --------------
   The signature covers JSON with sorted keys and no whitespace. canonical()
   below reproduces exactly what server/receipt.py signs. Note that the payload
   carries every non-integer number as a decimal STRING, which is what makes
   this reproducible across the two languages — see the CANONICAL FORM note in
   receipt.py for why.
   ========================================================================== */
(function () {
  'use strict';

  var API_BASE = window.REELREAL_API_BASE || 'http://127.0.0.1:8000';
  var $ = function (sel) { return document.querySelector(sel); };

  var serverKey = null;       // { publicKey, keyId, ephemeral }
  var videoHash = null;       // SHA-256 hex of the chosen file, or null

  /* ------------------------------------------------------------------------
     Canonical JSON — must match server/receipt.py byte for byte
     ---------------------------------------------------------------------- */
  function canonical(value) {
    if (value === null || typeof value !== 'object') return JSON.stringify(value);
    if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
    return '{' + Object.keys(value).sort().map(function (k) {
      return JSON.stringify(k) + ':' + canonical(value[k]);
    }).join(',') + '}';
  }

  function bytesFromBase64(text) {
    // Tolerate the URL-safe alphabet and missing padding: receipts travel
    // through chat apps and URLs before they get here.
    var clean = String(text).trim().replace(/-/g, '+').replace(/_/g, '/');
    while (clean.length % 4) clean += '=';
    var binary = atob(clean);
    var out = new Uint8Array(binary.length);
    for (var i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
    return out;
  }

  function hex(buffer) {
    return Array.prototype.map.call(new Uint8Array(buffer), function (b) {
      return b.toString(16).padStart(2, '0');
    }).join('');
  }

  /* ------------------------------------------------------------------------
     The signing key
     ---------------------------------------------------------------------- */
  fetch(API_BASE + '/v1/public-key')
    .then(function (r) { return r.json(); })
    .then(function (info) {
      serverKey = info;
      $('#keyLine').textContent = info.ephemeral
        ? 'Key ' + info.keyId + ' — throwaway development key'
        : 'Key ' + info.keyId + ' · Ed25519';
    })
    .catch(function () {
      $('#keyLine').textContent = 'Signing key unavailable — server unreachable';
    });

  /* ------------------------------------------------------------------------
     Inputs
     ---------------------------------------------------------------------- */
  $('#pickReceipt').addEventListener('click', function () { $('#receiptFile').click(); });
  $('#receiptFile').addEventListener('change', function (e) {
    var file = e.target.files[0];
    if (!file) return;
    file.text().then(function (text) { $('#receiptText').value = text; });
  });

  $('#pickVideo').addEventListener('click', function () { $('#videoFile').click(); });
  $('#videoFile').addEventListener('change', function (e) {
    if (e.target.files[0]) hashVideo(e.target.files[0]);
  });

  ['dragenter', 'dragover'].forEach(function (type) {
    $('#videoDrop').addEventListener(type, function (e) {
      e.preventDefault(); this.classList.add('over');
    });
  });
  ['dragleave', 'drop'].forEach(function (type) {
    $('#videoDrop').addEventListener(type, function (e) {
      e.preventDefault(); this.classList.remove('over');
    });
  });
  $('#videoDrop').addEventListener('drop', function (e) {
    var file = e.dataTransfer.files[0];
    if (file) hashVideo(file);
  });

  function hashVideo(file) {
    $('#videoName').textContent = file.name;
    $('#hashLine').textContent = 'Hashing…';
    videoHash = null;

    if (!window.crypto || !crypto.subtle) {
      // crypto.subtle only exists in a secure context. Opening the page as a
      // file:// URL is the usual cause, and it is worth naming.
      $('#hashLine').textContent =
        'Cannot hash here: this page needs to be served over http://localhost ' +
        'or https, not opened as a file.';
      return;
    }

    file.arrayBuffer()
      .then(function (buf) { return crypto.subtle.digest('SHA-256', buf); })
      .then(function (digest) {
        videoHash = hex(digest);
        $('#hashLine').textContent = 'SHA-256 ' + videoHash;
      })
      .catch(function (err) {
        $('#hashLine').textContent = 'Could not hash this file: ' + err.message;
      });
  }

  /* ------------------------------------------------------------------------
     Verification
     ---------------------------------------------------------------------- */
  function parseReceipt() {
    var raw = $('#receiptText').value.trim();
    if (!raw) throw new Error('Paste a receipt first.');
    var parsed = JSON.parse(raw);
    // A full analysis result has the receipt nested inside it; accept either,
    // since "copy the whole thing" is what people actually do.
    if (parsed && parsed.receipt && parsed.receipt.payload) return parsed.receipt;
    return parsed;
  }

  /* Verify locally with WebCrypto. Returns null when this browser has no
     Ed25519, which is the signal to fall back to the server. */
  async function verifyLocally(receipt, publicKeyB64) {
    if (!window.crypto || !crypto.subtle) return null;
    var key;
    try {
      key = await crypto.subtle.importKey(
        'raw', bytesFromBase64(publicKeyB64), { name: 'Ed25519' }, false, ['verify']);
    } catch (err) {
      return null;                                   // no Ed25519 in this browser
    }
    var data = new TextEncoder().encode(canonical(receipt.payload));
    var ok = await crypto.subtle.verify(
      'Ed25519', key, bytesFromBase64(receipt.signature), data);
    return { valid: ok, checkedBy: 'this browser' };
  }

  async function verifyOnServer(receipt) {
    var response = await fetch(API_BASE + '/v1/verify-receipt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ receipt: receipt })
    });
    var body = await response.json();
    return { valid: !!body.valid, reason: body.reason, checkedBy: 'the server' };
  }

  $('#verify').addEventListener('click', async function () {
    var receipt;
    try {
      receipt = parseReceipt();
    } catch (err) {
      return show('invalid', 'Not a readable receipt', err.message, null, null);
    }

    if (!receipt || !receipt.payload || !receipt.signature) {
      return show('invalid', 'Not a receipt',
                  'A receipt needs a "payload" object and a "signature" string.',
                  null, null);
    }

    var outcome = null;
    if (serverKey && serverKey.publicKey) {
      try {
        outcome = await verifyLocally(receipt, serverKey.publicKey);
      } catch (err) {
        outcome = { valid: false, reason: err.message, checkedBy: 'this browser' };
      }
    }
    if (!outcome) {
      try {
        outcome = await verifyOnServer(receipt);
      } catch (err) {
        return show('invalid', 'Could not verify',
                    'No signing key and no server to ask: ' + err.message,
                    receipt.payload, null);
      }
    }

    /* The file check is separate from the signature check and reported
       separately. A valid signature on a receipt for a DIFFERENT video is a
       real scenario — a genuine report attached to the wrong clip — and
       collapsing the two checks into one word would hide exactly that. */
    var fileState = null;
    if (videoHash) {
      fileState = (videoHash.toLowerCase() === String(receipt.payload.fileSha256).toLowerCase())
        ? 'match' : 'mismatch';
    }

    var title, reason;
    if (!outcome.valid) {
      title = 'Signature does not check out';
      reason = outcome.reason ||
        'The payload was altered after signing, or a different key signed it.';
    } else if (fileState === 'mismatch') {
      title = 'Valid signature, but a different video';
      reason = 'The signature is genuine, and it is for a different file than the ' +
               'one you loaded. This receipt does not describe that video.';
    } else if (fileState === 'match') {
      title = 'Valid, and it is this video';
      reason = 'Signature checks out against key ' + (serverKey ? serverKey.keyId : '?') +
               ', and the file you loaded hashes to the same SHA-256. Checked by ' +
               outcome.checkedBy + '.';
    } else {
      title = 'Valid signature';
      reason = 'Checked by ' + outcome.checkedBy + '. Load the video too if you ' +
               'want to confirm the receipt refers to that exact file.';
    }

    if (outcome.valid && serverKey && serverKey.ephemeral) {
      reason += ' Note: this server is signing with a throwaway development ' +
                'key, so the signature is real but the identity behind it is not durable.';
    }

    show(!outcome.valid ? 'invalid' : fileState === 'mismatch' ? 'warn' : 'ok',
         title, reason, receipt.payload, fileState);
  });

  /* Deliberate tamper demo: flip a digit in the signed payload and re-verify.
     A verification tool that has only ever shown green is indistinguishable
     from one that always shows green. */
  $('#tamper').addEventListener('click', function () {
    var receipt;
    try {
      receipt = parseReceipt();
    } catch (err) {
      return show('invalid', 'Nothing to alter', err.message, null, null);
    }
    if (!receipt.payload) return;

    var before = receipt.payload.confidence;
    // Nudge the score by one in its last decimal place, keeping the same number
    // of places so the payload still looks entirely plausible. If there is no
    // score to move, rename the file instead - either edit must be caught.
    if (typeof before === 'string' && /^\d+\.\d+$/.test(before)) {
      var places = before.split('.')[1].length;
      receipt.payload.confidence = (Number(before) + Math.pow(10, -places)).toFixed(places);
    } else {
      receipt.payload.fileName = 'altered-' + (receipt.payload.fileName || 'clip.mp4');
    }

    $('#receiptText').value = JSON.stringify(receipt, null, 2);
    $('#verify').click();
  });

  /* ------------------------------------------------------------------------
     Rendering
     ---------------------------------------------------------------------- */
  /* Three outcomes, kept apart on purpose:
       'ok'      the signature checks out (and the file matches, if given)
       'warn'    the signature is genuine but describes a different video
       'invalid' the signature does not check out, or the receipt is unreadable
     Collapsing 'warn' into 'invalid' would say someone tampered with a receipt
     that nobody tampered with; collapsing it into 'ok' would let a receipt for
     one clip vouch for another. */
  function show(state, title, reason, payload, fileState) {
    var panel = $('#result');
    panel.classList.add('on');
    panel.classList.toggle('state-clean', state === 'ok');
    panel.classList.toggle('state-warn', state === 'warn');
    panel.classList.toggle('state-invalid', state === 'invalid');

    $('#vTitle').textContent = title;
    $('#vReason').textContent = reason;
    var badge = $('#vBadge');
    badge.textContent = state === 'ok' ? 'Verified'
      : state === 'warn' ? 'Different video'
      : 'Not verified';
    badge.classList.toggle('clean', state === 'ok');

    var fields = $('#vFields');
    fields.innerHTML = '';
    if (!payload) return;

    var rows = [
      ['File name', payload.fileName],
      ['File SHA-256', payload.fileSha256],
      ['Verdict', payload.verdict],
      ['Score', payload.confidence],
      ['Decision threshold', payload.decisionThreshold],
      ['Model', payload.modelVersion],
      ['Analysed at', payload.analysedAt],
      ['Analysis id', payload.analysisId]
    ];
    if (fileState) {
      rows.splice(2, 0, ['Your file', fileState === 'match'
        ? 'hashes to the same value' : 'hashes to something else']);
    }

    rows.forEach(function (row) {
      if (row[1] === undefined || row[1] === null || row[1] === '') return;
      var line = document.createElement('div');
      line.className = 'ev';
      var label = document.createElement('b');
      label.textContent = row[0];
      var value = document.createElement('span');
      value.textContent = row[1];
      line.appendChild(label);
      line.appendChild(value);
      fields.appendChild(line);
    });

    panel.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }
})();
