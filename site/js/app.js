/* ============================================================================
   REEL/REAL — WEBSITE UI CONTROLLER
   ----------------------------------------------------------------------------
   Website-only. Owns DOM wiring and rendering; owns no detection logic.

   The only contact with analysis is:
       ReelReal.analyzeVideo(file, { onProgress })
   ...plus ReelReal.decodeResult() for reports handed over from the extension.
   ========================================================================== */

(function () {
  'use strict';

  var $ = function (sel) { return document.querySelector(sel); };

  /* ------------------------------------------------------------------------
     Hero video pair - both loop silently, hovering one turns its sound on
     --------------------------------------------------------------------------
     Every browser blocks autoplay with sound, and allows it without. So both
     clips autoplay muted and keep looping, and hover only flips `muted`.

     There is a second rule underneath that one: Chrome and Firefox will PAUSE a
     media element that is unmuted before the user has interacted with the page
     at all. Hovering is not an interaction by that definition - only a click,
     tap or keypress is. So unmuting is attempted, and if the clip stops as a
     result it is silently re-muted and restarted. A stopped frame in the hero
     would look like a broken video; staying silent for one more moment does not.
     ---------------------------------------------------------------------- */
  var REDUCED_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  document.querySelectorAll('.slot').forEach(function (slot) {
    var video = slot.querySelector('video');
    var ghost = slot.querySelector('.ghost');

    // Only reveal the video once a frame is actually decodable, otherwise a
    // missing asset would show as a black rectangle instead of the placeholder.
    video.addEventListener('loadeddata', function () {
      ghost.style.display = 'none';
      // Autoplay can be refused (a data-saver setting, an aggressive extension).
      // Asking again once the first frame exists costs nothing and recovers it.
      if (!REDUCED_MOTION) video.play().catch(function () {});
    });
    video.addEventListener('error', function () { video.style.display = 'none'; });

    slot.addEventListener('mouseenter', function () {
      video.muted = false;
      video.volume = 1;

      var started = video.play();
      if (started && started.catch) {
        started.catch(function () { silence(video); });
      }
      // play() can resolve and the element still be paused a tick later when
      // the autoplay policy steps in, so check the outcome rather than trusting
      // the promise alone.
      setTimeout(function () { if (video.paused) silence(video); }, 120);
    });

    slot.addEventListener('mouseleave', function () {
      video.muted = true;
      // Keep looping on the way out - the pair is meant to be alive on the page,
      // not only while the pointer is on it.
      if (!REDUCED_MOTION) video.play().catch(function () {});
    });

    /* Someone who has asked for reduced motion gets the old behaviour: still
       until hovered. Two clips looping forever is exactly the kind of motion
       that setting exists to stop. */
    if (REDUCED_MOTION) {
      video.autoplay = false;
      video.pause();
      slot.addEventListener('mouseleave', function () { video.pause(); });
    }
  });

  function silence(video) {
    video.muted = true;
    video.play().catch(function () {});
  }

  /* ------------------------------------------------------------------------
     Is there actually a detection server?
     --------------------------------------------------------------------------
     detector.js falls back to a MOCKED result when the server cannot be
     reached at all (see MOCK_FALLBACK). That fallback is useful for working on
     the UI, and dangerous everywhere else: a simulated verdict looks exactly
     like a measured one. This banner exists so the difference is visible
     before anyone reads a number off the screen.

     The check is passive - nothing is blocked while it runs, and the banner
     only appears once a probe has actually failed.
     ---------------------------------------------------------------------- */
  var serverNote = $('#serverNote');
  var serverReachable = null;          // null = not yet known

  /* A deployed detector sleeps when nobody is using it, and the container it
     lives in takes a minute or two to wake. One failed probe therefore means
     "not awake yet" far more often than it means "not there", so the alarming
     version of this banner is only shown after several attempts have failed
     across roughly two and a half minutes. Until then the page says the server
     is waking, which is what is actually happening. */
  var PROBE_DELAYS_MS = [4000, 10000, 20000, 40000, 60000];
  var probeAttempt = 0;

  function isLocalApi() {
    return /^https?:\/\/(127\.0\.0\.1|localhost|\[::1\])(:|$|\/)/.test(ReelReal.apiBase());
  }

  function showServerNote(state) {
    serverReachable = (state === 'up') ? true : (state === 'down') ? false : null;

    if (state === 'up') {
      serverNote.classList.remove('on', 'server-note--waking');
      serverNote.textContent = '';
      return;
    }

    if (state === 'waking') {
      serverNote.classList.add('on', 'server-note--waking');
      serverNote.innerHTML =
        '<b>Waking the detector</b>' +
        '<span>The analysis server sleeps when nobody is using it, and takes a ' +
        'minute or two to start. You can choose a video meanwhile - this notice ' +
        'disappears by itself once the server answers.</span>';
      return;
    }

    /* Genuinely unreachable. The instruction differs by audience: telling a
       visitor to the public site to run uvicorn on their own machine is
       nonsense, and the only honest thing to say there is to try again later. */
    serverNote.classList.add('on');
    serverNote.classList.remove('server-note--waking');
    serverNote.innerHTML =
      '<b>No detection server at ' + ReelReal.apiBase() + '</b>' +
      '<span>Anything you analyse now produces a <b>simulated</b> result - invented ' +
      'numbers for working on the interface, not a measurement of your video. ' +
      (isLocalApi()
        ? 'Start the server with <code>cd server &amp;&amp; python -m uvicorn app:app --port 8000</code> and reload.'
        : 'The server may still be starting, or may be down. Please try again in a few minutes.') +
      '</span>';
  }

  function probeServer() {
    fetch(ReelReal.apiBase() + '/health', { method: 'GET' })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error('HTTP ' + r.status)); })
      .then(function () { showServerNote('up'); })
      .catch(function () {
        if (probeAttempt < PROBE_DELAYS_MS.length) {
          showServerNote('waking');
          setTimeout(probeServer, PROBE_DELAYS_MS[probeAttempt++]);
        } else {
          showServerNote('down');
        }
      });
  }
  probeServer();

  /* ------------------------------------------------------------------------
     Upload staging
     ---------------------------------------------------------------------- */
  var drop = $('#drop');
  var fileInput = $('#file');
  var staged = $('#staged');

  var currentFile = null;   // the File chosen from the picker or a drop
  var objectURL = null;     // blob: URL for playback; null for extension imports

  $('#pick').addEventListener('click', function () { fileInput.click(); });

  fileInput.addEventListener('change', function (e) {
    if (e.target.files[0]) stageFile(e.target.files[0]);
  });

  // preventDefault on dragover is what actually makes an element a drop target;
  // without it the browser just navigates to the dropped file.
  ['dragenter', 'dragover'].forEach(function (type) {
    drop.addEventListener(type, function (e) { e.preventDefault(); drop.classList.add('over'); });
  });
  ['dragleave', 'drop'].forEach(function (type) {
    drop.addEventListener(type, function (e) { e.preventDefault(); drop.classList.remove('over'); });
  });
  drop.addEventListener('drop', function (e) {
    var file = e.dataTransfer.files[0];
    if (file) stageFile(file);
  });

  function stageFile(file) {
    currentFile = file;

    if (objectURL) URL.revokeObjectURL(objectURL);  // don't leak the previous blob
    objectURL = URL.createObjectURL(file);

    $('#sName').textContent = file.name;
    $('#sMeta').textContent = ReelReal.formatSize(file.size) + ' · Ready';

    /* Preview the chosen clip immediately. Without this the first sight of the
       video is the report, 35 seconds later, with no way to check beforehand
       that the right file was picked. */
    var preview = $('#sPreview');
    var sVideo = $('#sVideo');
    if (objectURL) {
      sVideo.src = objectURL;
      preview.classList.add('on');
    } else {
      sVideo.removeAttribute('src');
      sVideo.load();                        // drop the decoded frames of the old clip
      preview.classList.remove('on');
    }

    staged.classList.add('on');
    resetProgress();
    staged.scrollIntoView({ block: 'center', behavior: 'smooth' });
  }

  /* Duration and resolution are only knowable once the browser has read the
     file's header, so the meta line fills itself in a moment after staging. */
  $('#sVideo').addEventListener('loadedmetadata', function () {
    if (!currentFile) return;
    var parts = [ReelReal.formatSize(currentFile.size)];
    if (isFinite(this.duration)) parts.push(ReelReal.formatClock(Math.round(this.duration)));
    if (this.videoHeight) parts.push(this.videoHeight + 'p');
    $('#sMeta').textContent = parts.join(' · ') + ' · Ready';
  });

  $('#clear').addEventListener('click', function () {
    staged.classList.remove('on');
    currentFile = null;

    /* Detach the blob from the player before revoking it. Revoking a URL that
       is still a <video> src leaves the element pointing at a dead object. */
    var sVideo = $('#sVideo');
    sVideo.pause();
    sVideo.removeAttribute('src');
    sVideo.load();
    $('#sPreview').classList.remove('on');

    if (objectURL) { URL.revokeObjectURL(objectURL); objectURL = null; }
    fileInput.value = '';
  });

  /* ------------------------------------------------------------------------
     Progress
     ---------------------------------------------------------------------- */
  var bar = $('#bar');
  var barFill = bar.querySelector('i');
  var barLabel = $('#barLabel');

  function resetProgress() {
    bar.classList.remove('on');
    barLabel.classList.remove('on');
    barFill.style.width = '0%';
  }

  /* ------------------------------------------------------------------------
     Run analysis
     ---------------------------------------------------------------------- */
  $('#run').addEventListener('click', async function () {
    if (!currentFile) return;

    bar.classList.add('on');
    barLabel.classList.add('on');
    $('#run').disabled = true;

    try {
      var result = await ReelReal.analyzeVideo(currentFile, {
        onProgress: function (p) {
          barFill.style.width = p.percent + '%';
          barLabel.textContent = p.stage;
        }
      });
      renderReport(result, { playbackURL: objectURL });
      // A mocked result means the probe was right, or the server died since.
      // A mocked result is proof the server really is unreachable, whatever the
      // probe thought - so it settles the question immediately.
      if (result.isMock && serverReachable !== false) {
        probeAttempt = PROBE_DELAYS_MS.length;
        showServerNote('down');
      } else if (!result.isMock && serverReachable !== true) {
        showServerNote('up');
      }
      rememberCheck(result);
    } catch (err) {
      barLabel.textContent = 'Analysis failed: ' + err.message;
    } finally {
      $('#run').disabled = false;
    }
  });

  /* ------------------------------------------------------------------------
     Report rendering
     --------------------------------------------------------------------------
     Pure function of (AnalysisResult, playbackURL). It never inspects the File,
     which is what lets an imported result from the extension render through the
     exact same path with playbackURL = null.
     ---------------------------------------------------------------------- */
  function renderReport(result, opts) {
    opts = opts || {};
    var isSynthetic = result.verdict === 'synthetic';

    $('#rName').textContent = result.fileName;
    $('#rMeta').textContent = [
      ReelReal.formatSize(result.fileSizeBytes),
      ReelReal.formatClock(result.durationSeconds),
      result.resolution,
      result.provenance.summary
    ].join(' · ');

    /* Shown as a percentage. The underlying value is unchanged — people read
       "83%" without hesitating and "0.83" with.

       Inconclusive gets a dash, never a number. The model refused to judge, and
       printing "Likely AI-manipulated: 0%" against that refusal would read as
       the opposite — a confident all-clear. Same reason the flagged duration
       below shows a dash rather than "none". */
    var undecided = result.verdict === 'inconclusive';
    $('#rScore').textContent = undecided
      ? '—'
      : Math.round(result.confidence * 100) + '%';

    /* How far the score sits from the decision threshold, not how accurate the
       model is. A clip scoring far either side is a clearer call than one
       sitting on the line; neither says anything about the model's reliability
       on this kind of footage, which is what the caveat below is for. */
    $('#rBand').textContent = undecided ? '—' : (result.confidenceLabel || '—');

    /* "none" is a finding and must not be shown when the model could not see
       enough of a face to look — that case gets a dash, not a clean answer. */
    var manipulated = result.manipulatedDurationSeconds;
    $('#rDur').textContent =
      result.verdict === 'authentic' ? 'none'
      : result.verdict === 'inconclusive' ? '—'
      : (typeof manipulated === 'number' ? manipulated.toFixed(1) + ' s' : '—');

    var badge = $('#rBadge');
    /* "No signs of manipulation" rather than anything implying the video is
       genuine. This model only looks at faces: a real face with a cloned voice,
       or an edited background, reaches this branch untouched. */
    badge.textContent = isSynthetic ? 'Likely AI-manipulated'
      : result.verdict === 'inconclusive' ? 'Not enough face visible to judge'
      : 'No signs of manipulation';
    /* "clean" is the green treatment. Inconclusive is not a pass, so it keeps
       the neutral/alert styling rather than being coloured like a clean bill. */
    badge.classList.toggle('clean', result.verdict === 'authentic');

    /* The panel itself carries the verdict state, so all three outcomes are
       visually distinct: flagged (default), measured-and-clean (green wash +
       the clean-note line), and not-enough-evidence (dashed, unresolved).
       Driven off result.verdict only — it adds no claim of its own. */
    var panel = $('#report .brutalist-panel');
    if (panel) {
      panel.classList.toggle('state-clean', result.verdict === 'authentic');
      panel.classList.toggle('state-unknown', result.verdict === 'inconclusive');
      panel.classList.toggle('state-flagged', result.verdict === 'synthetic');
    }

    $('#stamp').textContent = 'Analysed in ' + (result.processingTimeMs / 1000).toFixed(1) +
                              's · model ' + result.modelVersion +
                              (result.isMock ? ' · SIMULATED RESULT' : '');

    /* "What we measured": every row is a value this analysis produced. Where a
       measurement is genuinely absent it shows a dash rather than a zero, since
       "0 of 0 frames" reads as a finding when it means the opposite.

       result.artifacts is deliberately not rendered here. Five of its six rows
       are signals this model does not measure at all, and a panel of "not
       checked" placeholders says less than the numbers below. The extension
       still renders that array, so the backend keeps producing it. */
    var p = result.pipeline || {};
    var m = function (id, text) { $('#' + id).textContent = text; };
    var has = function (v) { return typeof v === 'number' && isFinite(v); };

    m('mFace', has(p.framesSampled) && p.framesSampled > 0
      ? p.framesScored + ' of ' + p.framesSampled + ' frames (' +
        Math.round((p.coverage || 0) * 100) + '%)'
      : '—');

    m('mFlagged', has(p.framesScored) && p.framesScored > 0
      ? p.framesFlagged + ' of ' + p.framesScored
      : '—');

    m('mRun', has(p.longestRun) && p.framesFlagged
      ? p.longestRun + (p.longestRun === 1 ? ' frame' : ' frames in a row')
      : '—');

    var seg = result.flaggedSegment;
    m('mSection', seg
      ? ReelReal.formatClock(seg.startSecond) + ' – ' + ReelReal.formatClock(seg.endSecond)
      : '—');

    m('mScore', has(p.meanScore) && p.framesScored
      ? p.meanScore.toFixed(2) + ' average, ' + (p.peakScore || 0).toFixed(2) + ' peak'
      : '—');

    /* Grad-CAM, stated as what it is. A named region means that area drew at
       least 1.5x the attention an evenly spread map would put there; "spread
       out" means the map was computed and nothing stood out, which is a result
       and not a missing value. A dash means Grad-CAM did not run. */
    m('mAttn', p.attentionRegion
      ? p.attentionRegion + ' (' + (p.attentionEnrichment || 0).toFixed(1) +
        '× more than average)'
      : (p.attentionMeasured ? 'Spread out, no single area stood out' : '—'));

    /* Headline + guidance: only present on real (non-mocked) results.
       Hide the whole div when empty so it takes no space in the mock path. */
    var headlineEl = $('#rHeadline');
    var guidanceEl = $('#rGuidance');
    var headlinePanelEl = $('#reportHeadline');
    if (p.headline) {
      headlineEl.textContent = p.headline;
      guidanceEl.textContent = p.guidance || '';
      headlinePanelEl.style.display = '';
    } else {
      headlineEl.textContent = '';
      guidanceEl.textContent = '';
      headlinePanelEl.style.display = 'none';
    }

    /* Evidence sentences: list from pipeline.evidence (real results only).
       Each entry is a plain-English sentence the pipeline wrote from a real number. */
    var evidenceList = $('#evidenceList');
    var evidencePanel = $('#evidencePanel');
    evidenceList.innerHTML = '';
    var sentences = (p.evidence && p.evidence.length) ? p.evidence : [];
    if (sentences.length) {
      sentences.forEach(function (s) {
        var li = document.createElement('li');
        li.textContent = s;
        evidenceList.appendChild(li);
      });
      evidencePanel.style.display = '';
    } else {
      evidencePanel.style.display = 'none';
    }

    /* Video preview. Results imported from the extension have no playable blob. */
    var video = $('#rVideo');
    var ghost = $('#vGhost');
    if (opts.playbackURL) {
      video.src = opts.playbackURL;
      video.style.display = 'block';
      ghost.style.display = 'none';
    } else {
      video.removeAttribute('src');
      video.style.display = 'none';
      ghost.style.display = 'grid';
      ghost.textContent = opts.ghostText || 'Preview unavailable';
    }

    renderTimeline(result, video, Boolean(opts.playbackURL));

    /* The signed receipt, offered only when the server actually issued one.
       A mocked result has no receipt and must never appear to have one: the
       whole point of the signature is that it distinguishes a real analysis
       from an invented one. */
    var saveBtn = $('#saveReceipt');
    var verifyLink = $('#openVerify');
    saveBtn.hidden = !result.receipt;
    verifyLink.hidden = !result.receipt;
    if (result.receipt) {
      saveBtn.onclick = function () { downloadReceipt(result.receipt, result.fileName); };
    }

    $('#report').classList.add('on');
    $('#report').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function renderTimeline(result, video, canSeek) {
    var bars = $('#bars');
    bars.innerHTML = '';

    // One <button> per second. Buttons (not divs) so the timeline is keyboard
    // navigable and screen-reader announceable for free.
    result.timeline.forEach(function (point) {
      var b = document.createElement('button');
      b.style.height = Math.max(12, point.score * 100) + '%';
      b.className = point.label === 'synthetic' ? 'f' : (point.label === 'uncertain' ? 'w' : '');
      b.title = ReelReal.formatClock(point.second) + ' — score ' + point.score.toFixed(2);
      b.setAttribute('aria-label', 'Second ' + point.second + ', score ' + point.score.toFixed(2));

      b.addEventListener('click', function () {
        if (canSeek) {
          video.currentTime = point.second;
          video.play().catch(function () {});
        }
        markCurrent(point.second);
      });

      bars.appendChild(b);
    });

    $('#axisEnd').textContent = ReelReal.formatClock(result.durationSeconds);

    var seg = result.flaggedSegment;
    $('#axisMid').textContent = seg
      ? 'Flagged segment: ' + ReelReal.formatClock(seg.startSecond) + '–' + ReelReal.formatClock(seg.endSecond)
      : 'No flags';
    $('#axisMid').style.color = seg ? 'var(--fake)' : 'var(--text-muted)';

    function markCurrent(index) {
      Array.prototype.forEach.call(bars.children, function (el, n) {
        el.classList.toggle('now', n === index);
      });
    }

    // Keep the highlighted bar in sync while the clip plays.
    video.ontimeupdate = function () { markCurrent(Math.floor(video.currentTime)); };
  }

  /* ------------------------------------------------------------------------
     Report actions
     ---------------------------------------------------------------------- */
  $('#again').addEventListener('click', function () {
    $('#report').classList.remove('on');
    $('#handoffNote').classList.remove('on');
    staged.classList.remove('on');
    // Clear any handed-off result so a refresh doesn't resurrect it.
    if (location.hash) history.replaceState(null, '', location.pathname + location.search);
    document.getElementById('detect').scrollIntoView({ behavior: 'smooth' });
  });

  /* ------------------------------------------------------------------------
     Sample clips
     --------------------------------------------------------------------------
     Filled from assets/samples/samples.json. The strip stays hidden when that
     file is missing, which is the normal state of a fresh clone: the test
     footage is licensed for research use and is not redistributable, so no
     clip is ever committed. scripts/make_samples.py builds the folder from
     clips you already hold.

     A sample is fetched as a real Blob and staged exactly like a file you
     picked yourself, so pressing one runs the same analysis over the same
     upload path. Nothing here shortcuts the detector.
     ---------------------------------------------------------------------- */
  /* Two sources, tried in order:

       assets/samples/  built by scripts/make_samples.py from whatever clips you
                        hold locally. Gitignored in full, because that footage is
                        research-licensed and cannot be redistributed.
       assets/demo/     two clips cleared for publication, committed to the repo.

     So a local demo shows your own dataset clips, and the deployed site - and a
     fresh clone, which has no local manifest either - still has something to
     press. Each manifest's files resolve against its own folder, so neither
     knows the other exists. */
  var SAMPLE_SOURCES = [
    { manifest: 'assets/samples/samples.json', base: 'assets/samples/' },
    { manifest: 'assets/demo/samples.json', base: 'assets/demo/' }
  ];

  function loadSamples(index) {
    index = index || 0;
    if (index >= SAMPLE_SOURCES.length) return;   // nothing installed: strip stays hidden
    var source = SAMPLE_SOURCES[index];

    fetch(source.manifest, { cache: 'no-store' })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error('no manifest')); })
      .then(function (manifest) {
        var clips = (manifest && manifest.clips) || [];
        if (!clips.length) return loadSamples(index + 1);

        var row = $('#sampleRow');
        clips.forEach(function (clip) {
          var button = document.createElement('button');
          button.className = 'sample';
          button.type = 'button';

          var name = document.createElement('b');
          name.textContent = clip.title || clip.file;
          var truth = document.createElement('span');
          /* The dataset's ground truth, not a prediction. "Real (Celeb-DF)"
             tells you what the clip is; it deliberately does not tell you what
             the detector will say about it. */
          truth.textContent = clip.groundTruth || 'unlabelled';
          truth.className = 'sample-truth';
          button.appendChild(name);
          button.appendChild(truth);
          if (clip.note) {
            var note = document.createElement('em');
            note.textContent = clip.note;
            button.appendChild(note);
          }

          button.addEventListener('click', function () {
            button.disabled = true;
            var previous = name.textContent;
            name.textContent = 'Loading...';
            fetch(source.base + encodeURIComponent(clip.file))
              .then(function (r) { return r.ok ? r.blob() : Promise.reject(new Error('HTTP ' + r.status)); })
              .then(function (blob) {
                // Wrapped in a File so the staging path, the preview and the
                // upload all behave exactly as they do for a chosen file.
                stageFile(new File([blob], clip.file, { type: blob.type || 'video/mp4' }));
              })
              .catch(function (err) {
                name.textContent = 'Could not load: ' + err.message;
              })
              .finally(function () {
                button.disabled = false;
                if (name.textContent === 'Loading...') name.textContent = previous;
              });
          });

          row.appendChild(button);
        });
        $('#samples').hidden = false;
      })
      .catch(function () { loadSamples(index + 1); });
  }
  loadSamples();

  /* ------------------------------------------------------------------------
     Recent checks
     --------------------------------------------------------------------------
     A short history in localStorage so a run of clips reads as a sequence
     rather than as one isolated verdict. Metadata and the signed receipt only
     - the video itself is never stored anywhere.
     ---------------------------------------------------------------------- */
  var HISTORY_KEY = 'reelreal.history';
  var HISTORY_MAX = 6;

  /* The same three phrases the report's badge uses. Kept here rather than
     reusing ReelReal.labelFor, which maps a numeric SCORE to a band and would
     silently answer "authentic" for every verdict string handed to it.
     Inconclusive keeps its own wording: it is not a pass. */
  var VERDICT_WORDS = {
    synthetic: 'Likely AI-manipulated',
    authentic: 'No signs of manipulation',
    inconclusive: 'Not enough face visible to judge'
  };

  function readHistory() {
    try {
      var raw = localStorage.getItem(HISTORY_KEY);
      var parsed = raw ? JSON.parse(raw) : [];
      return Array.isArray(parsed) ? parsed : [];
    } catch (err) {
      // Private windows and blocked site data throw on access rather than
      // returning null, so every read and write here is guarded.
      return [];
    }
  }

  function writeHistory(entries) {
    try { localStorage.setItem(HISTORY_KEY, JSON.stringify(entries)); }
    catch (err) { /* storage unavailable or full: history is a convenience */ }
  }

  function rememberCheck(result) {
    var entries = readHistory();
    entries.unshift({
      name: result.fileName,
      verdict: result.verdict,
      // Kept as the same rounded value the report shows, so the two can never
      // disagree; a simulated result is marked as one.
      confidence: result.confidence,
      simulated: !!result.isMock,
      at: Date.now(),
      receipt: result.receipt || null
    });
    writeHistory(entries.slice(0, HISTORY_MAX));
    renderHistory();
  }

  function renderHistory() {
    var entries = readHistory();
    var panel = $('#history');
    var row = $('#historyRow');
    row.innerHTML = '';

    if (!entries.length) { panel.hidden = true; return; }
    panel.hidden = false;

    entries.forEach(function (entry) {
      var chip = document.createElement('div');
      chip.className = 'hist hist--' + (entry.simulated ? 'sim' : entry.verdict);

      var name = document.createElement('b');
      name.textContent = entry.name;
      chip.appendChild(name);

      var verdict = document.createElement('span');
      verdict.textContent = entry.simulated
        ? 'simulated - not a measurement'
        : VERDICT_WORDS[entry.verdict] +
          (entry.verdict === 'inconclusive' || typeof entry.confidence !== 'number'
            ? '' : ' \u00b7 ' + Math.round(entry.confidence * 100) + '%');
      chip.appendChild(verdict);

      var when = document.createElement('em');
      when.textContent = new Date(entry.at).toLocaleTimeString();
      chip.appendChild(when);

      // A receipt is only offered when one was actually issued.
      if (entry.receipt) {
        var save = document.createElement('button');
        save.className = 'link-btn';
        save.textContent = 'receipt';
        save.addEventListener('click', function () {
          downloadReceipt(entry.receipt, entry.name);
        });
        chip.appendChild(save);
      }

      row.appendChild(chip);
    });
  }

  $('#clearHistory').addEventListener('click', function () {
    try { localStorage.removeItem(HISTORY_KEY); } catch (err) { /* nothing to do */ }
    renderHistory();
  });

  renderHistory();

  /* ------------------------------------------------------------------------
     Signed receipts
     ---------------------------------------------------------------------- */
  function downloadReceipt(receipt, fileName) {
    var blob = new Blob([JSON.stringify(receipt, null, 2)], { type: 'application/json' });
    var url = URL.createObjectURL(blob);
    var link = document.createElement('a');
    link.href = url;
    link.download = (fileName || 'analysis').replace(/\.[^.]+$/, '') + '.receipt.json';
    document.body.appendChild(link);
    link.click();
    link.remove();
    // Revoke on the next tick: revoking synchronously can cancel the download
    // before the browser has read the blob.
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  /* ------------------------------------------------------------------------
     Handoff from the Chrome extension
     --------------------------------------------------------------------------
     The popup opens this page as:  index.html#result=<base64url(JSON)>
     A fragment is chosen deliberately: it is never transmitted to a server, so
     the report stays as local as the extension's own analysis was.
     ---------------------------------------------------------------------- */
  function importHandoff() {
    var match = /[#&]result=([^&]+)/.exec(location.hash);
    if (!match) return;

    var result = ReelReal.decodeResult(match[1]);
    if (!result) return;

    var note = $('#handoffNote');
    note.textContent = 'Imported from the REEL/REAL extension · ' + result.fileName +
                       ' · the original video stays on your device, so playback is unavailable here.';
    note.classList.add('on');

    renderReport(result, {
      playbackURL: null,
      ghostText: 'Analysed in the extension — video not transferred'
    });
  }

  importHandoff();
})();
