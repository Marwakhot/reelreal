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
     Hero video pair — hover to preview
     ---------------------------------------------------------------------- */
  document.querySelectorAll('.slot').forEach(function (slot) {
    var video = slot.querySelector('video');
    var ghost = slot.querySelector('.ghost');

    // Only reveal the video once a frame is actually decodable, otherwise a
    // missing asset would show as a black rectangle instead of the placeholder.
    video.addEventListener('loadeddata', function () { ghost.style.display = 'none'; });
    video.addEventListener('error', function () { video.style.display = 'none'; });

    slot.addEventListener('mouseenter', function () { video.play().catch(function () {}); });
    slot.addEventListener('mouseleave', function () { video.pause(); });
  });

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
