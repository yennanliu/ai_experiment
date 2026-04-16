/* Shared utilities for index.html and history.html */

function scoreColor(n) {
  if (n >= 75) return 'var(--success)';
  if (n >= 50) return 'var(--warning)';
  return 'var(--danger)';
}

function fmtDate(ts) {
  try { return new Date(ts).toLocaleString(undefined, {month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'}); }
  catch { return ts; }
}

function esc(s) {
  return (s || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function vClass(text) {
  const t = (text || '').toLowerCase();
  if (t.includes('strong yes')) return 'v-strong-yes';
  if (t.includes('pass with') || t.includes('maybe')) return 'v-concern';
  if (t.includes('yes')) return 'v-yes';
  if (t.includes('pass')) return 'v-pass';
  if (t.includes('reject') || t.includes('no')) return 'v-no';
  return '';
}

function renderDiff(original, tailored, beforeId, afterId) {
  const aLines = (original || '').split('\n');
  const bLines = (tailored || '').split('\n');
  const bSet = new Set(bLines), aSet = new Set(aLines);
  let bHtml = '', aHtml = '';
  aLines.forEach(l => {
    bHtml += !bSet.has(l)
      ? `<span class="diff-remove">${esc(l)||'&nbsp;'}</span>`
      : (l.trim() ? `<span class="diff-ctx">${esc(l)}</span>` : '');
  });
  bLines.forEach(l => {
    aHtml += !aSet.has(l)
      ? `<span class="diff-add">${esc(l)||'&nbsp;'}</span>`
      : (l.trim() ? `<span class="diff-ctx">${esc(l)}</span>` : '');
  });
  document.getElementById(beforeId).innerHTML = bHtml || '<span style="color:var(--text-3);font-style:italic;">No removed lines</span>';
  document.getElementById(afterId).innerHTML  = aHtml || '<span style="color:var(--text-3);font-style:italic;">No added lines</span>';
}

function renderATSPanel(ats, containerId) {
  const score = ats?.score || 0;
  const missing = ats?.missing_keywords || [];
  const suggestions = ats?.suggestions || [];
  const col = scoreColor(score);
  const msg = score >= 75
    ? '✅ Likely to pass ATS filters'
    : score >= 50
      ? '⚠️ Borderline — add missing keywords'
      : '❌ High risk of being filtered out';

  document.getElementById(containerId).innerHTML = `
    <div class="ats-header">
      <div>
        <div class="ats-big" style="color:${col};">${score}</div>
        <div style="font-size:11px;color:var(--text-3);font-weight:600;">/100 ATS</div>
      </div>
      <div style="flex:1;">
        <div class="ats-track"><div class="ats-fill" style="width:${score}%;background:${col};"></div></div>
        <div class="ats-msg">${msg}</div>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
      <div>
        <div class="sec-title">Missing Keywords</div>
        <div class="tags">${missing.length ? missing.map(k=>`<span class="tag tag-miss">${esc(k)}</span>`).join('') : '<span class="tag tag-none">None 🎉</span>'}</div>
      </div>
      <div>
        <div class="sec-title">Suggestions</div>
        <div class="tags">${suggestions.length ? suggestions.map(k=>`<span class="tag tag-sugg">${esc(k)}</span>`).join('') : '<span class="tag tag-none">None</span>'}</div>
      </div>
    </div>`;
}

function renderFeedbackPanel(recruiter, hm, containerId) {
  // Support both dict (new format) and _raw string (legacy)
  const rVerdict  = recruiter?.verdict  || (recruiter?._raw ? '' : '');
  const rFeedback = recruiter?.feedback || recruiter?._raw || '';
  const hmScore   = hm?.score || 0;
  const hmVerdict = hm?.verdict || '';
  const hmRatio   = hm?.rationale || hm?._raw || '';
  const hmCol = scoreColor(hmScore);

  document.getElementById(containerId).innerHTML = `
    <div class="feedback-grid">
      <div class="feedback-card">
        <div class="fc-title">👤 Recruiter <span class="fc-sub">30-second scan</span></div>
        ${rVerdict ? `<div><span class="verdict ${vClass(rVerdict)}">${esc(rVerdict)}</span></div>` : ''}
        <div class="fb-text">${esc(rFeedback)}</div>
      </div>
      <div class="feedback-card">
        <div class="fc-title">🎯 Hiring Manager <span class="fc-sub">technical fit</span></div>
        ${hmVerdict ? `<div><span class="verdict ${vClass(hmVerdict)}">${esc(hmVerdict)}</span></div>` : ''}
        ${hmScore ? `
          <div class="fit-row">
            <span style="font-size:11px;color:var(--text-3);font-weight:700;white-space:nowrap;">FIT SCORE</span>
            <div class="fit-track"><div class="fit-fill" style="width:${hmScore}%;background:${hmCol};"></div></div>
            <span style="font-size:13px;font-weight:700;color:${hmCol};min-width:32px;text-align:right;">${hmScore}</span>
          </div>` : ''}
        <div class="fb-text">${esc(hmRatio)}</div>
      </div>
    </div>`;
}

async function downloadPDFContent(content, title, pdfOpts = {}) {
  const payload = {
    content,
    title,
    style:        pdfOpts.style        || 'classic',
    page_size:    pdfOpts.page_size    || 'A4',
    margin:       pdfOpts.margin       || 'normal',
    font_scale:   pdfOpts.font_scale   || 1.0,
    accent_color: pdfOpts.accent_color || '#4F46E5',
    max_pages:    pdfOpts.max_pages    || 0,
  };
  const res = await fetch('/export-pdf', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) { alert('PDF generation failed'); return; }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = title + '.pdf'; a.click();
  URL.revokeObjectURL(url);
}

function copyToClipboard(text, feedbackId) {
  navigator.clipboard.writeText(text).then(() => {
    const el = document.getElementById(feedbackId);
    if (!el) return;
    el.textContent = '✓ Copied!';
    setTimeout(() => el.textContent = '', 2000);
  });
}
