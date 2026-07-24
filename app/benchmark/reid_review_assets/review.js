const data = JSON.parse(document.getElementById('benchmark-data').textContent);
const storageKey = `algonext-reid-review:${data.video_id}`;
let savedState = null;
try { savedState = JSON.parse(localStorage.getItem(storageKey) || 'null'); } catch (_) { savedState = null; }
if (savedState?.windows && Array.isArray(savedState.windows)) {
  const byIndex = new Map(savedState.windows.map(window => [Number(window.window_index), window]));
  data.windows.forEach(window => {
    const saved = byIndex.get(Number(window.window_index));
    if (!saved) return;
    for (const field of ['target_visibility','candidate_state','target_candidate_id','selected_track_is_target','notes']) {
      if (Object.prototype.hasOwnProperty.call(saved, field)) window[field] = saved[field];
    }
  });
}
const app = document.getElementById('app');
document.getElementById('video-title').textContent = data.video_id;
const frameAnnotations = new Map(
  Array.isArray(savedState?.frames)
    ? savedState.frames.map(item => [String(item.frame_index), item.state])
    : []
);
const frameMetadata = new Map();

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function pct(value) { return value == null ? '—' : `${(Number(value)*100).toFixed(1)}%`; }
function boxStyle(box) {
  if (!box) return '';
  return `left:${box.x*100}%;top:${box.y*100}%;width:${box.w*100}%;height:${box.h*100}%`;
}
function frameKey(frame) { return String(frame.frame_index); }
function ensureFrame(frame) {
  const key = frameKey(frame);
  frameMetadata.set(key, frame);
  if (!frameAnnotations.has(key)) frameAnnotations.set(key, {status:'UNCERTAIN', bbox:null});
  return frameAnnotations.get(key);
}
function evidenceHtml(frame, annotatable) {
  ensureFrame(frame);
  const predicted = frame.bbox ? `<div class="box" style="${boxStyle(frame.bbox)}"></div>` : '';
  const controls = annotatable ? `<div class="frame-actions">
      <button data-frame-action="draw">Disegna box</button>
      <button data-frame-action="not-visible">Target assente</button>
      <button data-frame-action="uncertain">Incerto</button>
    </div>` : '';
  return `<div class="frame-card">
    <div class="frame-wrap ${annotatable ? 'annotatable' : ''}" data-frame-index="${frame.frame_index}">
      <img src="${esc(frame.image_path || '')}" alt="frame ${frame.frame_index}">
      ${predicted}<div class="box truth" hidden></div>
    </div>
    <div class="frame-info"><span>t=${Number(frame.time_sec).toFixed(3)}s · #${frame.frame_index}</span><span class="status"></span></div>
    ${controls}
  </div>`;
}
function candidateHtml(candidate) {
  const evidence = candidate.evidence || [];
  const images = evidence.length ? evidence.map(frame => evidenceHtml(frame, false)).join('') : '<p class="empty">Nessun box candidato persistito in questo artifact.</p>';
  return `<div class="candidate">
    <div class="candidate-head"><strong>ID ${esc(candidate.candidate_id)}</strong><span>score ${candidate.combined_score ?? '—'}</span></div>
    <div class="meta">appearance ${candidate.appearance_similarity ?? '—'} · overlap ${candidate.overlap_score ?? '—'} · geometry ${candidate.geometry_score ?? '—'}</div>
    <div class="candidate-evidence">${images}</div>
  </div>`;
}
function windowHtml(window) {
  const ctx = window.review_context || {};
  const candidates = ctx.candidates || [];
  const candidateIds = candidates.map(c => String(c.candidate_id));
  const listId = `candidate-list-${window.window_index}`;
  const reasons = (ctx.reason_codes || []).map(code => `<span class="badge">${esc(code)}</span>`).join('');
  const decisionClass = String(ctx.decision || '').toLowerCase();
  return `<article class="window" id="window-${window.window_index}" data-window-index="${window.window_index}">
    <div class="window-head">
      <div><h2>Finestra ${window.window_index} · ${Number(window.window_start).toFixed(1)}–${Number(window.window_end).toFixed(1)} s</h2>
      <div class="meta">Direzione: ${esc(ctx.direction || '—')} · copertura: ${ctx.coverage_pct ?? '—'}%</div></div>
      <div><span class="badge ${decisionClass}">${esc(ctx.decision || '—')}</span><span class="badge">selezionato ${esc(ctx.selected_candidate_id || '—')}</span><span class="badge">best ${esc(ctx.best_candidate_id || '—')} / ${ctx.best_score ?? '—'}</span><div>${reasons}</div></div>
    </div>
    <div class="review-fields">
      <label>Visibilità target<select data-field="target_visibility"><option value="UNCERTAIN">Incerta / non valutabile</option><option value="VISIBLE">Visibile</option><option value="NOT_VISIBLE">Non visibile</option></select></label>
      <label>Presenza tra candidati<select data-field="candidate_state"><option value="">—</option><option value="PRESENT">Presente, ID verificato</option><option value="ABSENT">Assente dai candidati</option><option value="UNVERIFIABLE">ID non verificabile</option></select></label>
      <label>ID candidato corretto<input data-field="target_candidate_id" list="${listId}" placeholder="es. 24"><datalist id="${listId}">${candidateIds.map(id => `<option value="${esc(id)}">`).join('')}</datalist></label>
      <label>Track selezionato = target?<select data-field="selected_track_is_target"><option value="">Non giudicato</option><option value="true">Sì</option><option value="false">No</option></select></label>
      <label style="grid-column:1/-1">Note<textarea data-field="notes" placeholder="Occlusione, replay, maglia simile, cambio camera…"></textarea></label>
    </div>
    <div class="section"><h3>Frame per annotazione target</h3><div class="grid">${(window.evidence_frames || []).map(frame => evidenceHtml(frame, true)).join('')}</div></div>
    <div class="section"><h3>Candidati locali persistiti</h3><div class="grid">${candidates.length ? candidates.map(candidateHtml).join('') : '<p class="empty">Questo artifact non contiene evidence bbox dei candidati. Usa UNVERIFIABLE invece di dedurre un ID.</p>'}</div></div>
  </article>`;
}
app.innerHTML = data.windows.map(windowHtml).join('');

function getWindowElement(index) { return document.querySelector(`.window[data-window-index="${index}"]`); }
function syncWindow(index, persist = true) {
  const window = data.windows[index];
  const root = getWindowElement(index);
  const visibilityControl = root.querySelector('[data-field="target_visibility"]');
  const visibility = visibilityControl.value;
  const candidate = root.querySelector('[data-field="candidate_state"]');
  const candidateId = root.querySelector('[data-field="target_candidate_id"]');
  const selectedControl = root.querySelector('[data-field="selected_track_is_target"]');
  const accepted = window.review_context?.decision === 'ACCEPTED';
  if (visibility !== 'VISIBLE') {
    candidate.value = '';
    candidate.disabled = true;
    candidateId.value = '';
    candidateId.disabled = true;
  } else {
    candidate.disabled = false;
    candidateId.disabled = candidate.value !== 'PRESENT';
    if (candidate.value !== 'PRESENT') candidateId.value = '';
  }
  if (!accepted || visibility === 'UNCERTAIN') {
    selectedControl.value = '';
    selectedControl.disabled = true;
  } else if (visibility === 'NOT_VISIBLE') {
    selectedControl.value = 'false';
    selectedControl.disabled = true;
  } else {
    selectedControl.disabled = false;
  }
  window.target_visibility = visibility;
  window.candidate_state = visibility === 'VISIBLE' ? (candidate.value || null) : null;
  window.target_candidate_id = window.candidate_state === 'PRESENT' ? (candidateId.value.trim() || null) : null;
  const selected = selectedControl.value;
  window.selected_track_is_target = selected === '' ? null : selected === 'true';
  window.notes = root.querySelector('[data-field="notes"]').value.trim() || null;
  if (persist) updateProgress();
}
data.windows.forEach((window, index) => {
  const root = getWindowElement(index);
  root.querySelector('[data-field="target_visibility"]').value = window.target_visibility || 'UNCERTAIN';
  root.querySelector('[data-field="candidate_state"]').value = window.candidate_state || '';
  root.querySelector('[data-field="target_candidate_id"]').value = window.target_candidate_id || '';
  root.querySelector('[data-field="selected_track_is_target"]').value = window.selected_track_is_target == null ? '' : String(window.selected_track_is_target);
  root.querySelector('[data-field="notes"]').value = window.notes || '';
  root.querySelectorAll('[data-field]').forEach(input => input.addEventListener('change', () => syncWindow(index)));
  root.querySelector('[data-field="notes"]').addEventListener('input', () => syncWindow(index));
  syncWindow(index, false);
});

function renderFrameState(frameIndex) {
  const state = frameAnnotations.get(String(frameIndex));
  document.querySelectorAll(`.frame-wrap[data-frame-index="${frameIndex}"]`).forEach(wrap => {
    const truth = wrap.querySelector('.truth');
    if (state?.bbox) { truth.hidden = false; truth.style.cssText = boxStyle(state.bbox); } else { truth.hidden = true; }
    const card = wrap.closest('.frame-card');
    if (card) card.querySelector('.status').textContent = state?.status || 'UNCERTAIN';
  });
}
let drawing = null;
document.addEventListener('pointerdown', event => {
  const wrap = event.target.closest('.frame-wrap.annotatable');
  if (!wrap) return;
  if (event.target.closest('button')) return;
  const rect = wrap.getBoundingClientRect();
  drawing = {wrap, key: wrap.dataset.frameIndex, rect, x: Math.max(0, Math.min(1, (event.clientX-rect.left)/rect.width)), y: Math.max(0, Math.min(1, (event.clientY-rect.top)/rect.height))};
  wrap.setPointerCapture?.(event.pointerId);
});
document.addEventListener('pointermove', event => {
  if (!drawing) return;
  const x2 = Math.max(0, Math.min(1, (event.clientX-drawing.rect.left)/drawing.rect.width));
  const y2 = Math.max(0, Math.min(1, (event.clientY-drawing.rect.top)/drawing.rect.height));
  const box = {x:Math.min(drawing.x,x2),y:Math.min(drawing.y,y2),w:Math.abs(x2-drawing.x),h:Math.abs(y2-drawing.y)};
  frameAnnotations.set(drawing.key, {status:'TARGET_BOX', bbox:box}); renderFrameState(drawing.key);
});
document.addEventListener('pointerup', () => {
  if (!drawing) return;
  const state = frameAnnotations.get(drawing.key);
  if (!state?.bbox || state.bbox.w < .002 || state.bbox.h < .002) frameAnnotations.set(drawing.key, {status:'UNCERTAIN',bbox:null});
  renderFrameState(drawing.key); drawing = null; updateProgress();
});
document.addEventListener('click', event => {
  const button = event.target.closest('[data-frame-action]'); if (!button) return;
  const card = button.closest('.frame-card'); const wrap = card.querySelector('.frame-wrap'); const key = wrap.dataset.frameIndex;
  const action = button.dataset.frameAction;
  if (action === 'not-visible') frameAnnotations.set(key, {status:'NOT_VISIBLE',bbox:null});
  if (action === 'uncertain') frameAnnotations.set(key, {status:'UNCERTAIN',bbox:null});
  if (action === 'draw') wrap.scrollIntoView({block:'center',behavior:'smooth'});
  renderFrameState(key); updateProgress();
});

function validateWindows() {
  const errors = [];
  data.windows.forEach((window, index) => {
    syncWindow(index);
    if (window.target_visibility === 'VISIBLE' && !window.candidate_state) errors.push(`Finestra ${index}: specifica lo stato candidati.`);
    if (window.candidate_state === 'PRESENT' && !window.target_candidate_id) errors.push(`Finestra ${index}: specifica l'ID candidato.`);
    if (window.target_visibility === 'VISIBLE' && window.review_context?.decision === 'ACCEPTED' && window.selected_track_is_target == null) errors.push(`Finestra ${index}: giudica il track selezionato.`);
  });
  return errors;
}
function cleanWindowPayload() {
  return {schema_version:'reid-window-annotation-v1',video_id:data.video_id,identity:data.identity,fps:data.fps,windows:data.windows.map(window => ({
    window_index:window.window_index,window_start:window.window_start,window_end:window.window_end,target_visibility:window.target_visibility,candidate_state:window.candidate_state,target_candidate_id:window.target_candidate_id,selected_track_is_target:window.selected_track_is_target,evidence_frames:window.evidence_frames,notes:window.notes
  }))};
}
function framePayload() {
  const frames = [...frameAnnotations.entries()].map(([key,state]) => {
    const meta = frameMetadata.get(key); return {key,state,meta};
  }).filter(item => item.state.status !== 'UNCERTAIN').sort((a,b) => Number(a.key)-Number(b.key)).map(item => ({
    frame_index:Number(item.key),time_sec:item.meta.time_sec,objects:item.state.status === 'TARGET_BOX' ? [{identity:data.identity,ignore:false,bbox:item.state.bbox}] : []
  }));
  return {schema_version:'tracking-annotation-v1',video_id:data.video_id,fps:data.fps,frames};
}
function download(name, payload) {
  const blob = new Blob([JSON.stringify(payload,null,2)+'\n'],{type:'application/json'}); const url = URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=name; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
}
document.getElementById('export-windows').addEventListener('click', () => {
  const errors=validateWindows(); if(errors.length && !confirm(errors.join('\n')+'\n\nEsportare comunque?')) return; download(`${data.video_id}.reid-windows.json`,cleanWindowPayload());
});
document.getElementById('export-frames').addEventListener('click', () => {
  const payload=framePayload(); if(!payload.frames.length) {alert('Nessun frame è stato annotato.'); return;} download(`${data.video_id}.tracking-frames.json`,payload);
});
document.getElementById('jump-next').addEventListener('click', () => {
  const incomplete=data.windows.findIndex((window,index) => {syncWindow(index); return window.target_visibility==='UNCERTAIN' || (window.target_visibility==='VISIBLE' && window.review_context?.decision==='ACCEPTED' && window.selected_track_is_target==null);}); if(incomplete>=0) getWindowElement(incomplete).scrollIntoView({behavior:'smooth'}); else alert('Tutte le finestre hanno una decisione.');
});
document.getElementById('clear-local').addEventListener('click', () => {
  if (!confirm('Cancellare tutte le annotazioni salvate localmente per questo video?')) return;
  localStorage.removeItem(storageKey);
  location.reload();
});
function persistState() {
  const frames = [...frameAnnotations.entries()].map(([frame_index,state]) => ({frame_index:Number(frame_index),state}));
  const windows = data.windows.map(window => ({window_index:window.window_index,target_visibility:window.target_visibility,candidate_state:window.candidate_state,target_candidate_id:window.target_candidate_id,selected_track_is_target:window.selected_track_is_target,notes:window.notes}));
  try { localStorage.setItem(storageKey, JSON.stringify({windows,frames})); } catch (_) {}
}
function updateProgress() {
  const reviewed=data.windows.filter(window => window.target_visibility!=='UNCERTAIN').length;
  const annotated=[...frameAnnotations.values()].filter(value => value.status!=='UNCERTAIN').length;
  document.getElementById('progress').textContent=`Finestre reviewed ${reviewed}/${data.windows.length} · frame scored ${annotated} · autosalvataggio locale attivo`;
  persistState();
}
frameAnnotations.forEach((_, key) => renderFrameState(key));
updateProgress();
