// The pages of the local Badminton Coach: home, session, plan and progress.
// Everything comes from this server. Text from users or files is only ever set with textContent.

const $ = (id) => document.getElementById(id);

const CATEGORIES = [['technique', 'Technique'], ['contact', 'Hitting position'], ['footwork', 'Footwork'], ['tactics', 'Tactics']];
const SEVERITY = { priority: 'Work on first', improve: 'Work on', info: 'Note', strength: 'Doing well' };
const SEVERITY_ORDER = { priority: 0, improve: 1, info: 2, strength: 3 };
const BACKENDS = { fast: 'Fast', turbo: 'Turbo', replay: 'Replay of stored poses, not a live analysis' };
const DEFAULT_CHIPS = ['What should I work on first?', 'How do I fix it?', 'Show me the moment', 'What exercises help?',
  'What did I do well?', "What can't you measure?"];
const GEAR = { wall: 'A wall', mat: 'An exercise mat', band: 'A resistance band', dumbbell: 'A dumbbell',
  rope: 'A skipping rope', box: 'A sturdy box or step', kettlebell: 'A kettlebell', cable: 'A cable machine', bench: 'A bench' };
const VIEWER_TIMEOUT_MS = 12000;

function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props)) {
    if (value == null || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key === 'dataset') Object.assign(node.dataset, value);
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : String(value));
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

const button = (text, onclick, props = {}) => el('button', { type: 'button', ...props, onclick }, text);

async function api(path, { json, method } = {}) {
  const init = { method: method || (json === undefined ? 'GET' : 'POST'), headers: {} };
  if (json !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(json);
  }
  const res = await fetch(path, init);
  let data = null;
  try { data = await res.json(); } catch { data = null; }
  if (!res.ok) {
    const err = new Error((data && data.error) || `the app answered ${res.status}`);
    err.status = res.status;
    throw err;
  }
  return { status: res.status, data };
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

function withTimeout(promise, ms) {
  let timer;
  const late = new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('it took too long')), ms); });
  return Promise.race([promise, late]).finally(() => clearTimeout(timer));
}

function noteError(err) {
  (window.__appErrors ||= []).push(String((err && err.stack) || err));
}

function fmtSeconds(s) {
  if (s == null || !Number.isFinite(s)) return '';
  return s < 10 ? `${s.toFixed(1)} s` : `${Math.round(s)} s`;
}

function fmtClip(s) {
  if (!s) return '';
  if (s < 90) return `${Math.round(s)} s`;
  return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

function fmtDate(ts) {
  return new Date(ts * 1000).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

function fmtClock(t) {
  const m = Math.floor(t / 60);
  return `${m}:${(t - m * 60).toFixed(1).padStart(4, '0')}`;
}

function fmtValue(value, unit) {
  if (value == null || !Number.isFinite(value)) return '';
  switch (unit) {
    case 'deg': return `${Math.round(value)}°`;
    case 'share': case 'fraction': return `${Math.round(value * 100)}%`;
    case 's': return `${value.toFixed(2)} s`;
    case 'm': return `${value.toFixed(1)} m`;
    case 'count': return `${Math.round(value)}`;
    case 'heights': return `${value.toFixed(2)} × body height`;
    case 'heights/s': return `${value.toFixed(1)} body heights a second`;
    default: return value.toFixed(2);
  }
}

function fmtGuide(target, unit) {
  if (!Array.isArray(target) || target.length !== 2) return '';
  const [lo, hi] = target;
  if ((unit === 'share' || unit === 'fraction') && lo <= 0 && hi >= 1) return '';
  const bare = (v) => fmtValue(v, unit).replace(' × body height', '').replace(' body heights a second', '');
  const tail = fmtValue(hi, unit).slice(bare(hi).length);
  return `practice guide ${bare(lo)}–${bare(hi)}${tail}`;
}

function metricLine(finding) {
  const metrics = finding.metrics || [];
  const metric = metrics.find((m) => m.value != null) || null;
  if (!metric) {
    if (!metrics.length) return '';
    const guide = fmtGuide(metrics[0].target, metrics[0].unit);
    return `${metrics[0].label}: not measured in this clip${guide ? ` · ${guide}` : ''}`;
  }
  const guide = fmtGuide(metric.target, metric.unit);
  return `${metric.label}: ${fmtValue(metric.value, metric.unit)}${guide ? ` · ${guide}` : ''}`;
}

function backendWords(name, health) {
  if (health && health.backend && health.backend.name === name) return health.backend.label;
  return BACKENDS[name] || name || 'an unknown analyser';
}

function sampleTitle(stem) {
  const match = /^(\d+)-(.+)$/.exec(stem);
  const words = (match ? match[2] : stem).replace(/[-_]+/g, ' ');
  const nice = words.charAt(0).toUpperCase() + words.slice(1);
  return match ? `${match[1]} · ${nice}` : nice;
}

// ---------------------------------------------------------------------------------------------
// Home

let health = null;

async function loadHealth() {
  const { data } = await api('/api/health');
  health = data;
  const pill = $('warm');
  if (pill) {
    const warm = data.warm || {};
    pill.classList.toggle('ready', warm.state === 'ready');
    pill.classList.toggle('failed', warm.state === 'failed');
    if (warm.state === 'ready') {
      pill.textContent = `Analyser ready · ${BACKENDS[data.backend && data.backend.name] || 'unknown'}`;
      pill.title = data.backend ? data.backend.label : '';
    } else if (warm.state === 'failed') {
      pill.textContent = 'The analyser could not start';
      pill.title = warm.message || '';
    } else {
      pill.textContent = 'Warming up the analyser';
    }
  }
  return data;
}

async function watchWarmUp() {
  while (!health || !health.warm || health.warm.state === 'warming') {
    await sleep(1000);
    try { await loadHealth(); } catch { /* the next try says more */ }
  }
}

function showHomeError(message) {
  const box = $('home-error');
  box.textContent = message;
  box.hidden = !message;
}

async function startHome() {
  $('home').hidden = false;
  document.title = 'Badminton Coach';
  const samples = $('samples');
  samples.replaceChildren(...(health.samples || []).map((stem) => button(sampleTitle(stem), () => trySample(stem), {
    dataset: { sample: stem }, 'aria-label': `Try sample clip ${stem}`,
  })));
  if (!(health.samples || []).length) samples.append(el('p', { class: 'muted', text: 'No sample clips were found.' }));
  wireDrop();
  await renderSessions();
}

async function renderSessions() {
  const { data } = await api('/api/sessions');
  const list = $('sessions');
  list.replaceChildren(...data.sessions.map((s) => {
    let state;
    if (s.status === 'error') state = s.error || 'Could not be analysed';
    else if (s.status === 'ready') state = `${s.to_work_on} to work on`;
    else state = 'Analysing';
    const timing = s.analysis_s != null ? `Analysed in ${fmtSeconds(s.analysis_s)} · ${BACKENDS[s.backend] || s.backend || ''}` : '';
    return el('li', {}, el('a', { href: `/?s=${encodeURIComponent(s.id)}`, dataset: { session: s.id } },
      el('span', { class: 'name', text: s.video_name }),
      el('span', { class: 'muted small', text: fmtDate(s.created) }),
      el('span', { class: 'small', text: state }),
      el('span', { class: 'muted small', text: [s.duration_s ? `${fmtClip(s.duration_s)} clip` : '', timing].filter(Boolean).join(' · ') })));
  }));
  $('no-sessions').hidden = data.sessions.length > 0;
}

async function trySample(stem) {
  showHomeError('');
  try {
    const { data } = await api('/api/sample', { json: { name: stem } });
    location.assign(`/?s=${encodeURIComponent(data.id)}`);
  } catch (err) {
    showHomeError(err.message);
  }
}

function wireDrop() {
  const drop = $('drop');
  const input = $('file');
  input.addEventListener('change', () => { if (input.files[0]) upload(input.files[0]); });
  for (const name of ['dragenter', 'dragover']) {
    drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.add('over'); });
  }
  for (const name of ['dragleave', 'drop']) {
    drop.addEventListener(name, (event) => { event.preventDefault(); drop.classList.remove('over'); });
  }
  drop.addEventListener('drop', (event) => {
    const file = event.dataTransfer && event.dataTransfer.files[0];
    if (file) upload(file);
  });
}

function upload(file) {
  showHomeError('');
  if (!/\.(mp4|mov|webm)$/i.test(file.name)) {
    showHomeError('Send an .mp4, .mov or .webm video.');
    return;
  }
  $('upload').hidden = false;
  const bar = $('upload-bar');
  const text = $('upload-text');
  const started = performance.now();
  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload');
  xhr.setRequestHeader('X-Filename', encodeURIComponent(file.name));
  xhr.upload.addEventListener('progress', (event) => {
    if (!event.lengthComputable) return;
    const share = event.loaded / event.total;
    bar.style.width = `${(share * 100).toFixed(1)}%`;
    text.textContent = `Uploading ${file.name}: ${Math.round(share * 100)}% · ${fmtSeconds((performance.now() - started) / 1000)}`;
  });
  xhr.addEventListener('load', () => {
    let data = null;
    try { data = JSON.parse(xhr.responseText); } catch { data = null; }
    if (xhr.status === 201 && data && data.id) {
      location.assign(`/?s=${encodeURIComponent(data.id)}`);
    } else {
      $('upload').hidden = true;
      showHomeError((data && data.error) || `The upload failed (${xhr.status}).`);
    }
  });
  xhr.addEventListener('error', () => {
    $('upload').hidden = true;
    showHomeError('The upload stopped. Is the app still running?');
  });
  text.textContent = `Uploading ${file.name}`;
  xhr.send(file);
}

// ---------------------------------------------------------------------------------------------
// Session

const view = {
  id: null, status: null, bundle: null, revision: null, player: 'near', selected: null,
  exercises: new Map(), context: {}, log: [], chatBusy: false, viewer: null, workingSince: null,
  exportTimer: null, lastLongPoll: 0, pending: null, swapping: false, stopped: false, problem: '',
};

function saveChat() {
  try {
    sessionStorage.setItem(`coach-chat:${view.id}`, JSON.stringify({ log: view.log.slice(-60), context: view.context }));
  } catch { /* storage full or blocked; the chat still works */ }
}

function restoreChat() {
  try {
    const saved = JSON.parse(sessionStorage.getItem(`coach-chat:${view.id}`) || 'null');
    if (saved && Array.isArray(saved.log)) {
      view.log = saved.log;
      view.context = saved.context && typeof saved.context === 'object' ? saved.context : {};
    }
  } catch { view.log = []; }
}

// Saved turns are drawn once the bundle is here, because their buttons point at its findings.
function drawChat() {
  $('chat-log').replaceChildren();
  for (const turn of view.log) drawTurn(turn);
}

async function startSession(id) {
  view.id = id;
  $('session').hidden = false;
  window.coachApp = { viewer: null, view };
  wireSession();
  try {
    const { data } = await api('/api/exercises');
    for (const exercise of data.exercises) view.exercises.set(exercise.id, exercise);
  } catch { /* exercise chips stay hidden */ }
  restoreChat();
  setChips(chipsFor([]));
  await follow();
}

async function follow() {
  let since = null;
  let failures = 0;
  for (;;) {
    let status;
    const query = since == null ? '' : `?since=${since}&wait=20`;
    try {
      ({ data: status } = await api(`/api/session/${encodeURIComponent(view.id)}${query}`));
      failures = 0;
    } catch (err) {
      if (err.status === 404) {
        showFailure('This session does not exist. It may have been deleted.');
        return;
      }
      failures += 1;
      setStatusText(`Lost touch with the app (${err.message}). Trying again.`);
      await sleep(Math.min(10000, 1000 * failures));
      continue;
    }
    try {
      await applyStatus(status);
      since = status.stamp;
    } catch (err) {
      setStatusText(`This result could not be shown: ${err.message}. Trying again.`);
      await sleep(2000);
    }
    if (status.status === 'error' || view.stopped) return;
  }
}

async function applyStatus(status) {
  view.status = status;
  document.title = `${status.video_name} · Badminton Coach`;
  $('title').textContent = status.video_name;
  // A tab without this conversation's saved chat learns of a stop for pain from the app.
  if (status.chat_stopped && view.context.pain_stop !== true) {
    view.context = { ...view.context, pain_stop: true };
    saveChat();
    setChips(chipsFor([]));
  }
  if (status.status === 'error') {
    showFailure(status.error ? status.error.message : 'Something went wrong while analysing this clip.');
    return;
  }
  if (status.revision != null && status.revision !== view.revision) {
    view.pending = status;
    if (view.viewer) swapRevisions();
    else await mountViewer(status);
  }
  if (view.stopped) return;
  renderLabels(status.labels || {});
  renderStatus(status);
}

// One revision at a time: the viewer, the cards and the chat all read the same bundle object.
async function mountViewer(status) {
  let Viewer;
  try {
    ({ Viewer } = await import('/static/viewer.js'));
  } catch (err) {
    stopWith('The clip viewer could not load', `${err.message}. Reload the page. If that does not help, restart the app.`);
    return;
  }
  const viewer = new Viewer();
  viewer.onExercise = (id) => openDrawer([id]);
  viewer.onRefine = (window_) => refineWindow(window_);
  viewer.onSelect = (findingId, player) => followViewer(findingId, player);
  $('working').hidden = true;
  $('layout').hidden = false;
  try {
    await withTimeout(viewer.mount($('viewer'), status.bundle_url), VIEWER_TIMEOUT_MS);
  } catch (err) {
    noteError(err);
    stopWith('This clip cannot be shown here', err && err.message === 'video'
      ? 'The analysis finished, but this browser could not play the video file. An MP4 (H.264) copy of the clip should work.'
      : `The analysis finished, but the result could not be shown (${err.message}). Reload the page to try again.`);
    return;
  }
  view.viewer = viewer;
  window.coachApp.viewer = viewer;
  view.bundle = viewer.bundle;
  view.revision = status.revision;
  if (!view.bundle.players.includes(view.player)) view.player = view.bundle.players[0] || 'near';
  renderGroups();
  drawChat();
  const start = findFinding(view.context.finding_id, view.context.player) || topFinding(view.player);
  if (start) select(start);
}

// A newer revision waits while a slow-motion moment plays, so the moment is not cut off. Only the newest
// waiting revision is shown.
async function swapRevisions() {
  if (view.swapping) return;
  view.swapping = true;
  try {
    let told = false;
    while (view.pending && view.pending.revision !== view.revision && !view.stopped) {
      if (view.viewer.state().mode === 'moment') {
        if (!told) renderStatus(view.status);
        told = true;
        await sleep(300);
        continue;
      }
      await swapRevision(view.pending);
    }
  } finally {
    view.swapping = false;
  }
  if (view.status && !view.stopped) renderStatus(view.status);
}

async function swapRevision(status) {
  const viewer = view.viewer;
  const before = viewer.bundle;
  try {
    await withTimeout(viewer.update(status.bundle_url), VIEWER_TIMEOUT_MS);
  } catch (err) {
    noteError(err);
    view.problem = `The newer result could not be shown (${err.message}). Reload the page to see it.`;
    view.pending = null;
  }
  if (viewer.bundle === before) return;
  view.bundle = viewer.bundle;
  view.revision = status.revision;
  renderGroups();
  // update() keeps the viewer's finding when the new revision still has it, without calling onSelect.
  const kept = viewer.state();
  followViewer(kept.findingId, kept.player);
}

function waitingRevision() {
  return !!(view.pending && view.viewer && view.pending.revision !== view.revision);
}

function findFinding(id, player) {
  if (!view.bundle || !id) return null;
  return view.bundle.findings.find((f) => f.id === id && f.player === player) || null;
}

function topFinding(player) {
  return view.bundle.findings
    .filter((f) => f.player === player)
    .sort((a, b) => (SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]) || (b.score - a.score))[0] || null;
}

// The viewer's own Previous, Next and Compare buttons change its selection; the cards follow it.
function followViewer(findingId, player) {
  const finding = findFinding(findingId, player);
  const key = finding ? `${finding.player}:${finding.id}` : null;
  if (key === view.selected) return;
  if (finding) {
    markSelected(finding, { scroll: true });
  } else {
    view.selected = null;
    for (const node of document.querySelectorAll('#groups .card')) node.classList.remove('selected');
  }
}

function stopWith(title, message) {
  view.stopped = true;
  showFailure(message, { title, analysed: true });
}

function showFailure(message, { title = 'This clip could not be analysed', analysed = false } = {}) {
  $('working').hidden = true;
  $('layout').hidden = true;
  $('notice').hidden = true;
  $('failure').hidden = false;
  $('failure-title').textContent = title;
  $('failure-text').textContent = message;
  $('failure-foot').hidden = analysed;
  setStatusText('');
}

function setStatusText(text) {
  $('status').textContent = text;
}

function renderStatus(status) {
  const first = (status.timings && status.timings.first) || {};
  const latest = status.timings && status.timings.latest;
  const analysed = view.bundle != null;
  $('layout').hidden = !analysed;
  $('failure').hidden = true;
  const busy = status.status === 'queued' || status.status === 'preview';
  if (busy) {
    if (view.workingSince == null) view.workingSince = performance.now();
    $('working').hidden = analysed;
    $('working-text').textContent = status.detail || 'Analysing the clip';
    tickWorking();
  } else {
    view.workingSince = null;
    $('working').hidden = true;
  }

  const status_ = $('status');
  status_.replaceChildren();
  if (analysed && first.total != null) {
    const clip = status.duration_s ? `, from a ${fmtClip(status.duration_s)} clip` : '';
    status_.append(el('strong', { text: `Ready in ${fmtSeconds(first.total)}${clip}.` }),
      ` Analysed with ${backendWords(first.backend, health)}.`);
    const extra = [];
    if (first.upload_s != null) extra.push(`upload ${fmtSeconds(first.upload_s)}`);
    if (first.queue_s != null && first.queue_s >= 0.05) extra.push(`waiting in line ${fmtSeconds(first.queue_s)}`);
    if (first.warm_wait_s) extra.push(`waiting for the analyser to warm up ${fmtSeconds(first.warm_wait_s)}`);
    if (extra.length) status_.append(` Not counted in that time: ${extra.join(', ')}.`);
    status_.append(el('span', { class: 'tag', text: 'Measured on this computer · experimental' }));
  }
  if (status.status === 'refining') {
    const windows = status.windows;
    const count = windows && windows.total ? ` (${windows.done} of ${windows.total})` : '';
    status_.append(el('br'), `${status.detail || 'Sharpening key moments'}${count}.`);
  } else if (latest && latest.total != null && status.revision > 0) {
    status_.append(el('br'), latest.kind === 'all'
      ? `Every frame measured in ${fmtSeconds(latest.total)}.`
      : `Key moments sharpened in ${fmtSeconds(latest.total)}.`);
  }
  if (analysed) {
    status_.append(el('br'), view.bundle.tier === 'refined'
      ? 'Every frame of the clip was measured.'
      : 'Quick analysis: some frames are estimated between measured ones.');
  }

  const notes = [];
  if (status.notice) notes.push(status.notice);
  if (status.long_clip) notes.push(`This clip is longer than ${health.tested_seconds} s, so it takes longer to analyse.`);
  if (view.problem) notes.push(view.problem);
  if (waitingRevision() && view.viewer.state().mode === 'moment') {
    notes.push('A newer result is ready. It shows when this slow-motion moment ends.');
  }
  $('notice').textContent = notes.join(' ');
  $('notice').hidden = notes.length === 0;

  const sharpen = $('sharpen');
  sharpen.hidden = !(health.backend && health.backend.sharpens_windows && view.bundle && view.bundle.tier === 'preview'
    && status.status === 'ready');
  // The server gives a time only for the analyser and clip size it was measured on.
  sharpen.textContent = status.sharpen_all_s ? `Sharpen all (up to about ${status.sharpen_all_s} s)` : 'Sharpen all';
  sharpen.title = 'Measures every frame again. This takes longer than the quick result.';
  $('export').disabled = !health.export || !analysed;
  $('export').title = health.export ? 'Experimental' : 'The review video maker is not available in this version';
  $('plan-open').disabled = !health.plan || !analysed;
}

function tickWorking() {
  if (view.workingSince == null || $('working').hidden) return;
  const so = Math.round((performance.now() - view.workingSince) / 1000);
  const detail = (view.status && view.status.detail) || 'Analysing the clip';
  $('working-text').textContent = so >= 2 ? `${detail} · ${so} s so far` : detail;
  clearTimeout(view.workingTimer);
  view.workingTimer = setTimeout(tickWorking, 1000);
}

function label(role) {
  const labels = (view.status && view.status.labels) || {};
  return labels[role] || (role === 'near' ? 'Near player' : 'Far player');
}

function renderLabels(labels) {
  for (const btn of document.querySelectorAll('.seg button[data-player]')) {
    const role = btn.dataset.player;
    btn.querySelector('.lbl').textContent = labels[role] || label(role);
    btn.setAttribute('aria-pressed', String(role === view.player));
    btn.disabled = !!view.bundle && !view.bundle.players.includes(role);
  }
  for (const role of ['near', 'far']) {
    const input = $(`name-${role}`);
    if (document.activeElement !== input && !input.dataset.dirty) {
      const value = labels[role] || '';
      input.value = /^(Near|Far) player$/.test(value) ? '' : value;
    }
  }
}

function setPlayer(role) {
  if (view.player === role) return;
  view.player = role;
  view.context = { ...view.context, player: role };
  if (view.selected && !view.selected.startsWith(`${role}:`)) view.selected = null;
  renderLabels((view.status && view.status.labels) || {});
  renderGroups();
  saveChat();
}

function sortedFindings(category) {
  return view.bundle.findings
    .filter((f) => f.player === view.player && f.category === category)
    .sort((a, b) => (SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]) || (b.score - a.score));
}

function renderGroups() {
  if (!view.bundle) return;
  const groups = $('groups');
  const scroll = window.scrollY;
  const focusKey = document.activeElement && document.activeElement.closest && document.activeElement.closest('.card')
    ? document.activeElement.closest('.card').dataset.key : null;
  groups.replaceChildren();
  for (const [category, title] of CATEGORIES) {
    const items = sortedFindings(category);
    const section = el('section', { class: 'group', dataset: { category } }, el('h2', { text: title }));
    if (!items.length) section.append(el('p', { class: 'muted small', text: 'Nothing to report here for this player in this clip.' }));
    for (const finding of items) section.append(card(finding));
    groups.append(section);
  }
  if (view.bundle.warnings && view.bundle.warnings.length) {
    groups.append(el('section', { class: 'group' }, el('h2', { text: 'Worth knowing' }),
      el('ul', { class: 'small muted' }, view.bundle.warnings.map((w) => el('li', { text: (health.warnings || {})[w] || w.replace(/_/g, ' ') })))));
  }
  if (focusKey) {
    const again = groups.querySelector(`.card[data-key="${CSS.escape(focusKey)}"]`);
    if (again) again.focus({ preventScroll: true });
  }
  window.scrollTo(0, scroll);
}

function card(finding) {
  const key = `${finding.player}:${finding.id}`;
  const node = el('article', { class: `card${key === view.selected ? ' selected' : ''}`, dataset: { key, finding: finding.id }, tabindex: '-1' });
  const top = el('div', { class: 'top' }, el('span', { class: `badge ${finding.severity}`, text: SEVERITY[finding.severity] || finding.severity }));
  if (finding.tier === 'refined') {
    top.append(el('span', { class: 'badge sharp', text: 'Sharpened', title: 'Measured on every frame around these moments' }));
  }
  node.append(top, el('h3', { text: finding.headline }));
  if (finding.explain) node.append(el('p', { class: 'explain', text: finding.explain }));
  const metric = metricLine(finding);
  if (metric) node.append(el('p', { class: 'metric', text: metric }));
  const row = el('div', { class: 'row' });
  const moments = finding.moments || [];
  const issue = moments.findIndex((m) => m.kind !== 'best');
  const best = moments.findIndex((m) => m.kind === 'best');
  if (issue >= 0) row.append(button('Show me in slow motion', () => showMoment(finding, issue), { class: 'primary', dataset: { action: 'moment' } }));
  if (best >= 0) row.append(button(issue >= 0 ? 'My best moment' : 'Show me in slow motion', () => showMoment(finding, best), { dataset: { action: 'best' } }));
  if (finding.severity !== 'strength') row.append(button('How do I fix it?', () => askAbout(finding, 'How do I fix it?'), { dataset: { action: 'how' } }));
  if (row.children.length) node.append(row);
  const exercises = (finding.exercise_ids || []).map((id) => view.exercises.get(id)).filter(Boolean);
  if (exercises.length) {
    node.append(el('div', { class: 'row' }, el('span', { class: 'small muted', text: 'Gym exercises:' }),
      exercises.map((e) => button(e.name, () => { select(finding); openDrawer([e.id]); }, { class: 'quiet', dataset: { exercise: e.id } }))));
  }
  node.addEventListener('click', (event) => { if (!event.target.closest('button')) select(finding); });
  return node;
}

function markSelected(finding, { scroll = false } = {}) {
  view.selected = `${finding.player}:${finding.id}`;
  if (finding.player !== view.player) setPlayer(finding.player);
  view.context = { ...view.context, finding_id: finding.id, player: finding.player };
  for (const node of document.querySelectorAll('#groups .card')) node.classList.toggle('selected', node.dataset.key === view.selected);
  if (scroll) {
    const node = document.querySelector(`#groups .card[data-key="${CSS.escape(view.selected)}"]`);
    if (node) node.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
  saveChat();
}

function select(finding, options) {
  markSelected(finding, options);
  if (view.viewer) view.viewer.focusFinding(finding.id, finding.player);
}

function showMoment(finding, index) {
  select(finding);
  closeDrawer();
  if (view.viewer) view.viewer.playMoment(finding.id, finding.player, index).catch(noteError);
}

function askAbout(finding, text) {
  select(finding);
  showTab('coach');
  sendChat(text);
}

function showTab(which) {
  const coach = which === 'coach';
  $('tab-findings').setAttribute('aria-selected', String(!coach));
  $('tab-coach').setAttribute('aria-selected', String(coach));
  $('panel-findings').hidden = coach;
  $('panel-coach').hidden = !coach;
  if (coach) $('ask-text').focus({ preventScroll: true });
}

// chat

function setChips(texts) {
  $('chips').replaceChildren(...texts.map((text) => button(text, () => sendChat(text))));
}

// Once the coach has stopped for pain, only its own suggestions are offered, never the training starters.
function chipsFor(suggestions) {
  if (view.context.pain_stop === true) return suggestions || [];
  return suggestions && suggestions.length ? suggestions : DEFAULT_CHIPS;
}

function drawTurn(turn) {
  const log = $('chat-log');
  const node = el('li', { class: `turn ${turn.speaker}` }, el('span', { text: turn.text }));
  if (turn.speaker === 'coach' && turn.reply) {
    const row = actionButtons(turn.reply);
    if (row.children.length) node.append(row);
  }
  log.append(node);
  log.scrollTop = log.scrollHeight;
  return node;
}

function findingFor(action, after) {
  if (!view.bundle) return null;
  const findings = view.bundle.findings || [];
  if (action.kind === 'finding') {
    const role = view.bundle.players[action.arg];
    return findings.find((f) => f.id === action.ref && f.player === role) || null;
  }
  return findings.find((f) => f.id === action.ref && (!after || f.player === after.player)) || null;
}

function actionButtons(reply) {
  const row = el('div', { class: 'row' });
  let finding = null;
  const exercises = [];
  for (const action of reply.actions || []) {
    if (action.kind === 'finding') {
      finding = findingFor(action);
      if (finding) {
        const target = finding;
        row.append(button('Show the card', () => { showTab('findings'); select(target, { scroll: true }); }));
      }
    } else if (action.kind === 'seek') {
      const target = findingFor(action, finding);
      if (target && target.moments[action.arg]) {
        row.append(button('Play this moment', () => showMoment(target, action.arg), { dataset: { action: 'seek' } }));
      }
    } else if (action.kind === 'exercise' && view.exercises.has(action.ref)) {
      exercises.push(action.ref);
      row.append(button(view.exercises.get(action.ref).name, () => openDrawer([action.ref]), { class: 'quiet', dataset: { exercise: action.ref } }));
    }
  }
  if (exercises.length > 1) row.append(button('Show them all', () => openDrawer(exercises)));
  return row;
}

function runActions(reply) {
  let finding = null;
  let seek = null;
  const exercises = [];
  for (const action of reply.actions || []) {
    if (action.kind === 'finding') {
      finding = findingFor(action);
      if (finding) select(finding);
    } else if (action.kind === 'seek') {
      const target = findingFor(action, finding);
      if (target && target.moments[action.arg]) seek = [target, action.arg];
    } else if (action.kind === 'exercise' && view.exercises.has(action.ref)) {
      exercises.push(action.ref);
    }
  }
  if (exercises.length) openDrawer(exercises);
  else if (seek) showMoment(seek[0], seek[1]);
}

function chatContext() {
  const context = { ...view.context, player: view.player };
  if (view.viewer) context.frame = view.viewer.state().frame;
  return context;
}

async function sendChat(text) {
  text = (text || '').trim();
  if (!text || view.chatBusy) return;
  if (!view.bundle) return;
  view.chatBusy = true;
  showTab('coach');
  const userTurn = { speaker: 'user', text };
  view.log.push(userTurn);
  drawTurn(userTurn);
  const waiting = el('li', { class: 'turn coach wait', text: 'Thinking…' });
  $('chat-log').append(waiting);
  const started = performance.now();
  try {
    const { data } = await api('/api/chat', { json: { session: view.id, text, context: chatContext() } });
    view.context = data.context || {};
    waiting.remove();
    const turn = { speaker: 'coach', text: data.reply.text, reply: data.reply, ms: Math.round(performance.now() - started) };
    view.log.push(turn);
    drawTurn(turn);
    runActions(data.reply);
    setChips(chipsFor(data.reply.suggestions));
  } catch (err) {
    waiting.textContent = err.message;
    if (err.status === 400) {
      const stopped = view.context.pain_stop === true;
      view.context = stopped ? { player: view.player, pain_stop: true } : { player: view.player };
    }
  } finally {
    view.chatBusy = false;
    saveChat();
  }
}

// exercise drawer

function exerciseCard(exercise) {
  const media = exercise.gif_url
    ? el('img', { src: exercise.gif_url, alt: `${exercise.name}, shown as a short moving picture`, loading: 'eager' })
    : el('p', { class: 'nogif', text: 'The moving picture for this exercise is not ready yet.' });
  const list = (items) => el('ul', {}, (items || []).map((item) => el('li', { text: item })));
  const row = el('div', { class: 'row' });
  if (exercise.easier && view.exercises.has(exercise.easier)) row.append(button('Easier version', () => openDrawer([exercise.easier])));
  if (exercise.harder && view.exercises.has(exercise.harder)) row.append(button('Harder version', () => openDrawer([exercise.harder])));
  const eased = tooHard().includes(exercise.id);
  const note = !eased ? null : exercise.easier_dose
    ? `You said this felt too hard, so do ${exercise.easier_dose} instead of ${exercise.sets_reps}.`
    : 'You said this felt too hard, so do fewer reps or shorter holds than this.';
  return el('article', { class: 'ex-card', dataset: { exercise: exercise.id } },
    media,
    el('h3', { text: exercise.name }),
    el('p', { class: 'dose', text: eased && exercise.easier_dose ? exercise.easier_dose : exercise.sets_reps }),
    note ? el('p', { class: 'eased', text: note }) : null,
    exercise.why ? el('p', { text: exercise.why }) : null,
    el('h4', { text: 'How to do it' }), list(exercise.cues),
    el('h4', { text: 'Avoid' }), list(exercise.mistakes),
    exercise.safety ? el('p', { class: 'safety', text: exercise.safety }) : null,
    row.children.length ? row : null);
}

function openDrawer(ids) {
  const exercises = ids.map((id) => view.exercises.get(id)).filter(Boolean);
  if (!exercises.length) return;
  if (view.viewer) view.viewer.leaveMoment();
  $('ex-list').replaceChildren(...exercises.map(exerciseCard));
  const drawer = $('drawer');
  if (drawer.hidden) view.drawerReturn = document.activeElement;
  drawer.hidden = false;
  drawer.scrollTop = 0;
  if (!$('panel-coach').contains(document.activeElement)) drawer.focus({ preventScroll: true });
  view.context = { ...view.context, exercise_id: exercises[0].id };
  saveChat();
}

function closeDrawer() {
  const drawer = $('drawer');
  if (drawer.hidden) return;
  drawer.hidden = true;
  if (view.drawerReturn && document.contains(view.drawerReturn)) view.drawerReturn.focus({ preventScroll: true });
}

// plan, export, sharpen, names

function tooHard() {
  const ids = Array.isArray(view.context.too_hard) ? view.context.too_hard : [];
  return [...new Set(ids.filter((id) => typeof id === 'string' && id))];
}

function openPlanDialog() {
  const gear = new Set();
  for (const exercise of view.exercises.values()) for (const item of exercise.equipment || []) gear.add(item);
  const fieldset = $('plan-gear');
  fieldset.replaceChildren(el('legend', { text: 'What do you have?' }),
    ...[...gear].sort().map((item) => el('label', {},
      el('input', { type: 'checkbox', name: 'gear', value: item, checked: true }), GEAR[item] || item)));
  $('plan-for').textContent = `For ${label(view.player)}, from this clip.`;
  const avoid = tooHard();
  const names = avoid.map((id) => (view.exercises.get(id) || { name: id }).name);
  $('plan-avoid').textContent = avoid.length
    ? `You said ${names.join(', ')} felt too hard. The plan swaps ${avoid.length === 1 ? 'it' : 'them'} for another exercise, or keeps an easier dose if nothing else fits.`
    : '';
  $('plan-avoid').hidden = !avoid.length;
  $('plan-msg').hidden = true;
  $('plan-dialog').showModal();
}

async function submitPlan(event) {
  event.preventDefault();
  const level = Number(document.querySelector('input[name=level]:checked').value);
  const weeks = Number($('plan-weeks').value);
  const equipment = [...document.querySelectorAll('input[name=gear]:checked')].map((input) => input.value);
  const message = $('plan-msg');
  message.hidden = true;
  try {
    const { data } = await api('/api/plan', {
      json: { session: view.id, player: view.player, weeks, level, equipment, avoid: tooHard() },
    });
    if (data.url) {
      location.assign(data.url);
      return;
    }
    message.textContent = data.message || 'There is nothing to plan yet.';
  } catch (err) {
    message.textContent = err.message;
  }
  message.hidden = false;
}

async function startExport() {
  const note = $('action-note');
  clearTimeout(view.exportTimer);
  try {
    let { data } = await api('/api/export', { json: { session: view.id } });
    const revision = view.revision;
    while (data.status === 'running') {
      const sofar = data.seconds ? ` ${fmtClip(data.seconds)} so far.` : '';
      note.textContent = `Making the review video (experimental).${sofar} You can keep using the page.`;
      await sleep(1500);
      ({ data } = await api(`/api/export/${encodeURIComponent(view.id)}?revision=${revision}`));
    }
    if (data.status === 'ready' && data.url) {
      const took = data.seconds ? ` It took ${fmtClip(data.seconds)}.` : '';
      note.replaceChildren(`The review video is ready (experimental).${took} `,
        el('a', { href: data.url, download: 'review.mp4', text: 'Save the review video' }));
      if (data.analysis_url) {
        note.append(' · ', el('a', { href: data.analysis_url, download: 'analysis.html', text: 'Save matching analysis' }));
      }
      if (data.evidence_url) {
        note.append(' · ', el('a', { href: data.evidence_url, download: 'analysis.json', text: 'Save evidence and narration' }));
      }
    } else {
      note.textContent = data.message || 'The review video could not be made.';
    }
  } catch (err) {
    note.textContent = err.message;
  }
}

async function sharpenAll() {
  const note = $('action-note');
  try {
    await api('/api/refine', { json: { session: view.id, all: true } });
    note.textContent = 'Measuring every frame. This takes longer than the quick analysis; the page updates by itself.';
  } catch (err) {
    note.textContent = err.message;
  }
}

// The viewer's "Preview estimate" button asks for the first estimated stretch of the clip.
async function refineWindow(window_) {
  const note = $('action-note');
  if (!window_) {
    note.textContent = 'There is no single stretch to sharpen. Use "Sharpen all" instead.';
    return;
  }
  const fps = (view.bundle && view.bundle.fps) || 30;
  const span = `${fmtClock(window_.start_frame / fps)}–${fmtClock((window_.end_frame + 1) / fps)}`;
  try {
    const { data } = await api('/api/refine', {
      json: { session: view.id, window: { start_frame: window_.start_frame, end_frame: window_.end_frame } },
    });
    note.textContent = data && data.queued
      ? `Measuring ${span} frame by frame. The page updates by itself.`
      : (data && data.message) || `${span} is already being measured.`;
  } catch (err) {
    note.textContent = err.message;
  }
}

async function saveNames(event) {
  event.preventDefault();
  const message = $('names-msg');
  const near = $('name-near').value.trim() || 'Near player';
  const far = $('name-far').value.trim() || 'Far player';
  try {
    const { data } = await api('/api/labels', { json: { session: view.id, near, far } });
    if (view.status) view.status.labels = data.labels;
    for (const role of ['near', 'far']) delete $(`name-${role}`).dataset.dirty;
    renderLabels(data.labels);
    message.textContent = 'Saved';
  } catch (err) {
    message.textContent = err.message;
  }
}

function wireSession() {
  for (const btn of document.querySelectorAll('.seg button[data-player]')) {
    btn.addEventListener('click', () => setPlayer(btn.dataset.player));
  }
  $('tab-findings').addEventListener('click', () => showTab('findings'));
  $('tab-coach').addEventListener('click', () => showTab('coach'));
  $('ask').addEventListener('submit', (event) => {
    event.preventDefault();
    const input = $('ask-text');
    const text = input.value;
    input.value = '';
    sendChat(text);
  });
  $('drawer-close').addEventListener('click', closeDrawer);
  $('plan-open').addEventListener('click', openPlanDialog);
  $('plan-form').addEventListener('submit', submitPlan);
  $('plan-cancel').addEventListener('click', () => $('plan-dialog').close());
  $('export').addEventListener('click', startExport);
  $('sharpen').addEventListener('click', sharpenAll);
  $('names').addEventListener('submit', saveNames);
  for (const role of ['near', 'far']) $(`name-${role}`).addEventListener('input', (event) => { event.target.dataset.dirty = '1'; });
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && !$('drawer').hidden && !$('plan-dialog').open) {
      closeDrawer();
      event.preventDefault();
    }
  });
}

// ---------------------------------------------------------------------------------------------
// Plan page

function humanFinding(id) {
  const last = id.split('.').pop() || id;
  return last.replace(/_/g, ' ');
}

async function startPlan() {
  const id = new URLSearchParams(location.search).get('id');
  const error = $('plan-error');
  $('print').addEventListener('click', () => window.print());
  if (!id) {
    error.textContent = 'No plan was chosen. Make one from a session page.';
    error.hidden = false;
    return;
  }
  let saved;
  try {
    ({ data: saved } = await api(`/api/plan/${encodeURIComponent(id)}`));
  } catch (err) {
    error.textContent = err.message;
    error.hidden = false;
    return;
  }
  const plan = saved.plan;
  $('back').href = `/?s=${encodeURIComponent(saved.session_id)}`;
  const headlines = new Map();
  try {
    const res = await fetch(`/api/session/${encodeURIComponent(saved.session_id)}/bundle`);
    if (res.ok) for (const f of (await res.json()).findings) headlines.set(f.id, f.headline);
  } catch { /* the plan still reads well without them */ }
  const weeks = plan.weeks.length;
  const who = saved.label || plan.player;
  document.title = `Practice plan for ${who} · Badminton Coach`;
  $('plan-heading').textContent = `${weeks === 1 ? 'One-week' : `${weeks}-week`} practice plan for ${who}`;
  const levelWords = { 1: 'new to gym work', 2: 'some gym experience', 3: 'trains often' }[plan.level] || '';
  const gear = plan.equipment == null ? 'any equipment'
    : plan.equipment.length ? plan.equipment.map((g) => (GEAR[g] || g).toLowerCase()).join(', ') : 'no equipment';
  $('plan-meta').textContent = `From ${saved.video_name || 'a session'}, made ${fmtDate(saved.created)}. Level: ${levelWords}. Using ${gear}.`;

  const body = $('plan-body');
  const exercises = new Map();
  try {
    const { data } = await api('/api/exercises');
    for (const e of data.exercises) exercises.set(e.id, e);
  } catch { /* names and doses are in the plan itself */ }
  if (plan.findings && plan.findings.length) {
    body.append(el('section', {}, el('h2', { text: 'What this plan works on' }),
      el('ul', {}, plan.findings.map((f) => el('li', { text: headlines.get(f) || humanFinding(f) })))));
  }
  body.append(el('section', {}, el('h2', { text: 'Warm-up, every session' }), el('p', { text: plan.warm_up })));
  plan.weeks.forEach((week, w) => {
    const section = el('section', { class: `week${w === 0 ? ' first' : ''}` }, el('h2', { text: `Week ${week.week}` }));
    if (week.focus) section.append(el('p', { text: week.focus }));
    for (const session of week.sessions) {
      section.append(el('h3', { text: `Session ${session.session}` }));
      section.append(el('ol', { class: 'blocks' }, session.blocks.map((block) => {
        const exercise = exercises.get(block.exercise_id);
        const media = exercise && exercise.gif_url ? el('img', { src: exercise.gif_url, alt: block.name }) : el('span');
        return el('li', {}, media, el('div', {},
          el('strong', { text: block.name }), ' ', el('span', { class: 'dose', text: `· ${block.dose}` }),
          block.reason ? el('p', { class: 'why', text: block.reason }) : null));
      })));
    }
    body.append(section);
  });
  body.append(el('section', {}, el('p', { class: 'safety', text: plan.safety })));
}

// ---------------------------------------------------------------------------------------------
// Progress page

const SVG = 'http://www.w3.org/2000/svg';

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  for (const child of children) if (child != null) node.append(child instanceof Node ? child : String(child));
  return node;
}

function chart(metric) {
  const width = 520;
  const height = 190;
  const pad = { left: 70, right: 16, top: 14, bottom: 36 };
  const values = metric.points.map((p) => p.value);
  const target = Array.isArray(metric.target) ? metric.target : null;
  const pool = values.concat(target ? target.filter((v) => Number.isFinite(v)) : []);
  let lo = Math.min(...pool);
  let hi = Math.max(...pool);
  if (metric.unit === 'share' || metric.unit === 'fraction') { lo = Math.max(0, lo); hi = Math.min(1, hi); }
  if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
  const span = hi - lo;
  lo -= span * 0.08;
  hi += span * 0.08;
  const x = (i) => pad.left + (values.length === 1 ? (width - pad.left - pad.right) / 2 : i * (width - pad.left - pad.right) / (values.length - 1));
  const y = (v) => pad.top + (hi - v) * (height - pad.top - pad.bottom) / (hi - lo);
  const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': `${metric.label} across sessions` });
  if (target) {
    const top = y(Math.min(hi, target[1]));
    const bottom = y(Math.max(lo, target[0]));
    if (bottom > top) root.append(svg('rect', { class: 'band', x: pad.left, y: top, width: width - pad.left - pad.right, height: bottom - top }));
  }
  root.append(svg('line', { class: 'axis', x1: pad.left, y1: height - pad.bottom, x2: width - pad.right, y2: height - pad.bottom }));
  root.append(svg('text', { x: pad.left - 8, y: y(hi - span * 0.08) + 4, 'text-anchor': 'end' }, fmtValue(hi - span * 0.08, metric.unit).split(' ')[0]));
  root.append(svg('text', { x: pad.left - 8, y: y(lo + span * 0.08) + 4, 'text-anchor': 'end' }, fmtValue(lo + span * 0.08, metric.unit).split(' ')[0]));
  if (values.length > 1) {
    root.append(svg('polyline', { class: 'line', points: values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ') }));
  }
  metric.points.forEach((p, i) => {
    root.append(svg('circle', { class: 'pt', cx: x(i).toFixed(1), cy: y(p.value).toFixed(1), r: 5 },
      svg('title', {}, `${fmtDate(p.created)} · ${p.video_name}: ${fmtValue(p.value, metric.unit)}`)));
    if (values.length <= 8 || i === 0 || i === values.length - 1) {
      root.append(svg('text', { x: x(i).toFixed(1), y: height - 12, 'text-anchor': 'middle' },
        new Date(p.created * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })));
    }
  });
  return root;
}

const TREND_WORDS = { better: 'Better', worse: 'Worse', same: 'About the same', first: 'One reading' };

async function startProgress() {
  const select_ = $('player');
  const empty = $('progress-empty');
  const error = $('progress-error');
  let players;
  try {
    ({ data: { players } } = await api('/api/players'));
  } catch (err) {
    error.textContent = err.message;
    error.hidden = false;
    return;
  }
  if (!players.length) {
    empty.textContent = 'Name a player on a session page first. Progress only joins sessions of the same name, so different people are never mixed.';
    empty.hidden = false;
    $('pick').hidden = true;
    return;
  }
  select_.replaceChildren(...players.map((name) => el('option', { value: name, text: name })));
  const wanted = new URLSearchParams(location.search).get('player');
  if (wanted && players.includes(wanted)) select_.value = wanted;
  const draw = async () => {
    const name = select_.value;
    history.replaceState(null, '', `/progress?player=${encodeURIComponent(name)}`);
    error.hidden = true;
    const charts = $('charts');
    charts.replaceChildren();
    try {
      const { data } = await api(`/api/progress?player=${encodeURIComponent(name)}`);
      empty.hidden = data.metrics.length > 0;
      empty.textContent = 'No readings for this player yet. Readings the analysis was not sure of are not shown here.';
      for (const metric of data.metrics) {
        const guide = fmtGuide(metric.target, metric.unit);
        const better = { lower_better: 'Lower is better.', higher_better: 'Higher is better.', in_range: 'Inside the band is better.' }[metric.direction] || '';
        charts.append(el('section', { class: 'chart-card', dataset: { metric: metric.name } },
          el('h3', { text: metric.label }),
          el('p', { class: `trend ${metric.trend.word}`, text: `${TREND_WORDS[metric.trend.word] || ''}: ${metric.trend.text}` }),
          chart(metric),
          el('p', { class: 'small muted', text: [guide ? guide.charAt(0).toUpperCase() + guide.slice(1) + '.' : '', better].filter(Boolean).join(' ') })));
      }
    } catch (err) {
      error.textContent = err.message;
      error.hidden = false;
    }
  };
  select_.addEventListener('change', draw);
  await draw();
}

// ---------------------------------------------------------------------------------------------

async function main() {
  const page = document.body.dataset.page;
  if (page === 'plan') return startPlan();
  if (page === 'progress') return startProgress();
  await loadHealth();
  if (health.warm && health.warm.state === 'warming') watchWarmUp();
  const id = new URLSearchParams(location.search).get('s');
  if (id) return startSession(id);
  return startHome();
}

main().catch((err) => {
  noteError(err);
  const box = document.createElement('p');
  box.className = 'failure';
  box.textContent = `The page could not start: ${err.message}`;
  document.body.append(box);
});
