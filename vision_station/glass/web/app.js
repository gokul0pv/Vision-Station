// Vision Glass -- the application. Talks to server.py; draws everything.
const S = {
  st: null, view: 'run', pid: null, results: {}, preview: {}, history: [],
  sel: null, schema: {}, showAll: false, seq: -1, refVer: 0,
  cv: { k: 0.2, x: 0, y: 0, fitted: false, mode: 'select', full: false },
  armed: null, finder: null, drag: null, draft: null,
};
const QUICK = ['presence', 'ocv', 'ocr', 'code_read', 'count_template', 'measure_length',
               'color_match', 'defect_diff', 'count_blob', 'find_line', 'find_circle', 'align'];
const THRESHOLD = { presence: 'min_score', count_template: 'min_score', align: 'min_score',
                    ocv: 'min_confidence', ocr: 'min_confidence' };
const NO_REGION = ['distance', 'image_process', 'data_analysis', 'results_analysis', 'intersect',
                   'construct', 'fixture_npoint', 'calibrate_npoint'];

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} :
    { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
const job = () => (S.st?.jobs || []).find(j => j.pid === S.pid) || (S.st?.jobs || [])[0];
const label = ty => (S.st?.labels || {})[ty] || ty;
const toolsOf = j => (j?.tools || []);
const groupOf = t => (t.params?.group) || t.id.replace(/_?\d+$/, '') || t.id;

// ================================================================ start
async function boot() {
  applyTheme(localStorage.getItem('vg-theme') || 'auto');
  document.documentElement.classList.toggle('flat', localStorage.getItem('vg-flat') === '1');
  $('#runBtn').innerHTML = icon('run') + '<span>Run</span>';
  $('#settingsBtn').innerHTML = icon('gear');
  $('#openApp').innerHTML = icon('folder') + '<span>Open application</span>';
  $('#runBtn').onclick = runNow;
  $('#settingsBtn').onclick = openSettings;
  $('#themeBtn').onclick = () => {
    const order = ['auto', 'light', 'dark'];
    applyTheme(order[(order.indexOf(localStorage.getItem('vg-theme') || 'auto') + 1) % 3]);
  };
  $('#openApp').onclick = e => {
    const r = e.currentTarget.getBoundingClientRect();
    showMenu(r.left, r.top - 8 - Math.min(320, 40 * (S.st.apps.length + 1)), S.st.apps.map(a => ({
      label: a.name, icon: 'folder', checked: a.id === S.st.app,
      onClick: async () => { S.st = await api('/api/open', { app: a.id }); S.pid = null;
                             S.results = {}; S.preview = {}; S.sel = null; S.cv.fitted = false; render(); } })));
  };
  glassSwitch(false, on => api('/api/continuous', { on }).then(() => poll()), $('#contSwitch'));
  window.addEventListener('keydown', onKey);
  window.addEventListener('resize', () => { if (S.view === 'build') fitCanvas(); });
  await refresh();
  render();
  await fromAddress();
  loop();
}

// #view=build&theme=dark&run=1&select=ocv&zoom=3 -- open in a given state
// (a link to a view, and how the screenshots of this front end are made)
async function fromAddress() {
  const q = new URLSearchParams(location.hash.slice(1));
  if (!q.toString()) return;
  if (q.get('theme')) applyTheme(q.get('theme'));
  if (q.get('flat')) document.documentElement.classList.toggle('flat', q.get('flat') === '1');
  if (q.get('run')) { await runNow(); }
  if (q.get('view')) { S.view = q.get('view'); render(); }
  if (q.get('select')) {
    const t = toolsOf(job()).find(x => x.type === q.get('select') || x.id === q.get('select'));
    if (t) { S.sel = t.id; await new Promise(r => setTimeout(r, 900)); drawOverlay(); renderInspector(); }
  }
  if (q.get('zoom')) { await new Promise(r => setTimeout(r, 300)); zoomBy(Number(q.get('zoom'))); }
  if (q.get('finder')) { S.finder = { mode: q.get('finder'), items: [], area: null, names: {}, thr: {} }; renderBuild(); }
  if (q.get('settings')) openSettings(q.get('settings'));
  if (q.get('menu')) offerChecks(innerWidth / 2 - 120, 200, [100, 100, 50, 50], 'rect');
}

function applyTheme(t) {
  localStorage.setItem('vg-theme', t);
  document.documentElement.dataset.theme = t;
  $('#themeBtn').innerHTML = icon(t === 'dark' ? 'moon' : t === 'light' ? 'sun' : 'gauge');
  $('#themeBtn').title = 'Appearance: ' + t;
}

async function refresh() {
  S.st = await api('/api/state');
  if (!S.pid || !S.st.jobs.find(j => j.pid === S.pid)) S.pid = S.st.jobs[0]?.pid || null;
  if (S.st.seq !== S.seq) {
    S.seq = S.st.seq;
    S.results = await api('/api/results');
    return true;
  }
  return false;
}
async function loop() {
  for (;;) {
    const busy = S.st?.continuous || S.st?.running;
    await new Promise(r => setTimeout(r, busy ? 600 : 2500));
    try { await poll(); } catch (e) { /* the server may be restarting */ }
  }
}
async function poll() {
  const changed = await refresh();
  renderChrome();
  if (changed && S.view === 'run') renderRun();
  if (changed && S.view === 'history') renderHistory();
}

// ================================================================ chrome
function render() { renderChrome(); renderView(); }
function renderChrome() {
  const st = S.st;
  $('#appName').textContent = (st.apps.find(a => a.id === st.app) || {}).name || 'No application';
  const nav = $('#nav');
  const views = [['run', 'gauge', 'Run'], ['build', 'hammer', 'Build'], ['history', 'clock', 'History']];
  if (!nav.querySelector('.nav-glider')) {
    // built once, so the highlight can spring from one item to the next
    nav.innerHTML = '';
    nav.append(h('div', { class: 'nav-glider' }));
    for (const [v, ic, txt] of views)
      nav.append(h('div', { class: 'nav-item', 'data-v': v, html: icon(ic) + `<span>${txt}</span>`,
                            onclick: () => { S.view = v; render(); } }));
  }
  const items = $$('.nav-item', nav);
  items.forEach(it => it.classList.toggle('on', it.dataset.v === S.view));
  const on = items.find(it => it.dataset.v === S.view);
  if (on) nav.querySelector('.nav-glider').style.transform = `translateY(${on.offsetTop}px)`;
  const jl = $('#jobList'); jl.innerHTML = '';
  for (const j of st.jobs) {
    const r = S.results[j.pid];
    const cls = r ? (r.verdict === 'PASS' ? 'pass' : 'fail') : '';
    jl.append(h('div', { class: 'job-item' + (S.view === 'build' && j.pid === S.pid ? ' on' : ''),
      html: `<span class="dot ${cls}"></span><span>${esc(j.name)}</span><span class="v">${r ? (r.verdict === 'PASS' ? 'Accept' : 'Reject') : ''}</span>`,
      onclick: () => { S.pid = j.pid; S.sel = null; S.cv.fitted = false; S.view = 'build'; render(); } }));
  }
  $('#viewTitle').innerHTML = S.view === 'build'
    ? `Build <span class="crumb">/ ${esc(job()?.name || '')}</span>`
    : { run: 'Run', history: 'History' }[S.view];
  const c = st.counts, tot = c.accept + c.reject;
  const last = Object.values(S.results).sort((a, b) => b.seq - a.seq)[0];
  $('#kpis').innerHTML = [
    ['Accepted', c.accept, c.accept ? 'pass' : ''], ['Rejected', c.reject, c.reject ? 'fail' : ''],
    ['Yield', tot ? (100 * c.accept / tot).toFixed(1) + '%' : '–', ''],
    ['Last cycle', last ? Math.round(last.total_ms) + ' ms' : '–', '']]
    .map(([k, v, cls]) => `<div class="kpi"><div class="cap">${k}</div><div class="num ${cls}${
      S.kpiWas && S.kpiWas[k] !== undefined && S.kpiWas[k] !== v ? ' bump' : ''}">${v}</div></div>`).join('');
  S.kpiWas = { Accepted: c.accept, Rejected: c.reject };
  $('#contSwitch').set(st.continuous);
}

function renderView() {
  const v = $('#view'); v.innerHTML = '';
  if (S.shown !== S.view) {
    S.shown = S.view;
    v.classList.remove('view-enter'); void v.offsetWidth; v.classList.add('view-enter');
  }
  if (!S.st.jobs.length) {
    v.append(h('div', { class: 'empty glass panel', html:
      '<div><b>No application open</b><br>Open one from the sidebar - Vision Station\'s projects are here too.</div>' }));
    return;
  }
  if (S.view === 'run') renderRun();
  else if (S.view === 'build') renderBuild();
  else renderHistory();
}

// ================================================================ annotation (SVG)
const COL = { pass: 'var(--ok)', fail: 'var(--ng)', error: 'var(--warn)', warn: 'var(--warn)', search: '#38bdf8' };
function svgEl(tag, attrs) {
  const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) e.setAttribute(k, v);
  return e;
}
/** The results of a run drawn over its picture: outlines in their shapes,
    and each check's words on a tag attached to its box. px: screen pixels
    per image pixel, so lines and tags keep a readable size. */
function drawResults(svg, tools, px, onlyFailures) {
  const g = svgEl('g', {});
  const tagH = Math.max(13, Math.min(22, 16)) / px;
  const taken = [];
  for (const t of tools || []) {
    let box = null;
    for (const gr of t.graphics || []) {
      const pts = gr.points || [];
      if (!pts.length) continue;
      const col = COL[gr.status] || COL[t.status] || 'var(--ok)';
      const w = (gr.role === 'region' ? 2.4 : 1.6);
      let el;
      if (gr.kind === 'rect' && pts.length >= 2) {
        const [a, b] = pts;
        el = svgEl('rect', { x: Math.min(a[0], b[0]), y: Math.min(a[1], b[1]),
                             width: Math.abs(b[0] - a[0]), height: Math.abs(b[1] - a[1]), rx: 3 / px });
      } else if (gr.kind === 'circle' && pts.length >= 2) {
        el = svgEl('circle', { cx: pts[0][0], cy: pts[0][1], r: Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]) });
      } else if (gr.kind === 'line' && pts.length >= 2) {
        el = svgEl('line', { x1: pts[0][0], y1: pts[0][1], x2: pts[1][0], y2: pts[1][1] });
      } else if (pts.length >= 3) {
        el = svgEl('polygon', { points: pts.map(p => p.join(',')).join(' ') });
      } else {
        el = svgEl('circle', { cx: pts[0][0], cy: pts[0][1], r: 3 / px });
      }
      el.setAttribute('fill', gr.role === 'region' ? 'none' : 'none');
      el.setAttribute('stroke', col);
      el.setAttribute('stroke-width', w / px);
      if (gr.status === 'search') el.setAttribute('stroke-dasharray', `${5 / px} ${4 / px}`);
      el.setAttribute('class', 'region');
      g.append(el);
      if (gr.role === 'region' || !box) {
        const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
        const b = [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
        if (gr.kind === 'circle' && pts.length >= 2) {
          const r = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]);
          b[0] = pts[0][0] - r; b[1] = pts[0][1] - r; b[2] = pts[0][0] + r; b[3] = pts[0][1] + r;
        }
        if (gr.role === 'region' || !box) box = b;
      }
    }
    const lines = t.caption || [];
    if (!box || !lines.length || (onlyFailures && t.status === 'pass')) continue;
    g.append(tag(box[0], box[1], lines, COL[t.status] || 'var(--ok)', tagH, taken));
  }
  svg.append(g);
}
function tag(x, y, lines, col, th, taken) {
  const fs = th * 0.68, pad = th * 0.35;
  const w = Math.max(...lines.map(l => l.length)) * fs * 0.6 + 2 * pad;
  const hgt = th * lines.length;
  let top = y - hgt;
  for (const q of taken) if (Math.abs(q[0] - x) < w && top < q[1] + q[2] && q[1] < top + hgt) top = q[1] - hgt;
  taken.push([x, top, hgt]);
  const g = svgEl('g', {});
  g.append(svgEl('rect', { x, y: top, width: w, height: hgt, rx: th * 0.3, fill: col, opacity: .94 }));
  lines.forEach((l, i) => {
    const t = svgEl('text', { x: x + pad, y: top + th * (i + 0.72), fill: '#fff', 'font-size': fs,
                              'font-weight': i ? 500 : 700, 'font-family': 'Inter, -apple-system, sans-serif' });
    t.textContent = l; g.append(t);
  });
  return g;
}

// ================================================================ run view
function renderRun() {
  const v = $('#view'); v.innerHTML = '';
  const st = S.st;
  const wrap = h('div', { class: 'run' });
  const main = h('div', { class: 'run-main' });
  // the verdict
  const res = st.jobs.map(j => S.results[j.pid]).filter(Boolean);
  const verdict = !res.length ? 'idle' : res.every(r => r.verdict === 'PASS') ? 'pass' : 'fail';
  const fails = res.flatMap(r => r.tools.filter(t => t.status !== 'pass')
                                       .map(t => `${t.tool_id}: ${t.message || t.status}`));
  const c = st.counts, tot = c.accept + c.reject, yieldP = tot ? c.accept / tot : 0;
  const seqNow = Math.max(0, ...res.map(r => r.seq || 0));
  const fresh = seqNow !== S.verdictSeq; S.verdictSeq = seqNow;
  const R = 33, L = 2 * Math.PI * R;
  const orb = h('div', { class: 'orb', html:
    `<svg class="ring" viewBox="0 0 76 76"><circle class="track" cx="38" cy="38" r="${R}"/>` +
    `<circle class="arc" cx="38" cy="38" r="${R}" stroke-dasharray="${L}" stroke-dashoffset="${L}"/></svg>` +
    `<div class="glyph">${icon(verdict === 'pass' ? 'check' : verdict === 'fail' ? 'close' : 'run')}</div>` });
  // the arc grows to the yield after it is on the page, so it is seen moving
  requestAnimationFrame(() => requestAnimationFrame(() => {
    const a = orb.querySelector('.arc'); if (a) a.style.strokeDashoffset = String(L * (1 - yieldP)); }));
  main.append(h('div', { class: `verdict glass panel ${verdict}${fresh && res.length ? ' fresh' : ''}` },
    orb,
    h('div', { class: 'grow', style: { minWidth: 0 } },
      h('div', { class: 'word' }, { idle: 'Ready', pass: 'Accept', fail: 'Reject' }[verdict]),
      h('div', { class: 'why' }, verdict === 'idle' ? 'Press Run to inspect a part - or switch Continuous on.' :
        verdict === 'pass' ? 'Every check passed.' : fails.slice(0, 3).join('  ·  ') +
        (fails.length > 3 ? `  ·  +${fails.length - 3} more` : ''))),
    h('div', { class: 'yield' }, h('b', {}, tot ? (100 * yieldP).toFixed(1) + '%' : '–'),
      h('span', {}, tot ? `yield · ${tot} part${tot === 1 ? '' : 's'}` : 'yield'))));
  if (st.barcode.enabled) main.append(barcodeBar());
  // a tile per job
  const tiles = h('div', { class: 'tiles', style: { gridTemplateColumns: `repeat(${st.jobs.length > 1 ? 2 : 1}, minmax(0,1fr))` } });
  for (const j of st.jobs) tiles.append(runTile(j));
  main.append(tiles);
  wrap.append(main);
  // the queue
  const q = h('div', { class: 'queue glass panel' }, h('h3', {}, 'Results'));
  const rows = h('div', { class: 'rows' });
  api('/api/history').then(hist => {
    S.history = hist;
    if (!hist.length) rows.append(h('div', { class: 'faint', style: { padding: '8px 10px' } }, 'No parts inspected yet'));
    hist.slice(0, 40).forEach((e, i) => rows.append(h('div', { class: 'qrow', style: { '--i': i } },
      h('span', { class: 't' }, e.time), h('span', {}, e.barcode ? `${e.job}  ·  ${e.barcode}` : e.job),
      h('span', { class: 'badge ' + (e.verdict === 'PASS' ? 'pass' : 'fail') }, e.verdict === 'PASS' ? 'OK' : 'NG'),
      e.failed.length ? h('span', { class: 'f' }, e.failed.join(', ')) : null)));
  });
  q.append(rows);
  wrap.append(q);
  v.append(wrap);
}
function runTile(j) {
  const r = S.results[j.pid];
  const tile = h('div', { class: 'tile glass panel' });
  tile.append(h('div', { class: 'tile-head' }, h('span', { class: 'name' }, j.name),
    h('span', { class: 'faint' }, r ? `${Math.round(r.total_ms)} ms` : j.source), h('div', { class: 'grow' }),
    h('span', { class: 'badge ' + (r ? (r.verdict === 'PASS' ? 'pass' : 'fail') : '') }, r ? (r.verdict === 'PASS' ? 'ACCEPT' : 'REJECT') : 'IDLE'),
    h('button', { class: 'glass-btn small', onclick: () => { S.pid = j.pid; S.view = 'build'; S.cv.fitted = false; render(); } }, 'Edit')));
  const stage = h('div', { class: 'stage' });
  const size = (r && r.image_size) || j.size;
  if (size) {
    const fit = h('div', { class: 'fit' });
    const img = h('img', { src: r ? `/img/last/${j.pid}.jpg?w=1800&s=${r.seq}` : `/img/reference/${j.pid}.jpg?w=1800&v=${S.refVer}`, draggable: 'false' });
    const svg = svgEl('svg', { class: 'ov', viewBox: `0 0 ${size[0]} ${size[1]}`, preserveAspectRatio: 'none' });
    fit.append(img, svg);
    stage.append(fit);
    const place = () => {
      const W = stage.clientWidth, H = stage.clientHeight;
      if (!W || !H) return;
      const k = Math.min(W / size[0], H / size[1]);
      fit.style.width = size[0] * k + 'px'; fit.style.height = size[1] * k + 'px';
      svg.innerHTML = '';
      if (r) drawResults(svg, r.tools, k, r.tools.length > 12);
    };
    new ResizeObserver(place).observe(stage);
  }
  tile.append(stage);
  return tile;
}
function barcodeBar() {
  const st = S.st, bc = st.barcode;
  const field = h('input', { class: 'field', id: 'scan', placeholder: 'Scan the part\'s barcode', autocomplete: 'off' });
  const state = h('span', { class: 'badge' }, bc.pending ? `${bc.pending}  ready` : 'Waiting for a scan');
  field.addEventListener('keydown', async e => {
    if (e.key !== 'Enter' || !field.value.trim()) return;
    const r = await api('/api/barcode/scan', { code: field.value });
    if (!r.ok) { state.className = 'badge fail'; state.textContent = 'Wrong barcode - ' + r.why; toast('Wrong barcode: ' + r.why, 'fail'); field.select(); return; }
    state.className = 'badge pass'; state.textContent = r.code; field.value = '';
    if (r.run && r.run.results) await poll();
  });
  setTimeout(() => field.focus(), 50);
  const rule = [bc.prefix ? 'starts with ' + bc.prefix : '', bc.length ? bc.length + ' characters' : ''].filter(Boolean).join(', ') || 'any code';
  return h('div', { class: 'barcode glass panel lens-soft' }, h('span', { html: icon('barcode') }), h('b', {}, 'Barcode'),
    field, state, h('span', { class: 'faint' }, 'Rule: ' + rule));
}

// ================================================================ build view
function renderBuild() {
  const v = $('#view'); v.innerHTML = '';
  const j = job();
  if (!j) return;
  if (!j.size) { v.append(h('div', { class: 'empty glass panel' }, 'This inspection has no teach picture yet.')); return; }
  const wrap = h('div', { class: 'build' });
  const cw = h('div', { class: 'canvas-wrap' });
  const canvas = h('div', { class: 'canvas', id: 'canvas' });
  const world = h('div', { class: 'world', id: 'world' });
  const img = h('img', { id: 'teachImg', draggable: 'false', width: j.size[0], height: j.size[1],
                         src: `/img/reference/${j.pid}.jpg?w=2600&v=${S.refVer}` });
  const svg = svgEl('svg', { id: 'ov', width: j.size[0], height: j.size[1], viewBox: `0 0 ${j.size[0]} ${j.size[1]}` });
  world.append(img, svg);
  canvas.append(world);
  cw.append(canvas);
  // the tool rail
  const rail = h('div', { class: 'float rail glass lens' });
  for (const [m, ic, tip] of [['select', 'cursor', 'Select and move  (V)'], ['pan', 'hand', 'Move the picture  (H, or Space)'],
                              ['rect', 'rect', 'Draw a box  (R)'], ['ellipse', 'ellipse', 'Draw an ellipse  (E)']])
    rail.append(h('button', { class: 'glass-btn icon' + (S.cv.mode === m ? ' on' : ''), title: tip, html: icon(ic),
                              onclick: () => setMode(m) }));
  rail.append(h('div', { class: 'rail-sep' }));
  rail.append(h('button', { class: 'glass-btn icon' + (S.finder ? ' on' : ''), title: 'Find parts / text (AI)', html: icon('sparkle'),
                            onclick: () => { S.finder = S.finder ? null : { mode: 'parts', items: [], area: null, names: {}, thr: {} }; renderBuild(); } }));
  rail.append(h('button', { class: 'glass-btn icon', title: 'Undo  (Ctrl+Z)', html: icon('undo'), onclick: undo }));
  cw.append(rail);
  // zoom
  const zb = h('div', { class: 'float zoombar glass lens' },
    h('button', { class: 'glass-btn icon ghost', html: icon('zoomout'), title: 'Zoom out', onclick: () => glide(() => zoomBy(0.8)) }),
    h('span', { class: 'z', id: 'zoomLabel' }, ''),
    h('button', { class: 'glass-btn icon ghost', html: icon('zoomin'), title: 'Zoom in', onclick: () => glide(() => zoomBy(1.25)) }),
    h('button', { class: 'glass-btn icon ghost', html: icon('fit'), title: 'Fit  (F)', onclick: () => glide(() => { S.cv.fitted = false; fitCanvas(); }) }));
  cw.append(zb);
  if (S.armed) cw.append(h('div', { class: 'float note glass lens' },
    h('span', {}, `Drag on the picture where the ${label(S.armed)} check goes`),
    h('button', { class: 'glass-btn small', onclick: () => { S.armed = null; setMode('select'); } }, 'Cancel')));
  if (S.finder) cw.append(finderPanel());
  wrap.append(cw);
  wrap.append(inspector());
  v.append(wrap);
  bindCanvas(canvas);
  requestAnimationFrame(() => { if (!S.cv.fitted) fitCanvas(); else applyView(); });
  if (!S.preview[j.pid]) livePreview();
}

function fitCanvas() {
  const j = job(), c = $('#canvas');
  if (!c || !j) return;
  const k = Math.min((c.clientWidth - 40) / j.size[0], (c.clientHeight - 40) / j.size[1]);
  S.cv.k = k; S.cv.x = (c.clientWidth - j.size[0] * k) / 2; S.cv.y = (c.clientHeight - j.size[1] * k) / 2;
  S.cv.fitted = true;
  applyView();
}
function applyView() {
  const w = $('#world'); if (!w) return;
  w.style.transform = `translate(${S.cv.x}px, ${S.cv.y}px) scale(${S.cv.k})`;
  $('#zoomLabel').textContent = Math.round(S.cv.k * 100) + '%';
  // more detail once zoomed in
  const j = job(), img = $('#teachImg');
  if (img && !S.cv.full && S.cv.k * devicePixelRatio * Math.max(...j.size) > 2600 * 1.15) {
    S.cv.full = true; img.src = `/img/reference/${j.pid}.jpg?w=99999&v=${S.refVer}`;
  }
  drawOverlay();
}
/** A view change from a button or key, animated on a spring. The wheel and
    dragging stay immediate: the picture must be where the hand is. */
function glide(fn) {
  const w = $('#world');
  if (w) { w.classList.add('glide'); clearTimeout(glide.t); glide.t = setTimeout(() => w.classList.remove('glide'), 600); }
  fn();
}
function zoomBy(f, cx, cy) {
  const c = $('#canvas');
  cx = cx ?? c.clientWidth / 2; cy = cy ?? c.clientHeight / 2;
  const k = Math.max(0.02, Math.min(16, S.cv.k * f));
  S.cv.x = cx - (cx - S.cv.x) * (k / S.cv.k); S.cv.y = cy - (cy - S.cv.y) * (k / S.cv.k);
  S.cv.k = k; applyView();
}
function setMode(m) {
  S.cv.mode = m;
  if (m === 'select' || m === 'pan') S.armed = S.armed && m !== 'select' ? S.armed : null;
  renderBuild();
}
const toImage = (e) => {
  const r = $('#canvas').getBoundingClientRect();
  return [(e.clientX - r.left - S.cv.x) / S.cv.k, (e.clientY - r.top - S.cv.y) / S.cv.k];
};

// ---- the teach picture's checks, editable
function drawOverlay() {
  const svg = $('#ov'), j = job();
  if (!svg || !j) return;
  svg.innerHTML = '';
  const px = S.cv.k;
  const pv = S.preview[j.pid] || {};
  const byId = Object.fromEntries((pv.tools || []).map(t => [t.tool_id, t]));
  const taken = [];
  for (const t of toolsOf(j)) {
    const p = t.params || {};
    if (!p.roi && !(p.points && p.points.length > 2)) continue;
    const r = byId[t.id], sel = S.sel === t.id;
    const col = sel ? 'var(--sel)' : r ? (COL[r.status] || 'var(--ok)') : 'rgba(200,205,220,.85)';
    const g = svgEl('g', { 'data-id': t.id, style: 'cursor:move' });
    const [x, y, w, hh] = p.roi || [0, 0, 0, 0];
    let shape;
    if (p.shape === 'polygon' && p.points?.length > 2)
      shape = svgEl('polygon', { points: p.points.map(q => q.join(',')).join(' ') });
    else if (p.shape === 'ellipse')
      shape = svgEl('ellipse', { cx: x + w / 2, cy: y + hh / 2, rx: w / 2, ry: hh / 2 });
    else shape = svgEl('rect', { x, y, width: w, height: hh, rx: 3 / px });
    if (p.angle) shape.setAttribute('transform', `rotate(${p.angle} ${x + w / 2} ${y + hh / 2})`);
    shape.setAttribute('fill', sel ? 'rgba(167,139,250,.14)' : 'rgba(255,255,255,0.001)');
    shape.setAttribute('stroke', col);
    shape.setAttribute('stroke-width', (sel ? 2.4 : 2) / px);
    if (sel) shape.setAttribute('class', 'region-sel');
    if (!r && !sel) shape.setAttribute('stroke-dasharray', `${6 / px} ${4 / px}`);
    if (p.enabled === false) shape.setAttribute('opacity', '.4');
    g.append(shape);
    svg.append(g);
    // its result on a tag -- where the box is big enough on screen to carry
    // one, or it failed, or it is the one selected: at a whole-board zoom
    // twenty tags piled on each other over the parts
    const roomy = Math.max(w, hh) * px >= 34;
    if (r && (r.caption || []).length && (roomy || sel || r.status !== 'pass'))
      svg.append(tag(x, y, r.caption, col === 'var(--sel)' ? (COL[r.status] || 'var(--ok)') : col, 18 / px, taken));
    if (sel && p.shape !== 'polygon' && !p.angle) {
      for (const [hx, hy, k] of handles(x, y, w, hh)) {
        const hd = svgEl('circle', { cx: hx, cy: hy, r: 5.5 / px, class: 'handle', 'data-h': k,
                                     'stroke-width': 2.2 / px, style: `cursor:${k}-resize` });
        svg.append(hd);
      }
      svg.append(tag(x, y + hh + 26 / px, [`${Math.round(w)} × ${Math.round(hh)}`], 'var(--sel)', 16 / px, []));
    }
  }
  if (S.draft) {
    const d = S.draft;
    const el = S.cv.mode === 'ellipse'
      ? svgEl('ellipse', { cx: d[0] + d[2] / 2, cy: d[1] + d[3] / 2, rx: d[2] / 2, ry: d[3] / 2 })
      : svgEl('rect', { x: d[0], y: d[1], width: d[2], height: d[3], rx: 3 / px });
    el.setAttribute('fill', 'color-mix(in srgb, var(--sel) 14%, transparent)'); el.setAttribute('stroke', 'var(--sel)');
    el.setAttribute('stroke-width', 2 / px); el.setAttribute('stroke-dasharray', `${7 / px} ${5 / px}`);
    ants(el, px);
    svg.append(el);
  }
  if (S.finder) drawFinder(svg, px);
}
/** Marching ants on a selection outline: the dash runs round it; redrawn
    shapes pick the march up where it was. */
function ants(el, px) {
  el.classList.add('ants');
  el.style.setProperty('--ants', String(-12 / px));
  el.style.animationDelay = `-${performance.now() % 600}ms`;
}
function sizeChip(e, w, hh) {
  const c = $('#sizeChip'); if (!c) return;
  if (!e) { c.classList.remove('on'); return; }
  c.textContent = `${Math.round(w)} × ${Math.round(hh)}`;
  c.style.left = (e.clientX + 18) + 'px'; c.style.top = (e.clientY + 18) + 'px';
  c.classList.add('on');
}
function handles(x, y, w, hh) {
  return [[x, y, 'nw'], [x + w / 2, y, 'n'], [x + w, y, 'ne'], [x + w, y + hh / 2, 'e'],
          [x + w, y + hh, 'se'], [x + w / 2, y + hh, 's'], [x, y + hh, 'sw'], [x, y + hh / 2, 'w']];
}

function bindCanvas(c) {
  c.classList.toggle('draw', S.cv.mode === 'rect' || S.cv.mode === 'ellipse');
  c.classList.toggle('pan', S.cv.mode === 'pan');
  c.addEventListener('wheel', e => {
    e.preventDefault();
    const r = c.getBoundingClientRect();
    zoomBy(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top);
  }, { passive: false });
  c.addEventListener('mousedown', e => {
    const p = toImage(e);
    const hEl = e.target.closest('[data-h]'), gEl = e.target.closest('[data-id]');
    if (e.button === 1 || S.cv.mode === 'pan' || S.space) {
      S.drag = { kind: 'pan', sx: e.clientX, sy: e.clientY, x: S.cv.x, y: S.cv.y };
      grabbing(true);
    } else if (S.finder && S.finder.drawing) {
      S.drag = { kind: 'draw', p0: p }; S.draft = [p[0], p[1], 0, 0];
    } else if (S.cv.mode === 'rect' || S.cv.mode === 'ellipse') {
      S.drag = { kind: 'draw', p0: p }; S.draft = [p[0], p[1], 0, 0];
    } else if (hEl && S.sel) {
      const t = toolsOf(job()).find(t => t.id === S.sel);
      S.drag = { kind: 'resize', h: hEl.dataset.h, p0: p, roi: [...t.params.roi] };
    } else if (gEl && !(S.finder && gEl.closest('.fbox'))) {
      const id = gEl.dataset.id;
      if (S.sel !== id) { S.sel = id; renderInspector(); }
      const t = toolsOf(job()).find(t => t.id === id);
      S.drag = { kind: 'move', p0: p, roi: [...(t.params.roi || [0, 0, 0, 0])],
                 pts: t.params.points ? t.params.points.map(q => [...q]) : null };
      drawOverlay();
    } else {
      if (S.sel) { S.sel = null; renderInspector(); drawOverlay(); }
      S.drag = { kind: 'pan', sx: e.clientX, sy: e.clientY, x: S.cv.x, y: S.cv.y };
      grabbing(true);
    }
  });
  window.onmousemove = e => {
    const d = S.drag; if (!d) return;
    if (d.kind === 'pan') { S.cv.x = d.x + e.clientX - d.sx; S.cv.y = d.y + e.clientY - d.sy; applyView(); return; }
    const p = toImage(e), dx = p[0] - (d.p0 ? d.p0[0] : 0), dy = p[1] - (d.p0 ? d.p0[1] : 0);
    if (d.kind === 'draw') {
      let w = p[0] - d.p0[0], hh = p[1] - d.p0[1];
      if (e.shiftKey) { const m = Math.max(Math.abs(w), Math.abs(hh)); w = Math.sign(w || 1) * m; hh = Math.sign(hh || 1) * m; }
      S.draft = [Math.min(d.p0[0], d.p0[0] + w), Math.min(d.p0[1], d.p0[1] + hh), Math.abs(w), Math.abs(hh)];
      sizeChip(e, Math.abs(w), Math.abs(hh));
      drawOverlay(); return;
    }
    const t = toolsOf(job()).find(t => t.id === S.sel); if (!t) return;
    let [x, y, w, hh] = d.roi;
    if (d.kind === 'move') {
      t.params.roi = [Math.round(x + dx), Math.round(y + dy), w, hh];
      if (d.pts) t.params.points = d.pts.map(q => [q[0] + dx, q[1] + dy]);
    } else {
      if (d.h.includes('w')) { x += dx; w -= dx; } if (d.h.includes('e')) w += dx;
      if (d.h.includes('n')) { y += dy; hh -= dy; } if (d.h.includes('s')) hh += dy;
      if (w < 4) { w = 4; } if (hh < 4) { hh = 4; }
      t.params.roi = [Math.round(x), Math.round(y), Math.round(w), Math.round(hh)];
      sizeChip(e, w, hh);
    }
    d.moved = true;
    drawOverlay();
  };
  window.onmouseup = async e => {
    const d = S.drag; S.drag = null;
    sizeChip(null); grabbing(false);
    if (!d) return;
    if (d.kind === 'draw') {
      const box = S.draft; S.draft = null; drawOverlay();
      if (!box || box[2] < 6 || box[3] < 6) return;
      if (S.finder && S.finder.drawing) { S.finder.area = box.map(Math.round); S.finder.drawing = false;
                                          delete document.documentElement.dataset.ptr; renderBuild(); return; }
      const shape = S.cv.mode === 'ellipse' ? 'ellipse' : 'rect';
      if (S.armed) { const ty = S.armed; S.armed = null; S.cv.mode = 'select'; return addCheck(ty, box, shape); }
      return offerChecks(e.clientX, e.clientY, box, shape);
    }
    if ((d.kind === 'move' || d.kind === 'resize') && d.moved) {
      const t = toolsOf(job()).find(t => t.id === S.sel);
      const params = { roi: t.params.roi };
      if (t.params.points) params.points = t.params.points;
      const r = await api('/api/tool/update', { pid: job().pid, id: t.id, params });
      if (r.message) toast(r.message, 'pass');
      await afterEdit();
    }
  };
}

/** The pointer's hand closes while the picture is being moved. */
function grabbing(on) {
  if (on) document.documentElement.dataset.ptr = 'grabbing';
  else if (document.documentElement.dataset.ptr === 'grabbing') delete document.documentElement.dataset.ptr;
  $('#canvas')?.classList.toggle('panning', !!on);
  if (typeof Pointer !== 'undefined') Pointer.refresh();
}
function offerChecks(x, y, box, shape) {
  const items = [{ section: 'Add a check on this area' }];
  for (const ty of QUICK.slice(0, 8)) items.push({ label: label(ty), icon: TOOL_ICON[ty] || 'ok',
    onClick: () => addCheck(ty, box, shape) });
  items.push({ label: 'More checks…', icon: 'plus', onClick: () => pickCheck(ty => addCheck(ty, box, shape)) });
  if (S.sel) items.push('-', { label: `Move ${S.sel} here`, icon: 'cursor', onClick: async () => {
    await api('/api/tool/update', { pid: job().pid, id: S.sel, params: { roi: box.map(Math.round), shape } }); await afterEdit(); } });
  showMenu(x, y, items);
}
async function addCheck(ty, box, shape) {
  const r = await api('/api/tool/add', { pid: job().pid, type: ty, roi: box ? box.map(Math.round) : null, shape: shape || 'rect' });
  S.sel = r.id;
  toast(r.message || `Added ${label(ty)}`, 'pass');
  await afterEdit();
}
function beginAdd(ty) {
  if (NO_REGION.includes(ty)) return addCheck(ty, null);
  S.armed = ty; S.cv.mode = 'rect'; renderBuild();
}
function pickCheck(cb) {
  const body = h('div', { class: 'picker-grid' });
  const legacy = ['text_detect', 'ocr_auto', 'yolo_defect', 'anomaly'];
  let sh;
  const all = Object.keys(S.st.labels).filter(t => !legacy.includes(t));
  const ordered = QUICK.concat(all.filter(t => !QUICK.includes(t)));
  for (const ty of ordered) body.append(h('div', { class: 'pick', onclick: () => { sh.close(); cb(ty); } },
    h('span', { html: icon(TOOL_ICON[ty] || 'ok') }), h('b', {}, label(ty))));
  sh = sheet('Add a check', 'Pick what to check. Then drag its area on the picture.', body, [{ label: 'Cancel' }]);
}
async function afterEdit() {
  await refresh();
  renderChrome();
  if (S.view === 'build') { drawOverlay(); renderInspector(); }
  livePreview();
}
let previewTimer = null;
function livePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    const j = job(); if (!j) return;
    try { S.preview[j.pid] = await api('/api/preview', { pid: j.pid }); } catch (e) { return; }
    if (S.view === 'build') { drawOverlay(); renderInspector(); }
  }, 200);
}
async function undo() {
  const r = await api('/api/undo', { pid: job().pid });
  toast(r.undone ? 'Undone' : 'Nothing to undo', r.undone ? 'idle' : 'warn');
  await afterEdit();
}

// ---- the inspector: checks, the selected check's settings
function inspector() { const el = h('div', { class: 'inspector', id: 'inspector' }); setTimeout(renderInspector); return el; }
async function renderInspector() {
  const box = $('#inspector'); if (!box) return;
  const j = job();
  const pv = Object.fromEntries(((S.preview[j.pid] || {}).tools || []).map(t => [t.tool_id, t]));
  box.innerHTML = '';
  // the list
  const list = h('div', { class: 'card glass panel fill' });
  const head = h('div', { style: { display: 'flex', alignItems: 'center', gap: '8px', margin: '0 4px 10px' } },
    h('h3', { style: { margin: 0 } }, 'Checks'), h('span', { class: 'faint' }, String(toolsOf(j).length)), h('div', { class: 'grow' }),
    h('button', { class: 'glass-btn small', html: icon('plus') + 'Add', onclick: () => pickCheck(beginAdd) }));
  const filt = h('input', { class: 'field', placeholder: 'Filter', style: { height: '30px', marginBottom: '8px' } });
  const rows = h('div', { class: 'rows' });
  const fill = () => {
    rows.innerHTML = '';
    const q = filt.value.toLowerCase();
    for (const t of toolsOf(j)) {
      if (q && !(t.id + ' ' + label(t.type)).toLowerCase().includes(q)) continue;
      const r = pv[t.id];
      rows.append(h('div', { class: 'check-row' + (S.sel === t.id ? ' on' : '') + (t.params.enabled === false ? ' off' : ''),
        onclick: () => { S.sel = t.id; drawOverlay(); renderInspector(); } },
        h('div', { class: 'ic', html: icon(TOOL_ICON[t.type] || 'ok') }),
        h('div', { class: 'nm' }, h('b', {}, t.id), h('span', {}, label(t.type))),
        r ? h('span', { class: 'pill ' + r.status }, r.status === 'pass' ? 'OK' : r.status === 'fail' ? 'NG' : 'ERR') : null));
    }
  };
  filt.oninput = fill; fill();
  list.append(head, toolsOf(j).length > 6 ? filt : null, rows);
  box.append(list);
  // the selected check
  const t = toolsOf(j).find(x => x.id === S.sel);
  if (!t) {
    box.append(h('div', { class: 'card glass panel' },
      h('div', { class: 'faint' }, 'Select a check, or drag a box on the picture to add one.'),
      h('div', { style: { display: 'flex', gap: '8px', marginTop: '12px', flexWrap: 'wrap' } },
        h('button', { class: 'glass-btn small', html: icon('photo') + 'Teach on last part', onclick: teachOnLast }),
        h('button', { class: 'glass-btn small', html: icon('camera') + 'Image source', onclick: () => openSettings('source') }))));
    return;
  }
  const props = h('div', { class: 'card glass panel props', style: { maxHeight: '58%' } });
  props.append(h('div', { style: { display: 'flex', alignItems: 'center', gap: '8px' } },
    h('div', { class: 'grow' }, h('b', { style: { fontSize: '15px' } }, t.id), h('div', { class: 'faint' }, label(t.type))),
    h('button', { class: 'glass-btn icon small', title: 'Delete', html: icon('trash'), onclick: async () => {
      await api('/api/tool/delete', { pid: j.pid, ids: [t.id] }); S.sel = null; toast(`Deleted ${t.id} - Ctrl+Z brings it back`); await afterEdit(); } })));
  const r = pv[t.id];
  if (r) props.append(h('div', { class: 'result-line', html:
    `<span class="pill ${r.status}">${r.status === 'pass' ? 'OK' : r.status === 'fail' ? 'NG' : 'ERROR'}</span>  ` +
    `<b>${esc((r.caption || []).join('  ·  '))}</b><br><span class="faint">On the teach picture · score ${Number(r.score).toFixed(2)}` +
    `${r.message ? ' · ' + esc(r.message) : ''}</span>` }));
  // the group of the same kind
  const members = toolsOf(j).filter(x => x.type === t.type && groupOf(x) === groupOf(t));
  const key = THRESHOLD[t.type];
  if (members.length > 1 && key) {
    const val = Number(t.params[key] ?? (key === 'min_score' ? 0.65 : 0.5));
    const g = h('div', { class: 'group' }, h('b', {}, `Group ${groupOf(t)}  ·  ${members.length} checks`));
    g.append(sliderRow(`${key === 'min_score' ? 'Accept threshold' : 'Reading confidence'} for all ${members.length}`, val, 0, 1, 0.01,
      async v => { await api('/api/tool/update-many', { pid: j.pid, ids: members.map(m => m.id), params: { [key]: v } });
                   toast(`${members.length} checks set to ${v.toFixed(2)}`, 'pass'); await afterEdit(); }));
    props.append(g);
  }
  // settings
  const enabled = h('span', { class: 'switch' }), critical = h('span', { class: 'switch' });
  glassSwitch(t.params.enabled !== false, v => setParam(t, 'enabled', v), enabled);
  glassSwitch(t.params.critical !== false, v => setParam(t, 'critical', v), critical);
  props.append(formRow('Run this check', null, enabled), formRow('A failure rejects the part', null, critical));
  const sch = S.schema[t.type] || (S.schema[t.type] = await api('/api/schema?type=' + t.type));
  const rowsS = sch.rows.filter(rw => S.showAll || !sch.essential || sch.essential.includes(rw.key));
  for (const rw of rowsS) props.append(propRow(t, rw));
  if (sch.essential && sch.rows.length > sch.essential.length)
    props.append(h('button', { class: 'glass-btn small ghost', onclick: () => { S.showAll = !S.showAll; renderInspector(); } },
      S.showAll ? 'Show fewer settings' : `Show all settings (${sch.rows.length})`));
  box.append(props);
}
function sliderRow(text, val, lo, hi, step, onCommit) {
  const num = h('span', { class: 'num' }, val.toFixed(2));
  const s = h('input', { type: 'range', class: 'slider', min: lo, max: hi, step, value: val });
  const paint = () => { s.style.setProperty('--p', (100 * (s.value - lo) / (hi - lo)) + '%'); num.textContent = Number(s.value).toFixed(2); };
  s.oninput = paint; s.onchange = () => onCommit(Number(s.value)); paint();
  return h('div', { class: 'prop' }, h('label', {}, text), h('div', { class: 'rowline' }, s, num));
}
function propRow(t, rw) {
  const cur = t.params[rw.key] ?? rw.default;
  let ctl;
  if (rw.kind === 'bool') {
    ctl = glassSwitch(!!cur, v => setParam(t, rw.key, v));
    return h('div', { class: 'prop' }, formRow(rw.label, rw.help, ctl));
  }
  if (rw.kind === 'choice' && rw.extra) {
    ctl = glassSelect(rw.extra.map(o => ({ value: o, label: String(o) })), cur, v => setParam(t, rw.key, v), { width: '100%' });
  } else if ((rw.kind === 'float' || rw.kind === 'int') && rw.extra && rw.extra.length >= 2 && rw.extra[1] <= 1 && rw.extra[0] >= 0) {
    return h('div', { class: 'prop' }, sliderRow(rw.label, Number(cur ?? 0), rw.extra[0], rw.extra[1], rw.extra[2] || 0.01,
      v => setParam(t, rw.key, v)), rw.help ? h('div', { class: 'help' }, rw.help) : null);
  } else {
    const isJson = typeof cur === 'object' && cur !== null;
    ctl = h('input', { class: 'field' + (rw.kind === 'text' ? '' : ' mono'), value: isJson ? JSON.stringify(cur) : (cur ?? '') });
    ctl.onchange = () => {
      let v = ctl.value;
      if (rw.kind === 'float') v = parseFloat(v); else if (rw.kind === 'int') v = parseInt(v, 10);
      else if (isJson || /list|dict/.test(rw.kind)) { try { v = JSON.parse(v); } catch (e) { toast('Not valid: ' + v, 'fail'); return; } }
      setParam(t, rw.key, v);
    };
  }
  return h('div', { class: 'prop' }, h('label', {}, rw.label), ctl, rw.help ? h('div', { class: 'help' }, rw.help) : null);
}
async function setParam(t, key, value) {
  const r = await api('/api/tool/update', { pid: job().pid, id: t.id, params: { [key]: value } });
  if (r.message) toast(r.message, 'pass');
  await afterEdit();
}
async function teachOnLast() {
  try {
    const r = await api('/api/reference', { pid: job().pid });
    S.refVer++; S.cv.full = false;
    toast('Teach picture replaced' + (r.moved ? ' - every box moved with the part' : ''), 'pass');
    await afterEdit(); renderBuild();
  } catch (e) { toast(e.message, 'warn'); }
}

// ---- Find parts / text
function finderPanel() {
  const F = S.finder, j = job();
  const p = h('div', { class: 'float finder glass' });
  p.append(h('div', { style: { display: 'flex', alignItems: 'center' } }, h('b', { style: { fontSize: '15px' } }, 'Find parts and text'),
    h('div', { class: 'grow' }), h('button', { class: 'glass-btn icon small ghost', html: icon('close'), onclick: () => { S.finder = null; renderBuild(); } })));
  p.append(segmented([{ value: 'parts', label: 'Parts' }, { value: 'ocv', label: 'Verify text' }, { value: 'ocr', label: 'Read text' }],
    F.mode, m => { F.mode = m; F.items = []; renderBuild(); }));
  p.append(h('div', { style: { display: 'flex', gap: '8px', alignItems: 'center' } },
    h('span', { class: 'muted grow' }, F.area ? `Area ${F.area[2]} × ${F.area[3]}` : 'Area: the whole picture'),
    h('button', { class: 'glass-btn small', onclick: () => { F.drawing = true; document.documentElement.dataset.ptr = 'draw';
                                                                 toast('Drag a box round the area to search'); } }, 'Draw area'),
    F.area ? h('button', { class: 'glass-btn small ghost', onclick: () => { F.area = null; renderBuild(); } }, 'Whole') : null));
  const findBtn = h('button', { class: 'glass-btn prominent wide', html: icon('sparkle') + (F.mode === 'parts' ? 'Find parts' : 'Find text') });
  findBtn.onclick = async () => {
    findBtn.disabled = true; findBtn.innerHTML = icon('sparkle') + 'Looking…';
    try {
      const r = await api('/api/find', { pid: j.pid, mode: F.mode, area: F.area, sensitivity: 0.5 });
      F.items = r.items; F.names = {}; F.thr = {};
      toast(`${r.items.length} found in ${(r.ms / 1000).toFixed(1)} s (${r.device})`, 'idle');
    } catch (e) { toast(e.message, 'fail'); }
    renderBuild();
  };
  p.append(findBtn);
  const rows = h('div', { class: 'rows' });
  if (F.mode === 'parts') {
    const types = {};
    F.items.forEach(it => (types[it.type] = types[it.type] || []).push(it));
    for (const [ty, its] of Object.entries(types)) {
      const kept = its.some(i => i.keep);
      const nm = h('input', { class: 'field', placeholder: 'name', value: F.names[ty] || '', onclick: e => e.stopPropagation(),
                               oninput: e => { F.names[ty] = e.target.value; } });
      const thr = h('input', { class: 'field', type: 'number', min: 0.05, max: 0.99, step: 0.05, value: F.thr[ty] ?? 0.65,
                               style: { width: '70px' }, onclick: e => e.stopPropagation(), oninput: e => { F.thr[ty] = Number(e.target.value); } });
      rows.append(h('div', { class: 'frow' + (kept ? ' kept' : ''), onclick: () => { its.forEach(i => (i.keep = !kept)); renderBuild(); } },
        h('span', { class: 'tick', html: kept ? icon('check') : '' }),
        h('div', {}, h('div', { class: 'lbl' }, `${its[0].type_name}  ×  ${its.length}`), nm),
        h('div', {}, h('div', { class: 'faint', style: { fontSize: '11px' } }, 'threshold'), thr)));
    }
  } else {
    F.items.forEach(it => {
      const txt = h('input', { class: 'field', value: it.text, onclick: e => e.stopPropagation(), oninput: e => { it.text = e.target.value; } });
      rows.append(h('div', { class: 'frow' + (it.keep ? ' kept' : ''), onclick: () => { it.keep = !it.keep; renderBuild(); } },
        h('span', { class: 'tick', html: it.keep ? icon('check') : '' }), txt, h('span', { class: 'faint' }, it.score.toFixed(2))));
    });
  }
  p.append(rows);
  const n = F.items.filter(i => i.keep).length;
  p.append(h('button', { class: 'glass-btn prominent wide', disabled: !n, onclick: async () => {
    const items = F.items.filter(i => i.keep);
    const r = await api('/api/find/add', { pid: j.pid, mode: F.mode, items, names: F.names, thresholds: F.thr });
    toast(`Added ${r.added.length} checks - Ctrl+Z takes them all back`, 'pass');
    F.items = F.items.filter(i => !i.keep);
    await afterEdit(); renderBuild();
  } }, `Add ${n} ${F.mode === 'parts' ? 'Part present' : F.mode === 'ocv' ? 'Verify text' : 'Read text'} checks`));
  return p;
}
function drawFinder(svg, px) {
  const F = S.finder;
  if (F.area) {
    const a = F.area;
    const ar = svgEl('rect', { x: a[0], y: a[1], width: a[2], height: a[3], rx: 6 / px, fill: 'rgba(255,255,255,.04)', stroke: '#fff',
                               'stroke-width': 1.6 / px, 'stroke-dasharray': `${8 / px} ${6 / px}` });
    ants(ar, px);
    svg.append(ar);
  }
  const hues = [200, 30, 280, 140, 340, 60, 170, 0, 230, 100];
  F.items.forEach((it, i) => {
    const hue = F.mode === 'parts' ? hues[it.type % hues.length] : 140;
    const [x, y, w, hh] = it.box;
    const r = svgEl('rect', { x, y, width: w, height: hh, rx: 4 / px, class: 'fbox',
      fill: it.keep ? `hsla(${hue},90%,60%,.16)` : 'none', stroke: `hsl(${hue},90%,${it.keep ? 62 : 40}%)`,
      'stroke-width': 2 / px, style: 'cursor:pointer' });
    if (!it.keep) r.setAttribute('stroke-dasharray', `${4 / px} ${4 / px}`);
    r.addEventListener('mousedown', e => { e.stopPropagation(); it.keep = !it.keep; renderBuild(); });
    svg.append(r);
  });
}

// ================================================================ history
async function renderHistory() {
  const v = $('#view'); v.innerHTML = '';
  const hist = await api('/api/history');
  const p = h('div', { class: 'history glass panel' });
  p.append(h('div', { style: { display: 'flex', alignItems: 'center', margin: '0 8px 12px' } },
    h('h3', { style: { margin: 0 } }, 'This session'), h('div', { class: 'grow' }),
    h('button', { class: 'glass-btn small', onclick: async () => { await api('/api/reset', {}); await poll(); renderHistory(); } }, 'Reset counts')));
  const tbl = h('div', { class: 'htable' });
  for (const c of ['Time', 'Inspection', 'Result', 'Cycle', 'Barcode', 'Failed checks']) tbl.append(h('div', { class: 'h' }, c));
  for (const e of hist) {
    tbl.append(h('div', { class: 'mono' }, e.time), h('div', {}, e.job),
      h('div', {}, h('span', { class: 'badge ' + (e.verdict === 'PASS' ? 'pass' : 'fail') }, e.verdict === 'PASS' ? 'Accept' : 'Reject')),
      h('div', { class: 'mono' }, e.ms + ' ms'), h('div', { class: 'mono' }, e.barcode || '–'),
      h('div', { style: { color: 'var(--ng-text)' } }, e.failed.join(', ')));
  }
  p.append(h('div', { class: 'rows' }, hist.length ? tbl : h('div', { class: 'empty' }, 'No parts inspected yet.')));
  v.append(p);
}

// ================================================================ settings
function openSettings(tab) {
  const st = S.st, j = job();
  const body = h('div', {});
  const pane = h('div', { style: { marginTop: '14px' } });
  const tabs = segmented([{ value: 'source', label: 'Image source' }, { value: 'barcode', label: 'Barcode' },
                          { value: 'look', label: 'Appearance' }], tab || 'source', show);
  body.append(tabs, pane);
  let save = async () => true;
  function show(which) {
    pane.innerHTML = '';
    if (which === 'source') {
      const spec = j.source_spec || { mode: 'reference' };
      let mode = spec.mode;
      const folder = h('input', { class: 'field', value: spec.folder || '', placeholder: 'C:\\pictures\\line1' });
      const files = h('textarea', { class: 'field', style: { height: '80px', padding: '8px 12px' }, placeholder: 'one picture per line' }, (spec.paths || []).join('\n'));
      const cam = h('input', { class: 'field', type: 'number', min: 0, value: spec.camera || 0, style: { width: '90px' } });
      const sel = glassSelect([{ value: 'reference', label: 'The teach picture', icon: 'photo' }, { value: 'folder', label: 'A folder of pictures', icon: 'folder' },
                               { value: 'files', label: 'Picture files', icon: 'photo' }, { value: 'camera', label: 'Camera', icon: 'camera' }],
                              mode, m => { mode = m; }, { width: '240px' });
      pane.append(h('div', { class: 'faint', style: { marginBottom: '8px' } }, `For ${j.name} · now: ${j.source}`),
        formRow('Pictures come from', null, sel), formRow('Folder', 'For "A folder of pictures"', folder),
        formRow('Files', 'For "Picture files"', files), formRow('Camera number', 'Settings in Vision Station > Settings > Camera', cam));
      save = async () => {
        try { const r = await api('/api/source', { pid: j.pid, mode, folder: folder.value, files: files.value.split('\n').map(s => s.trim()), camera: Number(cam.value) });
              toast('Image source: ' + r.source, 'pass'); await poll(); return true; }
        catch (e) { toast(e.message, 'fail'); return false; }
      };
    } else if (which === 'barcode') {
      const bc = { ...st.barcode };
      const on = glassSwitch(bc.enabled, v => (bc.enabled = v));
      const prefix = h('input', { class: 'field mono', value: bc.prefix || '', placeholder: 'e.g. FIC', style: { width: '160px' } });
      const len = h('input', { class: 'field', type: 'number', min: 0, value: bc.length || 0, style: { width: '100px' } });
      const auto = glassSwitch(bc.auto_run !== false, v => (bc.auto_run = v));
      const casesw = glassSwitch(!!bc.match_case, v => (bc.match_case = v));
      pane.append(formRow('Check the barcode before inspecting', 'A part is inspected only after a valid scan; its pictures go in a folder named after it', on),
        formRow('Starts with', 'Empty: any start', prefix), formRow('Length', 'The whole code; 0: any length', len),
        formRow('Capital letters must match', null, casesw), formRow('Inspect as soon as a valid code is scanned', null, auto));
      save = async () => {
        const cfg = { enabled: bc.enabled, prefix: prefix.value.trim(), length: Number(len.value) || 0, match_case: bc.match_case,
                      auto_run: bc.auto_run !== false, source: bc.source || 'keyboard', port: bc.port || '', baud: bc.baud || 9600, folder: bc.folder || '' };
        await api('/api/barcode', { cfg }); toast('Barcode check ' + (cfg.enabled ? 'on' : 'off'), 'pass'); await poll(); renderView(); return true;
      };
    } else {
      const theme = localStorage.getItem('vg-theme') || 'auto';
      pane.append(formRow('Appearance', null, segmented([{ value: 'auto', label: 'Auto' }, { value: 'light', label: 'Light' }, { value: 'dark', label: 'Dark' }], theme, applyTheme)),
        formRow('Reduce transparency', 'Solid panels instead of frosted ones - lighter on the graphics card',
          glassSwitch(document.documentElement.classList.contains('flat'), v => { document.documentElement.classList.toggle('flat', v); localStorage.setItem('vg-flat', v ? '1' : '0'); })),
        formRow('Fluid pointer', 'A pointer that wraps buttons and turns into a crosshair, hand or arrow on the picture. Off: the Windows arrow',
          glassSwitch(localStorage.getItem('vg-pointer') !== '0', v => { localStorage.setItem('vg-pointer', v ? '1' : '0'); Pointer.apply(); })));
      save = async () => true;
    }
  }
  show(tab || 'source');
  sheet('Settings', 'Shared with Vision Station: the same projects, image sources and barcode rule.', body,
        [{ label: 'Cancel' }, { label: 'Save', prominent: true, onClick: () => save() }]);
}

// ================================================================ running and keys
async function runNow() {
  const btn = $('#runBtn'); btn.disabled = true; btn.classList.add('busy');
  try {
    const r = await api('/api/run', S.view === 'build' ? { pid: job().pid } : {});
    if (r.need_barcode) toast('Scan the part\'s barcode first', 'warn');
    else if (r.busy) toast('Still inspecting', 'warn');
    else {
      const vals = Object.values(r.results || {});
      const err = vals.find(x => x && x.error);
      if (err) toast(err.error, 'fail');
      else if (vals.length) {
        const ok = vals.every(x => x.verdict === 'PASS');
        toast(ok ? 'Accept - every check passed' : 'Reject - ' + vals.flatMap(x => x.tools.filter(t => t.status !== 'pass').map(t => t.tool_id)).join(', '), ok ? 'pass' : 'fail');
      }
    }
    await poll();
  } catch (e) { toast(e.message, 'fail'); }
  btn.disabled = false; btn.classList.remove('busy');
}
function onKey(e) {
  const typing = /INPUT|TEXTAREA/.test(document.activeElement?.tagName);
  if (e.key === 'F5') { e.preventDefault(); runNow(); return; }
  if (S.view === 'run' && S.st?.barcode?.enabled && !typing && e.key.length === 1) {
    const f = $('#scan'); if (f) { f.focus(); }
    return;
  }
  if (S.view !== 'build' || typing) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); undo(); return; }
  if (e.key === ' ') { S.space = true; e.preventDefault(); }
  const k = e.key.toLowerCase();
  if (k === 'v') setMode('select'); else if (k === 'h') setMode('pan');
  else if (k === 'r') setMode('rect'); else if (k === 'e') setMode('ellipse');
  else if (e.key === 'Escape') { S.armed = null; S.draft = null; if (S.finder?.drawing) S.finder.drawing = false;
                                 delete document.documentElement.dataset.ptr; setMode('select'); }
  else if ((e.key === 'Delete' || e.key === 'Backspace') && S.sel) {
    api('/api/tool/delete', { pid: job().pid, ids: [S.sel] }).then(async () => { toast(`Deleted ${S.sel} - Ctrl+Z brings it back`); S.sel = null; await afterEdit(); });
  } else if (k === 'f') glide(() => { S.cv.fitted = false; fitCanvas(); });
}
window.addEventListener('keyup', e => { if (e.key === ' ') S.space = false; });

boot().catch(e => { document.body.append(h('pre', { style: { position: 'fixed', top: '10px', left: '10px', zIndex: 99, color: 'red' } }, String(e.stack || e))); });
