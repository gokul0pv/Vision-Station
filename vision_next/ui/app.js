'use strict';
/* Vision Next interface -- "Fluid". Plain JavaScript, no build step.
   Three places: Run (operators), Build (recipes, every setting of every
   check), History (every part, filterable). The engine pushes results
   through a long poll; pictures arrive as JPEG at the size the screen needs;
   overlays are vector graphics drawn here from the result data. */

// ------------------------------------------------------------------- basics
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const SVG = 'http://www.w3.org/2000/svg';
const sleep = ms => new Promise(r => setTimeout(r, ms));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

function h(tag, attrs, ...kids) {
  const svg = tag.startsWith('svg:');
  const el = svg ? document.createElementNS(SVG, tag.slice(4)) : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else if (k === 'class') el.setAttribute('class', v);
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'value' && !svg) el.value = v;
    else if (k === 'checked' || k === 'selected' || k === 'disabled' || k === 'hidden' || k === 'multiple') el[k] = !!v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of kids.flat(9)) if (c != null && c !== false) el.append(c.nodeType ? c : document.createTextNode(String(c)));
  return el;
}

const IC = {
  play: 'M7 5.5v13l11-6.5z', loop: 'M17 2l4 4-4 4M3 11v-1a4 4 0 014-4h14M7 22l-4-4 4-4M21 13v1a4 4 0 01-4 4H3', stop: 'M7 7h10v10H7z',
  plus: 'M12 5v14M5 12h14', undo: 'M9 14L4 9l5-5M4 9h10a6 6 0 010 12h-3', redo: 'M15 14l5-5-5-5M20 9H10a6 6 0 000 12h3',
  save: 'M5 3h11l3 3v15H5zM8 3v6h8M8 21v-7h8v7', gear: 'M12 15.2a3.2 3.2 0 100-6.4 3.2 3.2 0 000 6.4zM19.4 13.5l1.6 1.3-2 3.4-2-.6a7.6 7.6 0 01-1.7 1l-.4 2.1h-3.8l-.4-2.1a7.6 7.6 0 01-1.7-1l-2 .6-2-3.4 1.6-1.3a7.8 7.8 0 010-2l-1.6-1.3 2-3.4 2 .6a7.6 7.6 0 011.7-1l.4-2.1h3.8l.4 2.1c.6.2 1.2.6 1.7 1l2-.6 2 3.4-1.6 1.3a7.8 7.8 0 010 2z',
  fit: 'M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5', grid: 'M4 4h7v7H4zM13 4h7v7h-7zM4 13h7v7H4zM13 13h7v7h-7z',
  camera: 'M3 8h4l2-3h6l2 3h4v11H3zM12 16.5a3.5 3.5 0 100-7 3.5 3.5 0 000 7z', folder: 'M3 6h6l2 2h10v11H3z',
  trash: 'M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3', home: 'M4 11l8-7 8 7v9H4zM10 20v-6h4v6', moon: 'M20 14.5A8 8 0 019.5 4 8 8 0 1020 14.5z',
  sun: 'M12 16a4 4 0 100-8 4 4 0 000 8zM12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4',
  up: 'M12 19V5M6 11l6-6 6 6', down: 'M12 5v14M18 13l-6 6-6-6', x: 'M6 6l12 12M18 6L6 18', check: 'M5 12.5l4.5 4.5L19 7.5',
  star: 'M12 3l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.4 6.8 19.1l1-5.8L3.5 9.2l5.9-.9z', upload: 'M12 16V4M7 9l5-5 5 5M4 20h16', download: 'M12 4v12M7 11l5 5 5-5M4 20h16',
  chev: 'M9 6l6 6-6 6', chevd: 'M6 9l6 6 6-6', more: 'M5 12h.01M12 12h.01M19 12h.01', search: 'M11 18a7 7 0 100-14 7 7 0 000 14zM20 20l-4-4',
  lock: 'M6 11h12v9H6zM8.5 11V8a3.5 3.5 0 017 0v3', unlock: 'M6 11h12v9H6zM8.5 11V8a3.5 3.5 0 016.7-1.4', copy: 'M8 8h11v11H8zM5 16V5h11',
  eye: 'M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 15a3 3 0 100-6 3 3 0 000 6z', eyeoff: 'M3 3l18 18M10.6 6.1A9.8 9.8 0 0112 6c6 0 10 6 10 6a17 17 0 01-3.1 3.8M6.6 6.6C3.8 8.5 2 12 2 12s4 7 10 7c1.6 0 3-.4 4.3-1',
  pointer: 'M5 3l14 8-6 2-3 6z', box: 'M4 6h16v12H4z', ellipse: 'M12 18c4.4 0 8-2.7 8-6s-3.6-6-8-6-8 2.7-8 6 3.6 6 8 6z', poly: 'M5 8l7-5 7 6-2 10H7z',
  live: 'M12 13a1 1 0 100-2 1 1 0 000 2zM8.5 15.5a5 5 0 010-7M15.5 8.5a5 5 0 010 7M5.6 18.4a9 9 0 010-12.8M18.4 5.6a9 9 0 010 12.8',
  history: 'M3 12a9 9 0 103-6.7L3 8M3 3v5h5M12 7v5l3 3', build: 'M14.7 6.3a4 4 0 00-5.4 5.4L3 18l3 3 6.3-6.3a4 4 0 005.4-5.4l-2.6 2.6-2.4-.6-.6-2.4z',
  barcode: 'M4 5v14M7 5v14M10 5v14M14 5v14M17 5v14M20 5v14', light: 'M9 18h6M10 21h4M12 3a6 6 0 00-3.5 10.9c.6.5 1 1.2 1 2.1h5c0-.9.4-1.6 1-2.1A6 6 0 0012 3z',
  line: 'M5 12h14M12 5l7 7-7 7', image: 'M4 5h16v14H4zM4 15l4-4 5 5 3-3 4 4M15 9.5a1.5 1.5 0 100-.01', cpu: 'M7 7h10v10H7zM10 3v4M14 3v4M10 17v4M14 17v4M3 10h4M3 14h4M17 10h4M17 14h4',
  info: 'M12 21a9 9 0 100-18 9 9 0 000 18zM12 11v6M12 7.5v.01', key: 'M15 7a4 4 0 11-3.8 5.2L4 19.5V22h3v-2h2v-2h2l1.2-1.2A4 4 0 0115 7z',
  pkg: 'M12 3l8 4.5v9L12 21l-8-4.5v-9zM4 7.5l8 4.5 8-4.5M12 12v9', import: 'M12 4v12M7 11l5 5 5-5M4 20h16', refresh: 'M20 12a8 8 0 11-2.3-5.7L20 9M20 4v5h-5',
  target: 'M12 21a9 9 0 100-18 9 9 0 000 18zM12 16a4 4 0 100-8 4 4 0 000 8zM12 12v.01', ruler: 'M3 17L17 3l4 4L7 21zM7 13l2 2M10 10l2 2M13 7l2 2',
  palette: 'M12 3a9 9 0 000 18c1 0 1.5-.7 1.5-1.5 0-.6-.3-1-.6-1.4-.3-.3-.5-.7-.5-1.1 0-.8.7-1.5 1.5-1.5H16a5 5 0 005-5c0-4-4-7.5-9-7.5zM7.5 12.5h.01M10 8h.01M15 8h.01',
  text: 'M5 6V4h14v2M12 4v16M9 20h6', count: 'M5 7h.01M5 12h.01M5 17h.01M9 7h10M9 12h10M9 17h10', surface: 'M3 17l6-6 4 4 8-8M15 7h6v6',
  logic: 'M8 6h8M8 12h8M8 18h5M4 6h.01M4 12h.01M4 18h.01', model: 'M12 3a4 4 0 014 4v1a4 4 0 01-8 0V7a4 4 0 014-4zM5 21a7 7 0 0114 0',
  locate: 'M12 2v4M12 18v4M2 12h4M18 12h4M12 17a5 5 0 100-10 5 5 0 000 10z', geometry: 'M4 20L20 4M4 4h6M4 4v6M20 20h-6M20 20v-6',
  grip: 'M9 6h.01M9 12h.01M9 18h.01M15 6h.01M15 12h.01M15 18h.01',
  rotcw: 'M20 12a8 8 0 11-2.3-5.7L20 9M20 4v5h-5', rotccw: 'M4 12a8 8 0 102.3-5.7L4 9M4 4v5h5', rot0: 'M4 18h16M7 18V6M7 6l-3 3M7 6l3 3',
  alignl: 'M4 3v18M8 7h12M8 13h7', alignh: 'M12 3v18M5 7h14M8 13h8', alignr: 'M20 3v18M4 7h12M9 13h7',
  alignt: 'M3 4h18M7 8v12M13 8v7', alignv: 'M3 12h18M7 5v14M13 8v8', alignb: 'M3 20h18M7 4v12M13 9v7',
  disth: 'M4 4v16M20 4v16M10 8h4v8h-4z', distv: 'M4 4h16M4 20h16M8 10h8v4H8z', samew: 'M4 12h16M7 9l-3 3 3 3M17 9l3 3-3 3', sameh: 'M12 4v16M9 7l3-3 3 3M9 17l3 3 3-3',
  zoomsel: 'M11 18a7 7 0 100-14 7 7 0 000 14zM20 20l-4-4M8 11h6M11 8v6', selall: 'M4 4h4M12 4h4M20 4v4M20 12v4M20 20h-4M12 20H8M4 20v-4M4 12V8',
  actual: 'M4 4h16v16H4zM9 9v6M14 9v6M11.5 10.5v.01M11.5 13.5v.01', link: 'M10 14a4 4 0 005.7 0l3-3a4 4 0 00-5.7-5.7l-1 1M14 10a4 4 0 00-5.7 0l-3 3a4 4 0 005.7 5.7l1-1', cmd: 'M9 6a3 3 0 10-3 3h12a3 3 0 10-3-3v12a3 3 0 103-3H6a3 3 0 103 3z',
};
function icon(n, cls) {
  const s = document.createElementNS(SVG, 'svg');
  s.setAttribute('viewBox', '0 0 24 24'); s.setAttribute('class', 'i' + (cls ? ' ' + cls : ''));
  const p = document.createElementNS(SVG, 'path'); p.setAttribute('d', IC[n] || IC.info); s.append(p);
  return s;
}
const GROUP_ICON = { 'Image': 'image', 'Locate': 'locate', 'Presence and count': 'count', 'Colour': 'palette', 'Geometry': 'geometry',
  'Surface': 'surface', 'Trained models': 'model', 'Text and codes': 'text', 'Logic': 'logic', 'Measurement': 'ruler' };
const store = {
  get(k, d) { try { const v = localStorage.getItem('vn.' + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem('vn.' + k, JSON.stringify(v)); } catch { } },
};

// -------------------------------------------------------------------- state
const S = {
  st: null, res: null, schema: null, stats: null, settings: null,
  view: store.get('view', 'run'), side: 0, sel: null, preview: null,
  buildPic: 'reference', runPic: 'last', tab: 'settings', draw: null, shape: 'rect',
  live: false, closed: store.get('closed', {}), histFilter: { verdict: '', q: '' }, showAllChecks: false,
  picks: new Set(),
};
const open = () => S.st && S.st.open;
const side = () => open() && open().sides[clamp(S.side, 0, open().sides.length - 1)];
const sideRes = (i = S.side) => S.res && S.res.sides && S.res.sides[i];
const lastResult = (i = S.side) => sideRes(i) && sideRes(i).result;
// every selected check (S.sel is the one shown in the inspector)
const sels = () => {
  if (!S.sel || !side()) return [];
  const have = new Set(side().tools.map(t => t.id));
  return (S.picks.has(S.sel) ? [...S.picks] : [S.sel]).filter(id => have.has(id));
};
const label = t => (S.schema.labels[t] || [t])[0];
const describe = t => (S.schema.labels[t] || ['', ''])[1];
const groupOf = t => (S.schema.groups.find(g => g[1].includes(t)) || ['Image'])[0];
const toolIcon = t => GROUP_ICON[groupOf(t)] || 'target';
const locked = () => S.st && S.st.locked;
const fmt = v => typeof v === 'number' ? (Number.isInteger(v) ? String(v) : Math.abs(v) >= 100 ? v.toFixed(1) : String(+v.toFixed(3)))
  : Array.isArray(v) ? v.map(fmt).join(', ') : v && typeof v === 'object' ? JSON.stringify(v) : String(v);

// ------------------------------------------------------------------- server
async function api(path, body, raw) {
  const opt = body === undefined ? {} : { method: 'POST', body: raw ? body : JSON.stringify(body), headers: raw ? {} : { 'Content-Type': 'application/json' } };
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({ error: r.statusText }));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}
function toast(msg, err) {
  const t = h('div', { class: 'toast' + (err ? ' err' : '') }, icon(err ? 'info' : 'check', 'sm'), msg);
  $('#toasts').append(t);
  setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 260); }, err ? 5200 : 2400);
}
const guard = fn => async (...a) => { try { return await fn(...a); } catch (e) { toast(e.message, true); } };

// ------------------------------------------------------------------- motion
document.addEventListener('pointermove', e => {
  const b = e.target.closest && e.target.closest('.btn');
  if (b) { const r = b.getBoundingClientRect(); b.style.setProperty('--mx', (e.clientX - r.left) + 'px'); b.style.setProperty('--my', (e.clientY - r.top) + 'px'); }
}, { passive: true });
document.addEventListener('pointerdown', e => {
  const b = e.target.closest && e.target.closest('.btn, .pick, .acard');
  if (!b || document.documentElement.dataset.motion === 'off') return;
  const r = b.getBoundingClientRect(), d = Math.max(r.width, r.height);
  const rp = h('span', { class: 'ripple', style: { width: d + 'px', height: d + 'px', left: (e.clientX - r.left - d / 2) + 'px', top: (e.clientY - r.top - d / 2) + 'px' } });
  b.append(rp); setTimeout(() => rp.remove(), 650);
}, { passive: true });

function tween(el, to, f = v => Math.round(v)) {
  if (!el) return;
  const from = parseFloat(el.dataset.v || '0') || 0;
  el.dataset.v = to;
  if (document.documentElement.dataset.motion === 'off' || from === to) { el.textContent = f(to); return; }
  const t0 = performance.now(), d = 520;
  const step = t => { const k = clamp((t - t0) / d, 0, 1), e = 1 - Math.pow(1 - k, 3); el.textContent = f(from + (to - from) * e); if (k < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}
// a segmented control whose pill glides to the chosen button
function seg(options, value, onpick, cls) {
  const el = h('div', { class: 'seg ' + (cls || '') }, h('span', { class: 'pill' }));
  for (const [v, text, ic] of options) {
    el.append(h('button', { class: v === value ? 'on' : '', 'data-v': v, onclick: () => { place(v); onpick(v); } }, ic ? icon(ic, 'sm') : null, text));
  }
  const place = v => {
    for (const b of $$('button', el)) b.classList.toggle('on', b.dataset.v === v);
    const b = $$('button', el).find(x => x.dataset.v === v), p = $('.pill', el);
    if (b) { p.style.left = b.offsetLeft + 'px'; p.style.width = b.offsetWidth + 'px'; p.style.opacity = 1; } else p.style.opacity = 0;
  };
  requestAnimationFrame(() => requestAnimationFrame(() => place(value)));
  el.place = place;
  return el;
}
function swapView(fn) {
  if (document.startViewTransition && document.documentElement.dataset.motion !== 'off') document.startViewTransition(fn);
  else fn();
}

// --------------------------------------------------------------------- boot
async function boot() {
  S.schema = await api('/api/schema');
  await refresh();
  S.settings = await api('/api/settings').catch(() => null);
  applyLook();
  if (!/[?&]still\b/.test(location.search)) poll();
  const m = /[?&]view=(\w+)/.exec(location.search); if (m) S.view = m[1];
  const ms = /[?&]sel=(\w+)/.exec(location.search); if (ms) S.sel = ms[1];
  const mt = /[?&]tab=(\w+)/.exec(location.search); if (mt) S.tab = mt[1];
  render();
  const mo = /[?&]open=(\w+)/.exec(location.search);
  if (mo) setTimeout(() => ({ settings: () => settingsSheet(/[?&]section=(\w+)/.exec(location.search)?.[1]), add: addSheet, palette })[mo[1]](), 300);
  stats(); setInterval(stats, 4000);
  bindKeys();
  window.addEventListener('resize', () => S.viewer && !S.viewer.userMoved && S.viewer.fit(false));
}
function applyLook() {
  const g = (S.settings && S.settings.general) || (S.st && S.st.general) || {};
  const theme = store.get('theme', g.theme || 'light');
  const dark = theme === 'dark' || (theme === 'auto' && matchMedia('(prefers-color-scheme: dark)').matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  document.documentElement.dataset.motion = g.motion === false ? 'off' : 'on';
}
async function refresh() {
  [S.st, S.res] = await Promise.all([api('/api/state'), api('/api/results')]);
  if (open() && S.side >= open().sides.length) S.side = 0;
  if (locked() && S.view === 'build') S.view = 'run';
}
async function reload() { await refresh(); render(); }
async function poll() {
  for (;;) {
    try {
      const r = await api('/api/wait?since=' + (S.res ? S.res.seq : 0));
      const ids = x => (x && x.sides || []).map(s => s.side).join();
      const changedApp = ids(r) !== ids(S.res), newer = !S.res || r.seq !== S.res.seq, lockChange = S.res && r.locked !== S.res.locked;
      const prevVerdict = overall();
      S.res = r;
      if (changedApp || lockChange) { await reload(); }
      else if (newer) onResults(prevVerdict);
    } catch { await sleep(1500); }
  }
}
async function stats() { try { S.stats = await api('/api/stats'); renderFoot(); } catch { } }

function overall(res = S.res) {
  const rs = (res && res.sides || []).map(s => s.result).filter(Boolean);
  if (!rs.length) return '';
  if (rs.some(r => r.verdict === 'FAIL' || r.verdict === 'ERROR')) return 'FAIL';
  if (rs.some(r => r.verdict === 'WARN')) return 'WARN';
  return 'PASS';
}
function onResults(prev) {
  renderTop(true);
  renderFoot();
  if (!open()) return;
  if (S.view === 'run') { updateRunPanel(true); renderStrip(); if (S.viewer) { S.viewer.refresh(); S.viewer.flash(overall()); } }
  else if (S.view === 'build') { if (S.buildPic === 'last' && S.viewer) S.viewer.refresh(); renderChecks(); renderInspector(); }
  else if (S.view === 'history') renderHistory();
}

// ------------------------------------------------------------------- render
function render() {
  if (S.viewer) { S.viewer.destroy(); S.viewer = null; }
  S.live = false;
  renderTop(false);
  renderMain();
  renderFoot();
}
function setView(v) {
  if (v === S.view) return;
  S.view = v; store.set('view', v); S.draw = null;
  // straight in: a page-transition snapshot of a big picture and a long list costs more than it shows
  if (S.viewer) { S.viewer.destroy(); S.viewer = null; } S.live = false;
  renderMain(); renderTop(false);
}

function renderTop(flash) {
  const top = $('#top');
  top.replaceChildren();
  top.append(h('div', { class: 'brand' }, h('img', { src: '/icon.svg', alt: '' }), 'Vision Next'));
  if (!open()) {
    top.append(h('div', { class: 'grow' }), themeBtn(), h('button', { class: 'btn ghost icon', title: 'Settings', onclick: () => settingsSheet() }, icon('gear')));
    return;
  }
  const o = open();
  top.append(
    h('div', { class: 'crumb', onclick: e => appMenu(e.currentTarget) }, h('span', {}, o.name), icon('chevd', 'sm')),
    seg([['run', 'Run', 'play'], ...(locked() ? [] : [['build', 'Build', 'build']]), ['history', 'History', 'history']], S.view, setView, 'big'),
    h('div', { class: 'grow' }));
  const c = S.res ? S.res.counts : { total: 0, pass: 0, fail: 0 };
  const parts = h('b', { 'data-v': '0' }), yld = h('b', { 'data-v': '0' }), cyc = h('b', { 'data-v': '0' });
  top.append(
    h('div', { class: 'stat' }, parts, h('span', {}, 'Parts')),
    h('div', { class: 'stat' }, yld, h('span', {}, 'Yield')),
    h('div', { class: 'stat' }, cyc, h('span', {}, 'Cycle')));
  const prev = S.topNums || { total: 0, y: 0, ms: 0 };
  parts.dataset.v = prev.total; yld.dataset.v = prev.y; cyc.dataset.v = prev.ms;
  const y = c.total ? 100 * c.pass / c.total : 0, ms = S.res ? S.res.cycle_ms || 0 : 0;
  tween(parts, c.total); tween(yld, y, v => c.total ? v.toFixed(1) + '%' : '–'); tween(cyc, ms, v => ms ? Math.round(v) + ' ms' : '–');
  S.topNums = { total: c.total, y, ms };
  const v = overall(), busy = S.res && S.res.busy;
  top.append(h('div', { class: 'verdict ' + v + (flash && v ? ' pop' : '') + (busy ? ' busy' : '') }, h('span', { class: 'ring' }), v === 'PASS' ? 'OK' : v === 'FAIL' ? 'NG' : v || 'READY'));
  const cont = S.res && S.res.continuous;
  top.append(
    h('button', { class: 'btn primary', title: 'Inspect once (Space, F5)', onclick: runOnce, disabled: cont }, icon('play'), 'Run'),
    open() && !locked() && S.settings && S.settings.general && S.settings.general.advanced !== false
      ? h('button', { class: 'btn icon', title: 'Normal run, or a model teaching run', disabled: cont, onclick: e => runMenu(e) }, icon('chevd', 'sm')) : null,
    h('button', { class: 'btn' + (cont ? ' on' : ''), title: 'Inspect continuously (F6)', onclick: toggleContinuous }, icon(cont ? 'stop' : 'loop'), cont ? 'Stop' : 'Continuous'));
  if (S.st.operator) {
    top.append(h('button', { class: 'btn ghost icon', title: locked() ? 'Unlock (supervisor)' : 'Lock (operator mode)', onclick: locked() ? unlockDialog : lockNow }, icon(locked() ? 'lock' : 'unlock')));
  }
  top.append(themeBtn(), h('button', { class: 'btn ghost icon', title: 'Settings', onclick: () => settingsSheet(), disabled: locked() }, icon('gear')));
}
function themeBtn() {
  const dark = document.documentElement.dataset.theme === 'dark';
  return h('button', { class: 'btn ghost icon', title: 'Light or dark', onclick: () => {
    swapView(() => { store.set('theme', dark ? 'light' : 'dark'); applyLook(); renderTop(false); renderFoot(); });
  } }, icon(dark ? 'sun' : 'moon'));
}

const runOnce = guard(async () => { await api('/api/run', {}); });
function runMenu(e) {
  const sd = side();
  menu({ currentTarget: e.currentTarget, type: 'click', clientX: 0, clientY: 0 }, [
    ['Normal run', 'play', runOnce, '', 'Space'],
    [sd.advanced ? 'Model teaching run (retrain)…' : 'Model teaching run…', 'model', () => teachingRun()],
    sd.advanced ? ['Remove advanced model…', 'trash', () => removeAdvanced(), 'danger'] : null,
  ]);
}
/* Model teaching run (Vision Station's advanced training): 16 pictures of a
   good part, then the side's own network trained on the graphics card to
   confirm its part and text checks. A window shows how far it is; Cancel
   deletes everything it made. */
const teachingRun = guard(async () => {
  const sd = side();
  if (sd.dirty) { toast('Save first (Ctrl+S): the teaching run reads the job as saved', true); return; }
  if (!await confirm1(`Model teaching run for ${sd.name}?`, 'Put a GOOD part under the camera, as it lies on the line. 16 pictures are taken, then a model is trained on the graphics card (10 to 30 minutes). Inspection is not possible meanwhile.')) return;
  await api('/api/teach/start', { side: sd.id });
  teachWindow();
});
const removeAdvanced = guard(async () => {
  const sd = side();
  if (!await confirm1(`Remove the advanced model of ${sd.name}?`, 'Its checks are judged by their own settings again. A new teaching run makes a new model.')) return;
  await api('/api/teach/remove', { side: sd.id }); await reload(); toast('Advanced model removed');
});
function teachWindow() {
  const bar = h('div', { class: 'teach-bar' }, h('span', {})), pct = h('b', {}, '0 %'), msg = h('div', { class: 'teach-msg' }, '…'), detail = h('div', { class: 'muted' }, ''),
    time1 = h('span', { class: 'muted' }, ''), cancel = h('button', { class: 'btn' }, 'Cancel'), close = h('button', { class: 'btn primary', hidden: true, onclick: () => { closeLayer(); reload(); } }, 'Close');
  const dlg = h('div', { class: 'dialog', style: { width: 'min(560px, calc(100vw - 32px))' } }, h('h2', {}, 'Model teaching run'), msg, h('div', { class: 'row', style: { margin: '14px 0 6px' } }, bar, pct), detail,
    h('div', { class: 'row', style: { marginTop: '12px' } }, time1, h('span', { class: 'grow' }), cancel, close));
  layer(dlg);
  $('#layer')._onclose = () => { };
  cancel.onclick = guard(async () => { cancel.disabled = true; cancel.textContent = 'Cancelling - deleting what it made…'; await api('/api/teach/cancel', {}); });
  const fmtT = s2 => s2 >= 3600 ? `${Math.floor(s2 / 3600)} h ${Math.round(s2 % 3600 / 60)} min` : s2 >= 60 ? `${Math.floor(s2 / 60)} min ${Math.round(s2 % 60)} s` : `${Math.round(s2)} s`;
  const tick = async () => {
    if (!dlg.isConnected) return;
    let t; try { t = await api('/api/teach'); } catch { setTimeout(tick, 1500); return; }
    const p = Math.round((t.progress || 0) * 100);
    bar.firstChild.style.width = p + '%'; pct.textContent = p + ' %';
    msg.textContent = t.message || ''; detail.textContent = t.epochs ? `${t.detail || ''}${t.detail ? ' · ' : ''}epoch ${t.epoch} of ${t.epochs}` : (t.detail || '');
    const el = t.started ? Date.now() / 1000 - t.started : 0;
    time1.textContent = `Time taken ${fmtT(el)}` + (t.eta_s ? ` · about ${fmtT(t.eta_s)} left` : '');
    cancel.disabled = cancel.disabled || t.can_cancel === false;
    if (t.done) {
      cancel.hidden = true; close.hidden = false;
      if (t.ok) {
        const r = t.result || {};
        msg.textContent = 'Training completed: the model is in use from the next run.';
        detail.textContent = [r.learned != null ? `${r.learned} checks learned` : '', r.precision != null ? `precision ${fmt(r.precision)}` : '', r.recall != null ? `recall ${fmt(r.recall)}` : ''].filter(Boolean).join(' · ');
      } else if (t.cancelled) msg.textContent = 'Cancelled - everything it made was deleted.';
      else { msg.textContent = 'Not trained: ' + (t.error || 'the training stopped'); msg.classList.add('ng-text'); }
      return;
    }
    setTimeout(tick, 1000);
  };
  tick();
}
const toggleContinuous = guard(async () => { await api('/api/continuous', { on: !(S.res && S.res.continuous) }); });
const lockNow = guard(async () => { await api('/api/operator/lock', {}); toast('Locked: operator mode'); });

function renderMain() {
  const main = $('#main');
  main.replaceChildren();
  if (!open()) { main.append(home()); return; }
  if (S.view === 'build' && !locked()) main.append(buildView());
  else if (S.view === 'history') main.append(historyView());
  else { S.view = 'run'; main.append(runView()); }
}

// --------------------------------------------------------------------- home
function hue(s) { let x = 0; for (const c of s) x = (x * 31 + c.charCodeAt(0)) % 360; return x; }
function home() {
  const el = h('div', { class: 'home' });
  const hr = new Date().getHours();
  el.append(h('div', { class: 'muted', style: { fontWeight: 600 } }, hr < 12 ? 'Good morning' : hr < 18 ? 'Good afternoon' : 'Good evening'),
    h('h1', {}, 'Applications'), h('p', { class: 'lead' }, 'Open one to inspect, or build a new one.'));
  const cards = h('div', { class: 'cards' });
  S.st.apps.forEach((a, i) => {
    const hh = hue(a.id);
    cards.append(h('div', { class: 'acard', style: { animationDelay: (i * 40) + 'ms' }, onclick: guard(async () => {
      await api('/api/open', { app: a.id }); S.side = 0; S.sel = null; S.preview = null; S.view = locked() ? 'run' : S.view; await reload(); if (S.view === 'build') preview();
    }) },
      h('div', { class: 'tile', style: { background: `linear-gradient(135deg, hsl(${hh} 85% 62%), hsl(${(hh + 40) % 360} 80% 50%))`, color: `hsl(${hh} 80% 50%)` } }, h('span', { style: { color: '#fff' } }, icon('target'))),
      h('h3', {}, a.name), h('div', { class: 'muted' }, `${a.sides} side${a.sides === 1 ? '' : 's'} · ${a.checks} checks`)));
  });
  if (!locked()) {
    cards.append(h('div', { class: 'acard new', style: { animationDelay: (S.st.apps.length * 40) + 'ms' }, onclick: newApplication }, icon('plus'), 'New application'));
    const imp = h('input', { type: 'file', accept: '.vnp,.zip,.vsp', hidden: true, onchange: importPackage });
    cards.append(h('label', { class: 'acard new', style: { animationDelay: ((S.st.apps.length + 1) * 40) + 'ms' } }, icon('pkg'), 'Import package', imp));
  }
  el.append(cards);
  if (S.st.legacy && !locked()) {
    const imp = h('div', { class: 'cards' }, ...[0, 1, 2].map(() => h('div', { class: 'acard skeleton', style: { minHeight: '90px' } })));
    el.append(h('h2', { style: { margin: '40px 0 4px', fontSize: '19px', letterSpacing: '-.02em' } }, 'From Vision Station'),
      h('p', { class: 'lead', style: { marginBottom: '16px', fontSize: '13.5px' } }, 'Copied here. Vision Station\'s own files stay as they are.'), imp);
    api('/api/import/list').then(r => {
      imp.replaceChildren();
      r.apps.forEach((a, i) => imp.append(h('div', { class: 'acard', style: { animationDelay: (i * 35) + 'ms', minHeight: '96px' }, onclick: guard(async () => {
        await api('/api/import', { id: a.id }); S.side = 0; await reload(); toast(`Imported ${a.name}`);
      }) }, h('div', { class: 'row' }, icon('import'), h('h3', {}, a.name)), h('div', { class: 'muted' }, `${a.sides} side(s) · ${a.checks} checks`))));
    }).catch(e => imp.replaceChildren(h('div', { class: 'muted' }, e.message)));
  }
  return el;
}
const newApplication = guard(async () => {
  const name = await prompt1('New application', 'A name for the part or product.', '');
  if (!name) return;
  await api('/api/app/new', { name }); S.view = 'build'; store.set('view', 'build'); await reload();
});
const importPackage = guard(async e => {
  const f = e.target.files[0]; if (!f) return;
  await api('/api/package/import', await f.arrayBuffer(), true); S.side = 0; await reload(); toast('Package imported');
});

// ------------------------------------------------------------------- viewer
/* The picture and its overlays. Zoom glides to its target; the stage is one
   CSS transform; outlines are SVG in picture pixels with strokes that keep
   their width at any zoom; captions keep their size on screen. */
class Viewer {
  constructor(o) {
    this.o = o;
    this.z = { s: 1, tx: 0, ty: 0 }; this.goal = null; this.size = null; this.key = '';
    this.el = h('div', { class: 'viewer' });
    this.stage = h('div', { class: 'stage' }, this.img = h('img', { alt: '', draggable: 'false' }), this.svg = h('svg:svg', {}));
    this.el.append(this.stage, this.prog = h('div', { class: 'progress' }), this.flashEl = h('div', { class: 'flash' }));
    this.caps = []; this.drag = null;
    this.bind();
    // the canvas changes size as the window maximises, panels open, the
    // sheet slides: the picture stays fitted until someone zooms or pans
    this.userMoved = false;
    this.ro = new ResizeObserver(() => { if (this.size && !this.userMoved) this.fit(false); });
    this.ro.observe(this.el);
  }
  destroy() { this.dead = true; this.ro && this.ro.disconnect(); window.removeEventListener('pointermove', this.mv); window.removeEventListener('pointerup', this.up); }
  toImg(e) { const r = this.el.getBoundingClientRect(); return [(e.clientX - r.left - this.z.tx) / this.z.s, (e.clientY - r.top - this.z.ty) / this.z.s]; }
  apply() {
    this.stage.style.transform = `translate(${this.z.tx}px, ${this.z.ty}px) scale(${this.z.s})`;
    if (this.tag && this.tag._x != null) this.tag.setAttribute('transform', `translate(${this.tag._x} ${this.tag._y}) scale(${1 / this.z.s})`);
    if (this.dims) this.dims.setAttribute('transform', `translate(${this.dims._x} ${this.dims._y}) scale(${1 / this.z.s})`);
    for (const p of $$('circle[data-r]', this.svg)) p.setAttribute('r', (+p.dataset.r || 6) / this.z.s);
    for (const p of $$('[data-sw]', this.svg)) p.setAttribute('stroke-width', +p.dataset.sw / this.z.s);
    for (const p of $$('.pt', this.svg)) p.setAttribute('r', Math.max(1.5, 2.6 / this.z.s));
    for (const p of $$('.fp-label', this.svg)) p.style.fontSize = (11 / this.z.s) + 'px';
    if (this.o.onZoom) this.o.onZoom(this.z.s);
  }
  animateTo(z) {
    if (document.documentElement.dataset.motion === 'off') { this.z = z; this.apply(); return; }
    this.goal = z;
    if (this.raf) return;
    const step = () => {
      if (this.dead || !this.goal) { this.raf = 0; return; }
      const g = this.goal, k = 0.22;
      this.z = { s: this.z.s + (g.s - this.z.s) * k, tx: this.z.tx + (g.tx - this.z.tx) * k, ty: this.z.ty + (g.ty - this.z.ty) * k };
      if (Math.abs(g.s - this.z.s) < g.s * 0.002 && Math.abs(g.tx - this.z.tx) < 0.3 && Math.abs(g.ty - this.z.ty) < 0.3) { this.z = g; this.goal = null; }
      this.apply();
      this.raf = this.goal ? requestAnimationFrame(step) : 0;
    };
    this.raf = requestAnimationFrame(step);
  }
  fit(animate = true) {
    if (!this.size) return;
    this.userMoved = false;
    const W = this.el.clientWidth, H = this.el.clientHeight;
    if (!W || !H) return;
    const s = Math.min(W / this.size[0], H / this.size[1]) * 0.96;
    const z = { s, tx: (W - this.size[0] * s) / 2, ty: (H - this.size[1] * s) / 2 };
    if (animate) this.animateTo(z); else { this.goal = null; this.z = z; this.apply(); }
  }
  zoomBy(k, cx, cy) {
    this.userMoved = true;
    const base = this.goal || this.z, W = this.el.clientWidth, H = this.el.clientHeight;
    cx = cx == null ? W / 2 : cx; cy = cy == null ? H / 2 : cy;
    const s = clamp(base.s * k, 0.02, 40);
    this.animateTo({ s, tx: cx - (cx - base.tx) * s / base.s, ty: cy - (cy - base.ty) * s / base.s });
    clearTimeout(this._zt); this._zt = setTimeout(() => this.refresh(true), 260);
  }
  flash(v) { if (!v) return; this.flashEl.className = 'flash'; void this.flashEl.offsetWidth; this.flashEl.className = 'flash ' + v; }
  busy(on) { this.prog.classList.toggle('on', !!on); }
  refresh(onlyPic) {
    if (this.dead) return;
    const size = this.o.size();
    const empty = $('.empty', this.el);
    if (!size) {
      this.img.removeAttribute('src'); this.svg.replaceChildren(); this.size = null;
      if (!empty) this.el.append(this.o.empty());
      return;
    }
    if (empty) empty.remove();
    const key = this.o.key() + ':' + size.join('x');
    if (key !== this.key || !this.size || this.size[0] !== size[0] || this.size[1] !== size[1]) {
      const first = !this.size; this.size = size; this.key = key;
      this.img.style.width = size[0] + 'px'; this.img.style.height = size[1] + 'px';
      this.svg.setAttribute('viewBox', `0 0 ${size[0]} ${size[1]}`); this.svg.setAttribute('width', size[0]); this.svg.setAttribute('height', size[1]);
      requestAnimationFrame(() => this.fit(!first));
    }
    const vw = this.el.clientWidth || 1200;
    const want = clamp(Math.ceil(vw * devicePixelRatio * Math.max(1, (this.goal || this.z).s * size[0] / vw) / 800) * 800, 800, size[0]);
    const url = this.o.url(want);
    if (url && this.img.dataset.url !== url && this._want !== url && !S.live) {
      // the new picture is decoded off screen, then put in place whole:
      // no blank frame between two pictures
      this._want = url;
      const pre = new Image();
      pre.alt = ''; pre.draggable = false; pre.src = url;
      pre.decode().then(() => {
        if (this.dead || S.live || this._want !== url) return;
        pre.style.width = this.img.style.width; pre.style.height = this.img.style.height; pre.dataset.url = url;
        this.img.replaceWith(pre); this.img = pre;
      }).catch(() => { this._want = null; });
    }
    if (!onlyPic) this.draw();
  }
  draw() {
    const svg = this.svg; svg.replaceChildren(); this.caps = []; this.regions = new Map(); this.tag = null;
    if (!this.size || S.live) return;
    const r = this.o.result();
    const tools = this.o.tools();
    const enabled = new Map(tools.map(t => [t.id, t.params && t.params.enabled !== false]));
    const P = p => p.map(q => q[0].toFixed(1) + ',' + q[1].toFixed(1)).join(' ');
    const regions = document.createDocumentFragment(), found = document.createDocumentFragment(), hits = document.createDocumentFragment();
    const sel = this.o.selected();
    const many = new Set(this.o.selection ? this.o.selection() : (sel ? [sel] : []));
    const one = many.size <= 1;
    const editSel = this.o.editable() && one && tools.find(x => x.id === sel);
    const status = new Map(), hitList = [];
    const addRegion = (id, pts, st, isSel, cursor) => {
      const poly = h('svg:polygon', { class: 'rg ' + st + (isSel ? ' sel' : ''), points: P(pts) });
      regions.append(poly);
      const xs = pts.map(q => q[0]), ys = pts.map(q => q[1]);
      hitList.push([(Math.max(...xs) - Math.min(...xs)) * (Math.max(...ys) - Math.min(...ys)) * (isSel ? 0 : 1),
        h('svg:polygon', { class: 'hit', points: P(pts), 'data-tool': id, style: cursor ? { cursor } : null })]);
      this.regions.set(id, { el: poly, x: Math.min(...xs), y: Math.min(...ys), w: Math.max(...xs) - Math.min(...xs) });
      status.set(id, st);
    };
    if (r && r.tools) {
      for (const t of r.tools) {
        if (enabled.get(t.tool_id) === false) continue;
        const isSel = many.has(t.tool_id);
        for (const g of t.graphics || []) {
          if (g.role === 'region') {
            if (editSel && isSel) continue; // drawn from its settings, below
            if (g.points && g.points.length > 1) addRegion(t.tool_id, g.points, t.status, isSel);
            continue;
          }
          const st = g.status || t.status;
          if (g.kind === 'point') found.append(h('svg:circle', { class: 'pt ' + st, cx: g.points[0][0], cy: g.points[0][1], r: Math.max(1.5, 2.6 / this.z.s) }));
          else if (g.kind === 'line') found.append(h('svg:line', { class: 'fd ' + st, x1: g.points[0][0], y1: g.points[0][1], x2: g.points[1][0], y2: g.points[1][1] }));
          else if (g.points && g.points.length > 1) found.append(h('svg:polygon', { class: 'fd ' + st, points: P(g.points) }));
        }
      }
    } else {
      for (const t of tools) {
        if (editSel && t.id === sel) continue;
        const out = outlineOf(t.params); if (!out) continue;
        addRegion(t.id, out, 'idle', many.has(t.id));
      }
    }
    if (editSel) {
      const out = outlineOf(editSel.params);
      if (out) addRegion(sel, out, 'edit', true, editSel.params.locked ? null : 'move');
    }
    // small regions over big ones (a component inside a board-sized box stays
    // reachable); the selected one on top of all
    hitList.sort((x, y) => y[0] - x[0]).forEach(x => hits.append(x[1]));
    svg.append(regions, found, hits);
    // Find parts and text: the area searched and every proposal, on top
    const ov = this.o.overlay && this.o.overlay();
    if (ov) {
      // As Vision Station's finder boxes: drag inside moves one, an edge or a
      // corner resizes it, just outside a corner turns it (Shift: 15 degree
      // steps), a click keeps or drops it. Grips show under the pointer.
      const g = h('svg:g', { class: 'finder' }), k = this.z.s;
      const box = (key, roi, ang, cls, colour, label) => {
        const cs = corners(roi, ang), gb = h('svg:g', { class: 'fpbox', 'data-fp': key });
        for (const q of cs) gb.append(h('svg:circle', { class: 'fpz', cx: q[0], cy: q[1], r: 16 / k, 'data-r': 16, 'data-z': 'turn', style: { cursor: ROTATE_CURSOR } }));
        gb.append(h('svg:polygon', { class: cls, points: P(cs), 'data-z': 'move', style: { stroke: colour } }));
        const sides = [['n', 0, 1, -90], ['e', 1, 2, 0], ['s', 2, 3, 90], ['w', 3, 0, 180]];
        for (const [z, i, j, dir] of sides) {
          gb.append(h('svg:line', { class: 'fpz edge', x1: cs[i][0], y1: cs[i][1], x2: cs[j][0], y2: cs[j][1], 'stroke-width': 12 / k, 'data-sw': 12, 'data-z': z, style: { cursor: resizeCursor(dir + ang) } }));
          gb.append(h('svg:circle', { class: 'fph', cx: (cs[i][0] + cs[j][0]) / 2, cy: (cs[i][1] + cs[j][1]) / 2, r: 3.5 / k, 'data-r': 3.5, style: { stroke: colour } }));
        }
        [['nw', -135], ['ne', -45], ['se', 45], ['sw', 135]].forEach(([z, dir], i) => {
          gb.append(h('svg:circle', { class: 'fph', cx: cs[i][0], cy: cs[i][1], r: 4.5 / k, 'data-r': 4.5, style: { stroke: colour } }));
          gb.append(h('svg:circle', { class: 'fpz', cx: cs[i][0], cy: cs[i][1], r: 8 / k, 'data-r': 8, 'data-z': z, style: { cursor: resizeCursor(dir + ang) } }));
        });
        if (label) {
          const top = cs.reduce((a, q) => q[1] < a[1] ? q : a, cs[0]);
          gb.append(h('svg:text', { class: 'fp-label', x: top[0] + 2 / k, y: top[1] - 3 / k, style: { fill: colour, fontSize: (11 / k) + 'px' } }, label));
        }
        g.append(gb);
      };
      if (ov.area) box('a', ov.area, ov.areaAngle || 0, 'fp-area', null, null);
      ov.boxes.forEach((b, i) => box(String(i), b.box, b.angle || 0, 'fp' + (b.kept ? '' : ' off') + (b.hot ? ' hot' : ''), b.kept ? b.colour : null, b.kept ? b.label : null));
      svg.append(g);
    }
    this.status = status;
    // one name tag: the selected check's, or the one under the pointer
    this.tag = h('svg:g', { class: 'tag' }, h('svg:rect', { rx: 6, height: 20, y: -26 }), h('svg:text', { x: 8, y: -16 }));
    svg.append(this.tag);
    this.showTag(sel);
    // handles for the selected check, where regions are drawn (the reference)
    const t = editSel;
    if (t && t.params && !t.params.locked) {
      const p = t.params, hf = document.createDocumentFragment();
      const grab = (x, y, k, cur, cls, r) => {
        hf.append(h('svg:circle', { class: 'handle ' + (cls || ''), cx: x, cy: y, r: (r || 6) / this.z.s, 'data-r': r || 6 }));
        hf.append(h('svg:circle', { class: 'grip-area', cx: x, cy: y, r: 13 / this.z.s, 'data-r': 13, 'data-handle': k, style: { cursor: cur } }));
      };
      if (p.shape === 'polygon' && p.points && p.points.length > 2) {
        // the bounding box's corners scale the whole shape (a region traced a
        // little tight); each corner of the shape moves on its own
        const xs = p.points.map(q => q[0]), ys = p.points.map(q => q[1]);
        const L = Math.min(...xs), T = Math.min(...ys), Rr = Math.max(...xs), B = Math.max(...ys);
        hf.append(h('svg:rect', { class: 'pbox', x: L, y: T, width: Rr - L, height: B - T }));
        for (const [k, x, y, cur] of [['bnw', L, T, 'nwse-resize'], ['bne', Rr, T, 'nesw-resize'], ['bse', Rr, B, 'nwse-resize'], ['bsw', L, B, 'nesw-resize']]) grab(x, y, k, cur, 'edge', 4);
        p.points.forEach((q, i) => grab(q[0], q[1], 'v' + i, 'crosshair'));
      } else if (p.roi) {
        const a = angleOf(p), cs = corners(p.roi, a);
        const mid = (i, j) => [(cs[i][0] + cs[j][0]) / 2, (cs[i][1] + cs[j][1]) / 2];
        const top = rotHandle(p.roi, a, this.z.s), m0 = mid(0, 1);
        hf.append(h('svg:line', { class: 'stalk', x1: m0[0], y1: m0[1], x2: top[0], y2: top[1] }));
        const spots = [['nw', cs[0], -135], ['n', mid(0, 1), -90], ['ne', cs[1], -45], ['e', mid(1, 2), 0],
                       ['se', cs[2], 45], ['s', mid(2, 3), 90], ['sw', cs[3], 135], ['w', mid(3, 0), 180]];
        for (const [k, q, dir] of spots) grab(q[0], q[1], k, resizeCursor(dir + a), k.length === 1 ? 'edge' : '', k.length === 1 ? 4 : 5.5);
        grab(top[0], top[1], 'rot', ROTATE_CURSOR, 'rot', 6);
      }
      svg.append(hf);
    }
  }
  // the name of a check over its region: no values, just which check it is
  showTag(id) {
    if (!this.tag) return;
    const reg = id && this.regions && this.regions.get(id);
    if (!reg) { this.tag.style.display = 'none'; return; }
    const st = this.status.get(id) || 'idle';
    const word = st === 'pass' ? '  OK' : st === 'fail' ? '  NG' : st === 'error' ? '  error' : '';
    const tool = this.o.tools().find(x => x.id === id);
    const text = id + (this.o.editable() ? (tool && tool.params && tool.params.locked ? '  (locked)' : '') : word);
    $('text', this.tag).textContent = text;
    const w = 16 + text.length * 7.2;
    $('rect', this.tag).setAttribute('width', w);
    this.tag.setAttribute('class', 'tag ' + (id === this.o.selected() ? 'sel' : st));
    this.tag._x = reg.x; this.tag._y = reg.y;
    this.tag.setAttribute('transform', `translate(${reg.x} ${reg.y}) scale(${1 / this.z.s})`);
    this.tag.style.display = '';
  }
  hover(id) {
    if (this._hov === id) return;
    if (this._hov && this.regions && this.regions.get(this._hov)) this.regions.get(this._hov).el.classList.remove('hov');
    this._hov = id;
    if (id && this.regions && this.regions.get(id)) this.regions.get(id).el.classList.add('hov');
    this.showTag(id || this.o.selected());
  }
  bind() {
    const el = this.el;
    el.addEventListener('wheel', e => { e.preventDefault(); const r = el.getBoundingClientRect(); this.zoomBy(Math.exp(-e.deltaY * 0.0016), e.clientX - r.left, e.clientY - r.top); }, { passive: false });
    el.addEventListener('dblclick', e => {
      if (S.draw) return;
      // what is under the pointer now: the first click may have redrawn it
      const t = document.elementFromPoint(e.clientX, e.clientY) || e.target, tool = selTool();
      const poly = this.o.editable() && tool && tool.params && !tool.params.locked && tool.params.shape === 'polygon' && (tool.params.points || []).length > 2 ? tool : null;
      if (poly && t.dataset && t.dataset.handle && t.dataset.handle[0] === 'v') {
        // a corner: removed (three stay at least)
        const pts = poly.params.points.map(q => q.slice()), i = +t.dataset.handle.slice(1);
        if (pts.length > 3) { pts.splice(i, 1); this.o.edit(polyParams(pts)); }
        return;
      }
      if (poly && t.dataset && t.dataset.tool === poly.id) {
        // an edge: a corner added where the nearest edge's middle is
        const p = this.toImg(e), pts = poly.params.points.map(q => q.slice());
        let best = Infinity, at = 0;
        pts.forEach((a, k) => { const b = pts[(k + 1) % pts.length], mx = (a[0] + b[0]) / 2 - p[0], my = (a[1] + b[1]) / 2 - p[1]; if (mx * mx + my * my < best) { best = mx * mx + my * my; at = k + 1; } });
        pts.splice(at, 0, [Math.round(p[0]), Math.round(p[1])]);
        this.o.edit(polyParams(pts));
        return;
      }
      this.fit();
    });
    el.addEventListener('pointerdown', e => {
      if (e.button !== 0 && e.button !== 1) return;
      const p = this.toImg(e), t = e.target;
      this.goal = null;
      if (e.button === 1) {
        // the middle button pans, whatever is under it
        this.drag = { kind: 'pan', x: e.clientX, y: e.clientY, tx: this.z.tx, ty: this.z.ty, moved: true, keep: true };
        el.classList.add('panning'); el.setPointerCapture && el.setPointerCapture(e.pointerId); e.preventDefault(); return;
      }
      const fpb = !S.draw && e.button === 0 && t.dataset && t.dataset.z && t.closest && t.closest('[data-fp]');
      if (fpb && this.o.overlayItem) {
        const it = this.o.overlayItem(fpb.dataset.fp);
        if (!it || !it.roi) return;
        this.drag = { kind: t.dataset.z, fp: fpb.dataset.fp, p0: p, params: { roi: it.roi.slice(), angle: it.angle || 0 } };
        holdCursor(getComputedStyle(t).cursor);
        el.setPointerCapture && el.setPointerCapture(e.pointerId);
        e.preventDefault(); return;
      }
      if (S.draw && e.button === 0) {
        if (S.draw.shape === 'polygon') {
          S.draw.pts = S.draw.pts || []; S.draw.pts.push(p); this.drawPoly(); e.preventDefault(); return;
        }
        this.drag = { kind: 'draw', p0: p, p1: p };
      } else if (t.dataset && t.dataset.handle && e.button === 0) {
        const tool = selTool();
        this.drag = { kind: t.dataset.handle, p0: p, params: JSON.parse(JSON.stringify(tool.params)), snap: this.snapLines([tool.id]), alt0: e.altKey, ctrl: e.ctrlKey };
        holdCursor(getComputedStyle(t).cursor);
      } else if (t.dataset && t.dataset.tool && e.button === 0) {
        const id = t.dataset.tool, tool = selTool();
        // Ctrl or Shift: add to the selection or take out of it
        if ((e.ctrlKey || e.shiftKey || e.metaKey) && this.o.toggle) { this.o.toggle(id); return; }
        const picked = this.o.selection ? this.o.selection() : [this.o.selected()];
        if (this.o.editable() && picked.length > 1 && picked.includes(id)) {
          // several regions move as one
          const items = side().tools.filter(x => picked.includes(x.id) && x.params && !x.params.locked && (x.params.roi || x.params.points))
            .map(x => ({ id: x.id, params: JSON.parse(JSON.stringify(x.params)) }));
          if (!items.length) return;
          this.drag = { kind: 'many', p0: p, items, dup: e.altKey };
          holdCursor('move');
        } else if (this.o.editable() && id === this.o.selected() && tool && !tool.params.locked && (tool.params.roi || tool.params.points)) {
          // Alt-drag: a copy; Shift: along one axis; Ctrl: no snapping
          this.drag = { kind: 'move', p0: p, params: JSON.parse(JSON.stringify(tool.params)), snap: this.snapLines([tool.id]), b0: boundsOf(tool.params), dup: e.altKey, ctrl: e.ctrlKey };
          holdCursor('move');
        } else { this.o.select(id); return; }
      } else if (this.o.editable() && this.o.band && e.button === 0) {
        // a drag on empty picture selects every region it touches
        this.drag = { kind: 'band', p0: p, p1: p, add: e.ctrlKey || e.shiftKey || e.metaKey };
      } else {
        this.drag = { kind: 'pan', x: e.clientX, y: e.clientY, tx: this.z.tx, ty: this.z.ty, moved: false };
        el.classList.add('panning');
      }
      el.setPointerCapture && el.setPointerCapture(e.pointerId);
      e.preventDefault();
    });
    this.mv = e => {
      if (!this.drag || this.dead) return;
      const d = this.drag;
      if (d.kind === 'pan') {
        this.z = { ...this.z, tx: d.tx + e.clientX - d.x, ty: d.ty + e.clientY - d.y };
        d.moved = d.moved || Math.abs(e.clientX - d.x) + Math.abs(e.clientY - d.y) > 3;
        if (d.moved) this.userMoved = true;
        this.apply(); return;
      }
      d.p1 = this.toImg(e); d.shift = e.shiftKey; d.alt = e.altKey || d.alt0; d.ctrl = e.ctrlKey;
      if (d.dup && (Math.abs(d.p1[0] - d.p0[0]) + Math.abs(d.p1[1] - d.p0[1])) * this.z.s > 3) {
        // as Vision Station: Alt and a drag leaves a copy of what is selected
        this.drag = null; holdCursor(null); duplicateTools(sels()); return;
      }
      if (d.kind === 'band') { this.drawBand(); return; }
      if (d.kind === 'many') { this.dragMany(d, false); return; }
      this.drawDraft();
    };
    this.up = async e => {
      if (!this.drag || this.dead) return;
      const d = this.drag; this.drag = null;
      el.classList.remove('panning');
      if (this.dims) { this.dims.remove(); this.dims = null; }
      holdCursor(null);
      if (d.kind === 'pan') {
        if (!d.moved && !d.keep && !(e.target.dataset && e.target.dataset.tool)) this.o.select(null);
        this.refresh(true); return;
      }
      if (d.kind === 'band') {
        const b = $('.band', this.svg); b && b.remove();
        const [x0, y0] = d.p0, [x1, y1] = d.p1;
        if ((Math.abs(x1 - x0) + Math.abs(y1 - y0)) * this.z.s < 4) { if (!d.add) this.o.select(null); return; }
        const L = Math.min(x0, x1), R = Math.max(x0, x1), T = Math.min(y0, y1), B = Math.max(y0, y1);
        const hit = [];
        for (const [id, reg] of this.regions || []) {
          const b = ptsBox(reg.el);
          if (b[2] >= L && b[0] <= R && b[3] >= T && b[1] <= B) hit.push(id);
        }
        this.o.band(hit, d.add);
        return;
      }
      if (d.kind === 'many') { if (d.p1) await this.dragMany(d, true); return; }
      if (d.fp != null) {
        const moved = d.p1 && (Math.abs(d.p1[0] - d.p0[0]) + Math.abs(d.p1[1] - d.p0[1])) * this.z.s >= 3;
        const res = moved && editResult(d);
        if (!moved) { if (d.fp !== 'a' && this.o.overlayClick) this.o.overlayClick(+d.fp); else this.draw(); return; }
        if (!res) { this.draw(); return; }
        await this.o.overlayEdit(d.fp, res); return;
      }
      if (!d.p1) return;
      const res = editResult(d);
      if (!res) { this.draw(); return; }
      if (d.kind === 'draw') { await finishDraw(res); return; }
      // the region stays where it was put while the server and the preview catch up
      const tool = selTool();
      if (tool) { for (const [k, v] of Object.entries(res)) { if (v == null) delete tool.params[k]; else tool.params[k] = v; } this.draw(); }
      await this.o.edit(res);
      if (tool && (tool.type === 'ocv' || tool.type === 'ocr')) await teachText([tool.id]);
    };
    window.addEventListener('pointermove', this.mv);
    el.addEventListener('contextmenu', e => {
      e.preventDefault();
      if (S.live) { menu(e, [['Save picture…', 'download', () => savePicture(), '', 'Ctrl+Shift+S']]); return; }
      if (S.draw || !this.size) return;
      const id = e.target.dataset && e.target.dataset.tool;
      const picked = this.o.selection ? this.o.selection() : [this.o.selected()];
      if (id && !picked.includes(id)) this.o.select(id);
      canvasMenu(e, this);
    });
    el.addEventListener('pointerover', e => { if (!this.drag) this.hover(e.target.dataset && e.target.dataset.tool || null); });
    el.addEventListener('pointerleave', () => this.hover(null));
    window.addEventListener('pointerup', this.up);
    // pictures dropped on the canvas are inspected once
    el.addEventListener('dragover', e => { e.preventDefault(); if (!$('.dropzone', el)) el.append(h('div', { class: 'dropzone' }, 'Drop a picture to inspect it')); });
    el.addEventListener('dragleave', e => { if (e.target === el) { const z = $('.dropzone', el); z && z.remove(); } });
    el.addEventListener('drop', guard(async e => {
      e.preventDefault(); const z = $('.dropzone', el); z && z.remove();
      const f = e.dataTransfer.files[0]; if (!f) return;
      this.busy(true);
      try { S.res = await api(`/api/picture/upload?side=${encodeURIComponent(side().id)}&name=${encodeURIComponent(f.name)}`, await f.arrayBuffer(), true); }
      finally { this.busy(false); }
      if (S.view === 'build') { S.buildPic = 'last'; renderMain(); } else onResults();
    }));
  }
  drawDraft() {
    const d = this.drag;
    let el = $('.draft', this.svg);
    const res = editResult(d, true);
    if (!res) return;
    const shape = d.kind === 'draw' ? (S.draw && S.draw.shape) : (d.params && d.params.shape);
    const pts = res.points && res.shape === 'polygon' ? res.points : outlineOf({ roi: res.roi, angle: res.angle || 0, shape: shape === 'ellipse' ? 'ellipse' : 'rect' });
    if (!el) { el = h('svg:polygon', { class: 'draft' }); this.svg.append(el); }
    el.setAttribute('points', pts.map(q => q[0] + ',' + q[1]).join(' '));
    // the lines it snapped to
    $$('.guide', this.svg).forEach(x => x.remove());
    if (d.guides && this.size) {
      for (const x of d.guides.x || []) this.svg.append(h('svg:line', { class: 'guide', x1: x, y1: 0, x2: x, y2: this.size[1] }));
      for (const y of d.guides.y || []) this.svg.append(h('svg:line', { class: 'guide', x1: 0, y1: y, x2: this.size[0], y2: y }));
    }
    // the size, beside the pointer
    if (res.roi) {
      if (!this.dims) { this.dims = h('svg:g', { class: 'dims' }, h('svg:rect', { rx: 6, height: 20, y: 6, x: 10 }), h('svg:text', { x: 18, y: 16 })); }
      if (!this.dims.isConnected) this.svg.append(this.dims);
      const text = d.kind === 'rot' || d.kind === 'turn' ? `${(+res.angle).toFixed(1)}°` : `${res.roi[2]} × ${res.roi[3]}`;
      $('text', this.dims).textContent = text;
      $('rect', this.dims).setAttribute('width', 16 + text.length * 7.4);
      this.dims._x = d.p1[0]; this.dims._y = d.p1[1];
      this.dims.setAttribute('transform', `translate(${d.p1[0]} ${d.p1[1]}) scale(${1 / this.z.s})`);
    }
  }
  drawBand() {
    const d = this.drag; let el = $('.band', this.svg);
    if (!el) { el = h('svg:rect', { class: 'band' }); this.svg.append(el); }
    const [x0, y0] = d.p0, [x1, y1] = d.p1;
    el.setAttribute('x', Math.min(x0, x1)); el.setAttribute('y', Math.min(y0, y1));
    el.setAttribute('width', Math.abs(x1 - x0)); el.setAttribute('height', Math.abs(y1 - y0));
  }
  /* What a dragged region snaps to: the picture's edges and middle, and every
     other region's edges and middle -- within 7 screen pixels, as Vision
     Station's canvas */
  snapLines(exclude) {
    if (!this.size) return null;
    const xs = [0, this.size[0], this.size[0] / 2], ys = [0, this.size[1], this.size[1] / 2];
    // every region as its check stores it: one added a moment ago is not
    // drawn yet until the preview catches up, but is snapped to all the same
    for (const t of this.o.tools()) {
      if (exclude.includes(t.id) || !t.params || t.params.enabled === false) continue;
      const b = boundsOf(t.params); if (!b) continue;
      xs.push(b.l, b.r, (b.l + b.r) / 2); ys.push(b.t, b.b, (b.t + b.b) / 2);
    }
    return { xs, ys, tol: 7 / this.z.s };
  }
  // several regions dragged together: drawn where they go, saved on release
  async dragMany(d, done) {
    let dx = Math.round(d.p1[0] - d.p0[0]), dy = Math.round(d.p1[1] - d.p0[1]);
    if (d.shift) { if (Math.abs(dx) >= Math.abs(dy)) dy = 0; else dx = 0; }
    const ups = {};
    for (const it of d.items) ups[it.id] = shifted(it.params, dx, dy);
    for (const t of side().tools) if (ups[t.id]) Object.assign(t.params, ups[t.id]);
    this.draw();
    if (done && (dx || dy)) await updateMany(ups);
  }
  // the view at 100 %, about the middle of what is shown
  actual() {
    if (!this.size) return;
    this.userMoved = true;
    const W = this.el.clientWidth, H = this.el.clientHeight, b = this.goal || this.z;
    const cx = (W / 2 - b.tx) / b.s, cy = (H / 2 - b.ty) / b.s;
    this.animateTo({ s: 1, tx: W / 2 - cx, ty: H / 2 - cy });
    clearTimeout(this._zt); this._zt = setTimeout(() => this.refresh(true), 260);
  }
  // the selected regions filling the view, with a margin
  zoomTo(ids) {
    const boxes = ids.map(id => this.regions && this.regions.get(id)).filter(Boolean).map(r => ptsBox(r.el));
    if (!boxes.length) { this.fit(); return; }
    const L = Math.min(...boxes.map(b => b[0])) - 40, T = Math.min(...boxes.map(b => b[1])) - 40;
    const R = Math.max(...boxes.map(b => b[2])) + 40, B = Math.max(...boxes.map(b => b[3])) + 40;
    const W = this.el.clientWidth, H = this.el.clientHeight, s = clamp(Math.min(W / (R - L), H / (B - T)), 0.02, 40);
    this.userMoved = true;
    this.animateTo({ s, tx: W / 2 - (L + R) / 2 * s, ty: H / 2 - (T + B) / 2 * s });
    clearTimeout(this._zt); this._zt = setTimeout(() => this.refresh(true), 260);
  }
  drawPoly() {
    let el = $('.draft', this.svg);
    if (!el) { el = h('svg:polyline', { class: 'draft' }); this.svg.append(el); }
    el.setAttribute('points', S.draw.pts.map(q => q[0] + ',' + q[1]).join(' '));
  }
}

// the resize arrow that points along a handle's direction (degrees, y down)
function resizeCursor(deg) {
  const d = ((Math.round(deg / 45) % 4) + 4) % 4;
  return ['ew-resize', 'nwse-resize', 'ns-resize', 'nesw-resize'][d];
}
const ROTATE_CURSOR = `url("data:image/svg+xml,${encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path d="M19 12a7 7 0 1 1-2.05-4.95M19 4v4h-4" fill="none" stroke="#fff" stroke-width="4.5" stroke-linecap="round" stroke-linejoin="round"/><path d="M19 12a7 7 0 1 1-2.05-4.95M19 4v4h-4" fill="none" stroke="#111" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>')}") 12 12, alias`;
// while a region is dragged, its cursor stays whatever the pointer crosses
function holdCursor(c) {
  let st = document.getElementById('cursor-hold');
  if (!c) { if (st) st.textContent = ''; return; }
  if (!st) { st = h('style', { id: 'cursor-hold' }); document.head.append(st); }
  st.textContent = `* { cursor: ${c} !important; }`;
}

// region geometry: centre, corners turned by the angle (degrees, y down)
const angleOf = p => { const a = +(p.angle || 0); return a > 180 ? a - 360 : a; };
function corners(roi, a) {
  const [x, y, w, hh] = roi, cx = x + w / 2, cy = y + hh / 2, r = a * Math.PI / 180, c = Math.cos(r), s = Math.sin(r);
  return [[-w / 2, -hh / 2], [w / 2, -hh / 2], [w / 2, hh / 2], [-w / 2, hh / 2]].map(([dx, dy]) => [cx + dx * c - dy * s, cy + dx * s + dy * c]);
}
function rotHandle(roi, a, z) {
  const [x, y, w, hh] = roi, cx = x + w / 2, cy = y + hh / 2, r = a * Math.PI / 180, d = hh / 2 + 28 / z;
  return [cx + d * Math.sin(r), cy - d * Math.cos(r)];
}
function outlineOf(p) {
  if (!p) return null;
  if (p.shape === 'polygon' && p.points && p.points.length > 2) return p.points;
  if (!p.roi) return null;
  if (p.shape === 'ellipse') {
    const [x, y, w, hh] = p.roi, cx = x + w / 2, cy = y + hh / 2, r = angleOf(p) * Math.PI / 180;
    return Array.from({ length: 40 }, (_, i) => { const t = i / 40 * Math.PI * 2, u = w / 2 * Math.cos(t), v = hh / 2 * Math.sin(t); return [cx + u * Math.cos(r) - v * Math.sin(r), cy + u * Math.sin(r) + v * Math.cos(r)]; });
  }
  return corners(p.roi, angleOf(p));
}
// what a drag on the canvas makes: params for a new or edited region
function editResult(d, live) {
  const R = n => Math.round(n), [x0, y0] = d.p0, [x1, y1] = d.p1 || d.p0;
  if (d.kind === 'draw') {
    const roi = [R(Math.min(x0, x1)), R(Math.min(y0, y1)), R(Math.abs(x1 - x0)), R(Math.abs(y1 - y0))];
    return roi[2] >= 4 && roi[3] >= 4 ? { roi, angle: 0 } : null;
  }
  const p = d.params, a = angleOf(p);
  let dx = x1 - x0, dy = y1 - y0;
  d.guides = null;
  if (d.kind === 'move') {
    if (d.shift) { if (Math.abs(dx) >= Math.abs(dy)) dy = 0; else dx = 0; }
    if (d.snap && d.b0 && !d.ctrl) {
      const b = d.b0, gx = [], gy = [];
      const sx = !(d.shift && dx === 0) && snapNear([b.l + dx, b.r + dx, (b.l + b.r) / 2 + dx], d.snap.xs, d.snap.tol);
      const sy = !(d.shift && dy === 0) && snapNear([b.t + dy, b.b + dy, (b.t + b.b) / 2 + dy], d.snap.ys, d.snap.tol);
      if (sx) { dx += sx[0]; gx.push(sx[1]); }
      if (sy) { dy += sy[0]; gy.push(sy[1]); }
      d.guides = { x: gx, y: gy };
    }
    const out = {};
    if (p.roi) out.roi = [R(p.roi[0] + dx), R(p.roi[1] + dy), p.roi[2], p.roi[3]];
    if (p.points && p.points.length) out.points = p.points.map(q => [R(q[0] + dx), R(q[1] + dy)]);
    if (p.shape === 'polygon') out.shape = 'polygon';
    out.angle = a;
    return out;
  }
  if (d.kind[0] === 'b' && d.kind.length === 3) {
    // the whole polygon, scaled about the opposite corner of its box
    const xs = p.points.map(q => q[0]), ys = p.points.map(q => q[1]);
    const L = Math.min(...xs), T = Math.min(...ys), Rr = Math.max(...xs), B = Math.max(...ys);
    const k = d.kind.slice(1), ax = k.includes('w') ? Rr : L, ay = k.includes('n') ? B : T, gx = k.includes('w') ? L : Rr, gy = k.includes('n') ? T : B;
    let sx = Math.abs(gx - ax) > 1e-6 ? (x1 - ax) / (gx - ax) : 1, sy = Math.abs(gy - ay) > 1e-6 ? (y1 - ay) / (gy - ay) : 1;
    sx = Math.max(0.05, Math.abs(sx)); sy = Math.max(0.05, Math.abs(sy));
    if (d.shift) sx = sy = Math.max(sx, sy);
    return polyParams(p.points.map(q => [R(ax + (q[0] - ax) * sx), R(ay + (q[1] - ay) * sy)]));
  }
  if (d.kind[0] === 'v') {
    const i = +d.kind.slice(1), pts = p.points.map(q => q.slice());
    pts[i] = [R(x1), R(y1)];
    const xs = pts.map(q => q[0]), ys = pts.map(q => q[1]);
    return { points: pts, shape: 'polygon', roi: [Math.min(...xs), Math.min(...ys), Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)] };
  }
  const [x, y, w, hh] = p.roi, cx = x + w / 2, cy = y + hh / 2;
  if (d.kind === 'turn') {
    let ang = a + (Math.atan2(y1 - cy, x1 - cx) - Math.atan2(y0 - cy, x0 - cx)) * 180 / Math.PI;
    ang = d.shift ? Math.round(ang / 15) * 15 : Math.round(ang * 10) / 10;
    ang = ((ang % 360) + 540) % 360 - 180;
    return { roi: p.roi, angle: ang };
  }
  if (d.kind === 'rot') {
    let ang = Math.atan2(y1 - cy, x1 - cx) * 180 / Math.PI + 90;
    ang = d.shift ? Math.round(ang / 15) * 15 : Math.round(ang * 10) / 10;
    if (ang > 180) ang -= 360;
    return { roi: p.roi, angle: ang };
  }
  // a corner: the opposite corner stays, in the region's own turned frame
  // a corner or a side: what is opposite stays, in the region's own turned frame
  const sign = { nw: [-1, -1], n: [0, -1], ne: [1, -1], e: [1, 0], se: [1, 1], s: [0, 1], sw: [-1, 1], w: [-1, 0] }[d.kind];
  if (!sign) return null;
  const r = a * Math.PI / 180, c = Math.cos(r), s = Math.sin(r);
  const lx = (x1 - cx) * c + (y1 - cy) * s, ly = -(x1 - cx) * s + (y1 - cy) * c;
  const corner = sign[0] !== 0 && sign[1] !== 0;
  let nw, nh, mx = 0, my = 0;
  if (p.shape === 'ellipse' || d.alt) {
    // about the centre: an ellipse always (it is defined by its centre), a
    // rectangle with Alt, as in every drawing application
    let hw = sign[0] ? Math.max(1, Math.abs(lx)) : w / 2, hh2 = sign[1] ? Math.max(1, Math.abs(ly)) : hh / 2;
    if (d.shift) {
      if (p.shape === 'ellipse') hw = hh2 = Math.max(sign[1] && !sign[0] ? hh2 : sign[0] && !sign[1] ? hw : hh2, 1);
      else if (corner) { const k = Math.max(hw / Math.max(1, w / 2), hh2 / Math.max(1, hh / 2)); hw = k * w / 2; hh2 = k * hh / 2; }
    }
    nw = 2 * hw; nh = 2 * hh2;
  } else {
    // from the corner or side opposite, which stays put
    const fx = -sign[0] * w / 2, fy = -sign[1] * hh / 2;
    let ux = lx - fx, uy = ly - fy;
    if (d.shift && corner) {
      const k = Math.max(Math.abs(ux) / Math.max(1, w), Math.abs(uy) / Math.max(1, hh));
      ux = Math.sign(ux || sign[0]) * k * w; uy = Math.sign(uy || sign[1]) * k * hh;
    }
    // an upright region's moving sides snap, as when it is moved; Ctrl: not
    if (d.snap && Math.abs(a) < 1e-6 && !d.ctrl && !(d.shift && corner)) {
      const gx = [], gy = [];
      if (sign[0]) { const sn = snapNear([cx + fx + ux], d.snap.xs, d.snap.tol); if (sn) { ux += sn[0]; gx.push(sn[1]); } }
      if (sign[1]) { const sn = snapNear([cy + fy + uy], d.snap.ys, d.snap.tol); if (sn) { uy += sn[0]; gy.push(sn[1]); } }
      d.guides = { x: gx, y: gy };
    }
    nw = sign[0] ? Math.abs(ux) : w; nh = sign[1] ? Math.abs(uy) : hh;
    mx = sign[0] ? fx + ux / 2 : 0; my = sign[1] ? fy + uy / 2 : 0;
  }
  if (nw < 4 || nh < 4) return null;
  const ncx = cx + mx * c - my * s, ncy = cy + mx * s + my * c;
  return { roi: [R(ncx - nw / 2), R(ncy - nh / 2), R(nw), R(nh)], angle: a };
}

// the nearest snapping line to any of vals within tol: [shift, line] or null
function snapNear(vals, lines, tol) {
  let best = null;
  for (const v of vals) for (const l of lines) { const dd = Math.abs(l - v); if (dd < tol && (!best || dd < best[2])) best = [l - v, l, dd]; }
  return best && [best[0], best[1]];
}
// a polygon's params from its corners
function polyParams(pts) {
  const xs = pts.map(q => q[0]), ys = pts.map(q => q[1]);
  return { points: pts, shape: 'polygon', roi: [Math.min(...xs), Math.min(...ys), Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)] };
}
// [left, top, right, bottom] of an SVG polygon's points
function ptsBox(el) {
  const pts = (el.getAttribute('points') || '').trim().split(/\s+/).map(q => q.split(',').map(Number));
  const xs = pts.map(q => q[0]), ys = pts.map(q => q[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}
// a region's params moved by dx, dy picture pixels
function shifted(p, dx, dy) {
  const out = {};
  if (p.roi) out.roi = [Math.round(p.roi[0] + dx), Math.round(p.roi[1] + dy), p.roi[2], p.roi[3]];
  if (p.points && p.points.length) out.points = p.points.map(q => [Math.round(q[0] + dx), Math.round(q[1] + dy)]);
  return out;
}
// the axis-aligned box round a region as it is seen (turned)
function boundsOf(p) {
  const pts = outlineOf(p); if (!pts) return null;
  const xs = pts.map(q => q[0]), ys = pts.map(q => q[1]);
  return { l: Math.min(...xs), t: Math.min(...ys), r: Math.max(...xs), b: Math.max(...ys) };
}

/* Right-click on the picture: what Vision Station's canvas offers. On a
   region, its arrangement; on empty picture, the view. */
function canvasMenu(e, v) {
  const build = S.view === 'build', L = build && !locked();
  const ids = build ? sels() : (S.sel ? [S.sel] : []), canEdit = L && v.o.editable();
  const ed = canEdit ? editableSel() : [];
  const items = [];
  if (ids.length) {
    if (L) {
      items.push(['Duplicate', 'copy', () => duplicateTools(ids), '', 'Ctrl+D']);
      items.push(['Delete', 'trash', () => deleteTools(ids), 'danger', 'Del']);
      items.push(null);
      items.push(['Rotate 90° clockwise', 'rotcw', () => rotateSel(90), '', ']', !ed.length]);
      items.push(['Rotate 90° anticlockwise', 'rotccw', () => rotateSel(-90), '', '[', !ed.length]);
      items.push(['Straighten (0°)', 'rot0', () => rotateSel(0, true), '', '', !ed.length]);
      const many = ed.length > 1;
      items.push(['Align and distribute', 'alignl', [
        ['Align left', 'alignl', () => arrange('align_left')],
        ['Align centres horizontally', 'alignh', () => arrange('align_hcenter')],
        ['Align right', 'alignr', () => arrange('align_right')], null,
        ['Align top', 'alignt', () => arrange('align_top')],
        ['Align centres vertically', 'alignv', () => arrange('align_vcenter')],
        ['Align bottom', 'alignb', () => arrange('align_bottom')], null,
        ['Distribute horizontally', 'disth', () => arrange('distribute_h'), '', '', ed.length < 3],
        ['Distribute vertically', 'distv', () => arrange('distribute_v'), '', '', ed.length < 3], null,
        ['Make the same width', 'samew', () => arrange('match_width'), '', '', !many],
        ['Make the same height', 'sameh', () => arrange('match_height'), '', '', !many],
      ], '', '', !ed.length]);
      const regs = side().tools.filter(t => ids.includes(t.id) && t.params && t.params.roi);
      if (regs.length && canEdit) {
        const all = regs.every(t => t.params.locked);
        items.push([all ? 'Unlock' : 'Lock position', all ? 'unlock' : 'lock', () => updateMany(Object.fromEntries(regs.map(t => [t.id, { locked: all ? null : true }])))]);
      }
      items.push(null);
    }
    items.push(['Zoom to selection', 'zoomsel', () => v.zoomTo(ids), '', 'Ctrl+2']);
  } else if (build) {
    items.push(['Select all regions', 'selall', selectAll, '', 'Ctrl+A']);
    if (!locked()) items.push(['Find parts and text…', 'search', () => openFinder()]);
  }
  items.push(['Fit to window', 'fit', () => v.fit(), '', 'Ctrl+0']);
  items.push(['Actual size', 'actual', () => v.actual(), '', 'Ctrl+1']);
  items.push(null);
  items.push(['Save picture…', 'download', () => savePicture(), '', 'Ctrl+Shift+S']);
  menu(e, items);
}
// the picture on screen at full resolution: the live camera, the last picture or the reference
const savePicture = guard(async () => {
  const what = S.live ? 'live' : (S.view === 'build' ? (S.buildPic === 'reference' ? 'reference' : 'last') : (S.runPic === 'reference' ? 'reference' : 'last'));
  const r = await api('/api/save_picture', { what, side: side().id, camera: side().source.camera || 0 });
  if (r.path) toast('Saved ' + r.path);
});
// the selected regions that can be turned and aligned: boxes, ellipses, not locked
function editableSel() {
  const ids = sels();
  return side().tools.filter(t => ids.includes(t.id) && t.params && t.params.roi && t.params.shape !== 'polygon' && !t.params.locked);
}
function selectAll() {
  if (S.view !== 'build') return;
  const ids = side().tools.filter(t => t.params && t.params.enabled !== false && outlineOf(t.params)).map(t => t.id);
  if (!ids.length) return;
  S.picks = new Set(ids); S.sel = ids.includes(S.sel) ? S.sel : ids[0];
  renderChecks(); renderInspector(); S.viewer && S.viewer.draw();
}
function toggleSel(id) {
  S.picks = new Set(sels());
  if (S.picks.has(id)) { S.picks.delete(id); if (S.sel === id) S.sel = [...S.picks].pop() || null; }
  else { S.picks.add(id); S.sel = id; }
  if (S.view === 'build') { renderChecks(); renderInspector(); }
  S.viewer && S.viewer.draw();
}
function bandSelect(ids, add) {
  const set = new Set(add ? sels() : []);
  ids.forEach(id => set.add(id));
  S.picks = set;
  S.sel = ids.length ? ids[ids.length - 1] : (set.has(S.sel) ? S.sel : [...set].pop() || null);
  renderChecks(); renderInspector(); S.viewer && S.viewer.draw();
}
function rotateSel(deg, absolute) {
  const ed = editableSel(); if (!ed.length) return;
  const ups = {};
  for (const t of ed) {
    let a = absolute ? deg : angleOf(t.params) + deg;
    a = ((a + 180) % 360 + 360) % 360 - 180; if (a === -180) a = 180;
    ups[t.id] = { angle: Math.round(a * 10) / 10 };
  }
  return updateMany(ups);
}
/* Align, distribute or match the size of the selected regions; one region
   is aligned to the picture, as Vision Station does. */
function arrange(op) {
  const ed = editableSel(); if (!ed.length) return;
  const bs = new Map(ed.map(t => [t, boundsOf(t.params)]));
  let ref;
  if (ed.length === 1) { const sz = side().reference || [0, 0]; ref = { l: 0, t: 0, r: sz[0], b: sz[1] }; }
  else ref = [...bs.values()].reduce((a, b) => ({ l: Math.min(a.l, b.l), t: Math.min(a.t, b.t), r: Math.max(a.r, b.r), b: Math.max(a.b, b.b) }));
  const ups = {};
  if (op.startsWith('align_')) {
    for (const [t, b] of bs) {
      const dx = { align_left: ref.l - b.l, align_right: ref.r - b.r, align_hcenter: (ref.l + ref.r) / 2 - (b.l + b.r) / 2 }[op] || 0;
      const dy = { align_top: ref.t - b.t, align_bottom: ref.b - b.b, align_vcenter: (ref.t + ref.b) / 2 - (b.t + b.b) / 2 }[op] || 0;
      if (Math.round(dx) || Math.round(dy)) ups[t.id] = shifted(t.params, dx, dy);
    }
  } else if ((op === 'distribute_h' || op === 'distribute_v') && ed.length >= 3) {
    const hz = op === 'distribute_h', key = t => { const b = bs.get(t); return hz ? (b.l + b.r) / 2 : (b.t + b.b) / 2; };
    const order = [...ed].sort((a, b) => key(a) - key(b));
    const a = key(order[0]), step = (key(order[order.length - 1]) - a) / (order.length - 1);
    order.forEach((t, i) => { const d = a + i * step - key(t); if (Math.round(d)) ups[t.id] = shifted(t.params, hz ? d : 0, hz ? 0 : d); });
  } else if ((op === 'match_width' || op === 'match_height') && ed.length >= 2) {
    const w = op === 'match_width', size = Math.max(...ed.map(t => t.params.roi[w ? 2 : 3]));
    for (const t of ed) {
      const [x, y, rw, rh] = t.params.roi, cx = x + rw / 2, cy = y + rh / 2;
      const nw = w ? size : rw, nh = w ? rh : size;
      ups[t.id] = { roi: [Math.round(cx - nw / 2), Math.round(cy - nh / 2), nw, nh] };
    }
  }
  if (Object.keys(ups).length) return updateMany(ups);
}
// arrow keys move the selection a pixel (Shift: ten); saved once the keys rest
let nudgeAcc = null, nudgeTimer = null;
function nudge(dx, dy) {
  const ids = sels();
  const ts = side().tools.filter(t => ids.includes(t.id) && t.params && !t.params.locked && (t.params.roi || t.params.points));
  if (!ts.length) return;
  nudgeAcc = nudgeAcc || {};
  for (const t of ts) {
    Object.assign(t.params, shifted(t.params, dx, dy));
    nudgeAcc[t.id] = t.params.points ? { roi: t.params.roi, points: t.params.points } : { roi: t.params.roi };
  }
  S.viewer && S.viewer.draw();
  clearTimeout(nudgeTimer);
  nudgeTimer = setTimeout(() => { const u = nudgeAcc; nudgeAcc = null; updateMany(u); }, 300);
}

// --------------------------------------------------------------------- run
function runView() {
  const multi = open().sides.length > 1;
  const v = h('div', { class: 'view run-view', style: { gridTemplateColumns: multi ? '168px 1fr 360px' : '1fr 360px' } });
  if (multi) v.append(h('div', { class: 'strip', id: 'strip' }));
  const stagecard = h('div', { class: 'card stagecard' });
  v.append(stagecard, h('div', { class: 'card panel', id: 'runpanel' }));
  S.viewer = new Viewer({
    size: () => { if (S.live) return S.liveSize; const r = S.runPic === 'last' ? lastResult() : null; return r && r.image_size && r.image_size[0] ? r.image_size : side().reference; },
    key: () => side().id + ':' + S.runPic,
    url: w => `/img/${S.runPic === 'reference' ? 'reference' : 'last'}/${encodeURIComponent(side().id)}.jpg?w=${w}&v=${(sideRes() || {}).img_seq || 0}-${side().ref_version}`,
    result: () => S.runPic === 'last' ? lastResult() : null,
    tools: () => side().tools, selected: () => S.sel, editable: () => false, names: () => false,
    select: id => { S.sel = id; S.viewer.draw(); updateRunPanel(false); },
    empty: () => h('div', { class: 'empty' }, h('div', {}, h('b', {}, 'No picture yet'), 'Press Run, or drop a picture here.', h('br'),
      h('button', { class: 'btn primary', onclick: runOnce }, icon('play'), 'Run'))),
    onZoom: s => { const b = $('#zoom'); if (b) b.textContent = Math.round(s * 100) + '%'; },
  });
  stagecard.append(S.viewer.el);
  const s = side();
  const tl = h('div', { class: 'float tl' }, seg([['last', 'Last picture'], ['reference', 'Reference']], S.runPic, p => { S.runPic = p; S.viewer.refresh(); }));
  if (s.source.mode === 'camera') tl.append(h('button', { class: 'btn sm' + (S.live ? ' on' : ''), title: 'Live picture from the camera, for aiming', onclick: e => toggleLive(e.currentTarget) }, icon('live', 'sm'), 'Live'));
  if (s.source.mode === 'camera') tl.append(h('button', { class: 'btn sm', id: 'livepause', hidden: !S.live, title: 'Hold the live picture still, to look at it closely',
    onclick: e => { S.livePaused = !S.livePaused; e.currentTarget.classList.toggle('on', S.livePaused); } }, icon('stop', 'sm'), 'Pause'));
  tl.append(h('button', { class: 'btn sm', title: 'Save the picture on screen at full resolution (Ctrl+Shift+S)', onclick: savePicture }, icon('download', 'sm'), 'Save image…'));
  stagecard.append(tl,
    h('div', { class: 'float br' }, h('button', { class: 'btn icon', title: 'Zoom out', onclick: () => S.viewer.zoomBy(1 / 1.4) }, icon('down', 'sm')),
      h('span', { class: 'zoombadge', id: 'zoom' }, '100%'), h('button', { class: 'btn icon', title: 'Zoom in', onclick: () => S.viewer.zoomBy(1.4) }, icon('up', 'sm')),
      h('button', { class: 'btn icon', title: 'Fit (F)', onclick: () => S.viewer.fit() }, icon('fit', 'sm'))),
    h('div', { class: 'float bl info', id: 'vinfo' }));
  S.animRun = true;
  requestAnimationFrame(() => { S.viewer.refresh(); renderStrip(); updateRunPanel(false); });
  return v;
}
async function toggleLive(btn) {
  S.live = !S.live; btn.classList.toggle('on', S.live);
  S.livePaused = false;
  const pb = $('#livepause'); if (pb) { pb.hidden = !S.live; pb.classList.remove('on'); }
  const cam = side().source.camera || 0;
  if (!S.live) { S.viewer.img.dataset.url = ''; S.viewer.refresh(); return; }
  S.viewer.svg.replaceChildren();
  // the next frame is asked for while this one is decoded: the camera's pace, not the round trip's
  const fetchFrame = () => fetch(`/img/live/${cam}.jpg?w=1600&t=${Date.now()}`).then(async r => { if (!r.ok) throw new Error(await r.text()); return r.blob(); });
  let next = fetchFrame();
  while (S.live && S.viewer && !S.viewer.dead) {
    if (S.livePaused) {
      // the frame on screen stays; none is asked for until Pause is pressed again
      await next.catch(() => { });
      while (S.livePaused && S.live && S.viewer && !S.viewer.dead) await sleep(120);
      next = fetchFrame();
      continue;
    }
    try {
      const b = await next;
      next = fetchFrame();
      const url = URL.createObjectURL(b);
      const im = new Image(); im.src = url; await im.decode();
      if (!S.live || !S.viewer) { URL.revokeObjectURL(url); break; }
      const old = S.viewer.img.src; S.viewer.img.src = url; if (old.startsWith('blob:')) URL.revokeObjectURL(old);
      if (!S.liveSize) { S.liveSize = side().reference || [im.naturalWidth * 2, im.naturalHeight * 2]; S.viewer.size = null; S.viewer.refresh(true); }
    } catch (e) { toast(e.message, true); S.live = false; btn.classList.remove('on'); break; }
  }
  next.catch(() => {});
  S.liveSize = null; S.livePaused = false;
  const pb2 = $('#livepause'); if (pb2) { pb2.hidden = true; pb2.classList.remove('on'); }
}
function renderStrip() {
  const el = $('#strip'); if (!el) return;
  el.replaceChildren();
  open().sides.forEach((s, i) => {
    const sr = sideRes(i), r = sr && sr.result;
    const url = `/img/last/${encodeURIComponent(s.id)}.jpg?w=320&v=${sr ? sr.img_seq : 0}-${s.ref_version}`;
    el.append(h('div', { class: 'strip-item' + (i === S.side ? ' on' : ''), style: { animationDelay: (i * 40) + 'ms' }, onclick: () => { if (i === S.side) return; S.side = i; S.sel = null; renderMain(); },
      oncontextmenu: e => { e.preventDefault(); sideMenu(e, i); } },
      h('div', { class: 'thumb', style: { backgroundImage: s.reference || (sr && sr.img_seq) ? `url("${url}")` : 'none' } }),
      r ? h('span', { class: 'chip badge ' + r.verdict.toLowerCase() }, r.verdict === 'PASS' ? 'OK' : 'NG') : null,
      h('div', { class: 'meta' }, h('span', { class: 'dot ' + (r ? r.verdict : '') }), s.name)));
  });
}
function updateRunPanel(fresh) {
  const panel = $('#runpanel'); if (!panel) return;
  const keep = panel.querySelector('.panel-body'); const sc = keep ? keep.scrollTop : 0;
  panel.replaceChildren();
  const s = side(), r = lastResult(), v = r ? r.verdict : '';
  const body = h('div', { class: 'panel-body' + (S.animRun ? ' anim' : ''), style: { paddingTop: '12px' } });
  S.animRun = false;
  const word = v === 'PASS' ? 'OK' : v === 'FAIL' ? 'NG' : v === 'ERROR' ? 'ERROR' : v === 'WARN' ? 'CHECK' : 'Ready';
  const bad = r ? r.tools.filter(t => t.status === 'fail' || t.status === 'error') : [];
  body.append(h('div', { class: 'hero ' + (v || 'idle') },
    h('div', { style: { fontSize: '12px', fontWeight: 650, opacity: .85, letterSpacing: '.06em', textTransform: 'uppercase' } }, s.name),
    h('div', { class: 'word' + (fresh ? ' pop' : '') }, word),
    h('div', { class: 'sub' }, r ? (r.aborted ? r.aborted : bad.length ? `${bad.length} check${bad.length === 1 ? '' : 's'} failed` : `All ${r.tools.length} checks passed`) : 'Press Run to inspect'),
    r ? h('div', { class: 'bigms' }, `${Math.round(r.total_ms)} ms · ${r.source}`) : null));
  const c = S.res ? S.res.counts : { total: 0, pass: 0, fail: 0 };
  const yld = c.total ? c.pass / c.total : 0, C = 2 * Math.PI * 20;
  const fg = h('svg:circle', { class: 'fg', cx: 27, cy: 27, r: 20, 'stroke-dasharray': `${(S.lastYield || 0) * C} ${C}` });
  body.append(h('div', { class: 'row', style: { marginTop: '12px', gap: '12px', alignItems: 'stretch' } },
    h('div', { class: 'kpi grow' }, h('b', { class: 'k1', 'data-v': S.kp1 || 0 }), h('span', {}, 'Good')),
    h('div', { class: 'kpi grow' }, h('b', { class: 'k2', 'data-v': S.kp2 || 0 }), h('span', {}, 'Rejected')),
    h('div', { class: 'kpi', style: { display: 'flex', alignItems: 'center', gap: '10px' } }, h('svg:svg', { class: 'donut', viewBox: '0 0 54 54' }, h('svg:circle', { class: 'bg', cx: 27, cy: 27, r: 20 }), fg),
      h('div', {}, h('b', { style: { fontSize: '17px' } }, c.total ? (100 * yld).toFixed(1) + '%' : '–'), h('span', {}, 'Yield')))));
  requestAnimationFrame(() => { fg.setAttribute('stroke-dasharray', `${yld * C} ${C}`); tween($('.k1', body), c.pass); tween($('.k2', body), c.fail); });
  S.lastYield = yld; S.kp1 = c.pass; S.kp2 = c.fail;
  if (S.st.barcode_rule) {
    const okCode = S.res && S.res.barcode;
    const inp = h('input', { class: 'input', placeholder: 'Scan or type the code', style: { height: '32px' }, onkeydown: guard(async e => {
      if (e.key !== 'Enter') return; const r = await api('/api/barcode/scan', { code: e.target.value }); e.target.value = ''; if (!r.ok) toast(r.why, true);
    }) });
    body.append(h('div', { class: 'barcode-card ' + (okCode ? 'ok' : S.res && /not good/.test(S.res.barcode_note) ? 'bad' : '') }, icon('barcode', 'lg'),
      h('div', { class: 'grow' }, h('b', {}, okCode || 'Waiting for a barcode'), h('div', { class: 'muted', style: { fontSize: '12px' } }, (S.res && S.res.barcode_note) || S.st.barcode_rule)),
      S.st.barcode_keyboard ? null : null), h('div', { style: { marginTop: '8px' } }, inp));
  }
  if (bad.length) {
    body.append(h('div', { class: 'section' }, `Failed (${bad.length})`));
    bad.forEach((t, i) => body.append(checkRow(t.tool_id, t.type, t, () => selectRun(t.tool_id), i)));
  }
  if (r) {
    body.append(h('div', { class: 'section' }, h('span', { style: { cursor: 'pointer' }, onclick: () => { S.showAllChecks = !S.showAllChecks; updateRunPanel(false); } },
      S.showAllChecks ? 'All checks ▾' : `All checks (${r.tools.length}) ▸`)));
    if (S.showAllChecks) r.tools.filter(t => !bad.includes(t)).forEach((t, i) => body.append(checkRow(t.tool_id, t.type, t, () => selectRun(t.tool_id), Math.min(i, 12))));
    if (r.not_ported) body.append(h('div', { class: 'note warn' }, `${r.not_ported} check(s) could not run here.`));
  }
  if (S.sel && r) { const t = r.tools.find(x => x.tool_id === S.sel); if (t) body.append(h('div', { class: 'section' }, 'Selected'), resultBox(t)); }
  panel.append(body);
  body.scrollTop = sc;
  const info = $('#vinfo'); if (info) info.textContent = r ? `${r.image_size[0]} × ${r.image_size[1]}  ·  ${r.source}` : (s.reference ? `${s.reference[0]} × ${s.reference[1]}  ·  reference` : 'no picture');
}
function selectRun(id) { S.sel = id; if (S.viewer) S.viewer.draw(); updateRunPanel(false); }
function checkRow(id, type, r, onclick, i = 0, tool) {
  const st = r ? r.status : '';
  const val = r ? (r.caption && r.caption[0]) || (r.status === 'skip' ? 'not run' : r.status === 'pass' ? 'OK' : r.status === 'fail' ? 'NG' : r.status) : '';
  const off = tool && tool.params && tool.params.enabled === false;
  return h('div', { class: 'check' + (id === S.sel ? ' on' : (S.view === 'build' && sels().includes(id) ? ' picked' : '')) + (off ? ' off' : ''), 'data-id': id, style: { animationDelay: (i * 22) + 'ms' }, onclick },
    h('div', { class: 'ti' }, icon(toolIcon(type), 'sm'), h('span', { class: 'dot ' + st })),
    h('div', { style: { minWidth: 0 } }, h('div', { class: 'name' }, id), h('div', { class: 'sub' }, r && r.message ? r.message : label(type))),
    h('div', { class: 'val', title: val }, val));
}
function resultBox(r) {
  const kv = h('div', { class: 'kv' });
  for (const [k, v] of Object.entries(r.values || {})) { if (k === 'fixture' || k === 'points') continue; kv.append(h('span', {}, k.replace(/_/g, ' ')), h('span', {}, fmt(v))); }
  kv.append(h('span', {}, 'time'), h('span', {}, r.ms + ' ms'));
  return h('div', { class: 'note' }, h('div', { class: 'row', style: { marginBottom: '8px' } }, h('span', { class: 'chip ' + r.status }, r.status === 'pass' ? 'OK' : r.status === 'fail' ? 'NG' : r.status), h('b', {}, r.tool_id)),
    r.message ? h('div', { style: { marginBottom: '8px' } }, r.message) : null, kv);
}

// ------------------------------------------------------------------- build
const selTool = () => side() && side().tools.find(t => t.id === S.sel);
const buildResult = () => S.buildPic === 'reference' ? S.preview : lastResult();
function resultFor(id) { const r = buildResult(); return r && r.tools.find(t => t.tool_id === id); }
function buildView() {
  const v = h('div', { class: 'view build-view' });
  v.append(h('div', { class: 'card panel', id: 'checks' }));
  const stagecard = h('div', { class: 'card stagecard' });
  v.append(stagecard, h('div', { class: 'card panel', id: 'inspector' }));
  S.viewer = new Viewer({
    size: () => { const r = S.buildPic === 'last' ? lastResult() : null; return r && r.image_size && r.image_size[0] ? r.image_size : side().reference; },
    key: () => side().id + ':' + S.buildPic,
    url: w => `/img/${S.buildPic}/${encodeURIComponent(side().id)}.jpg?w=${w}&v=${(sideRes() || {}).img_seq || 0}-${side().ref_version}`,
    result: buildResult, tools: () => side().tools, selected: () => S.sel, names: () => true,
    selection: sels, toggle: id => toggleSel(id), band: (ids, add) => bandSelect(ids, add),
    overlay: () => S.find ? finderOverlay() : null, overlayClick: i => finderToggle(i),
    overlayItem: k => finderItem(k), overlayEdit: (k, res) => finderEdited(k, res),
    editable: () => S.buildPic === 'reference' && !locked(),
    select: id => select(id),
    edit: async params => { await updateTool(S.sel, params); },
    empty: () => h('div', { class: 'empty' }, h('div', {}, h('b', {}, 'No reference picture'), 'Checks are drawn on a picture of a good part.', h('br'),
      h('div', { class: 'row', style: { justifyContent: 'center', marginTop: '14px' } },
        h('label', { class: 'btn primary' }, icon('upload'), 'Upload a picture', h('input', { type: 'file', accept: 'image/*', hidden: true, onchange: uploadReference })),
        h('button', { class: 'btn', onclick: sourceDialog }, icon('camera'), 'Picture source')))),
    onZoom: s => { const b = $('#zoom'); if (b) b.textContent = Math.round(s * 100) + '%'; },
  });
  stagecard.append(S.viewer.el);
  stagecard.append(
    h('div', { class: 'float tl' }, seg([['reference', 'Reference'], ['last', 'Last picture']], S.buildPic, p => { S.buildPic = p; preview(); S.viewer.refresh(); })),
    h('div', { class: 'float left' },
      ...[['rect', 'box', 'Box'], ['ellipse', 'ellipse', 'Ellipse'], ['polygon', 'poly', 'Polygon']].map(([k, ic, t]) =>
        h('button', { class: 'btn icon', 'data-shape': k, title: `Draw a ${t.toLowerCase()}, then choose its check`, onclick: () => {
          S.shape = k;
          if (S.draw && S.draw.shape === k && !S.draw.type) { cancelDraw(); return; }
          startDraw({ shape: k, label: 'the area to check' });
        } }, icon(ic))),
      h('span', { style: { height: '1px', width: '20px', background: 'var(--hair-2)', margin: '3px 0' } }),
      h('button', { class: 'btn icon', title: 'Add a check (A)', onclick: addSheet }, icon('plus')),
      h('button', { class: 'btn icon', title: 'Find parts and text (AI): every part or line of text in an area, as checks', onclick: () => openFinder() }, icon('search'))),
    h('div', { class: 'float tr' }, h('button', { class: 'btn sm', onclick: sourceDialog, title: 'Where this side\'s pictures come from' }, icon(side().source.mode === 'camera' ? 'camera' : 'folder', 'sm'), sourceText(side())),
      h('button', { class: 'btn sm primary', onclick: runOnce }, icon('play', 'sm'), 'Run')),
    h('div', { class: 'float br' }, h('button', { class: 'btn icon', onclick: () => S.viewer.zoomBy(1 / 1.4) }, icon('down', 'sm')),
      h('span', { class: 'zoombadge', id: 'zoom' }, '100%'), h('button', { class: 'btn icon', onclick: () => S.viewer.zoomBy(1.4) }, icon('up', 'sm')),
      h('button', { class: 'btn icon', title: 'Fit (F)', onclick: () => S.viewer.fit() }, icon('fit', 'sm'))));
  if (S.draw) stagecard.append(drawBanner());
  S.animChecks = true;
  requestAnimationFrame(() => { S.viewer.refresh(); renderChecks(); renderInspector(); preview(); });
  return v;
}
function sourceText(s) {
  return s.source.mode === 'camera' ? `Camera ${s.source.camera}` : (s.source.mode === 'folder' || s.source.mode === 'files') ? `${s.source.count} pictures` : 'Reference only';
}
function drawBanner() {
  const d = S.draw;
  return h('div', { class: 'float tc', id: 'drawbanner' }, h('span', { style: { padding: '0 10px', fontWeight: 600 } },
    d.shape === 'polygon' ? `Click the corners of ${d.label}, then Enter` : `Drag a box round ${d.label}`),
    d.shape === 'polygon' ? h('button', { class: 'btn sm primary', onclick: finishPolygon }, 'Done') : null,
    h('button', { class: 'btn sm', onclick: cancelDraw }, 'Cancel'));
}
function cancelDraw() {
  S.draw = null; const b = $('#drawbanner'); b && b.remove();
  S.viewer && S.viewer.el.classList.remove('drawing'); S.viewer && S.viewer.draw();
  $$('.float.left .btn[data-shape]').forEach(x => x.classList.remove('on'));
}
function startDraw(d) {
  if (!side().reference) { toast('Set a reference picture first', true); return; }
  S.draw = { ...d, shape: d.param ? 'rect' : (d.shape || S.shape) };
  if (S.buildPic !== 'reference') { S.buildPic = 'reference'; renderMain(); }
  const sc = $('.stagecard'); const old = $('#drawbanner'); old && old.remove();
  sc && sc.append(drawBanner());
  S.viewer && S.viewer.el.classList.add('drawing');
  $$('.float.left .btn[data-shape]').forEach(x => x.classList.toggle('on', x.dataset.shape === S.draw.shape));
}
const finishPolygon = guard(async () => {
  const pts = (S.draw && S.draw.pts || []).map(q => [Math.round(q[0]), Math.round(q[1])]);
  if (pts.length < 3) { toast('A polygon needs three corners or more', true); return; }
  const xs = pts.map(q => q[0]), ys = pts.map(q => q[1]);
  await finishDraw({ roi: [Math.min(...xs), Math.min(...ys), Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)], points: pts, shape: 'polygon', angle: 0 });
});
async function finishDraw(res, given) {
  const d = given || S.draw; cancelDraw();
  if (d.finder) { await finderDrawn(d.finder, res); return; }
  if (d.param) { await updateTool(d.id, { [d.param]: res.roi }); return; }
  if (d.reshape) { await updateTool(d.id, { roi: res.roi, points: res.points || null, shape: res.shape || d.shape, angle: 0 }); await teachText([d.id]); return; }
  if (!d.type) {
    // the area first, then what to check in it
    if (S.viewer) { S.viewer.draw(); const pts = res.points || outlineOf({ roi: res.roi, shape: d.shape }); S.viewer.svg.append(h('svg:polygon', { class: 'draft', points: pts.map(q => q.join(',')).join(' ') })); }
    // as Vision Station: the area drawn first can also become the selected check's new region
    const cur = selTool();
    const move = cur && (cur.params.roi || cur.params.points) ? { text: `Move ${cur.id} here`, sub: 'The selected check takes this area as its region', icon: 'box',
      fn: () => finishDraw(res, { reshape: true, id: cur.id, label: cur.id, shape: res.shape || d.shape }) } : null;
    addSheet(t => finishDraw.call(null, res, { ...d, type: t, label: label(t) }), 'What should this area check?', move);
    return;
  }
  const params = {};
  for (const r of S.schema.schema[d.type] || []) if (r.default !== null && r.default !== undefined) params[r.key] = r.default;
  params.shape = res.shape || (d.shape === 'polygon' ? 'rect' : d.shape) || 'rect';
  if (res.points) params.points = res.points;
  try {
    const at = S.sel ? side().tools.findIndex(t => t.id === S.sel) + 1 : undefined;
    const r = await api('/api/tool/add', { side: side().id, type: d.type, roi: res.roi, params, at });
    S.lastType = d.type; store.set('lastType', d.type);
    S.st = await api('/api/state'); S.sel = r.id; S.tab = 'settings';
    renderChecks(); renderInspector(); renderTop(false);
    if (d.type === 'count_template') startDraw({ param: 'template_roi', id: r.id, label: 'one sample of what to count' });
    if (d.type === 'ocv' || d.type === 'ocr') await teachText([r.id]);
    await preview();
  } catch (e) { toast(e.message, true); }
}
function select(id) {
  S.sel = id; S.picks = new Set(id ? [id] : []);
  if (id && S.view === 'build' && S.buildPic !== 'reference' && side().reference) {
    // regions are drawn and edited on the reference
    S.buildPic = 'reference';
    const segEl = $('.stagecard .float.tl .seg'); if (segEl && segEl.place) segEl.place('reference');
    if (S.viewer) S.viewer.refresh(true);
    preview();
  }
  if (S.view === 'build') { renderChecks(); renderInspector(); }
  if (S.viewer) S.viewer.draw();
  const row = $(`#checks .check[data-id="${CSS.escape(id || '')}"]`); row && row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}
function renderChecks() {
  const panel = $('#checks'); if (!panel) return;
  const sc = panel.querySelector('.panel-body'); const top = sc ? sc.scrollTop : 0;
  panel.replaceChildren();
  const s = side();
  panel.append(h('div', { class: 'panel-head' }, h('h3', {}, 'Checks'), h('span', { class: 'chip' }, s.tools.length), h('span', { class: 'grow' }),
    h('button', { class: 'btn sm icon ghost', title: 'Undo (Ctrl+Z)', disabled: !s.can_undo, onclick: () => undoRedo('undo') }, icon('undo', 'sm')),
    h('button', { class: 'btn sm icon ghost', title: 'Redo (Ctrl+Y)', disabled: !s.can_redo, onclick: () => undoRedo('redo') }, icon('redo', 'sm')),
    h('button', { class: 'btn sm' + (s.dirty ? ' primary' : ''), title: 'Save (Ctrl+S)', onclick: save }, icon(s.dirty ? 'save' : 'check', 'sm'), s.dirty ? 'Save' : 'Saved')));
  if (open().sides.length > 1 || !open().single) {
    const tabs = h('div', { class: 'row', style: { padding: '0 12px 8px', flexWrap: 'wrap', gap: '6px' } });
    open().sides.forEach((x, i) => tabs.append(h('button', { class: 'btn sm' + (i === S.side ? ' on' : ''), oncontextmenu: e => { e.preventDefault(); sideMenu(e, i); },
      onclick: () => { if (i === S.side) return; S.side = i; S.sel = null; S.preview = null; renderMain(); } }, h('span', { class: 'dot ' + ((lastResult(i) || {}).verdict || '') }), x.name, x.dirty ? ' •' : '')));
    if (!open().single) tabs.append(h('button', { class: 'btn sm icon ghost', title: 'Add a side', onclick: newSide }, icon('plus', 'sm')));
    panel.append(tabs);
  }
  const body = h('div', { class: 'panel-body' + (S.animChecks ? ' anim' : '') });
  S.animChecks = false;
  if (!s.tools.length) {
    body.append(h('div', { class: 'empty-note' }, h('div', { class: 'big' }, icon('target', 'lg')), h('b', {}, 'No checks yet'),
      h('div', { style: { margin: '6px 0 14px' } }, 'Add a check, then drag a box round the place on the picture.'), h('button', { class: 'btn primary', onclick: addSheet }, icon('plus'), 'Add a check')));
  }
  let dragId = null;
  s.tools.forEach((t, i) => {
    const row = checkRow(t.id, t.type, resultFor(t.id), e => (e.ctrlKey || e.metaKey || e.shiftKey) ? toggleSel(t.id) : select(t.id), Math.min(i, 14), t);
    row.draggable = true;
    row.prepend(h('span', { class: 'grip' }, icon('grip', 'sm')));
    row.addEventListener('contextmenu', e => { e.preventDefault(); e.stopPropagation(); if (!sels().includes(t.id)) select(t.id); checkMenu(e, t); });
    row.addEventListener('dragstart', e => { dragId = t.id; row.classList.add('dragging'); e.dataTransfer.effectAllowed = 'move'; });
    row.addEventListener('dragend', () => row.classList.remove('dragging'));
    row.addEventListener('dragover', e => { e.preventDefault(); const r = row.getBoundingClientRect(), after = e.clientY > r.top + r.height / 2; row.classList.toggle('dropafter', after); row.classList.toggle('dropbefore', !after); });
    row.addEventListener('dragleave', () => row.classList.remove('dropafter', 'dropbefore'));
    row.addEventListener('drop', guard(async e => {
      e.preventDefault(); const after = row.classList.contains('dropafter'); row.classList.remove('dropafter', 'dropbefore');
      if (!dragId || dragId === t.id) return;
      const from = s.tools.findIndex(x => x.id === dragId); let to = i + (after ? 1 : 0); if (from < to) to--;
      await api('/api/tool/move', { side: s.id, id: dragId, to }); S.st = await api('/api/state'); renderChecks(); preview();
    }));
    body.append(row);
  });
  // right-click on the list but not on a check: what to add, where pictures come from
  body.addEventListener('contextmenu', e => {
    e.preventDefault(); if (locked()) return;
    menu(e, [['Add a check…', 'plus', () => addSheet(), '', 'A'], ['Picture source…', 'camera', sourceDialog]]);
  });
  panel.append(body);
  body.scrollTop = top;
}
function checkMenu(e, t) {
  if (locked()) return;
  const ids = sels(), many = ids.length > 1 && ids.includes(t.id);
  if (many) {
    const texts = side().tools.filter(x => ids.includes(x.id) && (x.type === 'ocv' || x.type === 'ocr')).map(x => x.id);
    menu(e, [
      texts.length ? ['Expect what each reads now', 'text', () => teachText(texts, true)] : null,
      ['Duplicate ' + ids.length + ' checks', 'copy', () => duplicateTools(ids), '', 'Ctrl+D'],
      ['Enable all', 'eye', () => updateMany(Object.fromEntries(ids.map(id => [id, { enabled: null }])))],
      ['Disable all', 'eyeoff', () => updateMany(Object.fromEntries(ids.map(id => [id, { enabled: false }])))],
      null,
      ['Delete ' + ids.length + ' checks', 'trash', () => deleteTools(ids), 'danger', 'Del'],
    ]);
    return;
  }
  menu(e, [
    ['Open (all settings)', 'gear', () => { select(t.id); S.tab = 'settings'; renderInspector(); }],
    ['Inputs and links…', 'link', () => { select(t.id); S.tab = 'inputs'; renderInspector(); }],
    ['Rename…', 'text', () => renameTool(t.id), '', 'F2'],
    ['Duplicate', 'copy', () => duplicateTools([t.id]), '', 'Ctrl+D'],
    [t.params.enabled === false ? 'Enable' : 'Disable', t.params.enabled === false ? 'eye' : 'eyeoff', () => updateTool(t.id, { enabled: t.params.enabled === false })],
    ['Redraw region', 'box', () => startDraw({ reshape: true, id: t.id, label: t.id, shape: S.shape })],
    t.type === 'ocv' || t.type === 'ocr' ? ['Expect what it reads now', 'text', () => teachText([t.id], true)] : null,
    ['Pictures and tuning…', 'model', () => { select(t.id); S.tab = 'train'; renderInspector(); }],
    null,
    ['Move up', 'up', () => moveTool(t.id, -1)], ['Move down', 'down', () => moveTool(t.id, 1)],
    null,
    ['Delete', 'trash', () => deleteTools([t.id]), 'danger', 'Del'],
  ]);
}
function sideMenu(e, i) {
  if (locked()) return;
  const x = open().sides[i];
  menu(e, [
    ['Edit checks', 'build', () => { S.side = i; S.sel = null; S.picks = new Set(); S.preview = null; if (S.view === 'build') renderMain(); else setView('build'); }],
    null,
    ['Rename side…', 'text', async () => { const n = await prompt1('Rename side', '', x.name); if (n) { await api('/api/side/rename', { side: x.id, name: n }); await reload(); } }],
    ['Picture source…', 'camera', () => { S.side = i; sourceDialog(); }],
    S.settings && S.settings.general && S.settings.general.advanced !== false ? [x.advanced ? 'Model teaching run (retrain)…' : 'Model teaching run…', 'model', () => { S.side = i; teachingRun(); }] : null,
    x.advanced ? ['Remove advanced model…', 'trash', () => { S.side = i; removeAdvanced(); }, 'danger'] : null,
    null,
    ['Remove side…', 'trash', async () => { if (await confirm1('Remove ' + x.name + '?', 'Its folder is moved aside, not deleted.')) { await api('/api/side/remove', { side: x.id }); S.side = 0; await reload(); } }, 'danger'],
    ...(open().single ? [] : [null, ['Add a side…', 'plus', newSide]]),
  ]);
}

// ---------------------------------------------------------------- inspector
function renderInspector() {
  const panel = $('#inspector'); if (!panel) return;
  if (S.find && S.view === 'build') { renderFinder(panel); return; }
  const prevBody = panel.querySelector('.tabbody'); const top = prevBody ? prevBody.scrollTop : 0, keepTab = panel.dataset.tool === S.sel;
  panel.replaceChildren();
  panel.dataset.tool = S.sel || '';
  const t = selTool();
  if (!t) { panel.append(sidePanel()); return; }
  const r = resultFor(t.id);
  panel.append(h('div', { class: 'insp-head' }, h('div', { class: 'ti' }, icon(toolIcon(t.type), 'lg')),
    h('div', { class: 'grow', style: { minWidth: 0 } }, h('h3', {}, t.id), h('div', { class: 'muted' }, label(t.type))),
    r ? h('span', { class: 'chip ' + r.status }, r.status === 'pass' ? 'OK' : r.status === 'fail' ? 'NG' : r.status) : null,
    h('button', { class: 'btn icon ghost sm', title: 'More', onclick: e => checkMenu(e, t) }, icon('more', 'sm')),
    h('button', { class: 'btn icon ghost sm', title: 'Close (Esc)', onclick: () => select(null) }, icon('x', 'sm'))));
  const gb = groupBar(t);
  if (gb) panel.append(gb);
  const tabs = [['settings', 'Settings'], ['inputs', 'Inputs'], ['graphics', 'Graphics'], ['train', 'Train'], ['result', 'Result']];
  if (!tabs.find(x => x[0] === S.tab)) S.tab = 'settings';
  const bar = h('span', { class: 'bar' });
  const tb = h('div', { class: 'tabs' }, ...tabs.map(([k, n]) => h('button', { class: k === S.tab ? 'on' : '', 'data-k': k, onclick: () => { S.tab = k; renderInspector(); } }, n)), bar);
  panel.append(tb);
  requestAnimationFrame(() => { const b = $(`button[data-k="${S.tab}"]`, tb); if (b) { bar.style.left = b.offsetLeft + 'px'; bar.style.width = b.offsetWidth + 'px'; } });
  const fresh = panel._shown !== (S.sel + ':' + S.tab);
  panel._shown = S.sel + ':' + S.tab;
  const body = h('div', { class: 'tabbody' + (fresh ? ' enter' : '') });
  ({ settings: tabSettings, inputs: tabInputs, graphics: tabGraphics, train: tabTrain, result: tabResult })[S.tab](t, body, r);
  panel.append(body);
  if (keepTab) body.scrollTop = top;
}
/* As Vision Station's group bar: above a check's tabs when it belongs to a
   group of the same kind (the twelve inductors Find parts added) or when
   several are selected -- one threshold for all, select them all on the
   picture, turn them all on or off, text checks expect what they read now.
   A kind of part that fails on every board is put right once. */
const THRESHOLD_KEYS = { presence: ['min_score', 'Accept threshold'], count_template: ['min_score', 'Accept threshold'], align: ['min_score', 'Accept threshold'],
  ocv: ['min_confidence', 'Reading confidence'], ocr: ['min_confidence', 'Reading confidence'] };
const groupName = t => (t.params && t.params.group) || t.id.replace(/_?\d+$/, '') || t.id;
function groupMembers(t) {
  const g = groupName(t);
  return side().tools.filter(x => x.type === t.type && groupName(x) === g);
}
function groupBar(t) {
  const picked = sels(), multi = picked.length > 1;
  const members = multi ? side().tools.filter(x => picked.includes(x.id)) : groupMembers(t);
  if (members.length < 2) return null;
  const title = multi ? `${members.length} checks selected` : `Group ${groupName(t)} - ${members.length} ${label(t.type)} checks`;
  const keyed = members.filter(m => THRESHOLD_KEYS[m.type]);
  const box = h('div', { class: 'groupbar' }, h('div', { class: 'gb-title' }, icon('selall', 'sm'), h('b', {}, title)));
  if (keyed.length) {
    const [key, what] = THRESHOLD_KEYS[keyed[0].type];
    const vals = [...new Set(keyed.map(m => Math.round(+(m.params[key] ?? (key === 'min_score' ? 0.65 : 0.5)) * 1000) / 1000))].sort((a, b) => a - b);
    const inp = numInput(vals[0], 0.05, 0, 1, v => {
      const ups = {};
      for (const m of members) { const k = THRESHOLD_KEYS[m.type]; if (k) ups[m.id] = { [k[0]]: Math.round(v * 1000) / 1000 }; }
      updateMany(ups).then(() => toast(`${what} ${(+v).toFixed(2)} on ${Object.keys(ups).length} checks`));
    });
    box.append(h('div', { class: 'row gb-row' }, h('span', { class: 'grow' }, `${what} for all ${keyed.length}`, vals.length > 1 ? h('span', { class: 'faint' }, `  (now ${vals[0]} - ${vals[vals.length - 1]})`) : null), inp));
    const res = buildResult(), got = new Map(((res && res.tools) || []).map(x => [x.tool_id, x]));
    const sc = keyed.filter(m => got.has(m.id)).map(m => [got.get(m.id).score, m.id, got.get(m.id).status]);
    if (sc.length) {
      const lo = sc.reduce((a, b) => (b[0] < a[0] ? b : a)), nfail = sc.filter(x => x[2] !== 'pass').length;
      box.append(h('div', { class: 'help muted' }, `Last run: lowest score ${fmt(lo[0])} (${lo[1]})${nfail ? `  -  ${nfail} failed` : ''}`));
    }
  }
  const on = members.map(m => !m.params || m.params.enabled !== false), allOn = on.every(Boolean), noneOn = !on.some(Boolean);
  const sw2 = sw(allOn, v => {
    const ups = {}; for (const m of members) ups[m.id] = { enabled: v ? null : false };
    updateMany(ups).then(() => toast(`${members.length} checks turned ${v ? 'on' : 'off'}`));
  });
  if (!allOn && !noneOn) sw2.classList.add('mixed');
  const texts = members.filter(m => m.type === 'ocv' || m.type === 'ocr').map(m => m.id);
  box.append(h('div', { class: 'row gb-row', style: { flexWrap: 'wrap' } },
    h('button', { class: 'btn sm', onclick: () => {
      const ids = members.map(m => m.id);
      if (S.buildPic !== 'reference') { S.buildPic = 'reference'; renderMain(); }
      S.picks = new Set(ids); S.sel = ids.includes(S.sel) ? S.sel : ids[0];
      renderChecks(); renderInspector(); S.viewer && S.viewer.draw();
      setTimeout(() => S.viewer && S.viewer.zoomTo(ids), 30);
    } }, icon('target', 'sm'), 'Select all on the picture'),
    texts.length ? h('button', { class: 'btn sm', title: 'Each text check expects the text it reads on the teach picture', onclick: () => teachText(texts, true) }, icon('text', 'sm'), 'Expect what each reads now') : null,
    h('span', { class: 'grow' }), h('span', { class: 'muted', title: 'Turn every check of the group on or off' }, 'Run them'), h('div', { class: 'sw' }, sw2)));
  return box;
}
function sidePanel() {
  const s = side(), body = h('div', { class: 'tabbody' });
  const wrap = h('div', { class: 'panel', style: { height: '100%' } },
    h('div', { class: 'insp-head' }, h('div', { class: 'ti' }, icon('image', 'lg')), h('div', { class: 'grow' }, h('h3', {}, s.name), h('div', { class: 'muted' }, 'This side\'s picture and scale'))), body);
  body.append(h('div', { class: 'note' }, 'Select a check on the picture or in the list to see all its settings, or add one.'),
    h('div', { class: 'section' }, 'Picture'),
    h('div', { class: 'field' }, h('label', {}, 'Source'), h('button', { class: 'btn sm', onclick: sourceDialog }, icon(s.source.mode === 'camera' ? 'camera' : 'folder', 'sm'), sourceText(s))),
    h('div', { class: 'field' }, h('label', {}, 'Reference'), h('span', { class: 'muted', style: { textAlign: 'right' } }, s.reference ? `${s.reference[0]} × ${s.reference[1]}` : 'none')),
    h('div', { class: 'row', style: { flexWrap: 'wrap', margin: '10px 0' } },
      h('button', { class: 'btn sm', onclick: setReference, title: 'Use the last inspected picture' }, icon('star', 'sm'), 'Reference from last'),
      h('label', { class: 'btn sm' }, icon('upload', 'sm'), 'Upload reference', h('input', { type: 'file', accept: 'image/*', hidden: true, onchange: uploadReference }))),
    h('div', { class: 'section' }, 'Scale'),
    h('div', { class: 'field' }, h('label', {}, 'Millimetres per pixel (0 = not calibrated)'), numInput(s.mm_per_px || 0, 0.0001, 0, 100, v => projectSet({ mm_per_px: v }))),
    h('div', { class: 'section' }, 'Checks'),
    h('button', { class: 'btn primary', style: { width: '100%' }, onclick: addSheet }, icon('plus'), 'Add a check'));
  return wrap;
}
function numInput(val, step, min, max, set, isInt, allowEmpty) {
  const inp = h('input', { type: 'text', inputmode: 'decimal', value: val ?? '' });
  const commit = () => {
    if (allowEmpty && inp.value.trim() === '') { if (val !== '' && val != null) { val = ''; set(null); } return; }
    let n = isInt ? parseInt(inp.value, 10) : parseFloat(inp.value); if (isNaN(n)) { inp.value = val ?? ''; return; } if (min != null) n = Math.max(min, n); if (max != null) n = Math.min(max, n); if (n !== val) { val = n; set(n); } inp.value = n; };
  inp.addEventListener('change', commit);
  inp.addEventListener('keydown', e => { if (e.key === 'ArrowUp' || e.key === 'ArrowDown') { e.preventDefault(); bump(e.key === 'ArrowUp' ? 1 : -1, e.shiftKey ? 10 : 1); } });
  inp.addEventListener('wheel', e => { if (document.activeElement === inp) { e.preventDefault(); bump(e.deltaY < 0 ? 1 : -1, 1); } }, { passive: false });
  const st = +step || (isInt ? 1 : 0.1);
  const bump = (d, k) => { const cur = parseFloat(inp.value) || 0; inp.value = +(cur + d * st * k).toFixed(6); commit(); };
  return h('div', { class: 'num' }, inp, h('button', { onclick: () => bump(-1, 1), tabindex: -1 }, '−'), h('button', { onclick: () => bump(1, 1), tabindex: -1 }, '+'));
}
function sw(on, set) { const el = h('div', { class: 'switch' + (on ? ' on' : ''), role: 'switch', tabindex: 0 }); el.onclick = () => { on = !on; el.classList.toggle('on', on); set(on); }; return el; }
function field(text, control, help, wide) { return h('div', { class: 'field' + (wide ? ' wide' : '') }, h('label', {}, text), control, help ? h('div', { class: 'help' }, help) : null); }

function tabSettings(t, body) {
  const p = t.params || {}, set = (k, v) => updateTool(t.id, { [k]: v });
  if (!S.st.ported.includes(t.type)) body.append(h('div', { class: 'note warn' }, 'This check cannot run here; it is kept and saved.'));
  body.append(h('div', { class: 'note', style: { marginTop: '4px' } }, describe(t.type)));
  body.append(field('Name', h('input', { class: 'input', value: t.id, onchange: e => renameTool(t.id, e.target.value) })));
  body.append(field('Enabled', h('div', { class: 'sw' }, sw(p.enabled !== false, v => set('enabled', v)))));
  body.append(field('Rejects the part when it fails', h('div', { class: 'sw' }, sw(p.critical !== false, v => set('critical', v))), 'Off: a failure only warns'));
  if (p.roi || p.points) {
    body.append(field('Shape', seg([['rect', 'Box'], ['ellipse', 'Ellipse'], ['polygon', 'Polygon']], p.shape || 'rect', v => {
      if (v === 'polygon') startDraw({ reshape: true, id: t.id, label: 'the region', shape: 'polygon' });
      else set('shape', v);
    }), null, true));
    if (p.shape !== 'polygon') {
      body.append(field('Angle (°)', numInput(+(angleOf(p)).toFixed(2), 0.5, -180, 180, v => set('angle', v))));
      const roi = p.roi || [0, 0, 0, 0];
      const grid = h('div', { style: { display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '8px' } },
        ...['x', 'y', 'w', 'h'].map((k, i) => h('div', {}, h('div', { class: 'faint', style: { fontSize: '11px', margin: '0 0 3px 2px' } }, k),
          numInput(Math.round(roi[i]), 1, i > 1 ? 1 : null, null, v => { const r = roi.slice(); r[i] = v; set('roi', r); }, true))));
      body.append(field('Region (px)', grid, null, true));
    }
  }
  const rows = S.schema.schema[t.type] || [], ess = S.schema.essential[t.type];
  const main = ess ? rows.filter(r => ess.includes(r.key)) : rows, more = ess ? rows.filter(r => !ess.includes(r.key)) : [];
  const group = (title, list, key) => {
    if (!list.length) return;
    const closed = !!S.closed[key];
    const box = h('div', { hidden: closed });
    const gt = h('div', { class: 'group-title' + (closed ? ' closed' : ''), onclick: () => { S.closed[key] = !S.closed[key]; store.set('closed', S.closed); gt.classList.toggle('closed'); box.hidden = !box.hidden; } }, icon('chevd', 'sm'), title, h('span', { class: 'faint', style: { fontWeight: 500 } }, list.length));
    for (const r of list) box.append(paramField(t, r, p[r.key] === undefined ? r.default : p[r.key]));
    body.append(gt, box);
  };
  group('Main settings', main, 'main');
  group('More settings', more, 'more');
}
function paramField(t, r, val) {
  const set = v => updateTool(t.id, { [r.key]: v });
  const ex = r.extra, help = r.help;
  switch (r.kind) {
    case 'bool': return field(r.label, h('div', { class: 'sw' }, sw(!!val, set)), help);
    case 'int': case 'float': return field(r.label, numInput(val, ex ? ex[2] : null, ex ? ex[0] : null, ex ? ex[1] : null, set, r.kind === 'int'), help);
    case 'choice': return field(r.label, h('select', { class: 'input', onchange: e => set(e.target.value) }, ...(ex || []).map(o => h('option', { value: o, selected: o === val }, o.replace(/_/g, ' ')))), help);
    case 'toolref': {
      const opts = side().tools.filter(x => x.id !== t.id);
      return field(r.label, h('select', { class: 'input', onchange: e => set(e.target.value) }, h('option', { value: '' }, '—'), ...opts.map(x => h('option', { value: x.id, selected: x.id === val }, x.id))), help);
    }
    case 'intlist':
      if (r.key.endsWith('_roi')) return field(r.label, h('button', { class: 'btn sm', onclick: () => startDraw({ param: r.key, id: t.id, label: r.label.toLowerCase() }) }, icon('box', 'sm'), val ? 'Redraw' : 'Draw'), val ? `at ${val.map(Math.round).join(', ')}` : help);
      if (Array.isArray(val) && val.length === 2) {
        return field(r.label, h('div', { style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '6px' } },
          numInput(val[0], 1, null, null, v => set([v, val[1]])), numInput(val[1], 1, null, null, v => set([val[0], v]))), help);
      }
      return field(r.label, h('input', { class: 'input', value: (val || []).join(', '), onchange: e => set(e.target.value.split(/[,\s]+/).filter(Boolean).map(Number).filter(n => !isNaN(n))) }), help);
    case 'strlist': return field(r.label, chipsEditor(val || [], set, ['source', 'expect'].includes(r.key) ? 'red, green, blue' : 'comma separated'), help, true);
    case 'boollist': return field(r.label, boolsEditor(val || [], set), help, true);
    default: {
      if (r.key === 'expression' || r.key === 'pairs' || r.key === 'points') {
        return field(r.label, h('textarea', { class: 'input', rows: 2, value: typeof val === 'object' && val ? JSON.stringify(val) : val ?? '', onchange: e => set(e.target.value) }), help, true);
      }
      if (r.key === 'expect_counts') {
        return field(r.label, h('input', { class: 'input', value: val && typeof val === 'object' ? JSON.stringify(val) : val || '{}', placeholder: '{"screw": 4}', onchange: e => { try { set(JSON.parse(e.target.value || '{}')); } catch { toast('Write it as {"name": count}', true); } } }), help, true);
      }
      if (r.key === 'source' || r.key === 'points') {
        const ids = side().tools.filter(x => x.id !== t.id).map(x => x.id);
        const dl = 'dl_' + r.key;
        return field(r.label, h('div', {}, h('input', { class: 'input', list: dl, value: val ?? '', onchange: e => set(e.target.value) }), h('datalist', { id: dl }, ...ids.map(i => h('option', { value: i })))), help);
      }
      const long = String(val ?? '').length > 16 || /regex|text|model|contains|charset/.test(r.key);
      const inp = h('input', { class: 'input' + (/regex/.test(r.key) ? ' mono' : ''), value: val ?? '', onchange: e => set(e.target.value) });
      if (r.key === 'model') {
        const npz = /npz/i.test(r.label);
        return field(r.label, pathInput(inp, { kind: 'file', side: side().id, title: r.label,
          filters: npz ? [['Trained model (.npz)', ['npz']]] : [['ONNX model (.onnx)', ['onnx']]] }), help, true);
      }
      return field(r.label, inp, help, long);
    }
  }
}
function chipsEditor(list, set, ph) {
  return h('input', { class: 'input', value: list.join(', '), placeholder: ph, onchange: e => set(e.target.value.split(',').map(x => x.trim()).filter(Boolean)) });
}
function boolsEditor(list, set) {
  const wrap = h('div', { class: 'row', style: { flexWrap: 'wrap', gap: '6px' } });
  const draw = () => {
    wrap.replaceChildren(...list.map((b, i) => h('button', { class: 'btn sm' + (b ? ' on' : ''), onclick: () => { list = list.slice(); list[i] = !list[i]; set(list); draw(); } }, `${i + 1}: ${b ? 'on' : 'off'}`)),
      h('button', { class: 'btn sm icon ghost', title: 'One more', onclick: () => { list = [...list, true]; set(list); draw(); } }, icon('plus', 'sm')),
      list.length ? h('button', { class: 'btn sm icon ghost', title: 'One fewer', onclick: () => { list = list.slice(0, -1); set(list); draw(); } }, icon('x', 'sm')) : null);
  };
  draw();
  return wrap;
}
function tabInputs(t, body) {
  const p = t.params || {}, tools = side().tools, idx = tools.findIndex(x => x.id === t.id), before = tools.slice(0, idx);
  const fixtures = before.filter(x => x.type === 'align' || x.type === 'fixture_npoint');
  const pics = before.filter(x => ['image_process', 'affine_transform', 'polar_unwrap', 'image_combine'].includes(x.type));
  body.append(h('div', { class: 'section' }, 'Follows the part'),
    field('Fixture', h('select', { class: 'input', onchange: e => updateTool(t.id, { fixture: e.target.value }) },
      h('option', { value: 'auto', selected: !p.fixture || p.fixture === 'auto' }, 'Latest Locate (auto)'), h('option', { value: 'none', selected: p.fixture === 'none' }, 'None: picture coordinates'),
      ...fixtures.map(x => h('option', { value: x.id, selected: p.fixture === x.id }, x.id))), 'Which Locate check moves this region with the part'),
    h('div', { class: 'section' }, 'Reads the picture'),
    field('Picture', h('select', { class: 'input', onchange: e => updateTool(t.id, { capture: e.target.value }) },
      h('option', { value: 'default', selected: !p.capture || p.capture === 'default' }, 'The camera\'s'),
      ...pics.map(x => h('option', { value: x.id, selected: p.capture === x.id }, 'made by ' + x.id))), 'A filtered or transformed picture from an earlier check'));
  body.append(h('div', { class: 'section' }, 'Linked settings'), h('div', { class: 'help muted', style: { fontSize: '12px', margin: '-2px 0 6px' } }, 'A setting taken from an earlier check\'s result on every run, e.g. min_score ← locate.score'));
  const links = { ...(p.links || {}) };
  const nums = (S.schema.schema[t.type] || []).filter(r => r.kind === 'int' || r.kind === 'float' || r.kind === 'text');
  const vals = []; for (const x of before) { const rr = resultFor(x.id); vals.push(x.id + '.score', x.id + '.status'); if (rr) for (const k of Object.keys(rr.values || {})) vals.push(x.id + '.' + k); }
  const save = () => updateTool(t.id, { links: Object.keys(links).length ? links : null });
  for (const [k, v] of Object.entries(links)) {
    body.append(h('div', { class: 'row', style: { margin: '6px 0' } }, h('span', { class: 'chip accent' }, k), '←',
      h('input', { class: 'input mono', list: 'dl_links', value: v, onchange: e => { links[k] = e.target.value; save(); } }),
      h('button', { class: 'btn sm icon ghost', onclick: () => { delete links[k]; save(); } }, icon('x', 'sm'))));
  }
  const pick = h('select', { class: 'input' }, h('option', { value: '' }, 'Link a setting…'), ...nums.filter(r => !links[r.key]).map(r => h('option', { value: r.key }, r.label)));
  pick.onchange = () => { if (pick.value) { links[pick.value] = (vals[0] || ''); save(); } };
  body.append(pick, h('datalist', { id: 'dl_links' }, ...vals.map(v => h('option', { value: v }))));
}
function tabGraphics(t, body) {
  const g = { region: true, results: true, labels: true, search: true, ...(t.params.graphics || {}) };
  const set = (k, v) => { g[k] = v; updateTool(t.id, { graphics: g }); };
  body.append(h('div', { class: 'note' }, 'What this check draws on the picture: on screen, in saved result pictures and on the line PC alike.'),
    field('Its region', h('div', { class: 'sw' }, sw(g.region, v => set('region', v)))),
    field('What it found', h('div', { class: 'sw' }, sw(g.results, v => set('results', v))), 'Matches, edges, blobs, text lines'),
    field('Labels', h('div', { class: 'sw' }, sw(g.labels, v => set('labels', v)))),
    field('Where it searched', h('div', { class: 'sw' }, sw(g.search, v => set('search', v)))));
}
function tabResult(t, body, r) {
  if (!r) { body.append(h('div', { class: 'note' }, 'Not run yet. Results appear here after a preview or a run.')); return; }
  body.append(resultBox(r));
  if (r.caption && r.caption.length) body.append(h('div', { class: 'section' }, 'On the picture'), h('div', { class: 'note mono' }, r.caption.join('  ·  ')));
}
/* Example pictures for a check, as Vision Station's Train tab: OK and NG
   pictures, the whole recipe run on them (green: judged right, red: wrong),
   then Train (Surface), Tune threshold (Part present, Locate part). */
const TUNABLE = { presence: 'min_score', align: 'min_score' };
function tabTrain(t, body) {
  const model = ['yolo_detect', 'yolo_defect', 'classify', 'anomaly'].includes(t.type);
  if (model) body.append(h('div', { class: 'note' }, 'This check runs a model trained elsewhere (.onnx): set its file on the Settings tab. The pictures below still show how the whole recipe judges good and bad parts.'));
  const surface = t.type === 'trained_surface';
  body.append(h('div', { class: 'note' }, surface ? 'Learns the region from pictures of good parts and flags anything that differs. Add a few OK pictures (and NG ones if you have them), then Train.'
    : TUNABLE[t.type] ? 'Add pictures of good (OK) and bad (NG) parts, then Tune threshold: it is set midway between the lowest-scoring OK and the highest-scoring NG.'
    : 'Add pictures of good (OK) and bad (NG) parts, then Check them: every picture goes through the whole recipe and is marked green when judged right.'));
  const info = h('div', {}, h('div', { class: 'skeleton', style: { height: '80px', borderRadius: '14px' } }));
  body.append(info);
  let marks = {}, last = null, report = null;
  const call = guard(async (what, extra) => {
    const r = await api('/api/train/' + what, { side: side().id, id: t.id, ...(extra || {}) });
    if (what === 'evaluate' || what === 'tune') {
      marks = {}; for (const row of r.rows) marks[row.label + '/' + row.name] = row;
      report = r;
      if (r.tune) {
        const tu = r.tune;
        toast(tu.separates ? `${tu.param}: ${fmt(tu.old)} → ${fmt(tu.new)} (lowest OK ${fmt(tu.lowest_ok)}, highest NG ${fmt(tu.highest_ng)})`
          : `Not changed: the lowest OK (${fmt(tu.lowest_ok)}) does not score above the highest NG (${fmt(tu.highest_ng)})`, !tu.separates);
        S.st = await api('/api/state'); preview();
      }
      r.files = last && last.files; r.trained = last && last.trained;
    }
    if (what === 'run') { toast('Trained'); await preview(); }
    if (!r.files) { const l = await api('/api/train/list', { side: side().id, id: t.id }); r.files = l.files; r.trained = l.trained; r.ok = l.ok; r.ng = l.ng; }
    last = r; draw(r);
  });
  const thumbs = (label, names) => h('div', { class: 'train-thumbs' }, ...(names || []).map(n => {
    const m = marks[label + '/' + n];
    return h('div', { class: 'train-thumb' + (m ? (m.correct ? ' right' : ' wrong') : ''), title: m ? `${m.status} · score ${fmt(m.score)}${m.message ? ' · ' + m.message : ''}` : n },
      h('img', { src: `/api/train/img?side=${encodeURIComponent(side().id)}&id=${encodeURIComponent(t.id)}&label=${label}&name=${encodeURIComponent(n)}`, alt: '', loading: 'lazy' }),
      m ? h('span', { class: 'chip ' + (m.correct ? 'pass' : 'fail') }, fmt(m.score)) : null,
      h('button', { class: 'btn icon ghost sm mv', title: `Move to ${label === 'ok' ? 'NG' : 'OK'}`, onclick: () => call('move', { label, name: n }) }, icon(label === 'ok' ? 'down' : 'up', 'sm')),
      h('button', { class: 'btn icon ghost sm x', title: 'Remove from the training set', onclick: () => call('remove', { label, name: n }) }, icon('x', 'sm')));
  }));
  const draw = r => {
    const files = r.files || { ok: [], ng: [] };
    info.replaceChildren(h('div', {},
      h('div', { class: 'row', style: { flexWrap: 'wrap', marginTop: '4px' } },
        h('button', { class: 'btn sm', onclick: () => call('add', { label: 'ok', from: 'last' }) }, icon('plus', 'sm'), 'Last picture is OK'),
        h('button', { class: 'btn sm', onclick: () => call('add', { label: 'ng', from: 'last' }) }, icon('plus', 'sm'), 'Last picture is NG'),
        h('button', { class: 'btn sm', onclick: () => call('add', { label: 'ok', from: 'reference' }) }, icon('star', 'sm'), 'Reference is OK')),
      h('div', { class: 'section', style: { margin: '14px 2px 6px' } }, `OK pictures (${files.ok.length})`), thumbs('ok', files.ok),
      h('div', { class: 'section', style: { margin: '14px 2px 6px' } }, `NG pictures (${files.ng.length})`), thumbs('ng', files.ng),
      h('div', { class: 'row', style: { marginTop: '12px', flexWrap: 'wrap' } },
        surface ? h('button', { class: 'btn primary', disabled: !files.ok.length, onclick: e => { e.currentTarget.disabled = true; call('run'); } }, icon('model'), 'Train') : null,
        TUNABLE[t.type] ? h('button', { class: 'btn primary', disabled: !files.ok.length || !files.ng.length, onclick: () => call('tune') }, icon('target'), 'Tune threshold') : null,
        h('button', { class: 'btn', disabled: !files.ok.length && !files.ng.length, onclick: () => call('evaluate') }, icon('check'), 'Check them'),
        h('span', { class: 'grow' }),
        h('button', { class: 'btn ghost danger', disabled: !files.ok.length && !files.ng.length, onclick: async () => { if (await confirm1('Forget the training pictures?', 'A trained model stays until you train again.')) call('clear'); } }, 'Clear pictures')),
      report ? h('div', { class: 'note ' + (report.false_rejects || report.escapes ? 'warn' : 'good') },
        report.false_rejects || report.escapes ? `${report.false_rejects} good part${report.false_rejects === 1 ? '' : 's'} rejected, ${report.escapes} bad part${report.escapes === 1 ? '' : 's'} passed (escapes matter most).`
          : 'Every picture judged right.') : null,
      r.report ? h('div', { class: 'note ' + (r.report.separates ? 'good' : 'warn') }, h('div', { class: 'kv' },
        h('span', {}, 'threshold'), h('span', {}, fmt(r.report.threshold)), h('span', {}, 'worst OK'), h('span', {}, fmt(r.report.worst_ok)),
        h('span', {}, 'OK scores'), h('span', {}, fmt(r.report.ok_scores)), h('span', {}, 'NG scores'), h('span', {}, fmt(r.report.ng_scores)),
        h('span', {}, 'augmented'), h('span', {}, fmt(r.report.augmented)))) : null,
      surface ? h('div', { class: 'muted', style: { marginTop: '8px' } }, r.trained ? 'Trained: the model is in use.' : 'Not trained yet.') : null));
  };
  call('list');
}

// ------------------------------------------------- find parts and text (AI)
/* As Vision Station's part finder panel:
     Find     Parts -> a Part present check per kept part
              Text (verify) -> a Verify text check per kept line, expecting it
              Text (read)   -> a Read text check per kept line
     Area     the whole picture, or a box dragged round part of it
     Review   click a box to keep or drop it; a type's row keeps or drops all
              of its kind, names them ("inductor") and sets their threshold;
              Sensitivity refilters at once; Find all like this finds every
              copy of one part. Text: correct what each line must read.
     Add      one check per kept box, in one step (Ctrl+Z takes them all back) */
const FP_PALETTE = ['#00B3FF', '#FF9F1C', '#2EC4B6', '#E71D73', '#9B5DE5', '#F15BB5', '#00F5D4', '#FEE440', '#8AC926', '#FF595E',
  '#6A4C93', '#1982C4', '#FFCA3A', '#C77DFF', '#06D6A0', '#EF476F', '#118AB2', '#FFD166', '#73D2DE', '#D81159'];
const FP_MODES = [['parts', 'Parts → Part present'], ['ocv', 'Text → Verify text (OCV)'], ['ocr', 'Text → Read text (OCR)']];
const typeName = k => { let s2 = '', n = k + 1; while (n) { const r = (n - 1) % 26; s2 = String.fromCharCode(65 + r) + s2; n = Math.floor((n - 1) / 26); } return s2; };
function openFinder() {
  if (locked()) return;
  if (S.view !== 'build') setView('build');
  if (!side().reference) { toast('Set a reference picture first', true); return; }
  if (S.buildPic !== 'reference') { S.buildPic = 'reference'; renderMain(); }
  S.find = S.find || { mode: 'parts', area: null, areaAngle: 0, parts: [], dropped: new Set(), offTypes: new Set(), names: {}, thresholds: {},
    sens: 0.5, match: 0.6, seated: true, textConf: 0.5, busy: false, status: 'Choose what to find and where, then Find.' };
  S.sel = null; S.picks = new Set();
  renderChecks(); renderInspector(); S.viewer && S.viewer.draw();
}
function closeFinder() { S.find = null; renderInspector(); S.viewer && S.viewer.draw(); }
const fpText = () => S.find && S.find.mode !== 'parts';
function fpKept(i) { const f = S.find, p = f.parts[i]; return !f.dropped.has(i) && (fpText() || !f.offTypes.has(p.type)); }
function finderOverlay() {
  const f = S.find;
  return { area: f.area, areaAngle: f.areaAngle || 0, boxes: f.parts.map((p, i) => ({ box: p.box, angle: p.angle || 0, kept: fpKept(i), hot: f.hot === i || (f.hotType != null && p.type === f.hotType),
    colour: fpText() ? '#1F6FFF' : FP_PALETTE[p.type % FP_PALETTE.length], label: fpText() ? p.text : typeName(p.type) })) };
}
function finderToggle(i) {
  const f = S.find; if (!f) return;
  if (f.dropped.has(i)) { f.dropped.delete(i); if (!fpText()) f.offTypes.delete(f.parts[i].type); } else f.dropped.add(i);
  renderInspector(); S.viewer && S.viewer.draw();
}
// the area ('a') or a found box, as the canvas edits it
function finderItem(k) {
  const f = S.find; if (!f) return null;
  if (k === 'a') return f.area ? { roi: f.area, angle: f.areaAngle || 0 } : null;
  const p = f.parts[+k];
  return p ? { roi: p.box, angle: p.angle || 0 } : null;
}
// the upright box round the area, turned or not (null: the whole picture)
function fpBounds() {
  const f = S.find; if (!f || !f.area) return null;
  if (!f.areaAngle) return f.area;
  const b = boundsOf({ roi: f.area, angle: f.areaAngle });
  return [Math.floor(b.l), Math.floor(b.t), Math.ceil(b.r - b.l), Math.ceil(b.b - b.t)];
}
const fpAreaText = f => f.area ? `Area: ${f.area[2]} × ${f.area[3]} px${f.areaAngle ? `, turned ${(+f.areaAngle).toFixed(1)}°` : ''}` : 'Area: the whole picture';
/* A box moved, resized or turned on the picture: the check will use it as
   it is now; a text box is read again, upright to it. The area changed:
   Find again. */
const finderEdited = guard(async (k, res) => {
  const f = S.find; if (!f) return;
  if (k === 'a') {
    f.area = res.roi; f.areaAngle = res.angle || 0;
    f.status = `Area changed - press ${fpText() ? 'Find text' : 'Find parts'} again.`;
    renderInspector(); S.viewer && S.viewer.draw(); return;
  }
  const p = f.parts[+k]; if (!p) return;
  p.box = res.roi; p.angle = res.angle || 0;
  S.viewer && S.viewer.draw();
  if (!fpText()) return;
  const mode = f.mode;
  f.status = 'Reading the box again…'; renderInspector();
  try {
    const r = await api('/api/find/read_box', { side: side().id, box: p.box, angle: p.angle });
    if (S.find !== f || f.mode !== mode || !f.parts.includes(p)) return;
    p.text = r.text; p.score = r.score;
    f.status = 'Read again: ' + (r.text || 'no text in this box');
  } catch (e) { f.status = e.message; }
  if (S.find === f) { renderInspector(); S.viewer && S.viewer.draw(); }
});
async function finderDrawn(what, res) {
  const f = S.find; if (!f) return;
  if (what === 'area') { f.area = res.roi; f.areaAngle = 0; f.parts = []; f.dropped = new Set(); f.status = 'Area set: press Find.'; renderInspector(); S.viewer && S.viewer.draw(); return; }
  // Find all like this: every copy of the part drawn round, a new type
  f.busy = true; renderInspector();
  try {
    const r = await api('/api/find/like', { side: side().id, example: res.roi, area: fpBounds(), min_score: f.match });
    if (!r.found.length) { f.status = 'No copies found - lower Alike, or draw the box tighter round the part.'; return; }
    const t = f.parts.length ? Math.max(...f.parts.map(p => p.type)) + 1 : 0;
    const ov = (a, b) => Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
    // parts already found where a copy is are replaced by it
    const old = f.parts.filter(p => !r.found.some(c => ov(p.box, c.box) / Math.max(1, p.box[2] * p.box[3]) > 0.5));
    f.parts = [...old, ...r.found.map(c => ({ box: c.box, score: c.score, type: t }))];
    f.dropped = new Set();
    f.status = `Found ${r.found.length} copies (type ${typeName(t)}). Name them in the list.`;
  } catch (e) { f.status = e.message; } finally { f.busy = false; renderInspector(); S.viewer && S.viewer.draw(); }
}
const finderFind = async refilter => {
  const f = S.find; if (!f || f.busy) return;
  f.busy = true; if (!refilter) f.status = fpText() ? 'Reading text…' : 'Finding parts… (the first search of an area takes a few seconds)';
  renderInspector();
  try {
    const aa = f.area ? f.areaAngle || 0 : 0;
    const r = fpText() ? await api('/api/find/text', { side: side().id, area: f.area, area_angle: aa })
      : await api('/api/find/parts', { side: side().id, area: f.area, area_angle: aa, sensitivity: f.sens });
    f.parts = r.parts; f.dropped = new Set(); f.offTypes = new Set();
    if (fpText()) {
      f.names = {};
      // silkscreen noise -- a lone character, a doubtful read -- starts unticked
      f.parts.forEach((p, i) => { if ((p.text || '').trim().length < 2 || (p.score || 0) < 0.6) f.dropped.add(i); });
      f.status = f.parts.length ? `${f.parts.length} line${f.parts.length === 1 ? '' : 's'} of text in ${(r.ms / 1000).toFixed(1)} s. Click a row or its box to keep or drop it; correct the text it must read.` : 'No text found in the area.';
    } else {
      f.status = `${f.parts.length} parts of ${r.types} types in ${(r.ms / 1000).toFixed(1)} s (${r.device}). Click a box to drop it; untick a type to drop all of them.`;
    }
  } catch (e) { f.status = e.message; } finally { f.busy = false; renderInspector(); S.viewer && S.viewer.draw(); }
};
const finderAdd = guard(async () => {
  const f = S.find;
  const kept = f.parts.map((p, i) => [p, i]).filter(([, i]) => fpKept(i));
  if (!kept.length) return;
  const kind = f.mode === 'parts' ? 'presence' : f.mode;
  const defaults = {};
  for (const r of S.schema.schema[kind] || []) if (r.default !== null && r.default !== undefined) defaults[r.key] = r.default;
  const items = kept.map(([p, i]) => fpText() ? { box: p.box, angle: p.angle || 0, text: p.text, name: f.names[i] || '' } : p);
  const r = await api('/api/find/add', { side: side().id, mode: f.mode, parts: items, names: f.names, thresholds: f.thresholds,
    seated: f.seated, min_confidence: fpText() ? f.textConf : null, defaults });
  const what = f.mode === 'parts' ? 'Part present' : f.mode === 'ocv' ? 'Verify text' : 'Read text';
  // what was added leaves the list; the rest stays, to keep or drop
  const left = f.parts.filter((_, i) => !fpKept(i));
  f.parts = left; f.dropped = new Set(left.map((_, i) => i)); f.offTypes = new Set(); if (fpText()) f.names = {};
  f.status = `Added ${r.added} ${what} check${r.added === 1 ? '' : 's'}. Find more - another area or kind - or Close.`;
  S.st = await api('/api/state'); renderChecks(); renderInspector(); renderTop(false); S.viewer && S.viewer.draw(); preview();
  toast(`Added ${r.added} ${what} check${r.added === 1 ? '' : 's'} - Ctrl+Z takes them all back`);
});
function renderFinder(panel) {
  const f = S.find;
  panel.replaceChildren();
  panel.dataset.tool = '';
  const text = fpText();
  const keptN = f.parts.filter((_, i) => fpKept(i)).length;
  const what = f.mode === 'parts' ? 'Part present' : f.mode === 'ocv' ? 'Verify text' : 'Read text';
  const body = h('div', { class: 'tabbody' });
  panel.append(h('div', { class: 'insp-head' }, h('div', { class: 'ti' }, icon('search', 'lg')),
    h('div', { class: 'grow', style: { minWidth: 0 } }, h('h3', {}, 'Find parts and text'), h('div', { class: 'muted' }, 'AI on the reference picture')),
    h('button', { class: 'btn icon ghost sm', title: 'Close (Esc)', onclick: closeFinder }, icon('x', 'sm'))), body);
  body.append(h('div', { class: 'note' }, 'Find what is in an area of the picture, check the boxes, and add a separate check for each one you keep.'),
    field('Find', h('select', { class: 'input', disabled: f.busy, onchange: e => { f.mode = e.target.value; f.parts = []; f.dropped = new Set(); f.status = 'Press Find.'; renderInspector(); S.viewer.draw(); } },
      ...FP_MODES.map(([k, t]) => h('option', { value: k, selected: k === f.mode }, t))), null, true),
    field(fpAreaText(f), h('div', { class: 'row', style: { gap: '6px', justifyContent: 'flex-end' } },
      h('button', { class: 'btn sm', disabled: f.busy, onclick: () => startDraw({ finder: 'area', shape: 'rect', label: 'the area to search' }) }, icon('box', 'sm'), 'Draw area…'),
      h('button', { class: 'btn sm', disabled: f.busy || !f.area, onclick: () => { f.area = null; f.areaAngle = 0; f.parts = []; renderInspector(); S.viewer.draw(); } }, 'Whole picture')), null, true),
    h('button', { class: 'btn primary', style: { width: '100%', justifyContent: 'center', margin: '8px 0' }, disabled: f.busy, onclick: () => finderFind(false) },
      f.busy ? h('span', { class: 'spin' }) : icon('search', 'sm'), f.busy ? 'Working…' : text ? 'Find text' : 'Find parts'),
    h('div', { class: 'muted', style: { fontSize: '12.5px', margin: '0 2px 8px' } }, f.status || ''));
  if (!text) {
    const out = h('b', { style: { minWidth: '36px', textAlign: 'right' } }, Math.round(f.sens * 100));
    body.append(field('Sensitivity', h('div', { class: 'row', style: { width: '220px' } }, h('input', { type: 'range', class: 'range', min: 0, max: 100, value: Math.round(f.sens * 100), disabled: f.busy || !f.parts.length && !f.status.includes('parts'),
      oninput: e => out.textContent = e.target.value, onchange: e => { f.sens = e.target.value / 100; finderFind(true); } }), out), 'Higher finds fainter parts - and more that are not parts'));
  }
  // the list: a row per type (parts) or per line (text)
  const list = h('div', { class: 'fp-list' });
  if (!text) {
    const types = [...new Set(f.parts.map(p => p.type))].sort((a, b) => a - b);
    for (const t of types) {
      const n = f.parts.filter(p => p.type === t).length, on = !f.offTypes.has(t);
      const thr = numInput(f.thresholds[t] ?? 0.65, 0.05, 0.05, 0.99, v => { f.thresholds[t] = v; });
      $('input', thr).title = 'How closely each part of this kind must match the reference, 0 to 1, for every check of this kind';
      list.append(h('div', { class: 'fp-row' + (on ? '' : ' off'), onpointerenter: () => { f.hotType = t; S.viewer.draw(); }, onpointerleave: () => { f.hotType = null; S.viewer.draw(); } },
        h('div', { class: 'sw' }, sw(on, v => { if (v) { f.offTypes.delete(t); f.parts.forEach((p, i) => { if (p.type === t) f.dropped.delete(i); }); } else f.offTypes.add(t); renderInspector(); S.viewer.draw(); })),
        h('b', { style: { color: FP_PALETTE[t % FP_PALETTE.length] } }, typeName(t)), h('span', { class: 'muted' }, '× ' + n),
        h('input', { class: 'input', placeholder: 'name, e.g. inductor', value: f.names[t] || '', onchange: e => { f.names[t] = e.target.value.trim(); } }), thr));
    }
  } else {
    f.parts.forEach((p, i) => {
      const on = fpKept(i);
      list.append(h('div', { class: 'fp-row txt' + (on ? '' : ' off'), onpointerenter: () => { f.hot = i; S.viewer.draw(); }, onpointerleave: () => { f.hot = null; S.viewer.draw(); } },
        h('div', { class: 'sw' }, sw(on, () => finderToggle(i))),
        h('input', { class: 'input mono', value: p.text || '', title: f.mode === 'ocv' ? 'The text this check must read - correct it if it was misread' : 'The check fails unless this is read here - clear it for text that changes from part to part',
          onchange: e => { p.text = e.target.value; S.viewer.draw(); } }),
        h('input', { class: 'input', placeholder: 'name', value: f.names[i] || '', onchange: e => { f.names[i] = e.target.value.trim(); } }),
        h('span', { class: 'faint', title: 'how sure the reader was' }, fmt(p.score))));
    });
  }
  if (f.parts.length) body.append(h('div', { class: 'section', style: { margin: '10px 2px 6px' } }, text ? 'Lines' : 'Types'), list);
  if (text) body.append(field('Reading confidence for these checks', numInput(f.textConf, 0.05, 0.05, 0.99, v => { f.textConf = v; }), 'Lines read less surely than this are not trusted'));
  else {
    const mo = h('b', { style: { minWidth: '36px', textAlign: 'right' } }, f.match.toFixed(2));
    body.append(field('Find all like this', h('div', { class: 'row', style: { gap: '6px', justifyContent: 'flex-end', flexWrap: 'wrap' } },
      h('button', { class: 'btn sm', disabled: f.busy, onclick: () => startDraw({ finder: 'like', shape: 'rect', label: 'one part: every copy of it is found' }) }, icon('copy', 'sm'), 'Draw one part…'),
      h('span', { class: 'muted' }, 'alike'), h('input', { type: 'range', class: 'range', style: { width: '110px' }, min: 40, max: 95, value: Math.round(f.match * 100), oninput: e => { f.match = e.target.value / 100; mo.textContent = f.match.toFixed(2); } }), mo),
      'For rows of identical parts the network misses: every copy in the area, turned or not', true),
      field('Also check each part sits right', h('div', { class: 'sw' }, sw(f.seated, v => { f.seated = v; })), 'Lifted or tilted parts fail, not only missing ones'));
  }
  body.append(h('div', { class: 'row', style: { marginTop: '12px', flexWrap: 'wrap' } },
    h('button', { class: 'btn sm', disabled: !f.parts.length, onclick: () => { f.dropped = new Set(); f.offTypes = new Set(); renderInspector(); S.viewer.draw(); } }, 'Keep all'),
    h('button', { class: 'btn sm', disabled: !f.parts.length, onclick: () => { f.dropped = new Set(f.parts.map((_, i) => i)); renderInspector(); S.viewer.draw(); } }, 'Drop all'),
    h('span', { class: 'grow' }),
    h('button', { class: 'btn primary', disabled: !keptN || f.busy, onclick: finderAdd }, icon('plus', 'sm'), `Add ${keptN} ${what} check${keptN === 1 ? '' : 's'}`)));
}

// ------------------------------------------------------------- build actions
/* Verify text expects what it reads on the teach picture (Read text: what it
   contains) -- learnt when its box is set; force: also over typed-in text. */
const teachText = guard(async (ids, force) => {
  const ts = side().tools.filter(t => ids.includes(t.id) && (t.type === 'ocv' || t.type === 'ocr'));
  if (!ts.length) return;
  const r = await api('/api/tool/teach_text', { side: side().id, ids: ts.map(t => t.id), force: !!force });
  if (r.taught.length) { S.st = await api('/api/state'); renderChecks(); renderInspector(); renderTop(false); }
  if (r.notes.length) toast(r.notes.join(' · '), !r.taught.length);
  else if (force) toast('Nothing read: fit the boxes round the markings', true);
});
const updateTool = guard(async (id, params) => {
  await api('/api/tool/update', { side: side().id, id, params });
  S.st = await api('/api/state');
  renderChecks(); renderInspector(); renderTop(false);
  if (S.viewer) S.viewer.draw();
  await preview();
});
/* The recipe run on the reference (or last) picture while building. Asked
   for after every change, so: one at a time (changes made meanwhile are
   collected into the next one) and skipped when nothing it depends on changed. */
let previewWanted = false, previewRunning = null;
function previewKey() {
  const s = side();
  return s ? [s.id, S.buildPic, s.ref_version, (sideRes() || {}).img_seq, JSON.stringify(s.tools)].join('|') : '';
}
function preview() {
  if (S.view !== 'build' || !open() || locked()) return Promise.resolve();
  previewWanted = true;
  if (!previewRunning) previewRunning = runPreviews().finally(() => { previewRunning = null; });
  return previewRunning;
}
async function runPreviews() {
  while (previewWanted) {
    previewWanted = false;
    if (S.view !== 'build' || !open()) return;
    const key = previewKey();
    if (S.preview && S.previewKey === key) { if (S.viewer) S.viewer.draw(); continue; }
    const sid = side().id;
    if (S.viewer) S.viewer.busy(true);
    try {
      const r = await api('/api/preview', { side: sid, on: S.buildPic === 'last' ? 'last' : 'reference' });
      if (!side() || side().id !== sid) continue;
      S.preview = r; S.previewKey = key;
    } catch (e) { S.preview = null; S.previewKey = null; if (!/no picture/.test(e.message)) toast(e.message, true); }
    finally { if (S.viewer) S.viewer.busy(false); }
    if (previewWanted) continue; // another change came in: show the newest only
    if (S.viewer) S.viewer.draw();
    renderChecks(); renderInspector();
  }
}
const renameTool = guard(async (id, name) => {
  if (name === undefined) name = await prompt1('Rename check', 'Letters, digits and _.', id);
  if (!name || name === id) return;
  const r = await api('/api/tool/rename', { side: side().id, id, new_id: name });
  S.st = await api('/api/state'); S.sel = r.id; renderChecks(); renderInspector(); preview();
});
const deleteTools = guard(async ids => {
  if (!ids.length) return;
  await api('/api/tool/delete', { side: side().id, ids });
  if (ids.includes(S.sel)) S.sel = null;
  S.picks = new Set();
  S.st = await api('/api/state'); renderChecks(); renderInspector(); renderTop(false); preview();
  toast(`Deleted ${ids.join(', ')} - Ctrl+Z brings ${ids.length === 1 ? 'it' : 'them'} back`);
});
const deleteTool = id => deleteTools([id]);
const duplicateTools = guard(async ids => {
  if (!ids.length) return;
  const r = await api('/api/tool/duplicate', { side: side().id, ids });
  S.st = await api('/api/state');
  select(r.ids[r.ids.length - 1]);
  if (r.ids.length > 1) { S.picks = new Set(r.ids); renderChecks(); S.viewer && S.viewer.draw(); }
  renderTop(false); preview();
  toast(`Duplicated ${ids.join(', ')}`);
});
// several checks changed as one step (one undo)
const updateMany = guard(async ups => {
  if (!ups || !Object.keys(ups).length) return;
  for (const t of side().tools) if (ups[t.id]) for (const [k, v] of Object.entries(ups[t.id])) { if (v == null) delete t.params[k]; else t.params[k] = v; }
  S.viewer && S.viewer.draw();
  await api('/api/tool/update_many', { side: side().id, updates: ups });
  S.st = await api('/api/state');
  renderChecks(); renderInspector(); renderTop(false);
  if (S.viewer) S.viewer.draw();
  await preview();
});
const moveTool = guard(async (id, dir) => {
  const i = side().tools.findIndex(t => t.id === id); if (i < 0) return;
  await api('/api/tool/move', { side: side().id, id, to: Math.max(0, i + dir) });
  S.st = await api('/api/state'); renderChecks(); preview();
});
const undoRedo = guard(async w => { await api('/api/' + w, { side: side().id }); S.st = await api('/api/state'); renderChecks(); renderInspector(); renderTop(false); preview(); });
const save = guard(async () => { await api('/api/save', {}); S.st = await api('/api/state'); if (S.view === 'build') renderChecks(); toast('Saved'); });
const projectSet = guard(async set => { await api('/api/project/set', { side: side().id, set }); S.st = await api('/api/state'); renderChecks(); preview(); });
const setReference = guard(async () => { await api('/api/reference', { side: side().id }); S.st = await api('/api/state'); toast('Reference set from the last picture: save to keep it'); renderMain(); preview(); });
const uploadReference = guard(async e => {
  const f = e.target.files[0]; if (!f) return;
  await api('/api/reference/upload?side=' + encodeURIComponent(side().id), await f.arrayBuffer(), true);
  S.st = await api('/api/state'); S.buildPic = 'reference'; renderMain(); preview(); toast('Reference uploaded: save to keep it');
});
const newSide = guard(async () => {
  const name = await prompt1('New side', 'A name for this view of the part.', 'Side ' + (open().sides.length + 1));
  if (!name) return;
  await api('/api/side/new', { name }); await refresh(); S.side = open().sides.length - 1; S.sel = null; renderMain(); preview();
});

// ------------------------------------------------------------------ history
/* History, as Vision Station's: the parts of this session, then older ones from the run logs (Vision Next’s and Vision Station’s results.db); the production
   report from the results logs (yield, the checks rejecting most, CSV); and
   the saved result pictures, found by barcode, day or job. */
const PERIODS = [[1, 'Last hour'], [8, 'Last 8 hours'], [24, 'Last 24 hours'], [168, 'Last 7 days'], [0, 'Everything']];
function historyView() {
  const v = h('div', { class: 'view history-view' });
  const card = h('div', { class: 'card panel' });
  S.histTab = S.histTab || 'parts';
  const tabs = seg([['parts', 'Parts', 'history'], ['report', 'Report', 'grid'], ['find', 'Find picture', 'search']], S.histTab, t => { S.histTab = t; renderMain(); });
  const head = h('div', { class: 'hist-head' }, h('h2', {}, 'History'), tabs, h('span', { class: 'grow' }));
  if (S.histTab === 'parts') {
    const q = h('input', { class: 'input', placeholder: 'Search barcode, side, check, picture…', value: S.histFilter.q, style: { maxWidth: '320px' },
      oninput: e => { S.histFilter.q = e.target.value; clearTimeout(q._t); q._t = setTimeout(renderHistory, 200); } });
    head.append(h('span', { class: 'chip', id: 'histcount' }, '…'),
      seg([['', 'All'], ['pass', 'OK'], ['fail', 'NG']], S.histFilter.verdict, v => { S.histFilter.verdict = v; renderHistory(); }), q,
      h('button', { class: 'btn ghost', title: 'Start counting again', onclick: guard(async () => { if (await confirm1('Reset the counts and history?', 'Saved pictures and result files are kept.')) { await api('/api/reset_counts', {}); } }) }, icon('refresh', 'sm'), 'Reset'));
    card.append(head, h('div', { class: 'hist-grid', id: 'histgrid' }));
    requestAnimationFrame(renderHistory);
  } else if (S.histTab === 'report') {
    S.histHours = S.histHours ?? 24;
    head.append(h('select', { class: 'input', style: { width: 'auto' }, onchange: e => { S.histHours = +e.target.value; renderReport(); } },
      ...PERIODS.map(([k, t]) => h('option', { value: k, selected: k === S.histHours }, t))),
      h('button', { class: 'btn', onclick: renderReport }, icon('refresh', 'sm'), 'Refresh'),
      h('button', { class: 'btn primary', onclick: exportCsv }, icon('download', 'sm'), 'Export CSV…'));
    card.append(head, h('div', { class: 'panel-body', id: 'report', style: { padding: '4px 18px 18px' } }));
    requestAnimationFrame(renderReport);
  } else {
    const q = h('input', { class: 'input', placeholder: 'Type or scan a barcode, a day (2026-11-24) or a job…', value: S.findQ || '', style: { maxWidth: '420px' },
      oninput: e => { S.findQ = e.target.value; clearTimeout(q._t); q._t = setTimeout(renderFind, 200); } });
    head.append(q, h('span', { class: 'chip', id: 'findcount' }, '…'));
    card.append(head, h('div', { class: 'find-split' }, h('div', { class: 'find-list', id: 'findlist' }), h('div', { class: 'find-view', id: 'findview' },
      h('div', { class: 'empty-note' }, 'Choose a picture to see it here'))));
    requestAnimationFrame(() => { renderFind(); q.focus(); });
  }
  v.append(card);
  return v;
}
const exportCsv = guard(async () => {
  const r = await api('/api/export_csv', { hours: S.histHours || 0 });
  if (!r.files || !r.files.length) return;
  toast(`Exported ${r.files.length} file${r.files.length > 1 ? 's' : ''} to ${r.folder}`);
  api('/api/reveal', { path: r.files[0] }).catch(() => { });
});
const renderReport = guard(async () => {
  const el = $('#report'); if (!el) return;
  const r = await api('/api/report?hours=' + (S.histHours || 0));
  const fmtMs = ms => ms >= 1000 ? (ms / 1000).toFixed(2) + ' s' : Math.round(ms) + ' ms';
  const table = (cols, rows, empty) => rows.length ? h('table', { class: 'tbl' }, h('thead', {}, h('tr', {}, ...cols.map(c => h('th', {}, c)))),
    h('tbody', {}, ...rows.map(row => h('tr', {}, ...row.map(c => h('td', {}, c)))))) : h('div', { class: 'empty-note' }, empty);
  el.replaceChildren(
    h('h3', { class: 'sect-title' }, 'Yield by side'),
    table(['Side', 'Inspected', 'Accepted', 'Yield', 'Avg cycle'], r.sides.map(x => [x.name, x.total, x.passed, x.total ? x.yield_pct.toFixed(1) + ' %' : '–', x.total ? fmtMs(x.avg_ms) : '–']), 'No parts inspected in this period.'),
    h('h3', { class: 'sect-title' }, 'Checks rejecting parts, most first'),
    table(['Side', 'Check', 'Rejected parts'], r.pareto.map(x => [x.side, x.tool, x.count]), 'No rejected parts in this period.'));
});
const renderFind = guard(async () => {
  const list = $('#findlist'); if (!list) return;
  const r = await api('/api/pictures/find?limit=1000&q=' + encodeURIComponent(S.findQ || ''));
  const c = $('#findcount'); if (c) c.textContent = r.total > r.items.length ? `${r.items.length} of ${r.total}` : `${r.total}`;
  list.replaceChildren();
  if (!r.items.length) { list.append(h('div', { class: 'empty-note' }, r.total === 0 && !S.findQ ? `No saved pictures yet in ${(r.roots || [r.root]).join(', ')}. Settings > Result pictures chooses which are kept.` : 'Nothing matches')); return; }
  for (const it of r.items) {
    const row = h('div', { class: 'find-row', title: it.path, onclick: () => { $$('.find-row.on', list).forEach(x => x.classList.remove('on')); row.classList.add('on'); showFound(it); },
      ondblclick: () => api('/api/reveal', { path: it.path }).catch(e => toast(e.message, true)) },
      h('span', { class: 'chip ' + (it.verdict === 'OK' ? 'pass' : it.verdict === 'NG' ? 'fail' : '') }, it.verdict || '–'),
      h('b', {}, it.barcode || it.day), h('span', { class: 'muted' }, `${it.day} ${it.time}`), it.original ? h('span', { class: 'faint' }, 'original') : null);
    list.append(row);
  }
});
function showFound(it) {
  const v = $('#findview'); if (!v) return;
  v.replaceChildren(h('div', { class: 'find-pic' }, h('img', { src: '/api/file?path=' + encodeURIComponent(it.path), alt: '' })),
    h('div', { class: 'row', style: { gap: '8px', marginTop: '10px', flexWrap: 'wrap' } }, h('span', { class: 'muted', style: { flex: 1, minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' } }, it.rel),
      h('button', { class: 'btn sm', onclick: () => api('/api/reveal', { path: it.path, open: true }).catch(e => toast(e.message, true)) }, icon('image', 'sm'), 'Open picture'),
      h('button', { class: 'btn sm', onclick: () => api('/api/reveal', { path: it.path }).catch(e => toast(e.message, true)) }, icon('folder', 'sm'), 'Show in folder')));
}
const renderHistory = guard(async () => {
  const grid = $('#histgrid'); if (!grid) return;
  const r = await api(`/api/history?verdict=${S.histFilter.verdict}&q=${encodeURIComponent(S.histFilter.q)}&limit=200`);
  const c = $('#histcount'); if (c) c.textContent = `${r.items.length} of ${r.total}`;
  grid.replaceChildren();
  if (!r.items.length) { grid.append(h('div', { class: 'empty-note', style: { gridColumn: '1/-1' } }, h('div', { class: 'big' }, icon('history', 'lg')), h('b', {}, 'Nothing here yet'), h('div', {}, 'Every inspection appears here, newest first.'))); return; }
  // older parts' pictures are big saved files: their small copies load only once a card is in view
  const lazy = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    lazy.unobserve(e.target);
    const th = e.target, url = th._src;
    const img = new Image(); img.onload = img.onerror = () => { th.classList.remove('skeleton'); if (img.naturalWidth) th.style.backgroundImage = `url("${url}")`; };
    img.src = url;
  }), { rootMargin: '300px' });
  const today = new Date().toDateString();
  const when = t => {
    const d = new Date(t * 1000);
    return d.toDateString() === today ? d.toLocaleTimeString()
      : d.toLocaleDateString(undefined, { day: 'numeric', month: 'short' }) + ' ' + d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
  };
  r.items.forEach((it, i) => {
    const s0 = it.sides[0] || {};
    const failed = it.sides.flatMap(s => s.failed);
    const logged = it.seq == null;
    const src = logged ? (s0.picture ? '/api/file?thumb=1&path=' + encodeURIComponent(s0.picture) : '') : (s0.thumb ? `/img/hist/${it.seq}/${encodeURIComponent(s0.side)}.jpg` : '');
    const thumb = h('div', { class: 'thumb skeleton' },
      h('span', { class: 'chip ' + (it.verdict === 'PASS' ? 'pass' : 'fail') }, it.verdict === 'PASS' ? 'OK' : 'NG'));
    grid.append(h('div', { class: 'hcard', style: { animationDelay: Math.min(i, 20) * 25 + 'ms' }, title: logged ? `Inspected in ${it.app} (from its run log)` : '',
      onclick: () => logged ? loggedItem(it.log) : historyItem(it.seq) },
      thumb,
      h('div', { class: 'meta' }, h('div', { class: 'row' }, logged ? h('b', {}, when(it.t)) : h('b', {}, '#' + it.seq),
          h('span', { class: 'grow' }), h('span', { class: 'muted', style: { whiteSpace: 'nowrap' } }, logged ? (it.app === 'Vision Station' ? 'Vision Station' : 'earlier') : when(it.t))),
        h('div', { class: 'muted', style: { marginTop: '3px', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' } },
          (it.barcode ? it.barcode + ' · ' : '') + (failed.length ? failed.slice(0, 3).join(', ') + (failed.length > 3 ? '…' : '') : `${Math.round(it.ms)} ms`)))));
    if (!src) thumb.classList.remove('skeleton');
    else if (logged) { thumb._src = src; lazy.observe(thumb); }
    else { thumb.style.backgroundImage = `url("${src}")`; const img = new Image(); img.onload = img.onerror = () => thumb.classList.remove('skeleton'); img.src = src; }
  });
});
/* A part from the run logs (before this session, or inspected in Vision
   Station): its saved picture, with the results already drawn on it. */
const loggedItem = guard(async log => {
  const it = await api(`/api/history/logged?side=${log.side}&t=${log.t}&from=${log.from}`);
  const r = it.result;
  const bad = (r.tools || []).filter(t => t.status === 'fail' || t.status === 'error');
  const file = it.picture;
  const pic = h('div', { style: { position: 'relative', height: '62vh', borderRadius: '18px', overflow: 'hidden', background: 'var(--canvas)', marginTop: '14px', display: 'grid', placeItems: 'center' } },
    file ? h('img', { src: '/api/file?path=' + encodeURIComponent(file), alt: '', style: { maxWidth: '100%', maxHeight: '100%', objectFit: 'contain' } })
      : h('div', { class: 'empty-note' }, 'No picture was saved for this part'));
  layer(h('div', { class: 'dialog', style: { width: 'min(1100px, calc(100vw - 40px))' } },
    h('div', { class: 'row' }, h('h2', { style: { margin: 0 } }, it.name), h('span', { class: 'chip ' + (r.verdict === 'PASS' ? 'pass' : 'fail') }, r.verdict === 'PASS' ? 'OK' : 'NG'),
      h('span', { class: 'muted' }, new Date(r.timestamp * 1000).toLocaleString() + (r.barcode ? ' · ' + r.barcode : '') + (r.from === 'vs' ? ' · Vision Station' : '')),
      h('span', { class: 'grow' }), h('button', { class: 'btn icon ghost', onclick: closeLayer }, icon('x'))),
    pic,
    h('div', { class: 'row', style: { flexWrap: 'wrap', marginTop: '12px', gap: '6px' } },
      h('span', { class: 'chip' }, `${Math.round(r.total_ms)} ms`), r.source ? h('span', { class: 'chip' }, r.source) : null,
      h('span', { class: 'chip' }, `${(r.tools || []).length} checks`),
      ...bad.map(t => h('span', { class: 'chip fail', title: t.message || '' }, t.tool_id + (t.message ? ': ' + t.message.slice(0, 40) : ''))),
      h('span', { class: 'grow' }),
      file ? h('button', { class: 'btn sm', onclick: () => api('/api/reveal', { path: file, open: true }).catch(e => toast(e.message, true)) }, icon('image', 'sm'), 'Open picture') : null,
      file ? h('button', { class: 'btn sm', onclick: () => api('/api/reveal', { path: file }).catch(e => toast(e.message, true)) }, icon('folder', 'sm'), 'Show in folder') : null)));
});
const historyItem = guard(async seq => {
  const it = await api('/api/history/item?seq=' + seq);
  let k = 0;
  const host = h('div', { style: { position: 'relative', height: '62vh', borderRadius: '18px', overflow: 'hidden', background: 'var(--canvas)', marginTop: '14px' } });
  const info = h('div', {});
  const dlg = h('div', { class: 'dialog', style: { width: 'min(1100px, calc(100vw - 40px))' } },
    h('div', { class: 'row' }, h('h2', { style: { margin: 0 } }, `Part #${it.seq}`), h('span', { class: 'chip ' + (it.verdict === 'PASS' ? 'pass' : 'fail') }, it.verdict === 'PASS' ? 'OK' : 'NG'),
      h('span', { class: 'muted' }, new Date(it.t * 1000).toLocaleString() + (it.barcode ? ' · ' + it.barcode : '')), h('span', { class: 'grow' }),
      it.sides.length > 1 ? seg(it.sides.map((s, i) => [String(i), s.side]), '0', v => { k = +v; show(); }) : null,
      h('button', { class: 'btn icon ghost', onclick: closeLayer }, icon('x'))), host, info);
  layer(dlg);
  let viewer;
  const show = () => {
    const s = it.sides[k], r = s.result;
    if (viewer) { viewer.destroy(); host.replaceChildren(); }
    viewer = new Viewer({ size: () => r.image_size, key: () => 'h' + seq + k, url: () => `/img/hist/${seq}/${encodeURIComponent(s.side)}.jpg`, result: () => r,
      tools: () => [], selected: () => null, editable: () => false, names: () => false, select: () => { }, empty: () => h('div', { class: 'empty' }, 'No picture kept') });
    host.append(viewer.el);
    requestAnimationFrame(() => viewer.refresh());
    const bad = r.tools.filter(t => t.status === 'fail' || t.status === 'error');
    const file = (s.files || [])[0];
    info.replaceChildren(h('div', { class: 'row', style: { flexWrap: 'wrap', marginTop: '12px', gap: '6px' } },
      h('span', { class: 'chip' }, `${Math.round(r.total_ms)} ms`), h('span', { class: 'chip' }, r.source), ...bad.map(t => h('span', { class: 'chip fail', title: t.message }, t.tool_id + (t.message ? ': ' + t.message.slice(0, 40) : ''))),
      h('span', { class: 'grow' }),
      file ? h('button', { class: 'btn sm', onclick: () => api('/api/reveal', { path: file, open: true }).catch(e => toast(e.message, true)) }, icon('image', 'sm'), 'Open picture') : null,
      file ? h('button', { class: 'btn sm', onclick: () => api('/api/reveal', { path: file }).catch(e => toast(e.message, true)) }, icon('folder', 'sm'), 'Show in folder') : null,
      file ? h('button', { class: 'btn sm', onclick: guard(async () => { const x = await api('/api/save_picture', { what: 'hist', seq, side: s.side }); if (x.path) toast('Saved ' + x.path); }) }, icon('download', 'sm'), 'Save picture…')
        : h('span', { class: 'faint' }, 'Picture not saved: Settings > Result pictures')));
  };
  show();
});

// --------------------------------------------------------------- dialogs
function layer(el, sheet) {
  closeLayer(true);
  const L = $('#layer');
  const scrim = h('div', { class: 'scrim', onclick: () => closeLayer() });
  L.append(scrim, el);
  const f = el.querySelector('input:not([type=file]), select, textarea'); f && setTimeout(() => f.focus(), 60);
  L._sheet = sheet;
}
function closeLayer(now) {
  const L = $('#layer');
  if (!L.children.length) return;
  if (now) { L.replaceChildren(); return; }
  for (const c of L.children) c.classList.add('out');
  setTimeout(() => L.replaceChildren(), 220);
  if (L._onclose) { const f = L._onclose; L._onclose = null; f(); }
}
function prompt1(title, text, value) {
  return new Promise(res => {
    const inp = h('input', { class: 'input', value: value || '', style: { height: '40px', fontSize: '15px' } });
    const ok = () => { const v = inp.value.trim(); $('#layer')._onclose = null; closeLayer(); res(v); };
    inp.addEventListener('keydown', e => { if (e.key === 'Enter') ok(); });
    layer(h('div', { class: 'dialog' }, h('h2', {}, title), text ? h('p', { class: 'muted', style: { marginTop: 0 } }, text) : null, inp,
      h('div', { class: 'foot' }, h('button', { class: 'btn', onclick: () => closeLayer() }, 'Cancel'), h('button', { class: 'btn primary', onclick: ok }, 'OK'))));
    $('#layer')._onclose = () => res(null);
  });
}
function confirm1(title, text) {
  return new Promise(res => {
    layer(h('div', { class: 'dialog', style: { width: 'min(440px, calc(100vw - 32px))' } }, h('h2', {}, title), text ? h('p', { class: 'muted' }, text) : null,
      h('div', { class: 'foot' }, h('button', { class: 'btn', onclick: () => closeLayer() }, 'Cancel'),
        h('button', { class: 'btn primary', onclick: () => { $('#layer')._onclose = null; closeLayer(); res(true); } }, 'OK'))));
    $('#layer')._onclose = () => res(false);
  });
}
function menu(e, items) {
  closeMenu();
  // null is a dividing line: none first, last or twice in a row (a row left out leaves no gap)
  items = items.filter((x, i, a) => x !== null || (i > 0 && a.slice(0, i).some(y => y !== null) && a[i - 1] !== null && a.slice(i + 1).some(y => y !== null)));
  const m = h('div', { class: 'menu', id: 'menu' });
  fillMenu(m, items);
  document.body.append(m);
  const r = e.currentTarget && e.currentTarget.getBoundingClientRect ? e.currentTarget.getBoundingClientRect() : null;
  let x = e.clientX, y = e.clientY;
  if (r && e.type === 'click') { x = r.left; y = r.bottom + 6; }
  const mr = m.getBoundingClientRect();
  m.style.left = Math.min(x, innerWidth - mr.width - 10) + 'px'; m.style.top = Math.min(y, innerHeight - mr.height - 10) + 'px';
  setTimeout(() => document.addEventListener('pointerdown', closeMenuOutside), 0);
}
// rows: [text, icon, action or [sub rows], class, key, disabled]; null is a line
function fillMenu(m, items) {
  for (const it of items) {
    if (!it) { m.append(h('hr')); continue; }
    const [text, ic, fn, cls, kbd, off] = it;
    if (Array.isArray(fn)) {
      const b = h('button', { class: 'has-sub ' + (cls || ''), disabled: !!off }, icon(ic, 'sm'), text, h('span', { class: 'kbd' }, icon('chev', 'sm')));
      let sub = null;
      const show = () => {
        if (off || (sub && sub.isConnected)) return;
        $$('.menu.sub', m).forEach(x => x.remove());
        sub = h('div', { class: 'menu sub' }); fillMenu(sub, fn); m.append(sub);
        const mr = m.getBoundingClientRect(), sr = sub.getBoundingClientRect();
        sub.style.top = Math.max(-mr.top + 10, Math.min(b.offsetTop - 6, innerHeight - mr.top - sr.height - 10)) + 'px';
        sub.style.left = (mr.right + sr.width + 4 > innerWidth ? -sr.width - 4 : mr.width + 4) + 'px';
      };
      b.addEventListener('pointerenter', show); b.addEventListener('click', show);
      m.append(b);
      continue;
    }
    const b = h('button', { class: cls || '', disabled: !!off, onclick: guard(async () => { closeMenu(); await fn(); }) }, icon(ic, 'sm'), text, kbd ? h('span', { class: 'kbd' }, kbd) : null);
    // a plain row of the main menu closes an open sub-menu
    if (!m.classList.contains('sub')) b.addEventListener('pointerenter', () => $$('.menu.sub', m).forEach(x => x.remove()));
    m.append(b);
  }
}
function closeMenuOutside(e) { if (!e.target.closest('#menu')) closeMenu(); }
function closeMenu() { const m = $('#menu'); m && m.remove(); document.removeEventListener('pointerdown', closeMenuOutside); }
function appMenu(anchor) {
  const ev = { currentTarget: anchor, type: 'click', clientX: 0, clientY: 0 };
  const L = !locked();
  menu(ev, [
    ['All applications', 'home', async () => { await api('/api/close', {}); S.sel = null; S.preview = null; await reload(); }],
    L ? ['New application…', 'plus', newApplication] : null,
    null,
    L ? ['Save all', 'save', save, '', 'Ctrl+S'] : null,
    L ? ['Revert to saved', 'undo', async () => { if (await confirm1('Revert every side to its saved recipe?', 'Unsaved changes are lost.')) { await api('/api/revert', {}); await reload(); } }] : null,
    L && !open().single ? ['Add a side…', 'plus', newSide] : null,
    null,
    ['Export package', 'download', () => { const a = h('a', { href: '/api/package', download: '' }); document.body.append(a); a.click(); a.remove(); toast('Package saved to your Downloads folder'); }],
    L ? ['Import package…', 'pkg', () => { const i = h('input', { type: 'file', accept: '.vnp,.zip,.vsp', onchange: importPackage }); i.click(); }] : null,
    null,
    ['Reset counts', 'refresh', async () => { await api('/api/reset_counts', {}); }],
    L ? ['Settings…', 'gear', () => settingsSheet()] : null,
    L ? ['Check installation…', 'cpu', () => settingsSheet('system')] : null,
    ['Keyboard shortcuts', 'key', shortcutsDialog, '', '?'],
    ['Command palette', 'cmd', palette, '', 'Ctrl+K'],
  ].filter((x, i, a) => x !== null || (a[i - 1] !== null && i > 0)));
}

function addSheet(onPick, title, extra) {
  const pick = typeof onPick === 'function' ? onPick : t => startDraw({ type: t, label: label(t).toLowerCase() });
  const q = h('input', { class: 'input', placeholder: 'What do you want to check?', autofocus: true });
  const list = h('div', {});
  const draw = () => {
    const f = q.value.trim().toLowerCase();
    list.replaceChildren();
    let n = 0;
    if (extra && !f) {
      list.append(h('div', { class: 'picker', style: { marginTop: '14px' } }, h('div', { class: 'pick', onclick: () => { closeLayer(); extra.fn(); } },
        h('div', { class: 'ti' }, icon(extra.icon || 'box')), h('div', {}, h('b', {}, extra.text), h('span', {}, extra.sub || '')))));
    }
    const last = store.get('lastType', null);
    if (last && !f && S.schema.labels[last]) {
      list.append(h('div', { class: 'section', style: { margin: '14px 2px 8px' } }, 'Last used'),
        h('div', { class: 'picker' }, h('div', { class: 'pick', onclick: () => { closeLayer(); pick(last); } },
          h('div', { class: 'ti' }, icon(toolIcon(last))), h('div', {}, h('b', {}, label(last)), h('span', {}, describe(last))))));
    }
    for (const [title, sub, types] of S.schema.questions) {
      const ts = types.filter(t => !f || label(t).toLowerCase().includes(f) || describe(t).toLowerCase().includes(f) || t.includes(f) || title.toLowerCase().includes(f));
      if (!ts.length) continue;
      list.append(h('div', { class: 'section', style: { margin: '18px 2px 4px' } }, title), h('div', { class: 'muted', style: { fontSize: '12.5px', margin: '0 2px 10px' } }, sub));
      const grid = h('div', { class: 'picker' });
      for (const t of ts) {
        grid.append(h('div', { class: 'pick', style: { animationDelay: (n++ * 18) + 'ms' }, onclick: () => { closeLayer(); pick(t); } },
          h('div', { class: 'ti' }, icon(toolIcon(t))), h('div', {}, h('b', {}, label(t)), h('span', {}, describe(t)))));
      }
      list.append(grid);
    }
    if (!list.children.length) list.append(h('div', { class: 'empty-note' }, 'No check matches.'));
  };
  q.addEventListener('input', draw);
  q.addEventListener('keydown', e => { if (e.key === 'Enter') { const p = $('.pick', list); p && p.click(); } });
  draw();
  if (typeof onPick === 'function') $('#layer')._onclose = () => { if (S.viewer) S.viewer.draw(); };
  layer(h('div', { class: 'dialog', style: { width: 'min(780px, calc(100vw - 32px))' } }, h('div', { class: 'row' }, h('h2', { style: { margin: 0 } }, title || 'Add a check'), h('span', { class: 'grow' }),
    h('button', { class: 'btn icon ghost', onclick: () => closeLayer() }, icon('x'))),
    h('p', { class: 'muted', style: { margin: '4px 0 0' } }, 'Pick one, then drag a box round the place on the picture. Every check of Vision Station is here.'),
    h('div', { class: 'picker-search' }, icon('search'), q), list));
}

/* A path box with a Browse button: Windows' own Open dialog, shown by the
   server over the app window. opts: kind 'folder' | 'file', title, filters
   [[name, [ext]]], side (a file then comes relative to that side's folder,
   copied into its models folder when it was elsewhere), onpick(path). */
function pathInput(input, opts) {
  const b = h('button', { class: 'btn sm', type: 'button', title: opts.kind === 'folder' ? 'Choose a folder' : 'Choose a file', onclick: guard(async () => {
    b.disabled = true;
    try {
      const r = await api('/api/browse', { kind: opts.kind, title: opts.title || '', filters: opts.filters || [], current: input.value.trim(), side: opts.side });
      if (!r.path) return;
      input.value = r.path;
      if (r.copied) toast('Copied into this side\'s models folder');
      if (opts.onpick) opts.onpick(r.path); else input.dispatchEvent(new Event('change'));
    } finally { b.disabled = false; }
  }) }, icon('folder', 'sm'), 'Browse…');
  input.style.minWidth = '0';
  return h('div', { class: 'pathrow' }, input, b);
}
function sourceDialog() {
  const s = side();
  let mode = s.source.mode === 'files' ? 'folder' : s.source.mode;
  const folder = h('input', { class: 'input', value: s.source.folder || '', placeholder: 'C:\\pictures\\line 3' });
  const cam = numInput(s.source.camera || 0, 1, 0, 7, () => { }, true);
  const rows = h('div', { style: { marginTop: '14px' } });
  const draw = () => {
    rows.replaceChildren();
    if (mode === 'folder') rows.append(field('Folder of pictures', pathInput(folder, { kind: 'folder', title: 'Folder of pictures', onpick: () => { } }), 'Played in name order, then again from the start', true));
    if (mode === 'camera') rows.append(field('Camera number', cam, 'Addresses and settings: Settings > Cameras'));
    if (mode === 'reference') rows.append(h('div', { class: 'note' }, 'Every run inspects the reference picture: for building checks without a camera.'));
  };
  draw();
  layer(h('div', { class: 'dialog' }, h('h2', {}, `Pictures for ${s.name}`), h('p', { class: 'muted', style: { marginTop: 0 } }, 'Where each inspection takes its picture from.'),
    seg([['camera', 'Camera', 'camera'], ['folder', 'Folder', 'folder'], ['reference', 'Reference', 'star']], mode, v => { mode = v; draw(); }), rows,
    h('div', { class: 'foot' }, h('button', { class: 'btn', onclick: () => closeLayer() }, 'Cancel'),
      h('button', { class: 'btn primary', onclick: guard(async () => {
        await api('/api/source', { side: s.id, mode, folder: folder.value.trim(), camera: parseInt($('input', cam).value, 10) || 0 });
        closeLayer(); S.st = await api('/api/state'); renderMain(); toast('Picture source set: save to keep it');
      }) }, 'Use'))));
}
function unlockDialog() {
  const inp = h('input', { class: 'input', type: 'password', placeholder: 'Supervisor password', style: { height: '40px' } });
  const ok = guard(async () => { await api('/api/operator/unlock', { password: inp.value }); closeLayer(); toast('Unlocked'); });
  inp.addEventListener('keydown', e => { if (e.key === 'Enter') ok(); });
  layer(h('div', { class: 'dialog', style: { width: 'min(420px, calc(100vw - 32px))' } }, h('div', { class: 'row' }, icon('lock', 'lg'), h('h2', { style: { margin: 0 } }, 'Unlock')),
    h('p', { class: 'muted' }, 'Building checks and changing settings needs the supervisor password.'), inp,
    h('div', { class: 'foot' }, h('button', { class: 'btn', onclick: () => closeLayer() }, 'Cancel'), h('button', { class: 'btn primary', onclick: ok }, 'Unlock'))));
}

// ---------------------------------------------------------------- settings
async function settingsSheet(section) {
  S.settings = await api('/api/settings');
  const nav = h('nav', {}, h('h2', {}, 'Settings'));
  const content = h('div', { class: 'content' });
  const sections = [['general', 'General', 'gear'], ['cameras', 'Cameras', 'camera'], ['pictures', 'Result pictures', 'image'], ['line', 'Line communication', 'line'],
    ['light', 'Light', 'light'], ['barcode', 'Barcode', 'barcode'], ['operator', 'Operator mode', 'lock'], ['system', 'This computer', 'cpu']];
  let cur = section || store.get('settingsTab', 'general');
  const show = k => {
    cur = k; store.set('settingsTab', k); for (const b of $$('button', nav)) b.classList.toggle('on', b.dataset.k === k);
    content.replaceChildren(h('button', { class: 'btn icon ghost close', onclick: () => closeLayer() }, icon('x')));
    const keeps = RESTORE_KEEPS[k];
    if (keeps !== undefined && !locked()) content.append(h('button', { class: 'btn sm ghost restore', title: 'Put this page\u2019s settings back as they were when installed', onclick: guard(async () => {
      const title = sections.find(x => x[0] === k)[1];
      if (!await confirm1(`Restore the ${title} settings to their defaults?`, (keeps ? `Kept as they are: ${keeps}. ` : '') + 'This takes effect at once.')) return;
      await api('/api/settings/restore', { section: k }); toast(title + ': defaults restored'); settingsSheet(k);
      if (k === 'general') { S.settings = await api('/api/settings'); applyLook(); }
    }) }, icon('refresh', 'sm'), 'Restore defaults'));
    SETTINGS[k](content); content.scrollTop = 0;
  };
  for (const [k, t, ic] of sections) nav.append(h('button', { 'data-k': k, onclick: () => show(k) }, icon(ic, 'sm'), t));
  layer(h('div', { class: 'sheet' }, nav, content), true);
  show(cur);
}
// what Restore defaults leaves alone on each page (as Vision Station's KEEPS)
const RESTORE_KEEPS = {
  general: '', cameras: 'each camera\u2019s IP address or serial number', pictures: 'the folders pictures are saved in',
  line: 'whether the line is listened for, and its port (the PLC is set up for it)',
  light: 'whether Vision Next controls the light, its COM port and number of channels',
  barcode: 'how the scanner is connected (keyboard or COM port, baud) and the folder',
};
let saveTimer = null;
let pendingSettings = {};
function mergeInto(a, b) {
  for (const [k, v] of Object.entries(b)) {
    if (v && typeof v === 'object' && !Array.isArray(v) && a[k] && typeof a[k] === 'object' && !Array.isArray(a[k])) mergeInto(a[k], v);
    else a[k] = v;
  }
  return a;
}
function saveSettings(patch, quiet, now) {
  mergeInto(pendingSettings, JSON.parse(JSON.stringify(patch)));
  clearTimeout(saveTimer);
  const send = async () => {
    const p = pendingSettings; pendingSettings = {};
    if (!Object.keys(p).length) return;
    await api('/api/settings', p);
    if (!quiet) toast('Saved');
    S.st = await api('/api/state'); applyLook();
  };
  if (now) return guard(send)();
  saveTimer = setTimeout(guard(send), 300);
}
function sect(c, title, lead) { c.append(h('h3', {}, title), lead ? h('p', { class: 'lead' }, lead) : null); }
function bound(sec, key, kind, opts) {
  const s = S.settings[sec], v = s[key];
  const set = x => { s[key] = x; saveSettings({ [sec]: { [key]: x } }, true); if (opts && opts.after) opts.after(x); };
  if (kind === 'bool') return h('div', { class: 'sw' }, sw(!!v, set));
  if (kind === 'num') return numInput(v, opts && opts.step, opts && opts.min, opts && opts.max, set, opts && opts.int);
  if (kind === 'seg') return seg(opts.options, String(v), x => set(opts.int ? +x : x));
  if (kind === 'select') return h('select', { class: 'input', onchange: e => { const o = opts.options.find(x => String(x[0]) === e.target.value); set(o && typeof o[0] === 'number' ? o[0] : e.target.value); } }, ...opts.options.map(([k, t]) => h('option', { value: k, selected: String(v) === String(k) }, t)));
  if (kind === 'color') return h('input', { type: 'color', value: v, style: { width: '60px', height: '34px', border: 0, background: 'none', justifySelf: 'end' }, onchange: e => set(e.target.value) });
  if (kind === 'range') { const out = h('b', { style: { minWidth: '38px', textAlign: 'right' } }, fmt(v)); return h('div', { class: 'row' }, h('input', { type: 'range', class: 'range', min: opts.min, max: opts.max, step: opts.step, value: v, oninput: e => out.textContent = e.target.value, onchange: e => set(+e.target.value) }), out); }
  const inp = h('input', { class: 'input', value: v ?? '', placeholder: opts && opts.ph, onchange: e => set(e.target.value) });
  return opts && opts.browse ? pathInput(inp, { kind: opts.browse, title: opts.title }) : inp;
}
const SETTINGS = {
  general(c) {
    sect(c, 'General', 'How Vision Next looks and runs.');
    const theme = store.get('theme', S.settings.general.theme || 'light');
    c.append(h('div', { class: 'box' },
      field('Appearance', seg([['light', 'Light', 'sun'], ['dark', 'Dark', 'moon'], ['auto', 'Auto']], theme, v => { store.set('theme', v); S.settings.general.theme = v; saveSettings({ general: { theme: v } }, true); swapView(() => { applyLook(); renderTop(false); }); })),
      field('Motion', bound('general', 'motion', 'bool', { after: () => setTimeout(applyLook, 400) }), 'Springs, fades and glides. Off for slow PCs or remote desktops'),
      field('Run sides in parallel', bound('general', 'parallel', 'bool'), 'Every side of a part inspected at once, on all cores'),
      field('Model teaching run', bound('general', 'advanced', 'bool', { after: () => renderTop(false) }), 'Offers a model teaching run beside Run: the side\u2019s own AI model, trained on 16 pictures of a good part, confirms its part and text checks'),
      field('Model size for teaching', bound('general', 'model_size', 'select', { options: [['small', 'YOLO11 small (fastest)'], ['medium', 'YOLO11 medium'], ['large', 'YOLO11 large'], ['xlarge', 'YOLO11 extra large'],
        ['26-small', 'YOLO26 small'], ['26-medium', 'YOLO26 medium'], ['26-large', 'YOLO26 large'], ['26-xlarge', 'YOLO26 extra large']] }), 'Bigger learns more, trains longer and runs slower: go bigger only if small misses parts'),
      field('Run AI models on', bound('general', 'device', 'seg', { options: [['cpu', 'Processor'], ['gpu', 'Graphics card']] }), 'Text reading, YOLO, classifiers. The processor serves many checks at once and is usually faster here. Takes effect after a restart')));
  },
  cameras(c) {
    sect(c, 'Cameras', 'Basler cameras through pylon. A camera discovery cannot see is opened by its address. Empty fields leave the camera\'s own value.');
    const fw = h('div', {}); c.append(fw);
    api('/api/system').then(s => { const x = s.checks.find(k => k.fix === 'firewall'); if (x) fw.replaceChildren(h('div', { class: 'note warn' }, s.firewall_explanation, h('div', { style: { marginTop: '8px' } },
      h('button', { class: 'btn sm primary', onclick: allowFirewall }, icon('lock', 'sm'), 'Allow through Windows Firewall…')))); }).catch(() => { });
    const cams = S.settings.cameras.length ? S.settings.cameras : [{ ip: '' }];
    const saveC = () => saveSettings({ cameras: cams }, true);
    cams.forEach((cam, i) => {
      const f = (k, kind, opts) => {
        const v = cam[k];
        const set = x => { cam[k] = x === '' ? null : x; saveC(); };
        if (kind === 'num') { const inp = numInput(v ?? '', opts && opts.step, opts && opts.min, opts && opts.max, set, opts && opts.int, true); $('input', inp).placeholder = 'camera\'s own'; return inp; }
        if (kind === 'seg') return seg(opts.options, String(v ?? opts.options[0][0]), x => set(x === '' ? null : opts.int ? +x : x));
        if (kind === 'bool') return h('div', { class: 'sw' }, sw(!!v, set));
        if (kind === 'select') return h('select', { class: 'input', onchange: e => set(e.target.value || null) }, ...opts.options.map(([k2, t]) => h('option', { value: k2, selected: (v || '') === k2 }, t)));
        return h('input', { class: 'input', value: v ?? '', placeholder: opts && opts.ph, onchange: e => set(e.target.value.trim()) });
      };
      const status = h('div', { class: 'muted', style: { fontSize: '12.5px' } });
      const leave = ['', 'Leave'];
      c.append(h('div', { class: 'box' },
        h('div', { class: 'row', style: { padding: '12px 0 4px' } }, h('b', {}, `Camera ${i}`), h('span', { class: 'grow' }),
          h('button', { class: 'btn sm', onclick: guard(async () => { const list = await api('/api/camera/list', { ip: cam.ip || '' }); menu({ currentTarget: status, type: 'click' }, list.length ? list.map(d => [`${d.model || d.name}  ${d.ip || d.serial}`, 'camera', () => { cam.ip = d.ip || ''; saveC(); settingsSheet('cameras'); }]) : [['No camera found', 'info', () => { }]]); }) }, icon('search', 'sm'), 'Find'),
          h('button', { class: 'btn sm', onclick: guard(async () => { await api('/api/settings', { cameras: cams }); const r = await api('/api/camera/test', { camera: i }); status.textContent = `Picture ${r.size[0]} × ${r.size[1]} in ${r.ms} ms`; }) }, icon('camera', 'sm'), 'Test'),
          h('button', { class: 'btn sm', onclick: guard(async () => { const v = await api('/api/camera/values?camera=' + i); status.replaceChildren(h('div', { class: 'kv', style: { marginTop: '6px' } }, ...Object.entries(v).filter(x => x[1]).flatMap(([k, x]) => [h('span', {}, k.replace(/_/g, ' ')), h('span', {}, x)]))); }) }, icon('info', 'sm'), 'Now')),
        status,
        field('Address (GigE) or serial', f('ip', 'text', { ph: '192.168.3.6 — empty: the first found' })),
        (() => {
          // what the camera comes back with after a power cycle: a stored address
          // that does not fit the network leaves it unreachable after every restart
          const info = h('div', { class: 'help' }, 'Check to see the address the camera keeps after a restart');
          const keep = h('button', { class: 'btn sm', disabled: true, title: 'Store the camera’s current address and mask in the camera, so it is reachable after a power cycle', onclick: guard(async () => {
            const r = await api('/api/camera/keep_ip', { camera: i }); showIp(r.ip); toast('Address kept in the camera');
          }) }, icon('lock', 'sm'), 'Keep this address after restart');
          const showIp = ip => {
            if (!ip) { info.textContent = 'This camera has no IP settings (not a GigE camera).'; keep.disabled = true; info.className = 'help'; return; }
            const now = `${ip.current_ip} / ${ip.current_mask}`;
            const kept = ip.stored_on ? `${ip.stored_ip} / ${ip.stored_mask}` : 'none (it falls back to an automatic address)';
            info.textContent = ip.keeps_current ? `Now ${now} - kept after a restart.` : `Now ${now}, but after a restart: ${kept}. The camera will not be reachable after a power cycle - press Keep this address.`;
            info.className = 'help' + (ip.keeps_current ? '' : ' warn-text');
            keep.disabled = ip.keeps_current;
          };
          const check = h('button', { class: 'btn sm', onclick: guard(async () => { await api('/api/settings', { cameras: cams }); const r = await api('/api/camera/ip?camera=' + i); showIp(r.ip); }) }, icon('search', 'sm'), 'Check');
          return field('After a restart', h('div', { class: 'row', style: { gap: '6px', justifyContent: 'flex-end', flexWrap: 'wrap' } }, check, keep), info, true);
        })(),
        field('Exposure (µs)', f('exposure_us', 'num', { step: 100, min: 10 })),
        field('Exposure auto', f('exposure_auto', 'seg', { options: [leave, ['Off', 'Off'], ['Once', 'Once'], ['Continuous', 'Continuous']] }), 'Continuous auto exposure makes every picture different: off for inspection'),
        field('Gain', f('gain', 'num', { step: 1, min: 0 })),
        field('Gain auto', f('gain_auto', 'seg', { options: [leave, ['Off', 'Off'], ['Once', 'Once'], ['Continuous', 'Continuous']] })),
        field('White balance', f('white_balance', 'seg', { options: [leave, ['Off', 'Off'], ['Once', 'Once'], ['Continuous', 'Continuous']] }), 'Once: balance on a white sheet, then it stays'),
        field('Balance now', h('button', { class: 'btn sm', title: 'Point the camera at something white or grey and press: the balance is held from then on', onclick: guard(async () => {
          await api('/api/settings', { cameras: cams }); const r = await api('/api/camera/balance_once', { camera: i }); toast('White balanced' + (r.white_balance ? ' (now ' + r.white_balance + ')' : ''));
        }) }, icon('palette', 'sm'), 'Balance Once'), 'With the light on, over something white or grey; turn Continuous off once the lighting is fixed'),
        field('Pixel format', f('pixel_format', 'select', { options: [['', 'Leave'], ['BayerRG8', 'Bayer RG 8 (colour)'], ['BayerBG8', 'Bayer BG 8'], ['Mono8', 'Mono 8'], ['RGB8Packed', 'RGB 8'], ['BGR8Packed', 'BGR 8']] })),
        field('Gamma', f('gamma', 'num', { step: 0.1, min: 0 })),
        field('Black level', f('black_level', 'num', { step: 1, min: 0 })),
        field('Trigger', f('trigger', 'seg', { options: [leave, ['Off', 'Free run'], ['On', 'Triggered']] }), 'Triggered: each picture waits for the line\'s signal'),
        field('Trigger input', f('trigger_source', 'select', { options: [['', 'Leave'], ['Line1', 'Line 1'], ['Line2', 'Line 2'], ['Line3', 'Line 3'], ['Software', 'Software']] })),
        field('Trigger edge', f('trigger_edge', 'seg', { options: [leave, ['RisingEdge', 'Rising'], ['FallingEdge', 'Falling']] })),
        field('Trigger delay (µs)', f('trigger_delay_us', 'num', { step: 100, min: 0 })),
        field('Frame rate limit (0 = off)', f('frame_rate', 'num', { step: 1, min: 0 })),
        field('Rotation', f('rotation', 'seg', { options: [['0', '0°'], ['90', '90°'], ['180', '180°'], ['270', '270°']], int: true })),
        field('Mirror left-right', f('mirror_x', 'bool')), field('Mirror top-bottom', f('mirror_y', 'bool')),
        field('Packet size', f('packet_size', 'num', { step: 500, min: 576, int: true }), 'GigE: 8192 with jumbo frames on the network card'),
        field('Inter-packet delay', f('inter_packet_delay', 'num', { step: 100, min: 0, int: true })),
        field('Heartbeat timeout (ms)', f('heartbeat_ms', 'num', { step: 500, min: 500, int: true })),
        field('Picture timeout (ms)', f('timeout_ms', 'num', { step: 500, min: 200, int: true })),
        cams.length > 1 ? h('div', { style: { padding: '10px 0' } }, h('button', { class: 'btn sm ghost danger', onclick: () => { cams.splice(i, 1); saveC(); settingsSheet('cameras'); } }, icon('trash', 'sm'), 'Remove this camera')) : null));
    });
    c.append(h('button', { class: 'btn', onclick: () => { cams.push({ ip: '' }); S.settings.cameras = cams; saveC(); settingsSheet('cameras'); } }, icon('plus'), 'Add a camera'));
  },
  pictures(c) {
    sect(c, 'Result pictures', 'Pictures saved after each inspection, with the results drawn on them. Kept in a folder per day.');
    const R = 'results';
    if (S.settings.results.from_vs) c.append(h('div', { class: 'note' }, `Following Vision Station’s Settings > Results, so every part’s picture is in one place (${S.settings.results.folder}). Changing anything here makes Vision Next keep its own.`));
    c.append(h('div', { class: 'box' },
      field('Save', bound(R, 'which', 'seg', { options: [['none', 'None'], ['fail', 'Rejected'], ['pass', 'Accepted'], ['all', 'All']] })),
      field('Folder', bound(R, 'folder', 'text', { ph: 'empty: projects\\_results', browse: 'folder', title: 'Folder for result pictures' }), null, true),
      field('Subfolder', bound(R, 'subfolder', 'text', { ph: 'e.g. line 3' })),
      field('Annotated picture', bound(R, 'annotated', 'bool'), 'Outlines, OK / NG and what each check read or measured'),
      field('Results in words', bound(R, 'labels', 'bool')),
      field('Verdict banner', bound(R, 'banner', 'bool')),
      field('Original picture too', bound(R, 'original', 'bool')),
      field('File format', bound(R, 'format', 'seg', { options: [['jpg', 'JPEG'], ['png', 'PNG']] })),
      field('JPEG quality', bound(R, 'quality', 'range', { min: 50, max: 100, step: 1 })),
      field('OK colour', bound(R, 'ok_colour', 'color')),
      field('NG colour', bound(R, 'ng_colour', 'color')),
      field('Line thickness', bound(R, 'line_scale', 'range', { min: 0.5, max: 3, step: 0.1 }))));
  },
  line(c) {
    sect(c, 'Line communication', 'A PLC or robot talks to Vision Next over TCP/IP in text lines.');
    c.append(h('div', { class: 'box' },
      field('Listen for the line', bound('line', 'enabled', 'bool')),
      field('Port', bound('line', 'port', 'num', { step: 1, min: 1, max: 65535, int: true })),
      field('Send every result to all clients', bound('line', 'broadcast', 'bool'), 'For controllers that only listen'),
      field('State', h('span', { class: 'muted', style: { textAlign: 'right' } }, S.stats ? S.stats.line + (S.stats.clients ? ` · ${S.stats.clients} connected` : '') : ''))),
      h('div', { class: 'note mono', style: { whiteSpace: 'pre-wrap', lineHeight: 1.6 } },
        'TRIGGER            inspect every job → RESULT <job> PASS|FAIL|ERROR <ms> <failed> … DONE <verdict> <n>\nTRIGGER <job>      inspect one job\nGET <job>.<tool>.<value>  → VALUE <v>\nSTATUS             → READY | BUSY\nCODE <barcode>     → CODE OK <code> | CODE NG <why>\nPING → PONG    VERSION → VISIONSTATION 1'));
  },
  async light(c) {
    sect(c, 'Light', 'The light controller on a serial port (HIFLY DCS and OPT-protocol controllers, 9600 baud).');
    const ports = await api('/api/serial/ports').catch(() => []);
    const L = 'light', cfg = S.settings.light;
    const chans = () => cfg.channels || 1;
    // what the controller does now, and buttons that act at once
    const state = h('span', { class: 'chip' }, '…'), warn = h('div', {});
    const sliders = h('div', {});
    const show = r => {
      state.className = 'chip ' + (r.state === 'on' ? 'pass' : r.state === 'off' || r.state === 'unknown' ? '' : 'fail');
      state.textContent = r.state === 'on' ? 'On' : r.state === 'off' ? 'Off' : r.state === 'unknown' ? 'Not switched yet' : r.state;
      if (r.levels && r.levels.length) drawSliders(r.levels);
      const low = (r.levels || []).some(v => v < 15);
      warn.replaceChildren(low ? h('div', { class: 'note warn' }, `The brightness is very low (${r.levels.join(', ')} of 255): the light looks off even when on. Raise it with the slider.`) : '');
    };
    const act = guard(async (action, extra) => {
      if (!cfg.port) { toast('Choose the serial port first', true); return; }
      await saveSettings({}, true, true);
      show(await api('/api/light', { action, ...(extra || {}) }));
    });
    const drawSliders = levels => {
      sliders.replaceChildren();
      for (let ch = 0; ch < chans(); ch++) {
        const v = levels[ch] ?? 0;
        const out = h('b', { style: { minWidth: '64px', textAlign: 'right' } }, `${v} · ${Math.round(v / 2.55)}%`);
        sliders.append(field(`Brightness, channel ${ch + 1}`, h('div', { class: 'row', style: { width: '260px' } }, h('input', { type: 'range', class: 'range', min: 0, max: 255, value: v,
          oninput: e => out.textContent = `${e.target.value} · ${Math.round(e.target.value / 2.55)}%`,
          onchange: e => {
            const lv = (cfg.brightness || [255, 255, 255, 255]).slice(); lv[ch] = +e.target.value; cfg.brightness = lv; cfg.set_brightness = true;
            saveSettings({ light: { brightness: lv, set_brightness: true } }, true);
            act('set', { channel: ch + 1, value: lv[ch] });
          } }), out), null, false));
      }
    };
    drawSliders(cfg.brightness || [255, 255, 255, 255]);
    const live = h('div', { class: 'box' },
      h('div', { class: 'row', style: { padding: '12px 0' } }, h('b', {}, 'The light now'), state, h('span', { class: 'grow' }),
        h('button', { class: 'btn primary sm', onclick: () => act('on') }, icon('light', 'sm'), 'On'),
        h('button', { class: 'btn sm', onclick: () => act('off') }, 'Off'),
        h('button', { class: 'btn sm ghost', onclick: () => act('read'), title: 'Read the brightness from the controller' }, icon('refresh', 'sm'))),
      warn, sliders);
    const portSel = bound(L, 'port', 'select', { options: [['', 'Choose…'], ...ports.map(p => [p, p]), ...(cfg.port && !ports.includes(cfg.port) ? [[cfg.port, cfg.port + ' (not present)']] : [])],
      after: () => { saveSettings({}, true, true); setTimeout(() => act('read'), 400); } });
    c.append(h('div', { class: 'box' },
      field('Serial port', portSel),
      field('Channels', bound(L, 'channels', 'seg', { options: [['1', '1'], ['2', '2'], ['3', '3'], ['4', '4']], int: true, after: () => setTimeout(() => settingsSheet('light'), 350) }))),
      live,
      h('div', { class: 'box' },
        field('Control the light during inspection', bound(L, 'enabled', 'bool'), 'On: Vision Next switches the light for each picture as set below'),
        field('Always on', bound(L, 'always_on', 'bool'), 'Off: on only while a picture is taken and during continuous runs'),
        field('On before the picture (ms)', bound(L, 'on_before_ms', 'num', { step: 10, min: 0, int: true })),
        field('Off after the picture (ms)', bound(L, 'off_after_ms', 'num', { step: 10, min: 0, int: true })),
        field('Set the brightness at start', bound(L, 'set_brightness', 'bool'), 'Off: the brightness stored in the controller stays')));
    // automatic brightness from the camera's pictures, as Vision Station's
    const adjOut = h('div', { class: 'help' }, 'Finds the level that gives the picture the brightness of the side’s teach picture (or the fixed share below). Put a good part under the camera first.');
    const camSides = open() ? open().sides.filter(x => x.source.mode === 'camera') : [];
    const levels = cfg.job_levels || {};
    const own = Object.entries(levels).filter(([k]) => open() && open().sides.some(x => x.id === k));
    c.append(h('div', { class: 'box' },
      h('div', { class: 'row', style: { padding: '12px 0 4px' } }, h('b', {}, 'Automatic brightness')),
      field('Adjust now', h('div', { class: 'row', style: { gap: '6px', justifyContent: 'flex-end' } },
        h('button', { class: 'btn sm primary', disabled: !camSides.length, onclick: guard(async e => {
          const b = e.currentTarget; b.disabled = true; adjOut.textContent = 'Adjusting… (about ten pictures)';
          try {
            await saveSettings({}, true, true);
            const sd = camSides.find(x => x.id === side().id) || camSides[0];
            const r = await api('/api/light/adjust', { side: sd.id });
            cfg.brightness = [r.level, r.level, r.level, r.level]; cfg.set_brightness = true;
            adjOut.textContent = `Level ${r.level}: ${r.grey_pct.toFixed(0)} % grey (aim ${r.target_pct.toFixed(0)} %), ${r.white_pct.toFixed(1)} % white, after ${r.tries.length} pictures. Kept as the Brightness.`;
            drawSliders(cfg.brightness);
          } catch (err) { adjOut.textContent = 'Not adjusted: ' + err.message; } finally { b.disabled = false; }
        }) }, icon('light', 'sm'), 'Adjust now')), camSides.length ? null : 'Only for a side whose pictures come from a camera', true),
      adjOut,
      field('Keep adjusted', bound(L, 'auto', 'bool'), 'After every part: the level for the next one, from this part’s picture - so the pictures stay as bright as when the checks were taught'),
      field('Aim for', bound(L, 'target', 'seg', { options: [['reference', 'Teach picture'], ['fixed', 'Fixed grey']] })),
      field('Fixed grey (% of white)', bound(L, 'target_percent', 'num', { step: 1, min: 2, max: 95, int: true })),
      own.length ? field('Kept for each side', h('div', { style: { display: 'grid', gap: '6px', justifyItems: 'end' } }, ...own.map(([k, v]) => h('div', { class: 'row', style: { gap: '8px' } },
        h('span', {}, `${(open().sides.find(x => x.id === k) || {}).name || k}: level ${v}`),
        h('button', { class: 'btn sm ghost', title: 'Start again from the side’s own level', onclick: guard(async () => { await api('/api/light/forget', { side: k }); delete levels[k]; settingsSheet('light'); }) }, 'Forget')))), null, true) : null));
    if (cfg.port) act('read'); else show({ state: 'no port', levels: [] });
  },
  async barcode(c) {
    sect(c, 'Barcode', 'Inspect a part only after its barcode is scanned and found good; its pictures go in a folder named after the code.');
    const ports = await api('/api/serial/ports').catch(() => []);
    const B = 'barcode';
    c.append(h('div', { class: 'box' },
      field('Barcode-gated inspection', bound(B, 'enabled', 'bool')),
      field('Code starts with', bound(B, 'prefix', 'text', { ph: 'e.g. FIC' })),
      field('Length (0 = any)', bound(B, 'length', 'num', { step: 1, min: 0, int: true })),
      field('Match upper and lower case', bound(B, 'match_case', 'bool')),
      field('Inspect as soon as a good code is scanned', bound(B, 'auto_run', 'bool')),
      field('Scanner', bound(B, 'source', 'seg', { options: [['keyboard', 'USB (keyboard)'], ['serial', 'Serial port']] })),
      field('Serial port', bound(B, 'port', 'select', { options: [['', 'Choose…'], ...ports.map(p => [p, p])] })),
      field('Baud', bound(B, 'baud', 'select', { options: [[9600, '9600'], [19200, '19200'], [38400, '38400'], [57600, '57600'], [115200, '115200']] })),
      field('Folder for barcode folders', bound(B, 'folder', 'text', { ph: 'empty: the result pictures folder', browse: 'folder', title: 'Folder for barcode folders' }), null, true)));
  },
  operator(c) {
    sect(c, 'Operator mode', 'Locked, the line can run and watch but not build checks or change settings.');
    const set = !!S.settings.operator.password_set;
    const old = h('input', { class: 'input', type: 'password', placeholder: 'Present password' }), nw = h('input', { class: 'input', type: 'password', placeholder: 'New password (empty: none)' });
    c.append(h('div', { class: 'box' },
      field('Start locked', bound('operator', 'enabled', 'bool'), 'Vision Next opens in operator mode; a lock button appears at the top'),
      set ? field('Present password', old) : null, field(set ? 'New password' : 'Supervisor password', nw)),
      h('div', { class: 'row' }, h('button', { class: 'btn primary', onclick: guard(async () => { await api('/api/operator/password', { old: old.value, new: nw.value }); toast('Password saved'); S.settings = await api('/api/settings'); }) }, icon('key', 'sm'), 'Save password'),
        h('button', { class: 'btn', onclick: guard(async () => { closeLayer(); await api('/api/operator/lock', {}); }) }, icon('lock', 'sm'), 'Lock now')));
  },
  async system(c) {
    sect(c, 'This computer', 'What Vision Next uses here.');
    const s = await api('/api/system');
    const st = s.stats;
    c.append(h('div', { class: 'kpis' }, h('div', { class: 'kpi' }, h('b', {}, st.rss_mb + ' MB'), h('span', {}, 'memory now')),
      h('div', { class: 'kpi' }, h('b', {}, st.peak_mb + ' MB'), h('span', {}, 'memory peak')), h('div', { class: 'kpi' }, h('b', {}, s.cores), h('span', {}, 'cores'))));
    const box = h('div', { class: 'box' });
    for (const x of s.checks) {
      const chip = h('span', { class: 'chip ' + (x.ok ? 'pass' : 'fail'), style: { justifySelf: 'end' } }, x.ok ? 'ready' : 'missing');
      const fix = x.fix === 'firewall' ? h('button', { class: 'btn sm primary', title: s.firewall_explanation, onclick: allowFirewall }, icon('lock', 'sm'), 'Allow through Windows Firewall…') : null;
      box.append(field(x.what, fix ? h('div', { class: 'row', style: { gap: '8px', justifyContent: 'flex-end' } }, fix, chip) : chip, x.note));
    }
    c.append(h('div', { class: 'row', style: { justifyContent: 'flex-end', margin: '0 0 8px' } }, h('button', { class: 'btn sm', onclick: () => settingsSheet('system') }, icon('refresh', 'sm'), 'Check again')),
      box, h('div', { class: 'section' }, 'Folders'), h('div', { class: 'note mono' }, s.projects),
      h('div', { class: 'section' }, 'Keys'), keysTable());
  },
};

// every key, for This computer and the Keyboard shortcuts window
const KEYS = [
  ['Run', [['Space / F5', 'Run once'], ['F6', 'Continuous on / off'], ['1  2  3', 'Run, Build, History']]],
  ['Picture', [['Wheel', 'Zoom about the pointer'], ['Middle button drag', 'Move the picture'], ['Ctrl+0  /  F', 'Fit to window'], ['Ctrl+1', 'Actual size'],
    ['Ctrl+2', 'Zoom to the selection'], ['Ctrl+ +  /  Ctrl+ -', 'Zoom in / out'], ['Ctrl+Shift+S', 'Save the picture on screen']]],
  ['Regions (Build, on the reference)', [['Click, Ctrl / Shift+click', 'Select, add to the selection'], ['Drag on empty picture', 'Select every region it touches'],
    ['Ctrl+A', 'Select all regions'], ['Arrows  (Shift: 10 px)', 'Move the selection'], ['[  ]  (Shift: 15°)', 'Turn the selection a degree'],
    ['R  E  P', 'Draw a box, ellipse, polygon'], ['Enter', 'Finish a polygon'], ['Right-click', 'Duplicate, rotate, align, lock…']]],
  ['Checks', [['A', 'Add a check'], ['Ctrl+D', 'Duplicate'], ['Delete / Backspace', 'Delete'], ['F2', 'Rename'], ['Ctrl+Z / Ctrl+Y', 'Undo / redo'], ['Ctrl+S', 'Save']]],
  ['Anywhere', [['Ctrl+K', 'Command palette'], ['?', 'This list'], ['Esc', 'Cancel, close, clear the selection']]],
];
function keysTable() {
  return h('div', {}, ...KEYS.map(([group, rows]) => h('div', {}, h('div', { class: 'section', style: { margin: '12px 2px 6px' } }, group),
    h('div', { class: 'kv' }, ...rows.flatMap(([k, v]) => [h('span', {}, v), h('span', { class: 'kbd' }, k)])))));
}
function shortcutsDialog() {
  layer(h('div', { class: 'dialog', style: { width: 'min(620px, calc(100vw - 32px))' } },
    h('div', { class: 'row' }, h('h2', { style: { margin: 0 } }, 'Keyboard shortcuts'), h('span', { class: 'grow' }), h('button', { class: 'btn icon ghost', onclick: () => closeLayer() }, icon('x'))),
    h('div', { style: { maxHeight: '70vh', overflow: 'auto', marginTop: '6px' } }, keysTable())));
}
const allowFirewall = guard(async () => {
  toast('Windows asks for an administrator: answer Yes');
  const r = await api('/api/firewall/allow', {});
  toast(r.ok ? 'Allowed: the camera\u2019s pictures now reach Vision Next' : 'Not allowed - the administrator prompt was declined', !r.ok);
  if ($('#layer .sheet')) settingsSheet(store.get('settingsTab', 'system'));
});

// ----------------------------------------------------------- command palette
function palette() {
  const items = [
    ['Run once', 'play', runOnce], ['Continuous on/off', 'loop', toggleContinuous],
    ...(open() ? [['Run view', 'play', () => setView('run')], ...(locked() ? [] : [['Build view', 'build', () => setView('build')]]), ['History', 'history', () => setView('history')]] : []),
    ...(open() && !locked() ? [['Find parts and text (AI)…', 'search', () => { setView('build'); setTimeout(() => openFinder(), 60); }], ['Add a check…', 'plus', () => { setView('build'); setTimeout(addSheet, 50); }], ['Save all', 'save', save], ['Picture source…', 'camera', sourceDialog]] : []),
    ...(locked() ? [] : [['Settings', 'gear', () => settingsSheet()], ['Settings: cameras', 'camera', () => settingsSheet('cameras')], ['Settings: light', 'light', () => settingsSheet('light')]]),
    ['Light or dark', 'moon', () => { store.set('theme', document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'); swapView(() => { applyLook(); renderTop(false); }); }],
    ['Reset counts', 'refresh', () => api('/api/reset_counts', {})],
    ['Keyboard shortcuts', 'key', shortcutsDialog], ...(locked() ? [] : [['Check installation', 'cpu', () => settingsSheet('system')]]),
    ...S.st.apps.map(a => [`Open ${a.name}`, 'target', async () => { await api('/api/open', { app: a.id }); S.side = 0; S.sel = null; await reload(); }]),
    ...(open() ? side().tools.map(t => [`Check ${t.id}`, toolIcon(t.type), () => { if (S.view === 'run') selectRun(t.id); else { setView('build'); setTimeout(() => select(t.id), 60); } }]) : []),
  ];
  const inp = h('input', { placeholder: 'Type a command, an application or a check…' });
  const list = h('div', { class: 'list' });
  let k = 0, shown = [];
  const draw = () => {
    const f = inp.value.toLowerCase();
    shown = items.filter(i => i[0].toLowerCase().includes(f)).slice(0, 40);
    k = clamp(k, 0, shown.length - 1);
    list.replaceChildren(...shown.map((it, i) => h('div', { class: 'item' + (i === k ? ' on' : ''), onclick: () => go(i) }, icon(it[1], 'sm'), it[0])));
  };
  const go = i => { const it = shown[i]; closeLayer(); if (it) guard(it[2])(); };
  inp.addEventListener('input', () => { k = 0; draw(); });
  inp.addEventListener('keydown', e => {
    if (e.key === 'ArrowDown') { k = Math.min(shown.length - 1, k + 1); draw(); e.preventDefault(); }
    if (e.key === 'ArrowUp') { k = Math.max(0, k - 1); draw(); e.preventDefault(); }
    if (e.key === 'Enter') go(k);
  });
  draw();
  layer(h('div', { class: 'palette' }, inp, list));
}

// ------------------------------------------------------------------ footer
function renderFoot() {
  const f = $('#foot');
  f.replaceChildren();
  const spark = h('div', { class: 'spark', title: 'Last parts, newest on the right' });
  const items = (S.res && S.res.history || []).slice().reverse();
  items.forEach((x, i) => spark.append(h('i', { class: x.verdict, style: { height: clamp(4 + x.ms / 40, 4, 18) + 'px', animationDelay: (i === items.length - 1 ? 0 : 0) + 'ms' }, title: `#${x.seq} ${x.verdict} · ${Math.round(x.ms)} ms` })));
  f.append(spark, h('span', { class: 'grow' }));
  if (S.st && S.st.note) f.append(h('span', { class: 'foot-item' }, S.st.note));
  if (S.stats) {
    if (S.stats.light && S.stats.light !== 'off') f.append(h('span', { class: 'foot-item' }, icon('light', 'sm'), S.stats.light));
    if (S.stats.line && S.stats.line !== 'off') f.append(h('span', { class: 'foot-item' }, h('span', { class: 'dot live' }), S.stats.line));
    f.append(h('span', { class: 'foot-item' }, icon('cpu', 'sm'), `${S.stats.rss_mb} MB`), h('span', { class: 'foot-item' }, S.stats.inference.startsWith('off') ? 'AI off' : 'AI on ' + S.stats.inference));
  }
  f.append(h('span', { class: 'foot-item faint' }, 'v' + (S.st ? S.st.version : '')));
}

// --------------------------------------------------------------- keyboard
let scanBuf = '', scanT = 0;
function bindKeys() {
  window.addEventListener('contextmenu', e => {
    if (!e.defaultPrevented && !/INPUT|TEXTAREA/.test(e.target.tagName)) e.preventDefault();
  });
  window.addEventListener('keydown', e => {
    const typing = /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName);
    const L = $('#layer');
    if (e.key === 'Escape') {
      if ($('#menu')) closeMenu(); else if (L.children.length) closeLayer();
      else if (S.draw) cancelDraw(); else if (S.find && S.view === 'build') closeFinder(); else if (S.sel) { S.sel = null; S.picks = new Set(); if (S.view === 'build') { renderChecks(); renderInspector(); } else updateRunPanel(false); S.viewer && S.viewer.draw(); }
      return;
    }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); palette(); return; }
    if (S.draw && S.draw.shape === 'polygon' && e.key === 'Enter') { finishPolygon(); return; }
    // a USB barcode scanner types fast and ends with Enter
    if (!typing && S.st && S.st.barcode_keyboard && e.key.length === 1 && !e.ctrlKey) {
      const now = performance.now(); if (now - scanT > 80) scanBuf = ''; scanBuf += e.key; scanT = now;
    }
    if (!typing && S.st && S.st.barcode_keyboard && e.key === 'Enter' && scanBuf.length >= 3 && performance.now() - scanT < 80) {
      const code = scanBuf; scanBuf = ''; guard(async () => { const r = await api('/api/barcode/scan', { code }); if (!r.ok) toast(r.why, true); else toast('Barcode ' + r.code); })(); return;
    }
    if (typing || L.children.length || !open()) return;
    const ctrl = e.ctrlKey || e.metaKey;
    if (ctrl && !e.shiftKey && e.key.toLowerCase() === 's') { e.preventDefault(); if (!locked()) save(); return; }
    if (ctrl && e.key.toLowerCase() === 'z' && S.draw && S.draw.shape === 'polygon' && (S.draw.pts || []).length) {
      // while tracing: the last point placed, not the recipe's last change
      e.preventDefault(); S.draw.pts.pop(); S.viewer && (S.draw.pts.length ? S.viewer.drawPoly() : ($('.draft', S.viewer.svg) || { remove() {} }).remove()); return;
    }
    if (ctrl && e.key.toLowerCase() === 'z') { e.preventDefault(); if (S.view === 'build') undoRedo('undo'); return; }
    if (ctrl && e.key.toLowerCase() === 'y') { e.preventDefault(); if (S.view === 'build') undoRedo('redo'); return; }
    // Space on a focused button presses that button; it is not also a Run
    if (e.key === ' ' && document.activeElement && document.activeElement.tagName === 'BUTTON') return;
    if (e.key === ' ' || e.key === 'F5') { e.preventDefault(); if (!(S.res && S.res.continuous)) runOnce(); return; }
    if (e.key === 'F6') { e.preventDefault(); toggleContinuous(); return; }
    // the picture's keys, as in Vision Station's canvas
    const B = S.view === 'build' && !locked(), V = S.viewer, k = e.key.toLowerCase();
    if (ctrl && V) {
      if (k === 'a' && B) { e.preventDefault(); selectAll(); return; }
      if (k === 'd' && B) { e.preventDefault(); duplicateTools(sels()); return; }
      if (k === '0') { e.preventDefault(); V.fit(); return; }
      if (k === '1') { e.preventDefault(); V.actual(); return; }
      if (k === '2') { e.preventDefault(); V.zoomTo(S.view === 'build' ? sels() : (S.sel ? [S.sel] : [])); return; }
      if (k === '=' || k === '+') { e.preventDefault(); V.zoomBy(1.25); return; }
      if (k === '-') { e.preventDefault(); V.zoomBy(0.8); return; }
      if (k === 's' && e.shiftKey) { e.preventDefault(); savePicture(); return; }
    }
    if (ctrl) return;
    if (B && V && V.o.editable()) {
      if ('[]{}'.includes(e.key) && e.key.length === 1) { e.preventDefault(); const st = e.shiftKey ? 15 : 1; rotateSel('[{'.includes(e.key) ? -st : st); return; }
      const arrow = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
      if (arrow && sels().length) { e.preventDefault(); const st = e.shiftKey ? 10 : 1; nudge(arrow[0] * st, arrow[1] * st); return; }
    }
    if (B && !S.draw && side() && side().reference && (k === 'r' || k === 'e' || k === 'p')) {
      const shape = { r: 'rect', e: 'ellipse', p: 'polygon' }[k]; S.shape = shape;
      startDraw({ shape, label: 'the area to check' }); return;
    }
    if (B && e.key === 'F2' && S.sel) { e.preventDefault(); renameTool(S.sel); return; }
    if (B && e.key === 'Backspace' && sels().length) { e.preventDefault(); deleteTools(sels()); return; }
    if (e.key === '?') { shortcutsDialog(); return; }
    if (e.key === '1') setView('run');
    if (e.key === '2' && !locked()) setView('build');
    if (e.key === '3') setView('history');
    if (e.key.toLowerCase() === 'f' && S.viewer) S.viewer.fit();
    if (e.key.toLowerCase() === 'a' && S.view === 'build') addSheet();
    if (e.key === 'Delete' && S.view === 'build' && !locked() && sels().length) deleteTools(sels());
  });
}

boot().catch(e => document.body.append(h('pre', { style: { position: 'fixed', left: '20px', top: '80px', zIndex: 99 } }, 'Vision Next could not start: ' + e.message)));
