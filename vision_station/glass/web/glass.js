// Vision Glass components: menus and dropdowns, switches, segmented controls,
// toasts, sheets -- and the pointer. No framework, plain DOM.
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const k of kids.flat()) if (k != null && k !== false)
    el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  return el;
}
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ---------------------------------------------------------------- menus
let openMenu = null;
function closeMenu() {
  if (!openMenu) return;
  const m = openMenu; openMenu = null;
  m.classList.add('leave');
  setTimeout(() => m.remove(), 160);
}
document.addEventListener('mousedown', e => { if (openMenu && !openMenu.contains(e.target)) closeMenu(); }, true);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeMenu(); });

/** A menu at (x, y). items: [{label, icon?, hint?, checked?, onClick} | '-' | {section}] */
function showMenu(x, y, items) {
  if (openMenu) { openMenu.remove(); openMenu = null; }
  const m = h('div', { class: 'menu glass' });
  const live = [];
  for (const it of items) {
    if (it === '-') { m.append(h('div', { class: 'divider' })); continue; }
    if (it.section) { m.append(h('div', { class: 'section' }, it.section)); continue; }
    const row = h('div', { class: 'item', html:
      (it.checked !== undefined ? `<span class="check">${it.checked ? icon('check') : ''}</span>` : '') +
      (it.icon ? icon(it.icon) : '') + `<span>${esc(it.label)}</span>` +
      (it.hint ? `<span class="hint">${esc(it.hint)}</span>` : '') });
    row.addEventListener('click', () => { closeMenu(); it.onClick && it.onClick(); });
    m.append(row); live.push(row);
  }
  document.body.append(m);
  const r = m.getBoundingClientRect();
  const left = Math.max(10, Math.min(x, innerWidth - r.width - 10));
  const top = Math.max(10, Math.min(y, innerHeight - r.height - 10));
  m.style.left = left + 'px';
  m.style.top = top + 'px';
  // it grows out of the point it was opened from
  m.style.setProperty('--ox', Math.max(0, Math.min(r.width, x - left)) + 'px');
  m.style.setProperty('--oy', Math.max(0, Math.min(r.height, y - top)) + 'px');
  let hot = -1;
  m.tabIndex = -1; m.focus();
  m.addEventListener('keydown', e => {
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      hot = (hot + (e.key === 'ArrowDown' ? 1 : -1) + live.length) % live.length;
      live.forEach((x, i) => x.classList.toggle('hot', i === hot)); e.preventDefault();
    } else if (e.key === 'Enter' && hot >= 0) live[hot].click();
  });
  openMenu = m;
  return m;
}

/** A dropdown showing the choice; options [{value, label, icon?}] */
function glassSelect(options, value, onChange, { width } = {}) {
  const el = h('div', { class: 'gselect', tabindex: 0, style: width ? { width } : null });
  const set = v => {
    value = v;
    const o = options.find(o => o.value === v) || options[0] || { label: '' };
    el.innerHTML = (o.icon ? icon(o.icon) : '') + `<span class="val">${esc(o.label)}</span>` + icon('chevron');
  };
  set(value);
  const open = () => {
    const r = el.getBoundingClientRect();
    const m = showMenu(r.left, r.bottom + 6, options.map(o => ({
      label: o.label, icon: o.icon, checked: o.value === value,
      onClick: () => { set(o.value); onChange && onChange(o.value); } })));
    m.style.minWidth = r.width + 'px';
  };
  el.addEventListener('click', open);
  el.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(); } });
  el.setValue = set;
  el.getValue = () => value;
  return el;
}

/** A switch; returns the element with .set(bool) */
function glassSwitch(on, onChange, el) {
  el = el || h('span', { class: 'switch' });
  el.classList.toggle('on', !!on);
  el.setAttribute('role', 'switch');
  el.onclick = () => { const v = !el.classList.contains('on'); el.classList.toggle('on', v); onChange && onChange(v); };
  el.set = v => el.classList.toggle('on', !!v);
  return el;
}

/** A segmented control with a pill that springs to the choice. */
function segmented(options, value, onChange) {
  const el = h('div', { class: 'segmented' });
  const thumb = h('div', { class: 'thumb' });
  el.append(thumb);
  const btns = options.map(o => {
    const b = h('button', { class: o.value === value ? 'on' : '' }, o.label);
    b.onclick = () => { value = o.value; place(); onChange && onChange(o.value); };
    el.append(b); return b;
  });
  function place() {
    btns.forEach((b, i) => b.classList.toggle('on', options[i].value === value));
    const b = btns[options.findIndex(o => o.value === value)];
    if (b) { thumb.style.left = b.offsetLeft + 'px'; thumb.style.width = b.offsetWidth + 'px'; }
  }
  requestAnimationFrame(place);
  el.set = v => { value = v; place(); };
  return el;
}

// ---------------------------------------------------------------- toasts
function toast(text, tone = 'idle', ms = 4200) {
  if (!text) return;
  const ic = { pass: 'check', fail: 'close', warn: 'warn', idle: 'info' }[tone] || 'info';
  const t = h('div', { class: `toast glass ${tone}`, html: icon(ic) + `<div>${esc(text)}</div>` });
  const box = $('#toasts');
  box.prepend(t);
  while (box.children.length > 3) box.lastChild.remove();
  const go = () => { t.classList.add('leave'); setTimeout(() => t.remove(), 300); };
  t.onclick = go;
  setTimeout(go, ms);
}

// ---------------------------------------------------------------- sheets
function sheet(title, sub, body, buttons) {
  const scrim = h('div', { class: 'scrim' });
  const s = h('div', { class: 'sheet' }, h('h2', {}, title), sub ? h('div', { class: 'sub' }, sub) : null, body);
  const foot = h('div', { class: 'foot' });
  let closed = false;
  const close = () => {
    if (closed) return; closed = true;
    document.removeEventListener('keydown', onEsc);
    scrim.classList.add('leave');
    setTimeout(() => scrim.remove(), 200);
  };
  const onEsc = e => { if (e.key === 'Escape' && !openMenu) close(); };
  for (const b of (buttons || [{ label: 'Done', prominent: true }])) {
    foot.append(h('button', { class: 'glass-btn' + (b.prominent ? ' prominent' : ''),
      onclick: async () => { if (b.onClick && (await b.onClick()) === false) return; close(); } }, b.label));
  }
  s.append(foot);
  scrim.append(s);
  scrim.addEventListener('mousedown', e => { if (e.target === scrim) close(); });
  document.addEventListener('keydown', onEsc);
  $('#layer').append(scrim);
  return { close, el: s };
}

function formRow(label, help, control) {
  return h('div', { class: 'formrow' },
    h('div', { class: 'l' }, h('b', {}, label), help ? h('span', {}, help) : null), control);
}

// ---------------------------------------------------------------- light on buttons
// a soft highlight sits under the pointer while it crosses a button
document.addEventListener('pointermove', e => {
  const b = e.target.closest && e.target.closest('.glass-btn');
  if (!b) return;
  const r = b.getBoundingClientRect();
  b.style.setProperty('--mx', (e.clientX - r.left) + 'px');
  b.style.setProperty('--my', (e.clientY - r.top) + 'px');
}, { passive: true });

// ---------------------------------------------------------------- the pointer
// A dot exactly at the pointer -- never behind it, so aiming is as precise as
// the system arrow -- and a ring that follows on a spring. Over a control the
// ring becomes the control's highlight and leans a little toward the
// pointer; over the picture it turns into a crosshair (drawing), a hand
// (moving the picture), a four-way arrow (moving a check) or a resize arrow.
// Settings > Appearance > Fluid pointer switches it off (the system arrow
// comes back); it never runs on a touch screen.
const Pointer = (() => {
  const WRAP = '.glass-btn, .nav-item, .job-item, .menu .item, .gselect, .switch, .check-row, .frow, .pick, ' +
               '.segmented button, .qrow, .kpi, .badge, .toast';
  const RESIZE_DEG = { e: 0, w: 0, n: 90, s: 90, ne: -45, sw: -45, nw: 45, se: 45 };
  let dot, ring, glyph, x = -200, y = -200, rx = -200, ry = -200, rw = 34, rh = 34, rr = 17;
  let target = { kind: '', el: null }, down = false, space = false, started = false;

  const wanted = () => localStorage.getItem('vg-pointer') !== '0' && matchMedia('(pointer: fine)').matches;
  function apply() {
    const on = wanted();
    document.documentElement.classList.toggle('fluid-ptr', on);
    if (on && !started) start();
    if (dot) { dot.style.display = on ? '' : 'none'; ring.style.display = on ? '' : 'none'; }
  }
  function start() {
    started = true;
    dot = h('div', { id: 'ptr' });
    ring = h('div', { id: 'ptrRing' });
    glyph = h('span', {});
    ring.append(glyph);
    document.body.append(ring, dot);
    // size and place are animated here, frame by frame; CSS keeps the rest
    ring.style.transition = 'border-radius .3s, opacity .18s, background .18s, border-color .18s';
    document.addEventListener('mousemove', e => {
      x = e.clientX; y = e.clientY;
      document.documentElement.classList.add('ptr-in');
      if (!down) classify(e.target);
    }, { passive: true });
    document.addEventListener('mouseleave', () => document.documentElement.classList.remove('ptr-in'));
    document.addEventListener('mousedown', () => { down = true; ring.classList.add('down'); pulse(); }, true);
    document.addEventListener('mouseup', e => { down = false; ring.classList.remove('down'); classify(e.target); }, true);
    window.addEventListener('keydown', e => { if (e.key === ' ' && !space) { space = true; refresh(); } });
    window.addEventListener('keyup', e => { if (e.key === ' ') { space = false; refresh(); } });
    window.addEventListener('blur', () => document.documentElement.classList.remove('ptr-in'));
    requestAnimationFrame(frame);
  }
  function refresh() { const el = document.elementFromPoint(x, y); if (el) classify(el); }
  let squeeze = 1;
  function pulse() { squeeze = 0.82; }

  function classify(el) {
    let kind = '', host = null, deg = 0;
    const cv = el.closest && el.closest('.canvas');
    if (el.closest && el.closest('input:not([type=range]):not([type=checkbox]), textarea')) kind = 'text';
    else if (cv && !el.closest('.float')) {
      const hd = el.closest('[data-h]');
      if (document.documentElement.dataset.ptr === 'grabbing') kind = 'grab grabbing';
      else if (hd) { kind = 'resize'; deg = RESIZE_DEG[hd.dataset.h] || 0; }
      else if (cv.classList.contains('draw') || document.documentElement.dataset.ptr === 'draw') kind = 'cross';
      else if (cv.classList.contains('pan') || space) kind = 'grab';
      else if (el.closest('[data-id]') || el.closest('.fbox')) kind = 'move';
    } else if (el.closest) {
      host = el.closest(WRAP);
      if (host) {
        const r = host.getBoundingClientRect();
        if (r.width > 380 || r.height > 84) host = null;     // a big row: just a soft ring
        else kind = 'wrap';
      }
    }
    target = { kind, el: host };
    dot.className = kind;
    ring.className = kind;
    glyph.innerHTML = kind.startsWith('grab') ? icon('hand') : kind === 'move' ? icon('move') :
                      kind === 'resize' ? icon('resize') : '';
    glyph.firstChild && (glyph.firstChild.style.transform = `rotate(${deg}deg)`);
  }

  function frame() {
    requestAnimationFrame(frame);
    if (!dot || dot.style.display === 'none') return;
    let tx = x, ty = y, tw = 34, th = 34, tr = 17, k = 0.28;
    const kind = target.kind;
    if (kind === 'wrap' && target.el && target.el.isConnected) {
      const r = target.el.getBoundingClientRect();
      const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
      // the highlight leans toward the pointer: a magnet, not a cage
      tx = cx + (x - cx) * 0.12; ty = cy + (y - cy) * 0.12;
      tw = r.width + 8; th = r.height + 8;
      const br = parseFloat(getComputedStyle(target.el).borderTopLeftRadius) || 10;
      tr = Math.min(th / 2, br + 4); k = 0.24;
    } else if (kind === 'text') { tw = th = 10; tr = 5; }
    else if (kind === 'cross') { tw = th = 26; tr = 13; k = 0.5; }
    else if (kind.startsWith('grab') || kind === 'move' || kind === 'resize') {
      tw = th = kind === 'grab grabbing' ? 30 : 36; tr = tw / 2; k = 0.45; }
    tw *= squeeze; th *= squeeze;
    squeeze += (1 - squeeze) * 0.18;
    rx += (tx - rx) * k; ry += (ty - ry) * k;
    rw += (tw - rw) * 0.3; rh += (th - rh) * 0.3; rr += (tr - rr) * 0.3;
    dot.style.transform = `translate(${x}px, ${y}px)`;
    ring.style.width = rw + 'px'; ring.style.height = rh + 'px'; ring.style.borderRadius = rr + 'px';
    ring.style.transform = `translate(${rx - rw / 2}px, ${ry - rh / 2}px)`;
  }
  return { apply, refresh };
})();
document.addEventListener('DOMContentLoaded', () => Pointer.apply());
if (document.readyState !== 'loading') Pointer.apply();
