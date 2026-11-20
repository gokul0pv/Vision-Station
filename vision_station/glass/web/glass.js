// Liquid-glass components: menus and dropdowns, switches, segmented
// controls, toasts and sheets. No framework -- plain DOM.
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
function closeMenu() { if (openMenu) { openMenu.remove(); openMenu = null; } }
document.addEventListener('mousedown', e => { if (openMenu && !openMenu.contains(e.target)) closeMenu(); }, true);
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeMenu(); });

/** A glass menu at (x, y). items: [{label, icon?, hint?, checked?, onClick} | '-' | {section}] */
function showMenu(x, y, items) {
  closeMenu();
  const m = h('div', { class: 'menu glass lens-soft' });
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
  m.style.left = Math.min(x, innerWidth - r.width - 10) + 'px';
  m.style.top = Math.min(y, innerHeight - r.height - 10) + 'px';
  // arrow keys and Enter, as a native menu
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

/** A dropdown: a glass button showing the choice; options [{value, label, icon?}] */
function glassSelect(options, value, onChange, { width } = {}) {
  const el = h('div', { class: 'gselect', tabindex: 0, style: width ? { width } : null });
  const set = v => {
    value = v;
    const o = options.find(o => o.value === v) || options[0] || { label: '' };
    el.innerHTML = (o.icon ? icon(o.icon) : '') + `<span class="val">${esc(o.label)}</span>` + icon('chevron');
  };
  set(value);
  el.addEventListener('click', () => {
    const r = el.getBoundingClientRect();
    const m = showMenu(r.left, r.bottom + 6, options.map(o => ({
      label: o.label, icon: o.icon, checked: o.value === value,
      onClick: () => { set(o.value); onChange && onChange(o.value); } })));
    m.style.minWidth = r.width + 'px';
  });
  el.setValue = set;
  el.getValue = () => value;
  return el;
}

/** A switch; returns the element with .on and .set(bool) */
function glassSwitch(on, onChange, el) {
  el = el || h('span', { class: 'switch' });
  el.classList.toggle('on', !!on);
  el.onclick = () => { const v = !el.classList.contains('on'); el.classList.toggle('on', v); onChange && onChange(v); };
  el.set = v => el.classList.toggle('on', !!v);
  return el;
}

/** A segmented control with a sliding glass thumb. */
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
  const ic = { pass: 'ok', fail: 'warn', warn: 'warn', idle: 'info' }[tone] || 'info';
  const t = h('div', { class: `toast glass lens-soft ${tone}`, html: icon(ic) + `<div>${esc(text)}</div>` });
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
  const s = h('div', { class: 'sheet glass' }, h('h2', {}, title), sub ? h('div', { class: 'sub' }, sub) : null, body);
  const foot = h('div', { class: 'foot' });
  const close = () => scrim.remove();
  for (const b of (buttons || [{ label: 'Done', prominent: true }])) {
    foot.append(h('button', { class: 'glass-btn' + (b.prominent ? ' prominent' : ''),
      onclick: async () => { if (b.onClick && (await b.onClick()) === false) return; close(); } }, b.label));
  }
  s.append(foot);
  scrim.append(s);
  scrim.addEventListener('mousedown', e => { if (e.target === scrim) close(); });
  document.addEventListener('keydown', function k(e) {
    if (e.key === 'Escape') { close(); document.removeEventListener('keydown', k); }
  });
  $('#layer').append(scrim);
  return { close, el: s };
}

function formRow(label, help, control) {
  return h('div', { class: 'formrow' },
    h('div', { class: 'l' }, h('b', {}, label), help ? h('span', {}, help) : null), control);
}
