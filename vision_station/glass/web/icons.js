// Line icons in the manner of SF Symbols: 24-unit grid, round caps and joins.
const ICON_PATHS = {
  run: '<path d="M8 5.5v13l10.5-6.5z" fill="currentColor" stroke="none"/>',
  stop: '<rect x="6.5" y="6.5" width="11" height="11" rx="2.5" fill="currentColor" stroke="none"/>',
  gauge: '<path d="M4.5 16.5a8 8 0 1 1 15 0"/><path d="M12 12.5l4-4"/><circle cx="12" cy="13" r="1.3" fill="currentColor"/>',
  hammer: '<path d="M14.5 4.5l5 5-2 2-5-5z"/><path d="M12.5 6.5L4 15a2 2 0 0 0 0 2.8l2.2 2.2a2 2 0 0 0 2.8 0l8.5-8.5"/>',
  clock: '<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>',
  gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 13.5a7.6 7.6 0 0 0 0-3l2-1.6-2-3.4-2.4.9a7.5 7.5 0 0 0-2.6-1.5L14 2.5h-4l-.4 2.4A7.5 7.5 0 0 0 7 6.4l-2.4-.9-2 3.4 2 1.6a7.6 7.6 0 0 0 0 3l-2 1.6 2 3.4 2.4-.9a7.5 7.5 0 0 0 2.6 1.5l.4 2.4h4l.4-2.4a7.5 7.5 0 0 0 2.6-1.5l2.4.9 2-3.4z"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2.5v2M12 19.5v2M2.5 12h2M19.5 12h2M5.3 5.3l1.4 1.4M17.3 17.3l1.4 1.4M5.3 18.7l1.4-1.4M17.3 6.7l1.4-1.4"/>',
  moon: '<path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z"/>',
  folder: '<path d="M3.5 7a2 2 0 0 1 2-2h4l2 2.2h7a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2h-13a2 2 0 0 1-2-2z"/>',
  chevron: '<path d="M7 10l5 5 5-5"/>',
  check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
  cursor: '<path d="M6 3.5L18.5 12 13 13.3l3 6.2-2.3 1-3-6.2L6 18z"/>',
  hand: '<path d="M8 11V5.5a1.5 1.5 0 0 1 3 0V11M11 10V4a1.5 1.5 0 0 1 3 0v6M14 10.5V5.5a1.5 1.5 0 0 1 3 0V14a6.5 6.5 0 0 1-6.5 6.5h-.5a6 6 0 0 1-5-2.7L3.8 15a1.5 1.5 0 0 1 2.4-1.8L8 15V8.5"/>',
  rect: '<rect x="4" y="6" width="16" height="12" rx="2"/>',
  ellipse: '<ellipse cx="12" cy="12" rx="8.5" ry="6.5"/>',
  sparkle: '<path d="M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z"/><path d="M18.5 15.5l.7 1.8 1.8.7-1.8.7-.7 1.8-.7-1.8-1.8-.7 1.8-.7z"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  minus: '<path d="M5 12h14"/>',
  trash: '<path d="M4.5 7h15M9.5 7V4.5h5V7M6.5 7l1 12.5h9l1-12.5"/>',
  undo: '<path d="M9 14L4.5 9.5 9 5"/><path d="M4.5 9.5H14a5.5 5.5 0 0 1 0 11h-3"/>',
  fit: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
  zoomin: '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5M8 10.5h5M10.5 8v5"/>',
  zoomout: '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5M8 10.5h5"/>',
  barcode: '<path d="M4 6v12M7 6v12M10.5 6v12M13 6v12M16.5 6v12M20 6v12"/>',
  camera: '<path d="M4 8.5a2 2 0 0 1 2-2h2l1.5-2h5l1.5 2h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2z"/><circle cx="12" cy="13" r="3.5"/>',
  photo: '<rect x="3.5" y="5" width="17" height="14" rx="2.5"/><circle cx="9" cy="10" r="1.6"/><path d="M20.5 15.5l-5-5L6 19"/>',
  info: '<circle cx="12" cy="12" r="8.5"/><path d="M12 11v5"/><circle cx="12" cy="8" r=".9" fill="currentColor"/>',
  ok: '<circle cx="12" cy="12" r="8.5"/><path d="M8 12.3l2.8 2.8L16.2 9.5"/>',
  warn: '<path d="M12 4l9 15.5H3z"/><path d="M12 10v4.5"/><circle cx="12" cy="17" r=".8" fill="currentColor"/>',
  close: '<path d="M7 7l10 10M17 7L7 17"/>',
  repeat: '<path d="M17 3.5l3 3-3 3"/><path d="M4 11.5v-1a4 4 0 0 1 4-4h12M7 20.5l-3-3 3-3"/><path d="M20 12.5v1a4 4 0 0 1-4 4H4"/>',
  text: '<path d="M5 6.5V5h14v1.5M12 5v14M9 19h6"/>',
  layers: '<path d="M12 3.5l9 4.5-9 4.5-9-4.5z"/><path d="M3 12.5l9 4.5 9-4.5"/>',
  search: '<circle cx="10.5" cy="10.5" r="6"/><path d="M15 15l5 5"/>',
  save: '<path d="M5 4h11l3 3v12a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1z"/><path d="M8 4v5h7V4M8 20v-6h8v6"/>',
};
const TOOL_ICON = { presence: 'ok', ocv: 'text', ocr: 'text', code_read: 'barcode', count_template: 'layers',
  count_blob: 'layers', measure_length: 'rect', color_match: 'sun', defect_diff: 'search', align: 'fit',
  find_line: 'minus', find_circle: 'ellipse', color_classify: 'sun', led_glow: 'sun' };

function icon(name, cls) {
  const body = ICON_PATHS[name] || ICON_PATHS.info;
  return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75"
    stroke-linecap="round" stroke-linejoin="round" class="${cls || ''}">${body}</svg>`;
}
