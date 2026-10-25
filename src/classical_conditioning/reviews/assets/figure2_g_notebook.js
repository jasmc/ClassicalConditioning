/* Review-only controls. No requests, analysis execution, or figure-state writes. */
(() => {
  'use strict';
  const config = JSON.parse(document.getElementById('guide-config').textContent);
  const schema = 'figure2-g-review-notebook/1';
  const key = `${schema}:${config.guide_id}:${config.source_fingerprint}`;
  const positions = new Set(['unreviewed', 'questions', 'tentative', 'ready']);
  const textFields = ['understanding', 'questions', 'decision'];
  const sectionIds = new Set(config.sections.map(s => s.id));
  if (window.matchMedia('(max-width:760px)').matches)
    document.getElementById('chapter-menu').open = false;
  // Scale the exact native SVG as a CSS layer, preserving non-scaling strokes
  // relative to its fonts. This changes screen magnification only.
  const artwork = [...document.querySelectorAll('.artwork[data-native-width-mm]')];
  function fitArtwork() {
    for (const holder of artwork) {
      const width = Number(holder.dataset.nativeWidthMm) * 96 / 25.4;
      const height = Number(holder.dataset.nativeHeightMm) * 96 / 25.4;
      const scale = holder.clientWidth / width;
      const svg = holder.querySelector('svg');
      Object.assign(svg.style,{width:`${width}px`,height:`${height}px`,maxWidth:'none',transform:`scale(${scale})`,transformOrigin:'top left'});
      holder.style.height = `${height * scale}px`;
    }
  }
  for (const holder of artwork) new ResizeObserver(fitArtwork).observe(holder);
  fitArtwork();
  const status = document.getElementById('save-status');
  const message = document.getElementById('import-message');
  let storage = null, notes = {}, pending = null;
  const empty = () => ({understood:false, position:'unreviewed', understanding:'', questions:'', decision:''});
  const filled = v => v.understood || v.position !== 'unreviewed' || textFields.some(k => v[k]);
  function payload() {
    return {schema, guide_id:config.guide_id, source_fingerprint:config.source_fingerprint,
      exported_at:new Date().toISOString(), sections:structuredClone(notes)};
  }
  function validate(value) {
    if (!value || typeof value !== 'object' || value.schema !== schema ||
        typeof value.guide_id !== 'string' || typeof value.source_fingerprint !== 'string' ||
        !value.sections || typeof value.sections !== 'object' || Array.isArray(value.sections))
      throw new Error('This is not a supported Figure 2 G notebook export.');
    const clean = {}, ignored = [];
    for (const [id, note] of Object.entries(value.sections)) {
      if (!sectionIds.has(id)) { ignored.push(id); continue; }
      if (!note || typeof note !== 'object' || typeof note.understood !== 'boolean' || !positions.has(note.position))
        throw new Error(`Invalid review fields for section ${id}.`);
      const copy = empty(); copy.understood = note.understood; copy.position = note.position;
      for (const field of textFields) {
        if (typeof note[field] !== 'string' || note[field].length > 100000)
          throw new Error(`Invalid or oversized ${field} in ${id}.`);
        copy[field] = note[field];
      }
      clean[id] = copy;
    }
    if (Object.keys(clean).length === 0) throw new Error('No matching guide sections were found.');
    const warnings = [];
    if (value.guide_id !== config.guide_id) warnings.push('The export belongs to a different guide version.');
    if (value.source_fingerprint !== config.source_fingerprint) warnings.push('Its evidence fingerprint differs from this guide. Recheck any scientific preferences.');
    if (ignored.length) warnings.push(`${ignored.length} unknown section(s) will be omitted.`);
    return {sections:clean, warnings};
  }
  function paint() {
    for (const box of document.querySelectorAll('.notebook[data-section]')) {
      const value = notes[box.dataset.section] || empty();
      for (const field of box.querySelectorAll('[data-field]')) {
        if (field.type === 'checkbox') field.checked = value[field.dataset.field];
        else field.value = value[field.dataset.field];
      }
      box.querySelector('[data-local-status]').textContent = filled(value) ?
        (storage ? 'Notes restored/saved in this browser.' : 'Notes in memory only; export to keep them.') : 'No notes entered.';
    }
  }
  function save() {
    if (storage) {
      try {
        storage.setItem(key, JSON.stringify(payload()));
        status.textContent = `Saved in this browser at ${new Date().toLocaleTimeString()}. Export for a portable backup.`;
      } catch (_) {
        storage = null;
        status.textContent = 'Browser saving failed. Notes remain in this tab; export them to keep a copy.';
      }
    } else status.textContent = 'Browser storage unavailable. Notes remain in this tab; export them to keep a copy.';
  }
  function download(value, name) {
    const blob = new Blob([JSON.stringify(value, null, 2) + '\n'], {type:'application/json'});
    const url = URL.createObjectURL(blob), a = document.createElement('a');
    a.href = url; a.download = name; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }
  function prepareImport(value) {
    const result = validate(value);
    pending = result.sections;
    document.getElementById('import-warning').textContent = result.warnings.join(' ') ||
      'The guide version and evidence fingerprint match.';
    document.getElementById('import-preview').textContent =
      `${Object.keys(pending).length} sections; ${Object.values(pending).filter(filled).length} contain review notes.\n` +
      'Imported text stays in review fields and cannot execute code.';
    document.getElementById('import-dialog').showModal();
    return result.warnings;
  }
  function confirmImport() {
    if (!pending) return false;
    const old = payload();
    // Preserve current user notes explicitly before replacing them.
    if (Object.values(notes).some(filled)) download(old, 'figure2-g-notes-before-import.json');
    notes = pending; pending = null; paint(); save();
    document.getElementById('import-dialog').close();
    message.textContent = 'Imported review notes. Scientific analyses and figure selections are unchanged.';
    return true;
  }
  try {
    storage = window.localStorage;
    const probe = key + ':probe'; storage.setItem(probe,'1'); storage.removeItem(probe);
    const raw = storage.getItem(key);
    if (raw) notes = validate(JSON.parse(raw)).sections;
    status.textContent = raw ? 'Saved notes restored for this guide and evidence snapshot.' : 'Browser saving is available. No notes saved yet.';
  } catch (_) {
    storage = null;
    status.textContent = 'Browser storage unavailable or saved notes unreadable. Export notes to keep them.';
  }
  paint();
  for (const box of document.querySelectorAll('.notebook[data-section]')) {
    box.addEventListener('input', () => {
      const value = empty();
      for (const field of box.querySelectorAll('[data-field]'))
        value[field.dataset.field] = field.type === 'checkbox' ? field.checked : field.value;
      notes[box.dataset.section] = value; save();
      box.querySelector('[data-local-status]').textContent = storage ? 'Saved in this browser.' : 'In memory only; export to keep these notes.';
    });
  }
  document.getElementById('export-notes').addEventListener('click', () =>
    download(payload(), `${config.guide_id}-notes.json`));
  document.getElementById('import-notes').addEventListener('click', () => document.getElementById('import-file').click());
  document.getElementById('import-file').addEventListener('change', async event => {
    const file = event.target.files[0]; if (!file) return;
    try {
      if (file.size > 2 * 1024 * 1024) throw new Error('Notebook import exceeds 2 MB.');
      prepareImport(JSON.parse(await file.text()));
    } catch (err) { message.textContent = `Import rejected: ${err.message}`; }
    event.target.value = '';
  });
  document.getElementById('confirm-import').addEventListener('click', confirmImport);
  document.getElementById('cancel-import').addEventListener('click', () => {
    pending = null; document.getElementById('import-dialog').close(); message.textContent = 'Import cancelled; notes unchanged.';
  });
  document.getElementById('import-dialog').addEventListener('cancel', () => { pending = null; });
  document.getElementById('print-guide').addEventListener('click', () => window.print());
  let openBeforePrint = [];
  window.addEventListener('beforeprint', () => {
    openBeforePrint = [...document.querySelectorAll('details')].map(x => [x, x.open]);
    // Technical teaching details print; optional historical-image galleries and
    // interactive notebooks stay collapsed to keep the reading edition usable.
    for (const [el] of openBeforePrint)
      if (!el.classList.contains('notebook') && !el.classList.contains('archive-figure')) el.open = true;
  });
  window.addEventListener('afterprint', () => { for (const [el,open] of openBeforePrint) el.open = open; });
  const chapters = [...document.querySelectorAll('.chapter')];
  document.getElementById('guide-search').addEventListener('input', event => {
    const q = event.target.value.trim().toLowerCase();
    document.querySelectorAll('.search-hit').forEach(n => n.classList.remove('search-hit'));
    const matches = q.length < 2 ? [] : chapters.filter(c => !c.id.startsWith('source-inventory') && c.textContent.toLowerCase().includes(q));
    const result = document.getElementById('search-results'); result.replaceChildren();
    if (q.length < 2) return;
    result.append(`${matches.length} matching chapters. `);
    for (const c of matches) {
      const a = document.createElement('a'); a.href = '#' + c.id; a.textContent = c.querySelector('h2').textContent;
      a.style.display = 'block'; a.addEventListener('click', () => c.classList.add('search-hit')); result.append(a);
    }
  });
  const nav = new Map([...document.querySelectorAll('nav a')].map(a => [a.hash.slice(1),a]));
  function progress() {
    const total = document.documentElement.scrollHeight - innerHeight;
    document.getElementById('progress-fill').style.width = `${total > 0 ? Math.min(100,Math.max(0,scrollY/total*100)) : 0}%`;
    let active = chapters[0]; for (const c of chapters) if (c.getBoundingClientRect().top < 150) active = c;
    for (const [id,a] of nav) { a.classList.toggle('active', id === active.id); if (id === active.id) a.setAttribute('aria-current','location'); else a.removeAttribute('aria-current'); }
  }
  window.addEventListener('scroll',progress,{passive:true}); progress();
  // Small review-only interface used by automated notebook validation.
  window.GuideNotebook = Object.freeze({payload, validate, prepareImport, confirmImport,
    storageKey:key, guideId:config.guide_id, fingerprint:config.source_fingerprint});
})();
