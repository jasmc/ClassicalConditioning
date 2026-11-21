# HTML reading and review behavior

Use this reference when building a substantial analysis study guide with reader
notes. It generalizes the behavior of the Figure 2 G guide, not its scientific
definitions, chapter numbering, colors, storage location or results.

## Reading shell

- Offer an orientation, suggested reading order and glossary. Make it possible
  to learn the subject linearly and later jump directly to a result or method.
- Use a desktop chapter sidebar and topic search. On narrow screens collapse
  the chapter list so it does not displace the entire opening explanation.
  Keep wide tables in their own horizontal scroll containers.
- Put exact calculations, assumptions, code details and secondary output
  galleries in accessible disclosures. Keep the main result, interpretation and
  important limits visible beside each other.
- Display authentic saved figures with matching legends, source links and
  version/status captions. Keep current scientific selection, current layout,
  superseded previews and exploratory candidates distinguishable. Use the
  owning figure workflow for crops, SVG isolation and physical scaling; avoid
  ID collisions and preserve non-scaling strokes when enlarging SVGs.
- Supply a print stylesheet: remove navigation/controls, preserve headings,
  render table headers legibly, avoid splitting rows and figures, and expose
  useful technical text. Large optional image galleries can remain collapsed.
  Check an actual print rendering rather than assuming CSS rules worked.

## Review notebook

For each substantial section, useful fields are: “I can explain this without
AI,” review position, understanding in my words, open questions, and provisional
decision with reasons. These are advisory notes, never project approvals.

Use stable section/field IDs and an export schema with guide identity, guide
version, evidence fingerprint and export timestamp. Report whether browser
storage is available and whether edits were saved; export must still work if
storage is disabled or fails. Explain that storage belongs to a browser origin
and profile, so changing ports/devices may make notes appear absent.

Import should validate data before touching notes, show guide-version/evidence
mismatches, and require an explicit confirm or cancel. Preserve old notes with
an export/backup before replacement. Invalid or cancelled imports leave notes
unchanged. Assign imported content as plain text/values, never executable HTML.

On guide revisions, preserve compatible saved fields and flag their older
evidence/version; do not silently clear notes because the fingerprint changed.
Use a different guide identity when old section semantics are incompatible.
Do not present restored provisional decisions as new scientific approval.

## Optional reusable helper

Copy `assets/review-notebook.js` into the output or inline its bytes in the final
HTML. It provides the notebook, JSON export/import with downloadable backup,
chapter search/progress and print button. It performs no network requests or
scientific/project mutations. Equivalent framework code is fine.

The helper expects this markup contract; generate labels and sections to fit
the actual guide:

```html
<script type="application/json" id="guide-config">
{"guide_id":"stable-analysis-guide-id","guide_version":"v01",
 "source_fingerprint":"hash-of-bound-evidence",
 "sections":[{"id":"outcomes","title":"Outcome definitions"}]}
</script>
<input id="guide-search" type="search" aria-label="Find a topic">
<div id="search-results" aria-live="polite"></div>
<details id="chapter-menu" open><summary>Chapters</summary>
  <nav><a href="#outcomes">Outcome definitions</a></nav>
</details>
<div aria-hidden="true"><span id="progress-fill"></span></div>
<button id="export-notes">Export notes</button>
<button id="import-notes">Import notes</button>
<button id="print-guide">Print guide</button>
<input id="import-file" type="file" accept=".json,application/json" hidden>
<p id="save-status" role="status"></p><p id="import-message" role="status"></p>

<section class="chapter" id="outcomes"><h2>Outcome definitions</h2>
  <!-- Actual source-grounded explanation and outputs -->
  <details class="notebook no-print" data-section="outcomes">
    <summary>Your review notebook</summary>
    <label><input type="checkbox" data-field="understood">I can explain this</label>
    <label>Review position<select data-field="position">
      <option value="unreviewed">Not reviewed</option>
      <option value="questions">Questions remain</option>
      <option value="tentative">Provisional preference</option>
      <option value="ready">Ready for discussion</option>
    </select></label>
    <label>My understanding<textarea data-field="understanding"></textarea></label>
    <label>Open questions<textarea data-field="questions"></textarea></label>
    <label>Provisional decision and reason<textarea data-field="decision"></textarea></label>
    <span data-local-status></span>
  </details>
</section>
<dialog id="import-dialog"><h2>Import review notes</h2>
  <p id="import-warning"></p><pre id="import-preview"></pre>
  <p>Existing notes will be backed up before replacement.</p>
  <button id="confirm-import">Replace notes with this import</button>
  <button id="cancel-import">Cancel</button>
</dialog>
```

The optional `.artwork[data-native-width-mm][data-native-height-mm]` container
supports an inline SVG at its native physical size with a uniformly magnified
screen layer. Omit that markup when using image-based embedding or the project's
own SVG scaler. The helper never chooses crop bounds or scientific content.

Provide CSS for `.table-wrap`, `.active`, `.search-hit`, the notebook/disclosures
and print layout; the helper is not a theme. The helper exposes
`window.AnalysisReviewNotebook` for verification only. Its payload schema is
`analysis-review-guide-notebook/1`; storage is scoped to `guide_id`, with version
and evidence changes shown to the reader. It accepts up to 100,000 characters
per text field; its import-size limit scales with the number of sections so a
valid export can be reimported. Preserve this contract or update the helper and
tests together when adapting it.

## Verify against reader behavior

Use a fresh browser context, never the user's real notebook, for tests. Check:

- Chapter links/search, figures decoding, matching legends and narrow overflow.
- Save → reload → export retains text, checkbox and review position.
- Import mismatch preview; cancel leaves notes; confirm backs up old notes.
- Malformed imports cannot replace notes or execute text as HTML.
- Storage-disabled or quota-failed saving remains explicit, with export usable.
- Compatible notes survive a guide-version/evidence change with a visible warning.
- Print output has readable figures/tables and no empty/clipped layout surprises.

Wait for asynchronous import UI state rather than testing immediately after
file selection. Record what was checked, the exact final HTML identity and any
untested behavior. A passing browser/static check does not validate the science.
