"""Publish the verified, Figure-3-linked 3sTrace Figure 4 composition only."""
import os
from pathlib import Path

import pandas as pd
from PIL import Image

from classical_conditioning.artifacts import artifact_staging, publish_transaction
from classical_conditioning.figures.layout_frames import assess
from analyze_figures34_3strace import OUT, ROOT, bound, read, verify, write


DEST = OUT / 'figure4-figure3-score-draft'
REVIEWS = ['figure4', 'block-profiles-1', 'block-profiles-2', 'individual-catches', 'coverage']
COUNTS = {'Control reference': 90, 'Lower score': 30, 'Middle score': 29, 'Higher score': 30}
DESIGN = """# Figure 4: first 3sTrace composition tied to current Figure 3

The earlier classified preview used a saved legacy-WIP classifier and had only
57 fish in its four classified curves. It did **not** use the current Figure 3
continuous analysis. It is preserved separately. This new draft uses the exact
current Figure 3 acquisition scores, joined by fish and condition, for all 179
current-roster fish. No Figure 3 artifact or shared manuscript state is changed.

The current Figure 3 supplies continuous acquisition contrasts, not an approved
learner/nonlearner classifier. Here the 89 conditioned fish are displayed in
lower/middle/higher score thirds (30/29/30); all 90 controls are one reference.
These sample-dependent display strata are not biological classes, learner
thresholds, or confidence statements about individual learning. Equal-score ties
stay together. Exact cutpoints and all memberships are in analysis-spec.json and
figure3-score-membership.csv. In particular, score > 0 is not introduced as a
learner rule.

## Plans consulted and panel choices

Plans 04, 05 and 09 were read before building. Their exact bytes are bound in
start-ticket.json. Plan 04 allows continuous representations while Gate L remains
open; Plan 05 requires the full block/catch context and fish-first summaries;
Plan 09 places detailed catches, movement and coverage in supporting material.
The main figure should tell a temporal-response story rather than repeat the
Figure 3 score distribution or every output of the integrated analysis pipeline.

| Local panel | Included content | Reason |
|---|---|---|
| a | All-fish scaled total activity heatmaps: Control and 3sTrace, all nine ten-trial blocks plus pooled catches | Establishes acquisition/test progression without score-based selection; keeps the complete block sequence compact. |
| b | Pooled-catch scaled total activity by Figure 3 score thirds and the unsplit control reference | Shows the full zero-inclusive response profile on conditioned trials without the paired US. |
| c | Pooled-catch movement probability, same fish groups | Separates movement occurrence from intensity; a missing conditional-vigor value must not be read as immobility. |
| d | Pooled-catch signed conditional bout-ln-vigor, same groups | Retains the supported Figure 4 vigor estimand and reveals changes conditional on movement. |

Panels c/d are useful in this first draft because the named Plan 05 primary
outcome (Scaled total activity) and the operational Figure 4 manual (signed bout
ln vigor) are distinct estimands. They should be reviewed side by side before
selecting the final main-paper emphasis. They are not renamed or substituted
for one another: activity retains its saved per-trial scaling and valid zeros;
signed vigor retains its trial [-20,0) s baseline and undefined no-bout values.
Saved temporal configuration tags also differ across source recordings (121
mean_bout_vigor and 58 mean_total_activity); the current supported calculation
does not branch on that tag. This historical metadata issue remains recorded in
the parent temporal-config-variants.json and is not resolved by this rendering.

## Supporting views already made

Every view has a matching editable SVG, self-contained HTML and legend.

- [Main PNG](figure4.png), [main SVG](figure4.svg), [main review](figure4.html).
- All score-group curves, three quantities: [Pre and Train 1-4 PNG](block-profiles-1.png), [Train 5 and Test 1-3 plus pooled catches PNG](block-profiles-2.png).
- [Each of the five catches PNG](individual-catches.png), with all three quantities.
- [Contributing-fish coverage PNG](coverage.png), separated from movement occurrence; exact fish/trial counts are in per-bin-contributions.csv.

Thus the main panel a retains all nine configured blocks; full stratified curves
are not squeezed into the main page. Individual catches retain trials 25, 39,
53, 59 and 65. Trial 65 also belongs to Test 1. The roster has 90 controls and 89
conditioned fish; four timing-flagged recordings remain excluded. It is the
current source roster, not a newly approved primary technical cohort.

## Interpretation and next scientific decisions

The exact Figure 3 score is median G outcome on Pre 5-14 minus median G outcome
on late Train 40-64. Score construction therefore overlaps catches 53 and 59.
Differences between these retrospective score groups in pooled-catch curves
cannot independently validate learning or response timing. Test data have also
already been inspected; they are not described as prospectively held out.

No onset, peak, offset or timing significance is added. Those require reviewed
definitions and a disjoint classification/evaluation design if they are to
support independent timing claims. Ratio summaries and the score distribution
are not repeated merely because the pipeline can calculate them. The useful
next decision is the final continuous versus categorical Gate L representation;
if categories are selected, their approved classifier and matching cohort must
replace these display thirds through an authenticated join.

The manuscript Figure 4 README currently proposes A=Delay, B=3sTrace and
C=10sTrace. The a-d letters here belong only to this **3sTrace composition
candidate**; they do not reassign those manuscript assay slots. A full three-assay
assembly and common final scientific recipe remain future work.

## Verification and status

All 179 exact Figure 3 scores and memberships were checked. Raw saved trial bins
independently reproduce pooled-catch fish medians, group medians and contributing
fish counts for all three quantities. Lines use equal-fish medians and bands fish
IQRs; missing bins and no-bout conditional values remain distinct from zeros.
Protocol-verified timing marks show CS 0-10 s and expected paired US at 13 s.
Controls have unpaired reinforcers, which are not assumed absent from their
profile windows.

The main composition is 183 x 170 mm with one resolved nature-main profile.
Exact saved SVGs passed the shared layout-frame assessment at 600 dpi and all
five saved 300-dpi PNGs were visually inspected. This is not a complete Nature
production, font-embedding or scientific-approval audit. Gate L, primary cohort
approval and any freeze remain open. The previous legacy preview is preserved.
"""


def main():
    lock = DEST / 'publication.lock'
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    try:
        ticket = read(DEST / 'start-ticket.json')
        analysis = read(DEST / 'analysis-spec.json')
        completed = read(DEST / 'analysis-complete.json')
        rendered = read(DEST / 'render-complete.json')
        numerical = read(DEST / 'numerical-verification.json')
        inputs = [ticket['figure3_state'], ticket['score_source'],
                  ticket['preserve_legacy_review'], *ticket['inputs'],
                  *read(DEST / 'render-start.json')['inputs'], numerical['source']]
        existing = [bound(p) for p in sorted(DEST.iterdir())
                    if p.is_file() and p.name != 'publication.lock']
        for item in [*inputs, *completed['artifacts'], *rendered['artifacts'], *existing]:
            verify(item)

        # Resolve the exact score through the protected current Figure 3 registry.
        figure3 = read(ticket['figure3_state']['path'])
        verify(figure3['draft']['registry'])
        figure3_registry = read(figure3['draft']['registry']['path'])
        assert figure3_registry['B']['source'] == ticket['score_source']
        membership = pd.read_csv(DEST / 'figure3-score-membership.csv')
        scores = pd.read_csv(ticket['score_source']['path'])
        assert len(membership) == 179 and membership.fish_id.nunique() == 179
        assert membership.display_group.value_counts().to_dict() == COUNTS
        pd.testing.assert_frame_equal(
            membership[['fish_id', 'condition_id', 'acquisition_contrast']].reset_index(drop=True),
            scores[['fish_id', 'condition_id', 'acquisition_contrast']].reset_index(drop=True))
        assert numerical['status'] == 'passed' and numerical['group_counts'] == COUNTS

        layouts = {}
        visual = []
        for name in REVIEWS:
            layouts[name] = assess((DEST / (name + '.svg')).read_bytes(),
                                   read(DEST / (name + '-browser.json')))
            assert layouts[name]['status'] == 'passed', layouts[name]['issues']
            visual.append({'name': name, 'svg': bound(DEST / (name + '.svg')),
                           'png': bound(DEST / (name + '.png')),
                           'png_pixels': list(Image.open(DEST / (name + '.png')).size),
                           'png_dpi': 300})
        write(DEST / 'layout-evidence.json', {'status': 'passed', 'reviews': layouts})
        write(DEST / 'visual-review.json', {
            'status': 'passed within stated coverage', 'artifacts': visual,
            'actual_inspection': 'All five complete saved PNGs inspected: main composition, both block-profile pages, individual catches and coverage. Labels, timing guides, curves and legends visible; no obvious collisions or clipped content.',
            'limits': ['Complete Nature production/font-embedding audit remains open',
                       'Visual and layout checks do not validate learning categories or independent timing claims']})
        with (DEST / 'design.md').open('x', encoding='utf-8') as stream:
            stream.write(DESIGN)

        codes = [bound(ROOT / 'scripts' / name) for name in [
            'build_figure4_3strace_score_draft.py', 'check_figure4_3strace_score_draft.cjs',
            Path(__file__).name]]
        write(DEST / 'draft-registry.json', {
            'scope': '3sTrace-only Figure 4 composition; local a-d letters',
            'scientific_mapping_status': 'Explicit data-column and many-to-one membership mapping; not inferred from artwork',
            'figure3_state': ticket['figure3_state'], 'figure3_scores': ticket['score_source'],
            'membership': bound(DEST / 'figure3-score-membership.csv'),
            'group_data': bound(DEST / 'score-group-bins.parquet'),
            'within_fish_data': bound(DEST / 'fish-bins-with-score.parquet'),
            'condition_overview': bound(DEST / 'condition-overview.parquet'),
            'coverage': bound(DEST / 'per-bin-contributions.csv'),
            'quantity_definitions': analysis['quantities'],
            'roles': {
                'A': {'data': 'condition-overview.parquet', 'quantity': 'scaled_activity_median',
                      'x': 'time_s', 'rows': 'all nine blocks and pooled catches by group_order',
                      'population': 'all current-roster fish within condition; unsplit by score'},
                **{panel: {'data': 'score-group-bins.parquet', 'filter': "group_type == pooled_catch",
                           'x': 'time_s', 'line': signal + '_median',
                           'band': [signal + '_q25', signal + '_q75'],
                           'coverage': [signal + '_fish', signal + '_trials'],
                           'groups': 'exact membership in figure3-score-membership.csv'}
                   for panel, signal in [('B', 'scaled_activity'), ('C', 'movement_probability'),
                                         ('D', 'signed_log_vigor')]},
                'CS': {'seconds': [0, 10], 'source': 'authenticated all3sTrace protocol'},
                'expected_US': {'seconds': 13, 'meaning': 'paired training timing, not a conditioned catch US'}},
            'group_counts': COUNTS, 'display_rule': analysis['display_group_rule'],
            'no_learner_classification': True, 'validation_mode': analysis['validation_mode'],
            'supplements': {name: {'svg': bound(DEST / (name + '.svg')),
                                   'same_membership': True, 'all_three_quantities': True}
                            for name in REVIEWS[1:]},
            'design': bound(DEST / 'design.md'), 'code': codes,
            'status': 'Unfrozen draft registry; no complete freeze-element classification'})
        write(DEST / 'consistency-evidence.json', {
            'request_id': 'figure4-current-figure3-score-20261020',
            'named_relationship': 'Figure 4 display strata derive from current Figure 3 scores for the same fish',
            'scientific': {'status': 'verified', 'figure3_state': ticket['figure3_state'],
                           'source_scores': ticket['score_source'], 'fish': 179,
                           'properties': ['exact acquisition contrast values', 'unique fish/condition membership',
                                          'strict many-to-one join', 'controls unsplit', 'all fish retained'],
                           'numerical_evidence': bound(DEST / 'numerical-verification.json')},
            'layout': {'status': 'passed for shared invisible frames on all five exact SVGs',
                       'evidence': bound(DEST / 'layout-evidence.json')},
            'presentation': {'profile': 'nature-main', 'source_inputs': read(DEST / 'render-start.json')['inputs'],
                             'evidence': bound(DEST / 'visual-review.json')},
            'unresolved': ['No approved categorical classifier', 'Same-data score/catch overlap',
                           'Primary technical cohort gate remains open', 'Complete production audit remains open']})

        artifacts = [bound(p) for p in sorted(DEST.iterdir())
                     if p.is_file() and p.name != 'publication.lock']
        state = {
            'schema_version': 1, 'figure_id': 'figure4-3strace-score-draft',
            'scope': '3sTrace Figure 4 composition candidate only; local a-d letters',
            'default_display': 'draft', 'default_display_format': 'html-svg-and-legend',
            'whole_figure_frozen': False,
            'scientific_selection': {'status': 'unfrozen retrospective draft; Gate L and primary-cohort approval remain open', 'records': []},
            'draft': {'working_svg': bound(DEST / 'figure4.svg'), 'review': bound(DEST / 'figure4.html'),
                      'legend': bound(DEST / 'figure4-legend.md'), 'png': bound(DEST / 'figure4.png'),
                      'registry': bound(DEST / 'draft-registry.json'),
                      'layout_evidence': bound(DEST / 'layout-evidence.json'),
                      'visual_review': bound(DEST / 'visual-review.json')},
            'supplements': {name: {'svg': bound(DEST / (name + '.svg')), 'review': bound(DEST / (name + '.html')),
                                   'png': bound(DEST / (name + '.png')), 'legend': bound(DEST / (name + '-legend.md'))}
                            for name in REVIEWS[1:]},
            'figure3_state': ticket['figure3_state'], 'score_source': ticket['score_source'],
            'grouping_representation': 'observed continuous Figure 3 score thirds for display; not learner labels',
            'group_counts': COUNTS, 'membership': bound(DEST / 'figure3-score-membership.csv'),
            'analysis': bound(DEST / 'analysis-spec.json'), 'design': bound(DEST / 'design.md'),
            'consistency_requests': [bound(DEST / 'consistency-evidence.json')],
            'preserved_legacy_review': ticket['preserve_legacy_review'],
            'shared_assembly_pointers_changed': False, 'manuscript_assay_panel_numbering_changed': False,
            'main_canvas_mm': [183, 170], 'profile': 'nature-main', 'code': codes}

        def guard():
            for item in [*inputs, *existing, *artifacts, *codes]:
                verify(item)

        with artifact_staging(DEST, prefix='.publish-') as staging:
            manifest = staging / 'artifact-manifest.json'
            write(manifest, {'artifacts': artifacts})
            state['artifact_manifest'] = {'path': str(DEST / manifest.name), 'sha256': bound(manifest)['sha256']}
            pointer = staging / 'current-state.json'
            write(pointer, state)
            saved_refs = [{'path': str(DEST / p.name), 'sha256': bound(p)['sha256']} for p in [manifest, pointer]]
            def verify_published():
                guard()
                for item in saved_refs:
                    verify(item)
            guard()
            publish_transaction(((manifest, DEST / manifest.name), (pointer, DEST / pointer.name)),
                                staging, overwrite=False, verify_published=verify_published)
        print('Published Figure-3-linked 3sTrace Figure 4 draft; all 179 fish retained; protected states unchanged.')
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
