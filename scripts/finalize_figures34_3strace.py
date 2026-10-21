"""Guarded publication of an isolated scientific-draft review, never a freeze."""
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from classical_conditioning.artifacts import artifact_staging,publish_transaction
from classical_conditioning.figures.layout_frames import assess
from analyze_figures34_3strace import ROOT,OUT,BASE,read,write,bound,verify

def table(frame):
    return '| '+' | '.join(frame.columns)+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+'\n'.join(
        '| '+' | '.join(f'{v:.6g}' if isinstance(v,float) else str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))

def main():
    lock=OUT/'publication.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        for item in read(OUT/'render-start.json')['inputs']:verify(item)
        for item in read(OUT/'render-complete.json')['artifacts']:verify(item)
        for item in read(OUT/'render-complete.json')['atlas_sources']:verify(item)
        visual=read(OUT/'visual-review.json')
        for item in visual['artifacts']:verify(item)
        assert read(OUT/'numerical-verification.json')['result']=='passed'
        layouts=[]
        for browser in sorted((OUT/'browser-verification').glob('*.json')):
            if browser.name=='summary.json':continue
            stem=browser.stem
            svg=OUT/'individual-atlas'/(stem+'.svg') if stem.startswith('atlas_') else OUT/(stem+'.svg')
            layouts.append({'svg':bound(svg),'assessment':assess(svg.read_bytes(),read(browser))})
        write(OUT/'layout-evidence.json',{'artifacts':layouts,
            'coverage':'three summary SVGs and three representative atlas cards (both conditions); other atlas cards structurally checked only'})
        groups=pd.read_parquet(OUT/'group-bins.parquet')
        registry=[]
        registry.append({'artifact':bound(OUT/'figure3-score.svg'),'element_id':'figure3-score_axes_0',
            'scientific_role':'one acquisition contrast per source-roster fish',
            'source':bound(OUT/'acquisition-scores.csv'),'x':'condition_id, deterministic visual jitter',
            'y':'acquisition_contrast = pre_median - late_median',
            'uncertainty':'condition fish IQR; no individual interval or classifier'})
        for name,mask in [('figure4-condition-profiles',~groups.group_type.eq('individual_catch')),
                          ('figure4-individual-catches',groups.group_type.eq('individual_catch'))]:
            rows=groups[mask][['group_order','group_name']].drop_duplicates().sort_values('group_order')
            for row,(_,group) in enumerate(rows.iterrows()):
                for col,signal in enumerate(['signed_log_vigor','movement_probability','scaled_activity']):
                    registry.append({'artifact':bound(OUT/(name+'.svg')),'element_id':name+f'_axes_{row*3+col}',
                        'scientific_role':signal,'group_name':group.group_name,'group_order':int(group.group_order),
                        'source':bound(OUT/'group-bins.parquet'),'x':'time_s (s from CS)',
                        'y':signal+'_median','band':[signal+'_q25',signal+'_q75'],
                        'coverage':[signal+'_fish',signal+'_trials'],'conditions':['control','trace'],
                        'aggregation':'trial medians within fish, then equal-fish condition median/IQR'})
        for svg in sorted((OUT/'individual-atlas').glob('*.svg')):
            root=ET.fromstring(svg.read_bytes())
            assert root.get('width')=='183mm' and root.get('height')=='65mm'
            assert root.findall('.//{http://www.w3.org/2000/svg}text')
            assert root.findall('.//{http://www.w3.org/2000/svg}image')
            fish=svg.stem.removeprefix('atlas_')
            registry.append({'artifact':bound(svg),'fish_id':fish,
                'elements':{'heatmap':svg.stem+'_axes_0','trial_outcomes':svg.stem+'_axes_1','colorbar':svg.stem+'_axes_2'},
                'source_heatmap':bound(BASE/'fish'/f'{fish}-bins.parquet'),
                'source_trial_outcomes':bound(OUT/'observed-g-trial-outcomes.parquet'),
                'source_record':bound(BASE/'fish'/f'{fish}.json'),
                'heatmap_mapping':'acquisition_trial x bin_index -> delta_log_vigor; native managua_r ±0.25; undefined black',
                'trial_mapping':'trial_number -> legacy_log_median_difference; bars from acquisition-scores.csv',
                'status':'scientific draft; no classifier inputs or labels invented'})
        write(OUT/'draft-registry.json',{'mapping_basis':'explicit generator loops and authenticated tables; not inferred from appearance',
            'generator':bound(ROOT/'scripts/render_figures34_3strace.py'),'elements':registry,
            'coverage':'axes-level source mappings; not a full element-level freeze registry'})
        scores=pd.read_csv(OUT/'condition-score-summary.csv');coverage=pd.read_csv(OUT/'coverage-summary.csv')
        verification=read(OUT/'numerical-verification.json')
        analysis=read(OUT/'analysis.json')
        report=('# 3sTrace Figures 3 and 4 — scientific draft\n\n'
            'All results are retrospective, unfrozen and assay-specific. No shared assembly, selection or plan was changed.\n\n'
            '## What ran\n\n'
            '- Fresh processed-inventory reconciliation: 183 recordings; exactly four timing-flagged exclusions; 179 source-roster fish (90 Control, 89 3sTrace). This does not approve a C1 primary technical cohort.\n'
            '- Figure 3: authenticated G trial outcomes; Pre 5–14 minus late Train 40–64 continuous acquisition contrasts; complete heatmap/outcome atlas for all 179 fish. No example-fish filtering.\n'
            '- Figure 4: legacy_distal_angular_speed only; supported signed bout-log-vigor calculator at 0.5 s, baseline −20 to 0 s; movement probability; separate supported saved scaled total activity. Nine experiment-owned ten-trial blocks, pooled five catches and individual catches. Trials pool within fish before equal-fish condition median/IQR.\n'
            '- Corrected metric, movement, protocol and temporal-profile bytes, completion metadata and cached outcome hashes were authenticated. Focused-reader output exactly equalled the unmodified supported reader on 7,200 check bins. Unused camera/tracking bytes were not re-audited for the full run.\n'
            '- Existing scientific/figure regression suite: 13 tests passed. Numerical checks independently reconstruct pooled-catch aggregation and check the full 1,288,800-bin grid and missing-value mask.\n\n'
            '## Descriptive acquisition results\n\n'+table(scores)+'\n\n'
            'Each fish has an estimable score, but one finite observation per epoch is only mathematical estimability. Epoch counts remain explicit; no minimum scientific coverage rule or binary threshold is approved. Positive scores describe greater response suppression in the conditional positive moving-frame log-vigor outcome, not movement probability.\n\n'
            '## Coverage\n\n'+table(coverage)+'\n\n'
            f"There are {verification['control_unpaired_us_events_in_display']} recorded unpaired control reinforcer overlaps in the ±20 s CS windows, across {verification['controls_with_overlap']} controls. Exact fish/trial/latency rows are in control-unpaired-us-overlaps.csv. Conditioned paired-training timing passed the supported protocol check; the 13 s guide is not an observed US on catches.\n\n"
            'Saved temporal metadata retains historical aggregation tags: mean_bout_vigor for 121 fish and mean_total_activity for 58. The current supported temporal calculator does not branch on this tag; original source values and metadata remain intact. Saved Scaled total activity is explicitly provisional and separate from the recomputed signed signal. See temporal-config-variants.json.\n\n'
            '## Reviews and limits\n\n'
            '- figure3-score.html: score distribution and associated legend.\n'
            '- individual-atlas.html: all 179 authenticated individual heatmaps and observed outcomes.\n'
            '- figure4-condition-profiles.html: nine blocks and pooled catches, by condition.\n'
            '- figure4-individual-catches.html: the five configured catches.\n'
            '- Native SVGs and legends accompany each review. Nature-main is resolved once per render. Exact browser evidence and actual visual review are bound in layout-evidence.json and visual-review.json; unresolved layout/production coverage is retained.\n\n'
            'Gate L remains open: learner representation, calibration, scientific coverage, thresholds if any, and validation need approval. No categorical manifest, learner-stratified integrated command, classifier fit, population refit or independent learner claim was produced. C0/C1 approval remains separate. These same-data condition profiles do not validate learner classes. Figure 2 median-pooling display experiments did not change the G trial outcome.\n\n'
            'Reproduce with scripts/analyze_figures34_3strace.py, verify_figures34_3strace.py (inventory/numerical), render_figures34_3strace.py and check_figures34_3strace.cjs. Existing outputs are preserved; use the recorded per-fish caches for an interrupted run, and a newly scoped directory for a changed scientific recipe.\n')
        (OUT/'README.md').write_text(report,encoding='utf-8')
        files=[bound(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['publication.lock','artifact-manifest.json','assay-draft-state.json']]
        state={'scope':'3sTrace Figure 3/4 scientific draft review','scientific_selection':'none; unfrozen',
            'accepted_primary_cohort':False,'gate_L':'unresolved','shared_assembly_pointers_changed':False,
            'analysis':bound(OUT/'analysis.json'),'source_inventory':bound(OUT/'source-inventory.csv'),
            'numerical_verification':bound(OUT/'numerical-verification.json'),'visual_review':bound(OUT/'visual-review.json'),
            'layout_evidence':bound(OUT/'layout-evidence.json'),'layout_repair':bound(OUT/'layout-repair.json'),'registry':bound(OUT/'draft-registry.json'),
            'reviews':[{k:bound(OUT/(name+suffix)) for k,suffix in [('svg','.svg'),('html','.html'),('legend','-legend.md')]}
                for name in ['figure3-score','figure4-condition-profiles','figure4-individual-catches']],
            'atlas':{'html':bound(OUT/'individual-atlas.html'),'legend':bound(OUT/'individual-atlas-legend.md'),'fish':179},
            'code':[bound(ROOT/'scripts'/name) for name in ['analyze_figures34_3strace.py','verify_figures34_3strace.py','render_figures34_3strace.py','repair_figures34_3strace_layout.py','check_figures34_3strace.cjs','finalize_figures34_3strace.py']]}
        def guard():
            for item in files:verify(item)
            for item in read(OUT/'render-start.json')['inputs']:verify(item)
            for item in state['code']:verify(item)
        with artifact_staging(OUT,prefix='.publish-') as staging:
            manifest=staging/'artifact-manifest.json';write(manifest,{'files':files,'status':'scientific draft'})
            state['artifact_manifest']={'path':str(OUT/'artifact-manifest.json'),'sha256':bound(manifest)['sha256']}
            staged=staging/'assay-draft-state.json';write(staged,state)
            guard()
            publish_transaction(((manifest,OUT/'artifact-manifest.json'),(staged,OUT/'assay-draft-state.json')),
                staging,overwrite=False,verify_published=guard)
        print(report)
    finally:lock.unlink()

if __name__=='__main__':main()
