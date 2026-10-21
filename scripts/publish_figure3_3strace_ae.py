"""Publish only the verified external 3sTrace Figure 3 A–E draft pointer."""
import os
from pathlib import Path
import pandas as pd
from PIL import Image
from classical_conditioning.artifacts import artifact_staging,publish_transaction
from classical_conditioning.figures.layout_frames import assess
from analyze_figures34_3strace import OUT,ROOT,read,write,bound,verify

def main():
    dest=OUT/'figure3-a-e';lock=dest/'publication.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        ticket=read(dest/'start-ticket.json');build=read(dest/'build-complete.json');inputs=read(dest/'build-inputs.json')['inputs']
        guard_inputs=[ticket['parent_state'],ticket['shared_figure3_state'],*ticket['inputs'],*inputs]
        for item in guard_inputs:verify(item)
        for item in build['artifacts']:verify(item)
        layout=assess((dest/'figure3.svg').read_bytes(),read(dest/'browser.json'));assert layout['status']=='passed',layout['issues']
        movement=pd.read_csv(dest/'example-movement-and-events.csv')
        assert len(movement)==270 and not movement.duplicated(['fish_id','trial_number']).any()
        for col in ['baseline_fraction_moving','response_fraction_moving']:
            assert movement[col].dropna().between(0,1).all()
        visual={'artifacts':[bound(dest/'figure3.svg'),bound(dest/'figure3.png')],
            'review':'Actual complete PNG inspected: A–E and shared key present; labels and plotted observations readable; no observed overlaps or missing panels.',
            'png_pixels':list(Image.open(dest/'figure3.png').size),'png_dpi':300,
            'limits':'Basic visual and exact-frame verification only; no complete Nature production audit or independent scientific validation.'}
        write(dest/'layout-evidence.json',layout);write(dest/'visual-review.json',visual)
        codes=[bound(ROOT/'scripts'/x) for x in ['build_figure3_3strace_ae.py','check_figure3_3strace_ae.cjs','publish_figure3_3strace_ae.py']]
        registry={'scientific_status':'unfrozen descriptive','code':codes,
            'A':{'outcome':'median ln positive valid moving-frame response vigor minus baseline median ln vigor','baseline_s':[-15,0],'response_s':[0,13],'score_epochs':{'pre':[5,14],'late_train':[40,64]}},
            'B':{'source':bound(OUT/'acquisition-scores.csv'),'fish':179,'conditions':{'control':90,'trace':89},'quantity':'acquisition_contrast'},
            'C_E':{'selection':build['examples'],'heatmap':'source-bound acquisition_trial x bin_index -> delta_log_vigor; 1 s direct logged-frame medians','trial_outcome':bound(OUT/'observed-g-trial-outcomes.parquet'),
                'movement_and_actual_US':bound(dest/'example-movement-and-events.csv'),'movement_definition':'sum DeltaTimeMs on valid moving frames / sum DeltaTimeMs on valid frames; FrameStep=1 and finite selected metric; windows match A','labels':'descriptive magnitude only, no learner/nonlearner classes'}}
        write(dest/'draft-registry.json',registry)
        artifacts=[bound(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='publication.lock']
        state={'schema_version':1,'figure_id':'figure3-3strace','scope':'Figure 3 A–E only','default_display':'draft','default_display_format':'html-svg-and-legend',
            'whole_figure_frozen':False,'scientific_selection':{'status':'none; unfrozen descriptive','records':[]},
            'draft':{'working_svg':bound(dest/'figure3.svg'),'review':bound(dest/'review.html'),'legend':bound(dest/'legend.md'),'png':bound(dest/'figure3.png'),
                'registry':bound(dest/'draft-registry.json'),'layout_evidence':bound(dest/'layout-evidence.json'),'visual_review':bound(dest/'visual-review.json'),
                'panels':['A','B','C','D','E'],'examples':build['examples'],'final_size_mm':[183,212]},
            'protected_shared_state':ticket['shared_figure3_state'],'parent_assay_state':ticket['parent_state'],
            'limits':['Retrospective illustrative selection, not learner validation','Gate L and primary-cohort approval remain unresolved','No shared assembly pointer changed'], 'code':codes}
        def guard():
            for item in [*guard_inputs,*artifacts,*codes]:verify(item)
        with artifact_staging(dest,prefix='.publish-') as staging:
            m=staging/'artifact-manifest.json';write(m,{'artifacts':artifacts})
            state['artifact_manifest']={'path':str(dest/'artifact-manifest.json'),'sha256':bound(m)['sha256']}
            s=staging/'current-state.json';write(s,state)
            guard();publish_transaction(((m,dest/'artifact-manifest.json'),(s,dest/'current-state.json')),staging,overwrite=False,verify_published=guard)
        print('Published verified Figure 3 A-E draft only.')
    finally:lock.unlink()

if __name__=='__main__':main()
