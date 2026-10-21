"""Guarded publication of only the preliminary grouped 3sTrace Figure 4B."""
import os
from pathlib import Path
import pandas as pd
from PIL import Image
from classical_conditioning.artifacts import artifact_staging,publish_transaction
from classical_conditioning.figures.layout_frames import assess
from analyze_figures34_3strace import OUT,ROOT,read,bound,write,verify

def main():
    dest=OUT/'figure4-classified-preliminary';lock=dest/'publication.lock'
    with lock.open('x') as f:f.write(str(os.getpid()))
    try:
        ticket=read(dest/'start-ticket.json');build=read(dest/'build-complete.json');analysis=read(dest/'analysis.json')
        inputs=[ticket['parent_state'],ticket['protected_figure3'],*analysis['inputs']]
        for item in [*inputs,*build['artifacts']]:verify(item)
        flow=pd.read_csv(dest/'sample-flow.csv');assert len(flow)==179 and flow.fish_id.nunique()==179
        assert flow.classifier_label.eq('Unclassified').sum()==122
        layout=assess((dest/'figure4.svg').read_bytes(),read(dest/'browser.json'));assert layout['status']=='passed',layout['issues']
        write(dest/'layout-evidence.json',layout)
        write(dest/'visual-review.json',{'artifact':bound(dest/'figure4.svg'),'png':bound(dest/'figure4.png'),
            'actual_inspection':'Complete saved PNG inspected. Ten profile rows, four classifier groups, counts, timing guides, and contribution limitation visible. No obvious collisions or clipped marks.',
            'png_pixels':list(Image.open(dest/'figure4.png').size),'png_dpi':300,
            'limits':['No complete Nature production audit','Retrospective legacy groups, no independent validation','122 current-roster fish not included in four classified curves; retained as Unclassified in sample flow']})
        codes=[bound(ROOT/'scripts'/name) for name in ['build_figure4_3strace_preliminary.py','check_figure4_3strace_preliminary.cjs','publish_figure4_3strace_preliminary.py']]
        write(dest/'draft-registry.json',{'scientific_mapping':'explicit generator loops and many-to-one label join; not inferred from appearance',
            'classifier_table':analysis['classifier_table'],'classifier_manifest':analysis['classifier_manifest'],
            'classifier_execution_id':analysis['classifier_execution_id'],'metric':analysis['metric'],
            'membership':bound(dest/'sample-flow.csv'),'group_data':bound(dest/'group-bins.parquet'),'within_fish_data':bound(dest/'fish-bins.parquet'),
            'roles':{'x':'time_s, s from CS onset','y':'signed_median, signed bout ln vigor with trial [-20,0) baseline','uncertainty':['signed_q25','signed_q75'],'coverage':['signed_fish','signed_trials'],'groups':analysis['strata'],'CS_s':[0,10],'expected_paired_training_US_s':13},
            'validation_mode':'descriptive_same_data','status':'axes/data-level draft registry, not full freeze classification','code':codes})
        artifacts=[bound(p) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='publication.lock']
        state={'schema_version':1,'figure_id':'figure4-3strace','scope':'3sTrace Figure 4B only','default_display':'draft','default_display_format':'html-svg-and-legend',
            'whole_figure_frozen':False,'scientific_selection':{'status':'preliminary, no accepted selection or Gate L approval','records':[]},
            'draft':{'working_svg':bound(dest/'figure4.svg'),'review':bound(dest/'review.html'),'legend':bound(dest/'legend.md'),'png':bound(dest/'figure4.png'),
                'registry':bound(dest/'draft-registry.json'),'layout_evidence':bound(dest/'layout-evidence.json'),'visual_review':bound(dest/'visual-review.json')},
            'analysis':bound(dest/'analysis.json'),'membership':bound(dest/'sample-flow.csv'),'group_counts':analysis['strata'],'classifier_execution_id':analysis['classifier_execution_id'],
            'protected_figure3':ticket['protected_figure3'],'parent_state':ticket['parent_state'],'shared_assembly_pointers_changed':False,
            'limits':['57 classified fish in four curves; 122 current-roster fish remain unclassified/without saved labels','same-data legacy WIP labels are not validated learner categories','no single-control-fish IQR','full-roster classification remains incomplete'],'code':codes}
        def guard():
            for item in [*inputs,*artifacts,*codes]:verify(item)
        with artifact_staging(dest,prefix='.publish-') as staging:
            manifest=staging/'artifact-manifest.json';write(manifest,{'artifacts':artifacts})
            state['artifact_manifest']={'path':str(dest/'artifact-manifest.json'),'sha256':bound(manifest)['sha256']}
            pointer=staging/'current-state.json';write(pointer,state)
            guard();publish_transaction(((manifest,dest/'artifact-manifest.json'),(pointer,dest/'current-state.json')),staging,overwrite=False,verify_published=guard)
        print('Published preliminary 3sTrace Figure 4B; Figure 3 preserved.')
    finally:lock.unlink()

if __name__=='__main__':main()
