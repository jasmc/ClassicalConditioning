"""Independent acquisition-quality audit and a source-only protocol sensitivity."""
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
from .panel_g_run import sha,make_manifest,verify_stage,run_lock
from .panel_g_review import ReviewConfig,save_json
from .panel_g_legacy import run_cohort_sensitivities


def protocol_review_flag(acquisition):
    return bool(acquisition['alignment']['protocol_events_outside_acquisition'] or
                acquisition['protocol']['statistics']['invalid_duration_count']>0)


def run_technical_followup(parent,out):
    parent=Path(parent);out=Path(out);out.mkdir(exist_ok=False)
    expected=sha(parent/'artifact-manifest.json');verify_stage(parent,expected)
    for name in ['model-input.parquet','input-lineage.json','protected-originals.json']:
        shutil.copyfile(parent/name,out/name)
    def protected():
        for e in json.loads((out/'protected-originals.json').read_text()):
            if sha(Path(e['path']))!=e['sha256']:raise ValueError('Original G changed')
    with run_lock(out):
        source_root=Path(__file__).resolve().parents[4]
        source_files=[*sorted(Path(__file__).parent.glob('panel_g_*.py')),
                      source_root/'scripts/review_figure2_g_technical.py']
        snapshots=out/'source-snapshots';snapshots.mkdir()
        starting=[]
        for i,source in enumerate(source_files):
            # Module roots differ from the repository root by src/package/
            # analysis/inference. Resolve the wrapper explicitly below.
            if not source.is_file():
                source=Path(__file__).resolve().parents[4]/'scripts'/source.name
            if not source.is_file():raise ValueError('Missing technical source snapshot: '+str(source))
            starting.append({'path':str(source),'sha256':sha(source)})
            shutil.copyfile(source,snapshots/f'{i:02d}-{source.name}')
        save_json(out/'technical-sources.json',starting)
        protected();lineage=json.loads((out/'input-lineage.json').read_text())
        data=pd.read_parquet(out/'model-input.parquet');condition=data.groupby('fish_id',observed=True).condition_id.first().astype(str).to_dict()
        sources={(Path(e['path']).parent.name,Path(e['path']).name):e for e in lineage}
        rows=[];events=[];membership=[];used=[]
        for fish in sorted(condition):
            records=[sources[(fish,n)] for n in ['acquisition_summary.json','corrected-v1_preprocessing_summary.json','stimulus_events.parquet']]
            for e in records:
                if sha(Path(e['path']))!=e['sha256']:raise ValueError('Changed quality source')
            used.extend(records)
            acquisition=json.loads(Path(records[0]['path']).read_text());pre=json.loads(Path(records[1]['path']).read_text())
            camera=acquisition['camera']['statistics']['frames'];tracking=acquisition['tracking']['statistics']['frames']
            flag=protocol_review_flag(acquisition)
            rows.append({'fish_id':fish,'condition_id':condition[fish],'acquisition_status':acquisition['status'],
                'camera_missing_frames':camera['missing_frame_count'],'tracking_missing_frames':tracking['missing_frame_count'],
                'camera_duplicate_frames':camera['duplicate_events'],'tracking_duplicate_frames':tracking['duplicate_events'],
                'camera_reversed_frames':camera['reverse_events'],'tracking_reversed_frames':tracking['reverse_events'],
                'boundary_nonoverlap_frames':acquisition['alignment']['stream_nonoverlap_frame_span'],
                'coordinate_valid_fraction':pre['frame_valid_count']/pre['matched_frame_count'],
                'derivative_valid_fraction':pre['derivative_valid_count']/pre['matched_frame_count'],
                'invalid_protocol_durations':acquisition['protocol']['statistics']['invalid_duration_count'],
                'protocol_events_outside_acquisition':pre['protocol_timing']['events_outside_acquisition'],
                'protocol_review_flag':flag,'warnings':'; '.join(acquisition.get('warnings',[]))})
            p=pd.read_parquet(records[2]['path']);p['original_type_ordinal']=p.groupby('Type',sort=False).cumcount()+1
            low,high=pre['protocol_timing']['camera_absolute_ms_span']
            bad=p.loc[(p.End<p.Beg)|(p.Beg<low)|(p.End>high)]
            for r in bad.itertuples():events.append({'fish_id':fish,'condition_id':condition[fish],
                'type':r.Type,'original_type_ordinal':r.original_type_ordinal,'Beg':r.Beg,'End':r.End,
                'scope':'malformed/outside event evidence; intended timestamp not reconstructed'})
            membership.append({'recipe':'exclude_any_protocol_review_flag','condition_id':condition[fish],
                'fish_id':fish,'included':not flag,'failed_rules':'raw_protocol_timing_flag' if flag else '',
                'role':'conservative technical sensitivity; primary G unchanged'})
        pd.DataFrame(rows).to_csv(out/'technical-recording-audit.csv',index=False)
        pd.DataFrame(events).to_csv(out/'technical-protocol-rows.csv',index=False)
        pd.DataFrame(membership).to_csv(out/'legacy-sensitivity-membership.csv',index=False)
        save_json(out/'technical-policy-before-fitting.json',{'criterion':'exclude any raw protocol invalid duration or out-of-acquisition event',
            'registered_before_effect_fitting':True,'primary_approved':False,
            'boundary_offsets_alone_are_not_this_exclusion_rule':True,
            'sources':used,'parent':str(parent),'parent_manifest_sha256':expected,
            'source_script':{'path':str(Path(__file__)),'sha256':sha(Path(__file__))}})
        run_cohort_sensitivities(out,ReviewConfig(),workers=1,nboot=199)
        protected();verify_stage(parent,expected)
        for e in starting:
            if sha(Path(e['path']))!=e['sha256']:raise ValueError('Technical source changed during fitting')
        save_json(out/'technical-followup-complete.json',{'status':'audit_and_named_sensitivity_complete',
            'original_G_unchanged':True,'primary_cohort_changed':False})
        make_manifest(out)
