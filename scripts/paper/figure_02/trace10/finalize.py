"""Finalize saved-render/scientific evidence and guarded Figure 2 draft publication."""

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / 'pyproject.toml').is_file() and (p / 'src/classical_conditioning').is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / 'src'), str(_COMMAND_REPOSITORY / 'scripts')]

from pathlib import Path
import argparse,copy,json,re,sys,xml.etree.ElementTree as ET
REPO=_COMMAND_REPOSITORY
sys.path.insert(0,str(REPO/'src'))
from classical_conditioning.operations.candidate_paths import candidate_directory, evidence_file
from classical_conditioning.command_provenance import command_sources
import numpy as np
import pandas as pd
from classical_conditioning.figure_workspace import Snapshot,publish_state,tag,sha
from classical_conditioning.figures.layout_frames import assess
from scripts.analyze_figure2_trace10 import OUT,STATE,load,ref,save,check
from scripts.assemble_figure2_trace10 import BUILD,MODELS,DE,write,enc,numerical_inputs
from scripts.main_paper_figures import render as render_page

DEFAULT_FINAL=OUT/'final'

LEGEND='''Figure 2 | Population conditioning measures and exploratory comparisons.

(a–c) Control/conditioned population heatmaps: Delay, 28/29; 3sTrace, 90/89; 10sTrace, 32/29 fish. Direct 1-s median log positive detected-bout vigor, centred within fish/trial by finite baseline-bin medians [−15,0) s, then averaged with equal fish weight. Colours saturate at ±0.25; black is missing. Green: CS onset/offset; violet: expected US (10sTrace: 20 s during training). Display trials 1–90 correspond to acquisition 5–94.

(d–f) Fish/block median response-minus-baseline median log positive valid moving-frame vigor. Baseline [−15,0) s; response [0,9)/[0,13)/[0,20) s for Delay/3sTrace/10sTrace. PT/ET/LT: trials 10–14/65–69/90–94; minimum three eligible trials per five-trial block in every assay. Thin lines: paired fish; black: median/IQR. Display ±0.25; outside-view values remain in analysis. Two-sided exact sign tests (zero/paired changes) and Brunner–Munzel t tests (independent blocks/paired changes), separate Holm18 per panel. D values unchanged; E eligibility and D/E annotations corrected. Dagger: versus zero; stars: adjusted p<0.05/0.01/0.001/0.0001.

(g–i) Observed across-fish median log-vigor differences; pointwise 95% intervals from 5,000 whole-fish resamples (seed 10). Separate within-panel families: baseline-adjusted block interaction (Holm8), local condition mean (BH9), local slope (raw/BH9), phase-aware random-intercept/slope LMM change from average Pre5–14 (BH90). G row labels apply to H/I; H/I maximum three stars. Unavailable fits: n/a. Exploratory comparisons do not establish learning onset.

C/F/I share the approved 61-fish roster. Excluded 20230315_04 (timing flag) and 20250715_08 (short recording) remain in audit/sensitivity records. The 32 controls share historical 3sTrace acquisition identities, not independent new acquisitions across assays. Updated panels are unfrozen scientific drafts.'''

def prepare(final_dir=DEFAULT_FINAL):
    final_dir=candidate_directory(final_dir)
    numerical_inputs();e=load(BUILD/'assembly-evidence.json');check([e[k] for k in ['svg','html','legend','source','registry','roster','models','generator']])
    ticket=Snapshot([*command_sources(__file__),]);ticket.bind_hashes(load(BUILD/'starting-ticket.json'));ticket.check()
    ticket.bind_hashes(Snapshot([*command_sources(__file__),Path(__file__),BUILD/'assembly-evidence.json']).hashes)
    final_dir.mkdir(exist_ok=True);root=ET.fromstring(Path(e['svg']['path']).read_bytes())
    # Preserve initial bytes and fix only the generated I canvas's structural ID.
    for n in root.iter():
        if n.get('id')=='trace10-I__canvas-H':n.set('id','trace10-I__canvas-I')
        identity=n.get('id','')
        if identity.startswith('fig2__c') and '__vigor-cells-' in identity and n.tag==tag('line'):
            # Match the theme's physical 0.5pt guides through C's inherited scale.
            n.set('stroke-width',str(.5*25.4/72/(12.451279688759705/54.467716)))
    svg=ET.tostring(root,encoding='utf-8',xml_declaration=True)
    write(final_dir/'figure2.svg',svg);write(final_dir/'legend.md',LEGEND.encode())
    page=render_page([{'id':'fig2','svg_bytes':svg,'legend_text':LEGEND,'svg':ref(final_dir/'figure2.svg'),'legend':ref(final_dir/'legend.md'),'whole_figure_frozen':False}])[0];write(final_dir/'figure2.html',page)
    save(final_dir/'starting-ticket.json',ticket.hashes)
    save(final_dir/'candidate.json',{'svg':ref(final_dir/'figure2.svg'),'html':ref(final_dir/'figure2.html'),'legend':ref(final_dir/'legend.md'),'source_assembly':ref(BUILD/'assembly-evidence.json'),'generator':ref(Path(__file__)),'status':'candidate pending final render/scientific checks'})
    print(final_dir/'figure2.html')

def verify(final_dir=DEFAULT_FINAL, *, browser_evidence):
    final_dir=candidate_directory(final_dir,existing=True)
    browser_evidence=evidence_file(browser_evidence)
    browser_snapshot=Snapshot([browser_evidence,final_dir/"figure2.svg"])
    from scripts.trace10_panel_statistics import blocks,separate_tests
    roster,completion=numerical_inputs();e=load(BUILD/'assembly-evidence.json')
    root=ET.fromstring((final_dir/'figure2.svg').read_bytes());lookup={n.get('id'):n for n in root.iter() if n.get('id')}
    for key,h in e['protected_subtrees'].items():assert sha(ET.tostring(lookup[key]))==h
    report=load(browser_evidence);frame=assess((final_dir/'figure2.svg').read_bytes(),report)
    assert frame['status']=='passed',frame['issues']
    frame['browser_source']={'path':str(browser_evidence),'sha256':browser_snapshot.hashes[str(browser_evidence.resolve())]}
    save(final_dir/'layout-evidence.json',frame)
    ids=[r['recording_id'] for r in roster['selected']];plan=load(OUT/'plan.json')
    windows=pd.read_parquet(OUT/'windows.parquet');assert set(windows.fish_id)==set(ids) and len(windows)==61*90
    assert not windows.duplicated(['fish_id','trial_number']).any()
    pooled=pd.read_parquet(OUT/'C-pooled-bins.parquet');independent=[];coverage=[]
    for c in ['control','trace']:
        fish=[r['recording_id'] for r in roster['selected'] if r['condition']==c]
        arrays=[]
        for rid in fish:
            b=pd.read_parquet(OUT/'fish'/(rid+'-bins.parquet'))
            a=b.pivot(index='acquisition_trial',columns='bin_index',values='delta_log_vigor').reindex(index=range(5,95),columns=range(40)).to_numpy()
            assert np.nanmax(np.abs(np.nanmedian(a[:,5:20],axis=1)))<1e-12
            arrays.append(a)
        stack=np.array(arrays);expected=np.nanmean(stack,axis=0)
        actual=pooled.loc[pooled.condition_id.eq(c)].pivot(index='acquisition_trial',columns='bin_index',values='mean_delta_log_vigor').reindex(index=range(5,95),columns=range(40)).to_numpy()
        np.testing.assert_allclose(expected,actual,equal_nan=True,atol=1e-12)
        counts=np.isfinite(stack).sum(axis=0)
        actual_n=pooled.loc[pooled.condition_id.eq(c)].pivot(index='acquisition_trial',columns='bin_index',values='contributing_fish').reindex(index=range(5,95),columns=range(40)).to_numpy()
        np.testing.assert_array_equal(counts,actual_n)
        independent.append({'condition':c,'fish':len(fish),'matrix_sha256':sha(np.asarray(actual,dtype='<f8').tobytes()),'minimum_contributors':int(counts.min()),'maximum_contributors':int(counts.max())})
    for panel,wp in [('F',OUT/'windows.parquet'),('E',Path(load(DE/'calculation.json')['source_E_trials']['path']))]:
        expected=blocks(pd.read_parquet(wp));actual=pd.read_parquet((OUT if panel=='F' else DE)/(panel+'-fish-blocks.parquet'))
        pd.testing.assert_frame_equal(expected,actual)
        tests,_=separate_tests(panel,actual);saved=pd.read_csv((OUT if panel=='F' else DE)/(panel+'-statistics-Holm18.csv'))
        np.testing.assert_allclose(tests.p_holm18,saved.p_holm18,atol=1e-12)
        coverage.extend({'panel':panel,'condition':str(c),'block':str(b),'n':int(n)} for (c,b),n in actual.groupby(['condition_id','Selected block']).Eligible.sum().items())
    d=pd.read_parquet(DE/'D-fish-blocks.parquet');from analyze_figure2_trace3_matched import D_SOURCE
    pd.testing.assert_frame_equal(d,pd.read_parquet(D_SOURCE))
    dt,_=separate_tests('D',d);np.testing.assert_allclose(dt.p_holm18,pd.read_csv(DE/'D-statistics-Holm18.csv').p_holm18,atol=1e-12)
    coverage.extend({'panel':'D','condition':str(c),'block':str(b),'n':int(n)} for (c,b),n in d.groupby(['condition_id','Selected block']).Eligible.sum().items())
    all_windows=pd.read_parquet(OUT/'windows-observed.parquet')
    all_blocks=blocks(all_windows);sensitivity,_=separate_tests('F',all_blocks)
    sensitivity.to_csv(final_dir/'F-all63-sensitivity-Holm18.csv',index=False)
    all_windows.groupby(['condition_id','trial_number']).legacy_log_median_difference.agg(median='median',n='count').to_csv(final_dir/'I-all63-descriptive-sensitivity.csv')
    kept=windows.groupby(['condition_id','trial_number']).legacy_log_median_difference.agg(median='median',n='count')
    kept.to_csv(final_dir/'I-selected61-contributors.csv')
    pd.DataFrame(coverage).to_csv(final_dir/'DEF-contributing-fish.csv',index=False)
    protocol=[]
    for row in plan['cohort']:
        rid=row['recording_id'];p=row['protocol_audit'];lat=[a['latency_ms'] for t in p.get('training',[]) for a in t.get('near20',[])]
        # Audits use explicit source dictionaries; retain full original protocol for any alternate key layout.
        protocol.append({'recording_id':rid,'condition':row['condition'],'selected':rid in ids,'shared_control_identity':row['shared_control_identity'],'incomplete_protocol':row['incomplete_protocol'],'protocol_counts':p['counts'],'protocol_audit_bound_in_plan':ref(OUT/'plan.json')})
    diag=pd.read_csv(MODELS/'all-fit-diagnostics.csv')
    model_counts={name:len(pd.read_csv(MODELS/name)) for name in ['D-interaction-tests.csv','MR-local-block-tests.csv','phase-contrasts.csv']}
    assert model_counts=={'D-interaction-tests.csv':8,'MR-local-block-tests.csv':9,'phase-contrasts.csv':90}
    browser_snapshot.check()
    save(final_dir/'scientific-evidence.json',{'status':'passed scoped numerical/scientific checks; no freeze',
        'roster':ref(OUT/'common-roster.json'),'common_fish':61,'unique_control_acquisitions_shared_with_trace3':32,
        'C_independent_pooling_and_counts':independent,'DEF_separate_Holm18_recomputed':True,'D_values_exactly_preserved':True,
        'E_min3_recomputed_from_authenticated_trials':True,'coverage':coverage,'model_family_sizes':model_counts,
        'models':ref(MODELS/'completion.json'),'diagnostic_fit_count':len(diag),'primary_models_use_requested_random_slope':True,
        'available_outcomes_retained_for_excluded_fish':sorted(set(all_windows.fish_id)-set(ids)),
        'protocol_coverage':protocol,'excluded_sensitivity':{'F_all63':ref(final_dir/'F-all63-sensitivity-Holm18.csv'),'I_all63_descriptive':ref(final_dir/'I-all63-descriptive-sensitivity.csv')},
        'protected_subtrees':e['protected_subtrees'],'display_clipping':'D/E/F +/-0.25 and G/H/I inherited limits; all source values retained; out-of-view counts below',
        'out_of_view_F':int((pd.read_parquet(OUT/'F-fish-blocks.parquet').loc[lambda x:x.Eligible,'Fish median response / baseline'].abs()>.25).sum()),
        'limitations':['conditional on positive movement; not movement probability','shared controls are not new independent acquisitions across assays','boundary warnings retained; influence refits remain separate','exploratory model significance is not learning onset','production PDF/font/accessibility checks remain separate']})
    print('SCIENTIFIC AND SAVED-RENDER CHECKS PASSED')

def publish(note, final_dir=DEFAULT_FINAL):
    final_dir=candidate_directory(final_dir,existing=True)
    e=load(BUILD/'assembly-evidence.json');candidate=load(final_dir/'candidate.json');science=load(final_dir/'scientific-evidence.json');frame=load(final_dir/'layout-evidence.json')
    check([candidate[k] for k in ['svg','html','legend','source_assembly','generator']]);assert frame['status']=='passed' and frame['artifact_sha256']==candidate['svg']['sha256']
    snapshot=Snapshot([*command_sources(__file__),]);snapshot.bind_hashes(load(final_dir/'starting-ticket.json'))
    browser_source=frame.get('browser_source')
    if not browser_source:raise ValueError('Re-verify this candidate with explicit browser evidence before publication')
    path=str(evidence_file(browser_source['path']).resolve());digest=browser_source['sha256']
    snapshot.bind_hashes({path:digest})
    snapshot.check()
    state=load(STATE);d=state['draft'];registry=load(Path(d['semantic_registry']['path']))
    root=ET.fromstring(Path(candidate['svg']['path']).read_bytes());ids={n.get('id') for n in root.iter() if n.get('id')};frames=json.loads(next(n.text for n in root.iter() if n.get('id')=='layout-frames'))
    elements=registry.setdefault('element_registry',{})
    for key in list(elements):
        if str(elements[key].get('panel_id','')).lower() in ['c','d','e','f','i']:del elements[key]
    for p in ['D','E','F','I']:
        base=DE/'render' if p in ['D','E'] else BUILD;side=load(base/('panel-'+p+'.figure.json'));prefix=('matched-' if p in ['D','E'] else 'trace10-')+p+'__'
        registry.setdefault('matched_panel_registries',{})[p]=ref(base/('panel-'+p+'.figure.json'))
        for key,row in side['artist_registry'].items():
            k=prefix+key
            if k in ids:elements[k]={**row,'element_id':k,'panel_id':p.lower(),'source_panel_registry':ref(base/('panel-'+p+'.figure.json'))}
    elements.update(e['C_mappings'])
    for n in root.iter():
        key=n.get('id','')
        if key.startswith('fig2__c') and '__vigor-cells-' in key:elements[key]={'element_id':key,'panel_id':'c','scientific_role':'stimulus.guide' if 'CS-' in key or 'US20' in key else 'boundary.phase','source':ref(OUT/'plan.json'),'timing_s':20 if 'US20' in key else 10 if 'CS-offset' in key else 0 if 'CS-onset' in key else None}
    registry['presentation_layout']['allocations']=frames['allocations'];registry['presentation_layout']['final_size_mm']=frames['final_size_mm']
    registry['panel_relationships']=frames.get('panel_relationships',registry.get('panel_relationships',[]))
    registry['trace10_CFI_and_DE_correction']={'status':'verified scoped unfrozen scientific draft','roster':ref(OUT/'common-roster.json'),'scientific_evidence':ref(final_dir/'scientific-evidence.json'),'layout_evidence':ref(final_dir/'layout-evidence.json'),'assembly':ref(BUILD/'assembly-evidence.json'),'visual_review':note,'statistics':'separate Holm18 D/E/F; separate within-panel I model families','response_s':{'F_I':[0,20]},'common_roster_panels':['C','F','I'],'protected':['A','B','G','H']}
    registry.get('B_E_H_matched_expanded_cohort',{}).update({'D_E_settings_status':'superseded by trace10_CFI_and_DE_correction: minimum3 and separate Holm18; B/H unchanged'})
    axes=registry.setdefault('axis_inventory',{}).setdefault('axes',[])
    for a in axes:
        if a.get('panel') in ['C','D','E','F','I']:a['active_scientific_override']='trace10_CFI_and_DE_correction'
    for p in ['F','I']:
        for condition in (['control','trace'] if p=='F' else ['observed']):
            for dim in ['x','y']:axes.append({'id':p+'-'+condition+'-'+dim,'panel':p,'orientation':dim,'condition':condition,'quantity':('selected session block' if p=='F' else 'global CS trial') if dim=='x' else 'median log positive moving-frame response minus baseline','units':'categorical block' if dim=='x' and p=='F' else 'trial' if dim=='x' else 'dimensionless natural-log difference','group_ids':['trace10-'+p+'__axis__'+p.lower()+'-'+condition+'__'+dim],'scientific_registry':registry['matched_panel_registries'][p],'roster':ref(OUT/'common-roster.json')})
    registry['axis_inventory']['source_svg_sha256']=candidate['svg']['sha256']
    reg=final_dir/'semantic-registry.json';write(reg,enc(registry))
    history=final_dir/'previous-selected';history.mkdir(exist_ok=True)
    new=[]
    for name,p in [('state.json',STATE),('figure2.svg',Path(d['working_svg']['path'])),('figure2.html',Path(d['review']['path'])),('legend.md',Path(d['legend']['path'])),('semantic-registry.json',Path(d['semantic_registry']['path']))]:new.append((history/name,p.read_bytes()))
    svgpath=Path(d['working_svg']['path']);htmlpath=Path(d['review']['path']);legendpath=Path(d['legend']['path'])
    d['working_svg']={'path':str(svgpath),'sha256':candidate['svg']['sha256']};d['review']={'path':str(htmlpath),'sha256':candidate['html']['sha256']};d['legend']={'path':str(legendpath),'sha256':candidate['legend']['sha256'],'panel_ids':list('ABCDEFGHI')}
    d['semantic_registry']=ref(reg);d['layout_frames']=ref(final_dir/'layout-evidence.json');d['assembly_scale']['final_width_mm']=156.
    d['status']='C/F/I common61 approved10sTrace fish; D/E/F separate Holm18 with min3; unfrozen scientific drafts'
    d['trace10_scientific_draft']={'plan':ref(OUT/'plan.json'),'roster':ref(OUT/'common-roster.json'),'data':ref(OUT/'calculation.json'),'models':ref(MODELS/'completion.json'),'scope':['C','F','I','D/E authorized corrections'],'frozen':False,'scientific_evidence':ref(final_dir/'scientific-evidence.json')}
    d['matched_trace3_scientific_draft']['superseded_settings']='E minimum1 and joint D/E Holm36 superseded by trace10_scientific_draft; B/H and common179 roster retained'
    d['verification']={'layout_frames':ref(final_dir/'layout-evidence.json'),'scientific_data':ref(final_dir/'scientific-evidence.json'),'model_diagnostics':ref(MODELS/'completion.json'),'visual_review':{'status':'performed','note':note},'coverage':'Numerical identity, common roster, separate families, preserved A/B/G/H, exact600dpi paint/unclipped frame checks. Production font/PDF/accessibility and influence audits remain separate.'}
    d.setdefault('consistency_requests',[]).append({'id':'fig2-trace10-CFI-DE-min3-Holm18-20261020','scope':['A/B/C method','D/E/F method','G/H/I method','C/F/I common roster'],'author_direction':'Generate10sTrace panels matching columns1/2; separate panel stats; common min3; explicitly approved61-fish roster','status':'verified scoped numerical and exact-render evidence; no freeze','evidence':ref(final_dir/'scientific-evidence.json'),'presentation':ref(final_dir/'layout-evidence.json')})
    snapshot.bind_hashes(Snapshot([*command_sources(__file__),reg,final_dir/'candidate.json',final_dir/'layout-evidence.json',final_dir/'scientific-evidence.json',Path(__file__),Path(candidate['svg']['path']),Path(candidate['html']['path']),Path(candidate['legend']['path'])]).hashes)
    publish_state(STATE,state,snapshot,artifacts=new,evidence_updates=[(svgpath,Path(candidate['svg']['path']).read_bytes()),(htmlpath,Path(candidate['html']['path']).read_bytes()),(legendpath,Path(candidate['legend']['path']).read_bytes())])
    save(final_dir/'publication.json',{'status':'published-scientific-draft','state':ref(STATE),'svg':ref(svgpath),'html':ref(htmlpath),'legend':ref(legendpath),'visual_review':note,'no_freeze':True});print(htmlpath)

def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['prepare','verify','publish'])
    p.add_argument('--visual-review-note')
    p.add_argument('--candidate-dir',type=Path,default=DEFAULT_FINAL)
    p.add_argument('--browser-evidence',type=Path)
    a=p.parse_args()
    if a.operation=='prepare':prepare(a.candidate_dir)
    elif a.operation=='verify' and a.browser_evidence:verify(a.candidate_dir,browser_evidence=a.browser_evidence)
    elif a.operation=='publish' and a.visual_review_note:publish(a.visual_review_note,a.candidate_dir)
    else:p.error('verify requires --browser-evidence; publish requires an actual visual-review note')

if __name__=='__main__':cli()
