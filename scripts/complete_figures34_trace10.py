"""Verify and publish only the independent 10sTrace draft state on J:."""
from pathlib import Path
import html, io, json, sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from analyze_figures34_trace10 import OUT,DATA,PROJECT,load,ref,save,check
from classical_conditioning.figure_workspace import Snapshot,publish_state
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.layout_frames import assess
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme,style_axes

def atlas():
    scores=pd.read_csv(OUT/'acquisition-scores-all63.csv');trials=pd.read_parquet(OUT/'observed-trial-outcomes-all63.parquet')
    cards=[]; sources=[]; limits=[float(trials.legacy_log_median_difference.min())-.02,float(trials.legacy_log_median_difference.max())+.02]
    profile=load_presentation_profile('nature-main')
    with mpl.rc_context():
        theme=apply_theme(profile=profile)
        for row in scores.sort_values(['selected','condition_id','fish_id'],ascending=[False,True,True]).itertuples():
            fig=plt.figure(figsize=(3.5,3.3));fig.set_layout_engine('none')
            ax=fig.add_axes([.18,.45,.78,.48]);bp=DATA/'fish'/f'{row.fish_id}-bins.parquet'
            if row.selected:
                rec=load(DATA/'fish'/f'{row.fish_id}.json')
                bound=next(r for r in rec['outputs'] if Path(r['path'])==bp);check(bound);sources.append(bound)
                bins=pd.read_parquet(bp)
                matrix=bins.pivot(index='acquisition_trial',columns='bin_index',values='delta_log_vigor').reindex(index=range(5,95),columns=range(40)).to_numpy()
                cmap=mpl.colormaps['managua_r'].copy();cmap.set_bad('black')
                ax.imshow(np.ma.masked_invalid(matrix),extent=[-20,20,94.5,4.5],aspect='auto',interpolation='nearest',cmap=cmap,vmin=-.25,vmax=.25)
                ax.axvline(0,color=theme.cs_color,lw=theme.cs_linewidth);ax.axvline(10,color=theme.cs_color,lw=theme.cs_linewidth,ls='--')
                ax.set(xticks=[-20,0,20],yticks=[5,14,64,94]);style_axes(ax,theme=theme,xlabel='Time from CS (s)',ylabel='CS trial')
            else:
                ax.axis('off');ax.text(.5,.5,'Excluded from selected roster\nHeatmap not regenerated\nAvailable trial outcomes below',ha='center',va='center',transform=ax.transAxes,fontsize=theme.font_size)
            ax=fig.add_axes([.18,.14,.78,.18]);g=trials.loc[trials.fish_id.eq(row.fish_id)]
            ax.scatter(g.trial_number,g.legacy_log_median_difference,s=4,color='#00aeef' if row.condition_id=='control' else '#913619',linewidths=0)
            for a,b,v in [(5,14,row.pre_median),(40,64,row.late_median)]:ax.plot([a,b],[v,v],color='black',lw=1)
            ax.set(xlim=(4,95),ylim=limits,xticks=[5,40,64,94]);style_axes(ax,theme=theme,xlabel='CS trial',ylabel='Δ ln vigor')
            buf=io.StringIO();fig.savefig(buf,format='svg');plt.close(fig)
            status='Selected roster' if row.selected else 'EXCLUDED: '+row.exclusion_reason
            cards.append('<article><h3>'+row.fish_id+' · '+row.condition_id+'</h3>'+buf.getvalue()+'<p>'+html.escape(f'{status}. Contrast {row.acquisition_score:+.4f}; finite Pre {row.pre_n}/10; late Train {row.late_n}/25; observed outcomes {row.finite_outcomes}/{row.observed_trials}. Shared 3sTrace acquisition: {row.shared_control_identity}.')+'</p></article>')
    legend='All 63 processed identities. Selected roster: 32 Control and 29 10sTrace. Two excluded identities retain available outcomes. Heatmaps: direct 1 s median positive eligible moving-frame ln vigor, centred by finite baseline-bin medians [−15,0) s; managua_r saturates at ±0.25, black undefined; green CS 0/10 s. Trial dots: response [0,20) minus baseline [−15,0) median ln vigor; black bars: Pre 5–14 and late Train 40–64 medians; the contrast is Pre minus late Train. All cards share outcome limits, preserving observed extrema. No labels, thresholds, individual uncertainty or validation are approved. Undefined scores remain explicit. Controls share all3sTrace acquisition identities. This is a review atlas, not a submission page layout.'
    with (OUT/'all-fish-heatmap-atlas.html').open('x',encoding='utf-8') as f:f.write('<!doctype html><meta charset="utf-8"><style>body{font:14px Arial;color:black;background:white}main{display:grid;grid-template-columns:repeat(3,1fr)}article{border:1px solid #ddd;padding:10px}svg{width:100%;height:auto}</style><p>'+html.escape(legend)+'</p><main>'+''.join(cards)+'</main>')
    save(OUT/'atlas-source-manifest.json',{'heatmaps':sources,'trial_data':ref(OUT/'observed-trial-outcomes-all63.parquet'),'scores':ref(OUT/'acquisition-scores-all63.csv'),'outcome_limits':limits,'cards':63,'heatmaps_rendered':61,'excluded_heatmaps_regenerated':False,'legend':legend,'profile_inputs':profile.inputs,'production_layout_verified':False})
    print('Atlas complete: 63 cards, 61 authenticated heatmaps')

def finalize():
    state=OUT/'current-state.json'; snapshot=Snapshot([state,Path(__file__),ROOT/'scripts/analyze_figures34_trace10.py',ROOT/'scripts/render_figures34_trace10.py',ROOT/'scripts/check_figures34_trace10.cjs'])
    scores=pd.read_csv(OUT/'acquisition-scores-all63.csv');selected=scores.loc[scores.selected]
    trials=pd.read_parquet(OUT/'profile-trial-bins.parquet'); fish=pd.read_parquet(OUT/'profile-fish-bins.parquet');group=pd.read_parquet(OUT/'profile-condition-bins.parquet')
    assert len(selected)==61 and len(scores)==63 and trials.fish_id.nunique()==61
    assert not trials.duplicated(['fish_id','trial_number','time_s']).any()
    assert len(trials)==61*90*80
    assert not fish.duplicated(['fish_id','group_name','time_s']).any()
    assert not group.duplicated(['condition_id','group_name','time_s']).any()
    merged=fish.merge(scores[['fish_id','acquisition_score','estimable','acquisition_identity']],on='fish_id',validate='many_to_one')
    assert len(merged)==len(fish) and merged.acquisition_identity.notna().all()
    merged.to_parquet(OUT/'profile-fish-bins-with-continuous-scores.parquet',index=False)
    signals=['signed_log_vigor','movement_probability','scaled_activity']
    for s in signals:
        assert trials.loc[trials.coverage.lt(.9)|trials.coverage.isna(),s].isna().all()
        # Independent per-group count and numerical center check against saved fish table.
        for key,g in fish.groupby(['condition_id','group_name','time_s'],observed=True):
            out=group.loc[group.condition_id.eq(key[0])&group.group_name.eq(key[1])&group.time_s.eq(key[2])].iloc[0]
            vals=g[s].dropna().to_numpy();assert out[s+'_fish']==len(vals)
            if len(vals): np.testing.assert_allclose(out[s+'_median'],np.median(vals),rtol=1e-12,atol=1e-12)
    # Explicit pooled-catch check, including trial 65 in catch and Test1.
    caught=trials.loc[trials.trial_number.isin([25,39,53,59,65])]
    expected=caught.groupby(['fish_id','time_s'])[signals].median().sort_index()
    actual=fish.loc[fish.group_type.eq('pooled_catch')].set_index(['fish_id','time_s'])[signals].sort_index()
    np.testing.assert_allclose(actual,expected,equal_nan=True)
    for name in ['figure3','figure4']:
        result=assess((OUT/(name+'.svg')).read_bytes(),load(OUT/(name+'-browser.json')))
        save(OUT/(name+'-layout-evidence.json'),result)
        assert result['status']=='passed',result['issues']
    # Recheck immutable starting authority/cached inputs; no shared pointer is published.
    starting=load(OUT/'starting-ticket.json')
    for p,digest in starting.items():
        assert sha256_file(Path(p))==digest,p
    snapshot.hashes.update(starting)
    for item in load(OUT/'atlas-source-manifest.json')['heatmaps']:
        snapshot.hashes[str(Path(item['path']).resolve())]=item['sha256']
    for relative,digest in load(OUT/'atlas-source-manifest.json')['profile_inputs'].items():
        snapshot.hashes[str((ROOT/relative).resolve())]=digest
    for recording in load(OUT/'profile-source-manifest.json')['recordings']:
        # Completion metadata and saved profiles are compact immutable inputs;
        # their source frame hashes were authenticated by the supported reader.
        for item in recording['inputs']:
            if item['path'].endswith('.json') or 'temporal_outcomes' in Path(item['path']).name:
                snapshot.hashes[str(Path(item['path']).resolve())]=item['sha256']
    check(load(DATA/'final-spacing/scientific-evidence.json')['roster'])
    timing_rows=[]
    for row in scores.itertuples():
        pp=PROJECT/'Processed data'/row.fish_id/'stimulus_events.parquet'
        provenance=load(DATA/'fish'/f'{row.fish_id}.json') if (DATA/'fish'/f'{row.fish_id}.json').exists() else next(x for x in load(DATA/'calculation.json')['issues'] if x['recording_id']==row.fish_id)
        check(next(x for x in provenance['window_sources'] if Path(x['path'])==pp))
        protocol=pd.read_parquet(pp);cs=protocol.loc[protocol.Type.eq('Cycle')].sort_values('Beg');us=protocol.loc[protocol.Type.eq('Reinforcer'),'Beg'].to_numpy(float)
        latencies=[]
        if row.condition_id=='trace':
            for onset in cs.Beg.iloc[14:64]:
                delta=(us-float(onset))/1000.;latencies.extend(delta[np.abs(delta-20)<=.1].tolist())
        timing_rows.append({'fish_id':row.fish_id,'selected':row.selected,'condition_id':row.condition_id,'observed_cs':len(cs),'nominal_training_us_observations':len(latencies),'minimum_training_us_s':min(latencies) if latencies else None,'maximum_training_us_s':max(latencies) if latencies else None,'training_us_before20s':int(sum(t<20 for t in latencies)),'training_us_before19p9s':int(sum(t<19.9 for t in latencies)),'protocol':ref(pp)})
    save(OUT/'all-inventory-protocol-audit.json',{'fish':timing_rows,'score_response_window_s':[0,20],'timing_interpretation':'nominal response window retained; any recorded US before 20 s is explicitly counted, not silently shifted; existing strict-pre-US sensitivity applies only to its source candidate shortlist'})
    summary=[]
    for c,g in selected.groupby('condition_id'):
        summary.append({'condition':c,'cohort':len(g),'estimable':int(g.estimable.sum()),'score_median':float(g.acquisition_score.median()),'score_min':float(g.acquisition_score.min()),'score_max':float(g.acquisition_score.max()),'pre_n_range':[int(g.pre_n.min()),int(g.pre_n.max())],'late_n_range':[int(g.late_n.min()),int(g.late_n.max())]})
    coverage={s:{c:{'min_fish_per_bin':int(g[s+'_fish'].min()),'max_fish_per_bin':int(g[s+'_fish'].max()),'zero_fish_bins':int(g[s+'_fish'].eq(0).sum())} for c,g in group.groupby('condition_id')} for s in signals}
    review=load(OUT/'visual-review.json')
    for x in review['artifacts']:check(x)
    report=('10sTrace Figures 3/4 — descriptive scientific drafts\n\n'
            'Ran: authenticated cached-outcome acquisition contrasts for all63 identities, supported Figure4 signed profile recalculation for selected61 fish, saved scaled-activity/movement profiles, fish-first condition summaries for nine blocks, pooled catches and five individual catches, and Nature-profile SVG/HTML reviews. No refit or raw processing ran.\n\n'
            'Selected roster:32Control/29Trace; not a newly frozen C1 cohort. Score estimable:31Control/28Trace. Undefined:20250715_10(Control; no finite lateTrain) and20250715_06(Trace; no finite Pre). Both remain in the cohort. The two author-excluded fish20230315_04 and20250715_08 remain in the63-identity audit with available outcomes. All32controls retain author-authenticated historical all3sTrace acquisition linkage. Current checked 3sTrace inventory/source manifests overlap19of these controls, with identical camera/protocol/tracking raw-source hashes;13historically linked controls have no3sTrace processed manifest in the checked project. Do not infer32current-population overlaps or count shared acquisitions independently. Exact identity records are in shared-control-source-links.json.\n\n'
            'Outcome:median ln positive valid moving-frame vigor response[0,20) minus baseline[−15,0); score=Pre5–14 minus lateTrain40–64 medians. Figure2 median-pooling display experiments do not change this estimator. Mathematical estimability is not a coverage approval. Reproduced all63 saved candidate-selection scores; provisional flags and thresholds were not consumed.\n\n'
            'Profiles:0.5s bins, signed bout-log vigor with[−20,0) baseline, movement probability and separately defined scaled activity. Coverage<0.9 is missing; no-bout signed values are undefined. Trials pooled within fish, then condition median/IQR. Catches25,39,53,59,65;65also appears in Test1. Each selected conditioned fish passes the supported recorded-US timing gate. All per-bin fish/trial counts are saved. The score is joined continuously to fish bins without creating categories.\n\n'
            'Figure3/4 have no accepted scientific selections. This run does not update shared assemblies, plans, methods, Figures1/2 or freezes. GateL remains open: representation, coverage/precision, calibration, thresholds if any, independent validation and permitted claims. No categorical figure4-analyze or complete post-classifier integrated pipeline was run; it requires unavailable approved learner/cohort manifests. Same-data results and deterministic examples are descriptive only. Figure4 is condition-wide provisional work that remains useful independently. Nature profile and actual saved-render frame checks passed; submission compliance and atlas page layout are not certified.\n\n'
            'Numerical summary\n'+json.dumps(summary,indent=2)+'\n\nProfile coverage\n'+json.dumps(coverage,indent=2)+'\n')
    with (OUT/'report.md').open('x',encoding='utf-8') as f:f.write(report)
    save(OUT/'verification.json',{'summary':summary,'profile_coverage':coverage,'trial_bins':len(trials),'fish_bins':len(fish),'condition_bins':len(group),'unique_keys':'passed','low_coverage_mask':'passed','independent_equal_fish_median_check':'passed','pooled_catch_including65':'passed','many_to_one_continuous_join':'passed','starting_authority_recheck':'passed','scientific_acceptance':'none; Gate L unresolved','protocol_audit':ref(OUT/'all-inventory-protocol-audit.json'),'shared_control_provenance':ref(OUT/'shared-control-source-links.json'),'supported_method_tests':ref(OUT/'supported-method-tests.txt'),'report':ref(OUT/'report.md')})
    code_dir=OUT/'source-code';code_dir.mkdir(exist_ok=False)
    for p in [ROOT/'scripts/analyze_figures34_trace10.py',ROOT/'scripts/render_figures34_trace10.py',ROOT/'scripts/check_figures34_trace10.cjs',Path(__file__)]:
        with (code_dir/p.name).open('xb') as f:f.write(p.read_bytes())
    files=[p for p in OUT.iterdir() if p.is_file() and p.name not in ['current-state.json','artifact-manifest.json']]
    save(OUT/'artifact-manifest.json',{'files':[ref(p) for p in files],'code':[ref(p) for p in [ROOT/'scripts/analyze_figures34_trace10.py',ROOT/'scripts/render_figures34_trace10.py',ROOT/'scripts/check_figures34_trace10.cjs',Path(__file__)]],'source_snapshots':[ref(p) for p in code_dir.iterdir()]})
    snapshot.hashes.update({str(p.resolve()):sha256_file(p) for p in files+[OUT/'artifact-manifest.json']})
    record={'scope':'10sTrace Figures3/4 independent scientific draft','scientific_selection':{'status':'unfrozen; no accepted selection','records':[]},'Gate_L':'unresolved','default_display':'draft','draft':{name:{k:ref(OUT/(name+suffix)) for k,suffix in [('working_svg','.svg'),('review','.html'),('legend','-legend.md'),('layout_evidence','-layout-evidence.json')]} for name in ['figure3','figure4']},'atlas':ref(OUT/'all-fish-heatmap-atlas.html'),'report':ref(OUT/'report.md'),'manifest':ref(OUT/'artifact-manifest.json'),'shared_assembly_pointers_updated':False,'whole_figure_frozen':False}
    publish_state(state,record,snapshot)
    print(json.dumps({'summary':summary,'profile_coverage':coverage,'state':ref(state)},indent=2))

if __name__=='__main__':{'atlas':atlas,'finalize':finalize}[sys.argv[1]]()
