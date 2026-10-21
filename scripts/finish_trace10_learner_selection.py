"""Finalize the audited illustrative recommendation without replacing Figure 1."""
from pathlib import Path
import argparse,copy,json,math,sys,xml.etree.ElementTree as ET
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont,FontProperties
import numpy as np
import pandas as pd
from classical_conditioning.figure_workspace import Snapshot,publish_state,sha,tag
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme,style_axes
from classical_conditioning.figures.export import FigureProvenance,FigureMode,export_matplotlib_figure
from classical_conditioning.figures.layout_frames import assess
from analyze_trace10_learners import OUT,DATA,PROJECT,STATE,METHOD,CURRENT,load,ref,save,check

PREFERRED='20230309_07'
TEMP=Path(r'C:\Users\joaquim\.codex\visualizations\2026\10\18\01a14e0b-0191-7822-b4d7-47417feb0eff')
COMPARISON_LEGEND='''Fixed-10sTrace candidate comparison. Shared direct 1-s original-frame positive detected-bout median log-vigor heatmaps, centred within trial by median finite baseline-bin medians [−15,0) s; managua_r ±0.25, missing black. Acquisition 5–94 is displayed 1–90. Solid/dashed green: CS onset/10-s offset; violet: nominal training US at 20 s, checked against recorded events. Below: measured-time median log positive valid moving-frame response [0,20) minus baseline [−15,0); shared display ±0.7. Black bars show the scored Pre5–14 and lateTrain40–64 medians. Rank is among 24 feature-eligible conditioned fish; all 29 conditioned fish and 32 controls remain in the common roster. All 63 acquisition identities remain in the coverage/exclusion table. Preferred illustration: 20230309_07; comparator: 20230307_04. This is descriptive candidate selection, not a validated learner category, Figure 1 replacement or freeze.'''

def enc(x):return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def write(p,b):
    with Path(p).open('xb') as f:f.write(b)
def html_page(svg,legend):return ('<!doctype html><html><meta charset="utf-8"><style>body{margin:20px;background:white;color:black;font:14px Arial}main{max-width:1100px;margin:auto}svg{display:block;width:100%;height:auto}p{line-height:1.45}</style><main>'+svg.decode()+'<p>'+legend+'</p></main></html>').encode()

def prepare():
    calc=load(OUT/'calculation.json');check(calc['outputs']);comparison=load(OUT/'comparison.json');check([comparison[k] for k in ['svg','html','legend','registry','generator']])
    strict=load(OUT/'strict-pre-US-sensitivity.json');check([strict['actual_US_values']]+[r['strict_pre_US_values'] for r in strict['rows']])
    parents=[REPO/'docs/paper/figures/figure-01/current-state.json',REPO/'docs/paper/figures/figure-02/current-state.json']
    snapshot=Snapshot([*parents,STATE,Path(__file__),OUT/'comparison.json',OUT/'candidate-comparison.svg',OUT/'candidate-comparison.html',OUT/'comparison-legend.md',OUT/'calculation.json',OUT/'full-score-coverage-exclusion-table.csv',OUT/'strict-pre-US-sensitivity.json',METHOD])
    STATE.parent.mkdir(parents=True,exist_ok=True);save(OUT/'publication-starting-ticket.json',snapshot.hashes)
    scores=pd.read_csv(OUT/'full-score-coverage-exclusion-table.csv');original=scores.copy()
    # Correct a secondary audit field: trial coverage contains CS and US alignments.
    # Ranking/eligibility/outcome scores were computed from the CS-only windows and are unchanged.
    for index,row in scores.iterrows():
        c=pd.read_parquet(Path(row.tracking_coverage_source));c=c.loc[c.metric_id.eq('legacy_distal_angular_speed')&c.alignment.eq('CS')]
        for a,b in [(5,14),(40,64)]:scores.loc[index,f'valid_response_fraction_{a}_{b}']=c.loc[c.trial_number.between(a,b),'response_valid_fraction'].median()
    unchanged=[c for c in scores if not c.startswith('valid_response_fraction_')];pd.testing.assert_frame_equal(scores[unchanged],original[unchanged])
    scores.to_csv(OUT/'full-score-coverage-exclusion-table-verified.csv',index=False)
    save(OUT/'coverage-audit-correction.json',{'status':'verified CS-alignment audit fields','old_table':ref(OUT/'full-score-coverage-exclusion-table.csv'),'authoritative_table':ref(OUT/'full-score-coverage-exclusion-table-verified.csv'),'reason':'Supported coverage parquet contains both CS and US alignment rows; secondary tracking summaries now explicitly use CS only.','scores_ranks_feature_eligibility_and_operational_flags_unchanged':True})
    preferred=scores.set_index('fish_id').loc[PREFERRED];baseline=scores.set_index('fish_id').loc[CURRENT]
    profile=load_presentation_profile('nature-main');width,height=80.,96.;bp=DATA/'fish'/(PREFERRED+'-bins.parquet')
    bins=pd.read_parquet(bp);matrix=bins.pivot(index='acquisition_trial',columns='bin_index',values='delta_log_vigor').reindex(index=range(5,95),columns=range(40)).to_numpy()
    with mpl.rc_context():
        theme=apply_theme(profile=profile);fig=plt.figure(figsize=(width/25.4,height/25.4));fig.set_layout_engine('none');fig.patch.set_gid('preferred-canvas')
        ax=fig.add_axes([20/width,13/height,44/width,72/height]);cmap=mpl.colormaps['managua_r'].copy();cmap.set_bad('black')
        image=ax.imshow(np.ma.masked_invalid(matrix),extent=(-20,20,90.5,.5),aspect='auto',interpolation='nearest',cmap=cmap,vmin=-.25,vmax=.25);image.set_gid('preferred-heatmap-values')
        ax.set(xlim=(-20,20),ylim=(90.5,.5),xticks=[-20,0,10,20],yticks=[1,10,30,60,90])
        style_axes(ax,theme=theme,xlabel='Time from CS onset (s)',ylabel='Trial')
        for t,ls in [(0,'-'),(10,'--')]:ax.axvline(t,color=theme.cs_color,lw=theme.cs_linewidth,ls=ls)
        ax.plot([20,20],[10.5,60.5],color=theme.us_color,lw=theme.us_linewidth,ls=':',clip_on=False)
        for y in [10.5,60.5]:ax.axhline(y,color='white',lw=theme.axes_linewidth)
        ax.set_title('10sTrace · '+PREFERRED,fontsize=theme.axes_titlesize,pad=theme.axes_titlepad)
        extra=[]
        for label,y in [('Pre',5.5),('Train',35.5),('Test',75.5)]:
            artist=ax.text(-.34,y,label,transform=ax.get_yaxis_transform(),rotation=90,ha='center',va='center',fontsize=theme.font_size);artist.set_gid('phase-'+label)
        ca=fig.add_axes([68/width,29/height,2/width,40/height]);bar=fig.colorbar(image,cax=ca,ticks=[-.25,0,.25]);bar.ax.set_title('Δ ln vigor',fontsize=theme.font_size,pad=theme.axes_titlepad);bar.ax.tick_params(labelsize=theme.tick_labelsize)
        spec={'scope':'panel','final_size_mm':[width,height],'inset_mm':1.,'allocations':[{'id':'preferred-example','kind':'panel','box_mm':[0,0,width,height],'group_ids':['axes__preferred__main','axes__preferred-colourbar__main']}],'background_ids':['preferred-canvas'],'alignments':[],'data_axes':[]}
        provenance=FigureProvenance(figure_id='Preferred fixed10sTrace illustrative candidate',analysis_recipe='selected-direct1s-original-frame-log-median',source_file=str(Path(__file__)),source_symbol='prepare',source_hash=ref(Path(__file__))['sha256'],reproduction_snippet='.venv-trace/Scripts/python.exe scripts/finish_trace10_learner_selection.py prepare',input_artifacts=(ref(bp),ref(OUT/'full-score-coverage-exclusion-table-verified.csv'),ref(OUT/'strict-pre-US-sensitivity.json')),artist_mappings={'preferred-heatmap-values':{'scientific_role':'heatmap.vigor','fish_id':PREFERRED,'data':ref(bp),'baseline_s':[-15,0],'limits':[-.25,.25],'shape':[90,40],'missing':'black'}},analysis_identity={'fish_id':PREFERRED,'status':'preferred illustration; not a validated individual learner or Figure1 replacement','profile_inputs':profile.inputs,'native_size_mm':[width,height]})
        export_matplotlib_figure(fig,OUT/'preferred-single-fish',provenance,mode=FigureMode.PUBLICATION,panel_ids=['preferred','preferred-colourbar'],allow_dirty_publication=True,formats=('svg',),layout_frames=spec);plt.close(fig)
    single_legend=f'Preferred fixed-10sTrace illustrative candidate {PREFERRED}. Direct 1-s positive detected-bout original-frame log-vigor medians, centred within trial by median finite baseline-bin medians [−15,0) s; native managua_r ±0.25, black missing. Acquisition5–94 displayed1–90. Green: CS onset/10-s offset; violet: expected training US20 s. Source protocol and strict pre-US sensitivity are recorded. Its acquisition contrast is {preferred.primary_score:+.3f}, rank {int(preferred.rank_min3)}/24 eligible conditioned fish; this is a descriptive illustration, not a validated learner label, replacement or freeze.'
    write(OUT/'preferred-single-fish-legend.md',single_legend.encode());write(OUT/'preferred-single-fish.html',html_page((OUT/'preferred-single-fish.svg').read_bytes(),single_legend))
    save(OUT/'selection.json',{'preferred':PREFERRED,'comparator':CURRENT,'primary_rank':int(preferred.rank_min3),'eligible_conditioned_fish':24,'primary_score':float(preferred.primary_score),'current_score':float(baseline.primary_score),'reason':'Stable high acquisition rank, lower late-training and late-catch vigor, broad finite coverage, small baseline drift, clearer late CS-window suppression than comparator; highest raw score largely elevatedPre; runner-up has lower tracking validity.','alternatives':['20250714_08',CURRENT,'20230309_10'],'operational_flag':'Preferred fish does not meet pooled-control-tail <=0.05 rule; no validated individual learner claim','preferred_selected_after':'inspected exact seven-fish matched heatmap and outcome comparison','comparison':ref(OUT/'comparison.json'),'strict_pre_US_sensitivity':ref(OUT/'strict-pre-US-sensitivity.json'),'source_table':ref(OUT/'full-score-coverage-exclusion-table-verified.csv'),'font_resources':[ref(Path(findfont(FontProperties(family='Arial',weight=w),fallback_to_default=False))) for w in ['normal','bold']]})
    print(scores.loc[scores.fish_id.isin(load(OUT/'shortlist.json')['fish']),['fish_id','valid_response_fraction_5_14','valid_response_fraction_40_64']].to_string(index=False))
    snapshot.check();print('PREFERRED',PREFERRED)

def report_and_publish(note):
    calculation=load(OUT/'calculation.json');check(calculation['outputs']);selection=load(OUT/'selection.json');comparison=load(OUT/'comparison.json')
    check([selection['source_table'],selection['comparison'],selection['strict_pre_US_sensitivity']])
    artifacts={}
    for name,svgname,tempdir in [('comparison','candidate-comparison.svg','trace10-learner-comparison'),('preferred','preferred-single-fish.svg','trace10-preferred')]:
        report=load(TEMP/tempdir/'browser.json');evidence=assess((OUT/svgname).read_bytes(),report);assert evidence['status']=='passed',evidence['issues'];save(OUT/(name+'-layout-evidence.json'),evidence);artifacts[name]=ref(OUT/(name+'-layout-evidence.json'))
    snapshot=Snapshot([]);snapshot.hashes=load(OUT/'publication-starting-ticket.json');snapshot.check()
    scores=pd.read_csv(OUT/'full-score-coverage-exclusion-table-verified.csv');index=scores.set_index('fish_id');p=index.loc[PREFERRED];current=index.loc[CURRENT];strict=load(OUT/'strict-pre-US-sensitivity.json');strict_preferred=next(r for r in strict['rows'] if r['fish_id']==PREFERRED)
    flags=scores.loc[scores.provisional_strong_suppression_expression,'fish_id'].tolist()
    fields=['fish_id','rank_min3','primary_score','median_5_14','median_40_64','n_5_14','n_40_64','rank_25_39','rank_50_64','baseline_drift_late_minus_pre','heatmap_finite_cells']
    short=scores.loc[scores.fish_id.isin(load(OUT/'shortlist.json')['fish'])].sort_values('rank_min3')[fields]
    rows=['| Fish | Primary rank | Score | Pre median | Late median | Pre/late n | Mid/late-window rank | Finite cells |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in short.itertuples():rows.append(f'| {r.fish_id} | {int(r.rank_min3)} | {r.primary_score:+.3f} | {r.median_5_14:+.3f} | {r.median_40_64:+.3f} | {r.n_5_14}/{r.n_40_64} | {int(r.rank_25_39)}/{int(r.rank_50_64)} | {int(r.heatmap_finite_cells)}/3600 |')
    text=f'''# Fixed-10sTrace learners and Figure 1 candidate recommendation

## Recommendation

Prefer **{PREFERRED}** as the strongest credible illustration in this bounded comparison. It is rank **3/24** feature-eligible conditioned fish, with acquisition contrast **{p.primary_score:+.4f}**: Pre median **{p.median_5_14:+.4f}** (10/10 trials) to lateTrain40–64 **{p.median_40_64:+.4f}** (25/25). The current provisional comparator {CURRENT} is rank **12/24**, contrast **{current.primary_score:+.4f}**. This is a recommendation for an illustrative candidate, not proof that this fish learned or an instruction to replace selected Figure 1 artwork.

The preferred fish has **{int(p.heatmap_finite_cells)}/3600** finite heatmap cells ({p.heatmap_finite_cells/36:.1f}%), complete scheduled trial coverage, and no selected cadence-loss gate failure. The matched heatmap shows stronger late response-window suppression, with a more coherent decrease than the existing comparator. Its rank is 3 under all 2/5, 3/5 and 4/5 distributed coverage choices; Train25–39 rank3 and Train50–64 rank2. Trial or five-trial-block deletion gives ranks **2–4**, with block-deletion contrasts **{p.block_deletion_score_min:+.3f} to {p.block_deletion_score_max:+.3f}**, all positive. These ranges are robustness diagnostics, not confidence intervals.

## Learner identification and limitations

All63 acquisition identities are retained. The author-approved common C/F/I cohort is **61: 32 controls, 29 conditioned fixed-10sTrace fish**. The timing-flagged20230315_04 and shortened20250715_08 remain explicit exclusions with available outcomes and scores retained. Six controls and five conditioned fish fail the prespecified distributed feature-coverage rule, leaving **26 controls and24 conditioned fish** for the primary learner-feature comparison. Insufficient coverage is separate from failure to show a positive score.

The continuous score is median observed outcome over Pre5–14 minus lateTrain40–64. Outcome: median log positive valid moving-frame tail angular speed during response[0,20) minus baseline[−15,0). It measures intensity conditional on positive movement, not movement probability or total movement. Coverage requires at least3 finite trials in each constituent5-trial block, with2/5 and4/5 sensitivities; the method was recorded before inspecting the score distribution.

The strict prespecified provisional strong-suppression-expression rule flags **{', '.join(flags) if flags else 'no fish'}**: eligible, positive contrast, negative late median, descriptive pooled-control tail<=0.05, and positive block-deletion scores. The flagged fish20230307_01 has Pre+0.158 and late−0.001: its large score is mostly an unusually elevated Pre state. It is not my preferred learning illustration. The flag is a provisional phenotype, not a validated biological learner category. The preferred fish does **not** meet that pooled-control-tail criterion: its descriptive tail rank is **{p.descriptive_control_upper_tail:.3f}** against26 controls. Its2023-era reference tail is **{p.era_control_upper_tail:.3f}** against{int(p.era_control_reference_n)} controls, a coarse descriptive reference with unestablished exchangeability. Neither rank is a calibrated p value or probability of learning. Remainder fish are not labelled proven nonlearners.

## Candidate comparison

{chr(10).join(rows)}

20230309_10 (rank2) has large trial fluctuations, a markedly different mid-training rank23, baseline drift−0.199 log units and lower CS-aligned tracking-valid fractions than the preferred fish. It is a weaker illustration despite its slightly larger contrast. 20250714_08 (rank4) is a late-acquisition alternative, but mid-training rank19, baseline drift+0.254 and substantial missing intensity cells make it less clean. The legacy-highlighted comparator20230307_04 remains a useful conservative alternative: all3600 cells finite and smaller contrast, but no robust late-window suppression relative to the preferred example. 20250714_03 and20230314_07 are retained in the bounded comparison; the latter's late median is above zero. No ranking was redesigned to make the preferred image rank first.

## Timing, tracking and motor audits

All29 included conditioned fish passed the recorded-event fixed20-s check with46 paired training trials and four catches25/39/53/59. For the preferred fish, actual paired-US latency is **{p.actual_us_min_s:.3f}–{p.actual_us_max_s:.3f} s**. A separate[0,19.9) sensitivity ends at least **{strict_preferred['minimum_margin_before_actual_US_s']*1000:.0f} ms** before its earliest actual pairedUS. Its contrast remains **{strict_preferred['strict_pre_US_19p9s_score']:+.4f}**, close to the primary20-s contrast; the recommendation is not driven by a post-US response. Actual-US1-s pre/post intensity values and positive-frame counts are saved separately and never enter scoring, classification or candidate ordering.

The preferred fish's baseline log-vigor drift is **{p.baseline_drift_late_minus_pre:+.3f}**, and positive moving-frame fractions are **{p.positive_moving_frame_fraction_5_14:.3f} Pre** versus **{p.positive_moving_frame_fraction_40_64:.3f} late**. Its CS-aligned tracking-valid response fractions are **{p.valid_response_fraction_5_14:.3f} / {p.valid_response_fraction_40_64:.3f}**. These are audit quantities, not bout probability. Late catch-trial median is **{p.catch_late53_59_median:+.3f}** (two trials, descriptive only). LateTest90–94 median is **{p.median_90_94:+.3f}**, a recovery-direction change of **{p.test_recovery_lateTest_minus_lateTrain:+.3f}** from lateTrain. Catch and Test behavior are audits, not classifier requirements or prospectively held-out validation.

## Scientific and storage protections

Controls are shared acquisition identities with historical3sTrace inputs, not independent new acquisitions across assays. Increasing-trace fish are outside this fixed20-s inventory. Heatmaps use the selected cadence/detector, original-frame1-s log medians, per-trial median baseline-bin centering, native managua_r±0.25, black missing and no amplitude scaling. Trial outcomes use measured-time unbinned medians and a common±0.7 view; these estimators are distinct. Figure2 population-model significance was not used for individual ranking or learning-onset claims.

The original score table is preserved; the authoritative `full-score-coverage-exclusion-table-verified.csv` corrects only secondary tracking audit fields to select CS rather than mixed CS/US coverage rows. Scores, eligibility, rankings and operational flags are unchanged. Complete score/coverage/exclusion, deletion/rank sensitivities, control leave-one-out references, actual-US audits, strict-pre-US values, source hashes, editable SVG and render evidence are saved here. Native Arial text is editable; production PDF/font-embedding/accessibility and independent classifier validation remain separate. No selected Figure1/2 artwork or freeze is replaced by this comparison.
'''
    write(OUT/'selection-report.md',text.encode())
    # A readable self-contained report contains the recommendation, audit and source table.
    import html
    report_html=('<!doctype html><html><meta charset="utf-8"><style>body{max-width:1100px;margin:35px auto;padding:0 20px;font:16px/1.5 Arial;color:#111;background:white}pre{white-space:pre-wrap;font:inherit}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border-bottom:1px solid #ddd;text-align:right}a{color:#0645ad}</style><h1>10sTrace candidate recommendation: '+PREFERRED+'</h1><pre>'+html.escape(text)+'</pre><h2>Shortlist values</h2>'+short.to_html(index=False,float_format=lambda x:f'{x:.4f}')+'<p><a href="candidate-comparison.html">Matched editable comparison</a> · <a href="preferred-single-fish.html">Preferred single-fish heatmap</a> · <a href="full-score-coverage-exclusion-table-verified.csv">Complete score/coverage/exclusion table</a></p></html>').encode()
    write(OUT/'selection-report.html',report_html)
    new_legend=(COMPARISON_LEGEND+'\n').encode();new_html=html_page((OUT/'candidate-comparison.svg').read_bytes(),COMPARISON_LEGEND)
    history=OUT/'history';history.mkdir(exist_ok=True)
    additions=[(history/'comparison-before-caption-spacing.html',(OUT/'candidate-comparison.html').read_bytes()),(history/'comparison-before-caption-spacing-legend.md',(OUT/'comparison-legend.md').read_bytes())]
    state={'figure_id':'fig1-trace10-learner-candidates','default_display':'draft','default_display_format':'html-svg-and-legend','whole_figure_frozen':False,'preferred_fish':PREFERRED,'parent_figures_replaced':False,
        'draft':{'working_svg':ref(OUT/'candidate-comparison.svg'),'review':{'path':str(OUT/'candidate-comparison.html'),'sha256':sha(new_html)},'legend':{'path':str(OUT/'comparison-legend.md'),'sha256':sha(new_legend)},'scientific_registry':ref(OUT/'candidate-comparison.figure.json'),'single_fish':{'svg':ref(OUT/'preferred-single-fish.svg'),'html':ref(OUT/'preferred-single-fish.html')},'report':{'markdown':ref(OUT/'selection-report.md'),'html':ref(OUT/'selection-report.html')},'score_table':ref(OUT/'full-score-coverage-exclusion-table-verified.csv'),'selection':ref(OUT/'selection.json'),'method':ref(METHOD)},
        'verification':{'layout_frames':artifacts,'visual_review':{'status':'performed','note':note},'coverage_audit_correction':ref(OUT/'coverage-audit-correction.json'),'strict_pre_US':ref(OUT/'strict-pre-US-sensitivity.json'),'classification_status':'provisional rule and continuous scores; biological categories unvalidated','parent_state_hashes':{k:v for k,v in snapshot.hashes.items() if k.endswith('current-state.json') and k!=str(STATE.resolve())}}}
    snapshot.hashes.update(Snapshot([Path(__file__),OUT/'selection.json',OUT/'full-score-coverage-exclusion-table-verified.csv',OUT/'comparison-layout-evidence.json',OUT/'preferred-layout-evidence.json',OUT/'selection-report.md',OUT/'selection-report.html',OUT/'preferred-single-fish.svg',OUT/'preferred-single-fish.html']).hashes)
    publish_state(STATE,state,snapshot,artifacts=additions,evidence_updates=[(OUT/'candidate-comparison.html',new_html),(OUT/'comparison-legend.md',new_legend)])
    save(OUT/'completion.json',{'status':'recommendation-and-comparison-complete','preferred':PREFERRED,'state':ref(STATE),'selection':ref(OUT/'selection.json'),'report':ref(OUT/'selection-report.md'),'classification':'provisional; no validated individual learner category','parent_figures_replaced':False,'freeze':False})
    print(OUT/'selection-report.html')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','publish']);p.add_argument('--visual-review-note');a=p.parse_args()
    if a.operation=='prepare':prepare()
    elif a.visual_review_note:report_and_publish(a.visual_review_note)
    else:p.error('Actual visual-review note required')
