"""Full-window exploratory Figure 4 analysis; never extrapolate narrow fits.

Reuse hash-authenticated supported-reader trial values, recompute fish medians,
and restart the existing six-df spline ML/whole-fish bootstrap on [-20,20).
Old component and assembly versions remain immutable. Publication is separate
from fitting and requires an exact-artifact visual verification receipt.
"""
from pathlib import Path
import sys
ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import json
import io
import html
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from classical_conditioning.figure_workspace import Snapshot, sha, publish_state
from classical_conditioning.external_artifacts import resolve_artifact
from classical_conditioning.analysis.review.figure4_assay_profiles import ProfileSpec, model_data, fit_bootstrap, GROUPS

STATE = ROOT/'docs/paper/figures/figure-04/export-current-state.json'
DEFAULT = Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/figures/figure-04/assembly/versions/full-window-20261029')

def read(p): return json.loads(Path(p).read_text())
def write(p, obj):
    with Path(p).open('x') as f: json.dump(obj, f, indent=2, allow_nan=False); f.write('\n')
def ref(p): return {'path': str(p), 'sha256': sha(Path(p).read_bytes())}
def snapshot(directory):
    ticket = read(directory/'start-ticket.json')
    snap = Snapshot([directory/'start-ticket.json', Path(__file__),
        ROOT/'src/classical_conditioning/analysis/review/figure4_assay_profiles.py',
        ROOT/'src/classical_conditioning/analysis/review/figure4_profiles.py',
        ROOT/'src/classical_conditioning/analysis/inference/panel_g_fast_lmm.py'])
    snap.bind_hashes(ticket['starting_hashes']); snap.check()
    return ticket, snap

def analyze(directory, repetitions=399):
    ticket, snap = snapshot(directory)
    for assay, source in ticket['sources'].items():
        out = directory/assay
        if (out/'analysis-summary.json').exists():
            old = read(out/'analysis-summary.json')
            for r in old['artifacts']:
                assert ref(Path(r['path'])) == r
            assert old['source_hashes'] == snap.hashes
            continue
        out.mkdir(exist_ok=True)
        trials = pd.read_parquet(source['files']['trial-bins.parquet'])
        members = pd.read_csv(source['files']['membership.csv'])
        expected_times = np.arange(-20, 20, .5)+.25
        assert set(trials.fish_id) == set(members.fish_id)
        assert len(trials) == len(members)*5*80
        assert set(trials.trial_number) == {25,39,53,59,65}
        assert not trials.duplicated(['fish_id','trial_number','time_s']).any()
        assert np.array_equal(np.sort(trials.time_s.unique()), expected_times)
        assert not trials.loc[trials.coverage < .9, ['signed','movement']].notna().any().any()
        # Authenticate individual extraction checkpoints and compare their full
        # trial values to the master table, including missingness and identities.
        complete = read(source['complete'])
        for record in complete['fish_reports']:
            rp = resolve_artifact(record['path'], expected_sha256=record['sha256'])
            report = read(rp); r = report['output']
            p = resolve_artifact(r['path'], expected_sha256=r['sha256'])
            snap.bind_hashes({str(p):r['sha256']})
            part = pd.read_parquet(p)
            actual = trials.loc[trials.fish_id.eq(part.fish_id.iloc[0]), part.columns]
            pd.testing.assert_frame_equal(actual.reset_index(drop=True), part.reset_index(drop=True), check_dtype=False)
        fish = trials.groupby(['fish_id','condition_id','display_group','time_s'], sort=False).agg(
            signed=('signed','median'), signed_trials=('signed','count'),
            movement=('movement','median'), movement_trials=('movement','count')).reset_index()
        saved = pd.read_parquet(source['files']['fish-bins.parquet'])
        pd.testing.assert_frame_equal(fish, saved, check_dtype=False)
        fish.to_parquet(out/'fish-bins.parquet', index=False)
        support = fish.groupby(['display_group','time_s']).agg(signed_fish=('signed','count'),
            signed_trials=('signed_trials','sum'), movement_fish=('movement','count')).reset_index()
        support.to_csv(out/'group-support.csv', index=False)
        coverage = trials.groupby(['display_group','time_s']).agg(trials=('coverage','size'),
            adequate_trials=('coverage',lambda x: int((x>=.9).sum())),
            signed_trials=('signed','count'), minimum_coverage=('coverage','min')).reset_index()
        coverage.to_csv(out/'trial-coverage.csv', index=False)
        spec = ProfileSpec(-20.,20.)
        data = model_data(fish, spec)
        assert np.array_equal(np.sort(data.time_s.unique()), expected_times)
        for group in GROUPS:
            assert (support.loc[support.display_group.eq(group),'signed_fish']>0).all()
        print(f'{assay}: {len(data)} observed bins; {data.fish_id.nunique()} fish; full 80-bin fit', flush=True)
        result, grid, curves, draws, diagnostics = fit_bootstrap(data, expected_times,
            repetitions=repetitions, spec=spec,
            progress=lambda n,ok: print(f'{assay} whole-fish refits {n}/{repetitions}: {ok} accepted',flush=True))
        np.savez_compressed(out/'model-arrays.npz', times=expected_times, group_curves=curves, bootstrap_curves=draws)
        observed = fish.loc[fish.display_group.isin(GROUPS)].groupby(['display_group','time_s']).agg(
            mean=('signed','mean'), n=('signed','count')).reset_index()
        observed.to_csv(out/'observed-points.csv', index=False)
        residual = np.asarray(result.resid)
        residuals = data[['fish_id','display_group','time_s']].copy(); residuals['residual']=residual
        residuals.to_parquet(out/'residuals.parquet', index=False)
        pairs = residuals.merge(residuals.assign(time_s=residuals.time_s+.5), on=['fish_id','display_group','time_s'],suffixes=('','_previous'))
        old = read(source['analysis'])
        summary = {'scope': assay+' full-window exploratory Figure 4 panel a',
            'recipe': 'figure4-full-window-six-df-spline-20261029',
            'fit_window_s':[-20,20], 'bin_centers_s':expected_times.tolist(),
            'extrapolated_bins':0, 'observed_model_bins':len(data),'modeled_fish':int(data.fish_id.nunique()),
            'counts':{g:int(members.display_group.eq(g).sum())for g in GROUPS},
            'profile':old['profile'],'aggregation':old['aggregation'],'membership':old['membership'],
            'cs_s':[0,10],'expected_us_s':old['expected_us_s'],
            'diagnostics':diagnostics,'residual_excess_kurtosis':float(kurtosis(residual)),
            'adjacent_half_second_residual_correlation':float(pairs.residual.corr(pairs.residual_previous)),
            'support_ranges':{g:[int(support.loc[support.display_group.eq(g),'signed_fish'].min()),int(support.loc[support.display_group.eq(g),'signed_fish'].max())]for g in GROUPS},
            'limitations':old['limitations']+['Full-window refit retains six spline degrees of freedom per group; temporal smoothing differs from the narrow-window model.'],
            'source_hashes':snap.hashes,
            'artifacts':[ref(out/f)for f in ['fish-bins.parquet','group-support.csv','trial-coverage.csv','model-arrays.npz','observed-points.csv','residuals.parquet']]}
        write(out/'analysis-summary.json',summary)
        snap.check()
        print(json.dumps({k:summary[k]for k in ['scope','support_ranges','residual_excess_kurtosis','adjacent_half_second_residual_correlation']}),flush=True)

def render(directory):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from classical_conditioning.figures.theme import apply_theme, style_axes
    from classical_conditioning.figures.presentation_profiles import load_presentation_profile
    ticket, snap = snapshot(directory)
    profile = load_presentation_profile('nature-main')
    snap.include([ROOT/'configs/paper-figures/figure-elements.json'])
    arrays={a:np.load(directory/a/'model-arrays.npz')for a in ticket['sources']}
    summaries={a:read(directory/a/'analysis-summary.json')for a in arrays}
    for a in arrays:
        snap.include([directory/a/'analysis-summary.json'])
        for r in summaries[a]['artifacts']:snap.bind_hashes({r['path']:r['sha256']})
    values=[]
    for a,z in arrays.items():
        values.extend([z['group_curves'].min(),z['group_curves'].max(),*np.quantile(z['bootstrap_curves'],[.025,.975],axis=0).ravel()])
        points=pd.read_csv(directory/a/'observed-points.csv');values.extend(points['mean'].dropna())
    lo=min(-.30, np.floor((min(values)-.01)/.05)*.05);hi=max(.15,np.ceil((max(values)+.01)/.05)*.05)
    width,height=183.,73.
    colors={'delay':['#00aeef','#ec008c','#E69F00'],'3strace':['#00aeef','#f15a29','#777777']}
    with mpl.rc_context():
        theme=apply_theme(profile=profile)
        mpl.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'fig4-full-window-20261029','savefig.bbox':None})
        fig=plt.figure(figsize=(width/25.4,height/25.4));axes=[];mapping={}
        for i,a in enumerate(['delay','3strace']):
            x=15+i*91.5;ax=fig.add_axes([x/width,15/height,72/width,43/height]);axes.append(ax)
            ax.set_gid(a+'-panel-a');ax.set_xlim(-20,20);ax.set_ylim(lo,hi);ax.set_xticks([-20,-10,0,10,20])
            ax.axhline(0,color='#999999',lw=.35,zorder=0)
            ax.axvline(0,color=theme.cs_color,lw=theme.cs_linewidth,zorder=0)
            ax.axvline(summaries[a]['expected_us_s'],color=theme.us_color,lw=theme.us_linewidth,ls=':',zorder=0)
            ax.axvline(10,color=theme.cs_color,lw=theme.cs_linewidth,ls='--',zorder=0)
            z=arrays[a];t=z['times'];bands=np.quantile(z['bootstrap_curves'],[.025,.975],axis=0)
            obs=pd.read_csv(directory/a/'observed-points.csv')
            for k,g in enumerate(GROUPS):
                ax.fill_between(t,bands[0,k],bands[1,k],color=colors[a][k],alpha=.16,lw=0,gid=a+'-band-'+str(k))
                ax.plot(t,z['group_curves'][k],color=colors[a][k],lw=.8,label=f'{g} (n={summaries[a]["counts"][g]})',gid=a+'-fit-'+str(k))
                p=obs.loc[obs.display_group.eq(g)];ax.scatter(p.time_s,p['mean'],s=3,color=colors[a][k],alpha=.65,lw=0,gid=a+'-observed-'+str(k))
            style_axes(ax,theme=theme,xlabel='Time from CS (s)',ylabel='Signed bout ln vigor')
            ax.set_title('Delay'if a=='delay'else'3sTrace',fontsize=theme.axes_titlesize,pad=6)
            ax.legend(loc='lower right',frameon=False,fontsize=5,handlelength=2,borderaxespad=.35)
            fig.text((x-12)/width,65/height,chr(97+i),fontsize=8,fontweight='bold')
            mapping[a]={'xlim':[-20,20],'ylim':[float(lo),float(hi)],'bin_centers':t.tolist(),'observed_points_per_group':80,
                'group_order':list(GROUPS),'counts':summaries[a]['counts'],'colors':colors[a],
                'zero_figure_y':float(ax.transData.transform((0,0))[1]/fig.bbox.height),
                'observed_source':ref(directory/a/'observed-points.csv'),'model_source':ref(directory/a/'model-arrays.npz')}
        fig.savefig(directory/'figure-04.svg')
        fig.savefig(directory/'figure-04-600dpi.png',dpi=600)
        plt.close(fig)
    legend=('Figure 4. Full-window pooled catch response profiles. (a) Delay; (b) 3sTrace. '
        'Dots are equal-fish means of observed fish-level medians across catch trials 25, 39, 53, 59 and 65 in 0.5 s bins. '
        'The signed conditional bout ln-vigor estimator uses each trial’s [−20,0) s baseline and requires ≥0.9 tracking coverage; no-bout intensity remains missing. '
        'Solid colored curves are group-specific cubic spline Gaussian mixed-model fits (six degrees of freedom per group), refitted using actual data throughout [−20,+20) s. '
        'Shading shows pointwise 95% whole-fish bootstrap percentile ranges (399 requested refits); it is not a simultaneous band. '
        'Cyan: unsplit controls. Delay: magenta learner-flagged, orange nonlearner-flagged; 3sTrace: orange learner-flagged, gray nonlearner-flagged. '
        'Group sizes appear in each line key. Both panels use identical axes and aligned zero. Solid green: CS onset; dashed green: CS offset at 10 s; dotted red: expected paired-training US at 9 s (Delay) or 13 s (3sTrace). These are catch trials without a paired US. '
        'Saved learner flags are retrospective, unvalidated and overlap the evaluation trials; unclassified experimental fish are excluded. '
        'These exploratory fits do not validate learner status or CR peak/onset timing. Informative bout availability, temporal residual dependence and the untested penalized GAMM alternative remain limitations. '
        'Conditional fitted intensity in bins where a particular fish has no bout does not reconstruct its movement.')
    (directory/'legend.md').write_text(legend+'\n')
    raw=(directory/'figure-04.svg').read_text();raw=raw[raw.index('<svg'):]
    (directory/'figure-04.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Figure 4</title><style>body{margin:24px auto;max-width:1000px;font:15px Arial;color:#111}svg{width:100%;height:auto}p{line-height:1.5}</style></head><body>'+raw+'<p>'+html.escape(legend)+'</p></body></html>')
    evidence={'scope':'Delay and 3sTrace panel-a full-window refit','axes':mapping,'presentation_profile':profile.inputs,
        'source_hashes':snap.hashes,'scientific_freeze':False,'full_nature_compliance_certified':False,
        'frame_coverage':'Physical artwork bounds and matched data rectangles; detailed role/frame certification unresolved.'}
    assert mapping['delay']['zero_figure_y']==mapping['3strace']['zero_figure_y']
    write(directory/'presentation-evidence.json',evidence);snap.check()
    print(json.dumps({'ylim':[lo,hi],'svg':str(directory/'figure-04.svg')}))

def publish(directory):
    ticket,snap=snapshot(directory)
    evidence=read(directory/'presentation-evidence.json');snap.bind_hashes(evidence['source_hashes'])
    review=read(directory/'visual-review.json')
    assert review['passed']
    for r in review['artifacts']:assert ref(Path(r['path']))==r
    files=[p for p in directory.rglob('*')if p.is_file()and p.name!='version-lock.json']
    snap.include(files);snap.check()
    state=read(STATE);previous=state.get('presentation_lock')
    lock={'schema_version':1,'kind':'provisional-presentation-lock','date':'2026-10-29',
        'scope':'Delay and 3sTrace full [-20,20) analysis; two panels; common axes; PNG-based PDF',
        'author_authorization':ticket['authorization'],'files':{str(p.relative_to(directory)):ref(p)for p in files},
        'previous_lock':previous,'retain_other_versions':True,'publication_freeze_completed':False,
        'scientific_status':'Exploratory full-window refit, no method or learner approval','source_guard_snapshot':snap.hashes}
    raw=(json.dumps(lock,indent=2)+'\n').encode();lockpath=directory/'version-lock.json'
    state.update(scope=lock['scope']);state['scientific_selection']={'status':lock['scientific_status'],'records':[]}
    state['draft']={'status':'Author-locked provisional presentation','working_svg':ref(directory/'figure-04.svg'),
        'review':ref(directory/'figure-04.html'),'legend':ref(directory/'legend.md'),
        'exports':{'png':ref(directory/'figure-04-600dpi.png'),'pdf':ref(directory/'figure-04.pdf')},
        'analysis':{a:ref(directory/a/'analysis-summary.json')for a in ticket['sources']},
        'verification':ref(directory/'visual-review.json'),'presentation_evidence':ref(directory/'presentation-evidence.json')}
    state['presentation_lock']={'path':str(lockpath),'sha256':sha(raw),'status':'locked','kind':lock['kind'],
        'date':lock['date'],'scope':lock['scope'],'retain_other_versions':True,'future_edits_require_new_version':True,'publication_freeze_completed':False,'previous_lock':previous}
    publish_state(STATE,state,snap,artifacts=[(lockpath,raw)])
    print('Published corrected Figure 4 provisional presentation lock.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['analyze','render','publish']);p.add_argument('--directory',type=Path,default=DEFAULT);a=p.parse_args()
    {'analyze':analyze,'render':render,'publish':publish}[a.stage](a.directory)
