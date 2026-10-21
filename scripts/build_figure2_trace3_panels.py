"""Draft Figure 2 B/E/H from authenticated expanded fixed-3sTrace inputs.

Uses A's supported cadence/heatmap calculation, D's measured-time block
outcome and G's whole-fish descriptive bootstrap. No inference is run.
Calculation, rendering and guarded publication are separate operations.
"""
from __future__ import annotations

import argparse
import gc
import html
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
import numpy as np
import pandas as pd
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figure_workspace import Snapshot, publish_state
from classical_conditioning.figures.pooled_log_heatmap import analyze_recording, pool_equal_fish
from classical_conditioning.analysis.trial_outcomes import verify_candidate_trial_outcomes
from review_figure2_block_log_median import calculate_trials, read_windows, COL
from render_figure2_delay_legacy_metric_lme import bootstrap_trajectories

PROJECT = Path(r'F:\Digested Data\all3sTrace-full-v1')
OUT = Path('J:/ClassicalConditioning Outputs/current/reviews/figure-02-3strace')
BASELINE = REPO / 'records/figure-reviews/figure-02/trace3-panels-start-20261019.json'
STATE = REPO / 'docs/paper/figures/figure-02/panels-b-e-h/current-state.json'
PARENT = STATE.parent.parent / 'current-state.json'
SCIENCE = [REPO / 'src/classical_conditioning/figures/pooled_log_heatmap.py',
           REPO / 'scripts/review_figure2_block_log_median.py',
           REPO / 'scripts/render_figure2_delay_legacy_metric_lme.py',
           REPO / 'src/classical_conditioning/preprocessing/acquisition_timing.py',
           REPO / 'src/classical_conditioning/analysis/movement_state.py']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encoded(data):
    return (json.dumps(data, indent=2, allow_nan=False) + '\n').encode('utf-8')


def save_new(path, data):
    with Path(path).open('xb') as stream:
        stream.write(encoded(data))


def artifact(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha256_file(path)}


def guard_calculation_writes():
    """Keep source drives read-only and permanent writes inside this new scope."""
    root = OUT.resolve()
    old = {str(p.resolve()).casefold() for p in OUT.rglob('*') if p.is_file()
           and p.name not in {'calculation-stdout.log','calculation-stderr.log','.calculation.lock'}}
    def guard(event, args):
        paths=[]
        if event=='open' and args and isinstance(args[0], (str,bytes,os.PathLike)):
            mode, flags=args[1] or '', args[2] if len(args)>2 else 0
            if any(c in mode for c in 'wax+') or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC): paths=[args[0]]
        elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime'}: paths=[args[0]]
        elif event=='os.rename': paths=list(args[:2])
        for value in paths:
            if not isinstance(value,(str,bytes,os.PathLike)): continue
            p=Path(os.fsdecode(value)).resolve()
            if p.drive.lower() in {'d:','f:'}: raise PermissionError('Source drive is read-only: '+str(p))
            if p.drive.lower()=='j:' and p!=root and root not in p.parents: raise PermissionError('Writes outside panel scope: '+str(p))
            if str(p).casefold() in old: raise PermissionError('Existing panel evidence is immutable: '+str(p))
    sys.addaudithook(guard)


def plan():
    baseline = load(BASELINE)
    snapshot = Snapshot([])
    snapshot.hashes = baseline['hashes']
    snapshot.check(ignore=SCIENCE)  # Preserve the pre-edit code hash; bind its reviewed replacement below.
    if not OUT.parent.is_dir():
        raise FileNotFoundError('JOAQUIM figure storage unavailable')
    OUT.mkdir(exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError('Preserve existing panel work; use the sealed plan')
    rows = baseline['cohort']
    if len(rows) != 183 or len({r['recording_id'] for r in rows}) != 183:
        raise ValueError('Unexpected expanded cohort')
    meta = PROJECT / 'Metadata'
    sources = [BASELINE, *SCIENCE]
    for row in rows:
        rid = row['recording_id']
        sources += [Path(row['summary_path']), meta/f'{rid}_source_manifest.json',
                    meta/f'{rid}_candidate-trial-outcomes-corrected_complete.json']
    # The extra original fish is outside the 197-identity pickle intake; keep it
    # in the expanded cohort only after verifying the same supported lineage.
    verify_candidate_trial_outcomes(PROJECT, '20230803_12', metric_recipe='tail-candidate-corrected')
    selection = load(meta/'trace3-additions-20261018-verified-run-plan.json')
    held = {r['recording_id'] for r in selection['held']}
    if held & {r['recording_id'] for r in rows}:
        raise ValueError('Held fish entered the figure cohort')
    for row in rows:
        if row['condition'] not in {'trace', 'control'}:
            raise ValueError('Unexpected condition')
    save_new(OUT/'plan.json', {'cohort': rows, 'held': selection['held'],
             'source_hashes': Snapshot(sources).hashes, 'starting_state': baseline,
             'author_scope': 'B/E/H; always match current Delay A/D/G; no inferential statistics',
             'methods': {'B': 'selected A direct 1 s frame-log medians, median baseline bins [-15,0), equal fish mean',
                         'E': 'selected D positive valid moving-frame log median difference, measured times, response [0,13), PT10-14 ET65-69 LT90-94, min 1 eligible trial',
                         'H': 'selected G log-median difference, whole-fish 5000 percentile bootstrap, seed 10; no models/tests'}})
    print('PLAN: 183 fish (91 Control, 92 3sTrace)', flush=True)


def check_plan():
    selection = load(OUT/'plan.json')
    snapshot = Snapshot([])
    snapshot.hashes = selection['source_hashes']
    snapshot.check()
    return selection


def calculate(limit=None):
    selection = check_plan()
    cache = OUT/'fish'
    cache.mkdir(exist_ok=True)
    for index, row in enumerate(selection['cohort'], 1):
        if limit and index > limit:
            break
        rid, condition = row['recording_id'], row['condition']
        record_path = cache/(rid+'.json')
        if record_path.exists():
            record = load(record_path)
            for ref in record.get('outputs', []):
                if sha256_file(Path(ref['path'])) != ref['sha256']:
                    raise ValueError('Edited numerical cache: '+rid)
            print('CACHED '+rid, flush=True)
            continue
        if (cache/(rid+'-windows.parquet')).exists() or (cache/(rid+'-bins.parquet')).exists():
            raise FileExistsError('Uncommitted partial calculation preserved for reconciliation: '+rid)
        print(f'START {rid} [{index}/183]', flush=True)
        verify_candidate_trial_outcomes(PROJECT, rid, metric_recipe='tail-candidate-corrected')
        proc = PROJECT/'Processed data'/rid
        record = {'recording_id': rid, 'condition': condition, 'outputs': []}
        # E and H share the exact measured-time window outcome; B independently
        # uses A's cadence reconstruction and baseline-bin estimator.
        summary = load(Path(row['summary_path']))
        if summary['config']['baseline_window_s'] != [-15.0, 0.0] or summary['config']['response_window_s'] != [0.0, 13.0]:
            raise ValueError('Unexpected trial outcome windows: '+rid)
        cycles = pd.read_parquet(proc/'stimulus_events.parquet')
        cycles = cycles.loc[cycles.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
        if len(cycles) != 94:
            raise ValueError('Unexpected CS count: '+rid)
        intervals = [(float(t)-15000, float(t)+13000) for t in cycles.iloc[4:].Beg]
        metrics_path = proc/'frame_activity_candidates-corrected.parquet'
        movement_path = proc/'movement_state_candidates-corrected.parquet'
        metrics = read_windows(metrics_path, ['FrameID', 'AbsoluteTime', COL], intervals)
        movement = read_windows(movement_path, ['FrameID', 'AbsoluteTime', 'valid', 'moving'], intervals)
        trials = calculate_trials(metrics, movement, cycles, range(5,95), 13)
        trials['fish_id'], trials['condition_id'] = rid, condition
        trial_path = cache/(rid+'-windows.parquet')
        trials.to_parquet(trial_path, index=False)
        record['outputs'].append(artifact(trial_path))
        record['window_sources'] = [artifact(p) for p in [metrics_path, movement_path, proc/'stimulus_events.parquet']]
        for ref, key in zip(record['window_sources'], ['candidate_frames','movement_state','protocol']):
            if ref['sha256'] != summary['inputs'][key]:
                raise ValueError('Window input changed during calculation: '+rid)
        del metrics, movement
        gc.collect()
        try:
            matrix, bins, info = analyze_recording(PROJECT, rid, '', 'all3sTrace', condition)
            bin_path = cache/(rid+'-bins.parquet')
            bins.to_parquet(bin_path, index=False)
            record.update(heatmap='complete', heatmap_provenance=info)
            record['outputs'].append(artifact(bin_path))
            del matrix, bins
        except Exception as exc:
            traceback.print_exc()
            record.update(heatmap='review-required', heatmap_error=repr(exc))
        save_new(record_path, record)
        print(f'DONE {rid}; B={record["heatmap"]}; E/H=complete', flush=True)
        gc.collect()
    if limit:
        return
    records = [load(cache/(r['recording_id']+'.json')) for r in selection['cohort']]
    windows = pd.concat([pd.read_parquet(cache/(r['recording_id']+'-windows.parquet')) for r in selection['cohort']], ignore_index=True)
    windows.to_parquet(OUT/'window-logmedians.parquet', index=False)
    blocks = []
    for label, first, last in [('PT',10,14), ('ET',65,69), ('LT',90,94)]:
        group = windows.loc[windows.trial_number.between(first,last)].groupby(['fish_id','condition_id']).legacy_log_median_difference
        block = group.agg(value='median', eligible_trials='count').reset_index()
        block['block'] = label
        blocks.append(block)
    pd.concat(blocks, ignore_index=True).to_parquet(OUT/'blocks.parquet', index=False)
    boot_input = windows.rename(columns={'legacy_log_median_difference':'ratio'}).copy()
    boot_input['condition_id'] = boot_input.condition_id.replace({'trace':'delay'})
    trajectory, draws = bootstrap_trajectories(boot_input, n_boot=5000, seed=10)
    trajectory['condition_id'] = trajectory.condition_id.replace({'delay':'trace'})
    trajectory.to_parquet(OUT/'trajectory.parquet', index=False)
    # Persist the actual whole-fish draws for reproducible uncertainty bands.
    np.savez_compressed(OUT/'bootstrap-draws.npz', **{f'{c}_{k}':v for c, d in draws.items() for k,v in d.items()})
    issues = [r for r in records if r['heatmap'] != 'complete']
    if not issues:
        bins = pd.concat([pd.read_parquet(cache/(r['recording_id']+'-bins.parquet')) for r in selection['cohort']], ignore_index=True)
        pooled = pool_equal_fish(bins, {r['recording_id']:r['condition'] for r in selection['cohort']})
        pooled.to_parquet(OUT/'pooled-bins.parquet', index=False)
    check_plan()
    save_new(OUT/'calculation.json', {'status':'complete' if not issues else 'B-review-required',
             'cohort_counts':{'control':91,'trace':92}, 'heatmap_issues':issues,
             'plan':artifact(OUT/'plan.json'), 'fish_records':[artifact(cache/(r['recording_id']+'.json')) for r in selection['cohort']],
             'outputs':[artifact(p) for p in OUT.glob('*.parquet')], 'inference':'not run'})
    print('CALCULATION FINISHED', flush=True)


def render(include_heatmap=True):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from classical_conditioning.figures.presentation_profiles import load_presentation_profile
    from classical_conditioning.figures.theme import apply_theme, style_axes
    from classical_conditioning.figures.layout_frames import load_policy
    from classical_conditioning.figures.export import export_matplotlib_figure, FigureProvenance, FigureMode
    selection = check_plan()
    calculation = load(OUT/'calculation.json')
    if include_heatmap and calculation['status'] != 'complete':
        raise ValueError('B has unresolved technical coverage; preserve all fish and resolve before publication')
    for ref in calculation['outputs'] + calculation['fish_records']:
        if sha256_file(Path(ref['path'])) != ref['sha256']:
            raise ValueError('Calculation changed')
    profile = load_presentation_profile('nature-main')
    blocks = pd.read_parquet(OUT/'blocks.parquet')
    curves = pd.read_parquet(OUT/'trajectory.parquet')
    pool = pd.read_parquet(OUT/'pooled-bins.parquet') if include_heatmap else None
    height = 165 if include_heatmap else 93
    basename = 'figure2-B-E-H' if include_heatmap else 'figure2-E-H'
    colors = {'control':'#29abe2', 'trace':'#f15b2d'}
    with mpl.rc_context():
        theme = apply_theme(profile=profile)
        fig = plt.figure(figsize=(89/25.4,height/25.4))
        def axes(x,y,w,h):
            offset=0 if include_heatmap else 72
            return fig.add_axes([x/89,1-(y-offset+h)/height,w/89,h/height])
        baxes = [axes(14,8,27,52), axes(46,8,27,52)] if include_heatmap else []
        cax = axes(77,8,2,52) if include_heatmap else None
        eaxes = [axes(14,79,28,29), axes(48,79,28,29)]
        hax = axes(14,126,62,29)
        for ax, condition in zip(baxes, ['control','trace']):
            rows = pool.loc[pool.condition_id.eq(condition)]
            matrix = rows.pivot(index='acquisition_trial',columns='bin_index',values='mean_delta_log_vigor').reindex(index=range(5,95),columns=range(40)).to_numpy()
            cmap = mpl.colormaps['managua_r'].copy(); cmap.set_bad('black')
            image = ax.imshow(matrix, extent=[-20,20,90.5,.5], aspect='auto', interpolation='nearest', cmap=cmap,vmin=-.25,vmax=.25)
            ax.set(xticks=[-20,0,20],yticks=[1,30,60,90],xlim=(-20,20),ylim=(90.5,.5))
            style_axes(ax, theme=theme, xlabel='', ylabel='Trial' if condition=='control' else '',show_yticklabels=condition=='control')
            ax.spines[['top','right']].set_visible(True)
            for x in [0,10]: ax.axvline(x,color=theme.cs_color,lw=theme.cs_linewidth)
            if condition=='trace': ax.axvline(13,color=theme.us_color,lw=theme.us_linewidth)
            for y in [10.5,60.5]: ax.axhline(y,color='black',lw=theme.axes_linewidth)
            ax.set_title('Control' if condition=='control' else '3sTrace', color='black')
        if include_heatmap:
            bar = fig.colorbar(image,cax=cax,ticks=[-.25,0,.25])
            bar.set_label('Baseline-centred log vigor')
            fig.text(43.5/89,1-68/height,'Time from CS onset (s)',ha='center',va='center',fontsize=theme.axes_labelsize)
        for ax, condition in zip(eaxes,['control','trace']):
            data=blocks.loc[blocks.condition_id.eq(condition)]
            wide=data.pivot(index='fish_id',columns='block',values='value').reindex(columns=['PT','ET','LT'])
            for values in wide.to_numpy(): ax.plot([0,1,2],values,color=colors[condition],alpha=.18,lw=.4)
            q=np.nanquantile(wide.to_numpy(),[.25,.5,.75],axis=0)
            ax.errorbar([0,1,2],q[1],yerr=[q[1]-q[0],q[2]-q[1]],fmt='o',color='black',markersize=2,lw=profile.contract['styles']['median_iqr']['linewidth_pt'],capsize=2)
            ax.axhline(0,color='black',lw=theme.axes_linewidth,ls='--')
            ax.set(xticks=[0,1,2],xticklabels=['PT','ET','LT'],xlim=(-.35,2.35))
            style_axes(ax,theme=theme,xlabel='',ylabel='Δ log vigor' if condition=='control' else '',show_yticklabels=condition=='control')
            ax.set_title('Control' if condition=='control' else '3sTrace',color='black')
        limits=[min(ax.get_ylim()[0] for ax in eaxes),max(ax.get_ylim()[1] for ax in eaxes)]
        for ax in eaxes: ax.set_ylim(limits)
        for condition in ['control','trace']:
            data=curves.loc[curves.condition_id.eq(condition)].sort_values('trial_number')
            x=data.trial_number.to_numpy()-4
            hax.fill_between(x,data.ci_lower,data.ci_upper,color=colors[condition],alpha=.2,lw=0)
            hax.plot(x,data['median'],color=colors[condition],lw=profile.contract['styles']['summary_curve']['linewidth_pt'],label='Control' if condition=='control' else '3sTrace')
        hax.axhline(0,color='black',lw=theme.axes_linewidth,ls='--')
        for x in [10.5,60.5]: hax.axvline(x,color='black',lw=theme.axes_linewidth,ls='--')
        hax.set(xlim=(1,90),xticks=[1,30,60,90])
        style_axes(hax,theme=theme,xlabel='Trial',ylabel='Δ log vigor')
        hax.legend(loc='lower right',bbox_to_anchor=(1,1.02),ncol=2,frameon=False,borderaxespad=0,handlelength=1.5,columnspacing=.8)
        letters=[('b',2),('e',73),('h',120)] if include_heatmap else [('e',1),('h',48)]
        for letter,y in letters:
            artist=fig.text(2/89,1-y/height,letter,weight='bold',fontsize=profile.contract['styles']['panel_letter']['font_size_pt'],va='top')
            artist.set_gid('panel-letter-'+letter)
        ids=(['B-control','B-trace','B-colorbar'] if include_heatmap else [])+['E-control','E-trace','H']
        axes_ids=['axes__'+p.lower()+'__main' for p in ids]
        policy=load_policy()[0]
        frames={'scope':'figure','final_size_mm':[89,height],'inset_mm':policy['selected_inset_mm'],
                'allocations':([{'id':'B','kind':'panel','box_mm':[0,0,89,72],'group_ids':axes_ids[:3]+['panel-letter-b']}] if include_heatmap else [])+
                              [{'id':'E','kind':'panel','box_mm':[0,72 if include_heatmap else 0,89,47],'group_ids':axes_ids[-3:-1]+['panel-letter-e']},
                               {'id':'H','kind':'panel','box_mm':[0,119 if include_heatmap else 47,89,46],'group_ids':axes_ids[-1:]+['panel-letter-h']}],
                'background_ids':['patch_1'],'alignments':[],'data_axes':[]}
        # The shared B xlabel is explicitly owned, like all other figure text.
        if include_heatmap:
            fig.texts[0].set_gid('B-shared-xlabel'); frames['allocations'][0]['group_ids'].append('B-shared-xlabel')
        prov=FigureProvenance(figure_id='Figure 2 B E H',analysis_recipe='expanded-fixed-3strace-no-inference',source_file=str(Path(__file__)),source_symbol='render',source_hash=sha256_file(Path(__file__)),reproduction_snippet='.venv-trace/Scripts/python.exe scripts/build_figure2_trace3_panels.py --render',input_artifacts=tuple([artifact(OUT/'plan.json'),artifact(OUT/'calculation.json'),*calculation['outputs']]),analysis_identity={'methods':selection['methods'],'cohort_counts':calculation['cohort_counts'],'inference':'not run','presentation_profile_inputs':profile.inputs})
        export_matplotlib_figure(fig,OUT/basename,prov,mode=FigureMode.PUBLICATION,panel_ids=ids,allow_dirty_publication=True,formats=('svg',),layout_frames=frames)
        plt.close(fig)
    counts=blocks.groupby(['condition_id','block']).value.count().to_dict()
    legend=('(b) Control (91 fish) and 3sTrace (92 fish): equal-fish means of direct 1-s median log positive detected-bout vigor, centred by each fish/trial’s median finite baseline-bin value over [−15,0) s. Colours saturate at ±0.25; black is missing data. Green: CS onset/offset; violet: expected paired US at 13 s. Trials 1–90 represent acquisition trials 5–94.\n\n'
            '(e) Per-fish block medians of response minus baseline median log positive valid bout-frame vigor, using measured times: baseline [−15,0) s and response [0,13) s. PT, ET and LT are acquisition trials 10–14, 65–69 and 90–94; at least one eligible trial is required. Thin lines connect fish; black points and bars show medians and interquartile ranges. Eligible fish: '+', '.join(f'{c} {b}: {n}' for (c,b),n in sorted(counts.items()))+'.\n\n'
            '(h) Across-fish median trajectories of the same response-minus-baseline log-vigor outcome. Shading shows pointwise 95% percentile intervals from 5,000 whole-fish bootstrap resamples (seed 10), matching the descriptive Delay G recipe. Missing trials remain missing. No hypothesis tests, fitted models or significance annotations are included. These are unfrozen expanded-cohort drafts.')
    if not include_heatmap:
        legend='(e)'+legend.split('(e)',1)[1]
    with (OUT/(basename+'-legend.md')).open('x',encoding='utf-8') as stream: stream.write(legend+'\n')
    svg=(OUT/(basename+'.svg')).read_text(encoding='utf-8')
    page='<!doctype html><meta charset="utf-8"><title>Figure 2 B E H</title><style>body{margin:0;background:white}.sheet{width:89mm;margin:auto}svg{display:block;width:100%;height:auto}.legend{font:9pt/1.45 Georgia,serif;margin:12px 4px;white-space:pre-line}</style><main class="sheet">'+svg+'<div class="legend">'+html.escape(legend)+'</div></main>'
    with (OUT/(basename+'.html')).open('x',encoding='utf-8') as stream:stream.write(page)
    print('RENDERED '+str(OUT/(basename+'.html')),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--calculate',action='store_true')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--panels',choices=['BEH','EH'],default='BEH')
    args=parser.parse_args()
    if args.plan: plan()
    if args.calculate:
        lock=OUT/'.calculation.lock'
        save_new(lock,{'pid':os.getpid(),'script':artifact(Path(__file__))})
        try:
            guard_calculation_writes()
            calculate(args.limit)
        finally:
            lock.unlink()
    if args.render: render(include_heatmap=args.panels=='BEH')


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
