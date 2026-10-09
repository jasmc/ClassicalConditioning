"""Recover Figure 2 review panels from authenticated, existing processed data.

No raw recording is reprocessed. Ratios use existing trial outcomes; heatmaps
reuse the shared signed-bout review calculation with the frozen [-15,0) baseline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns

from assemble_svg_figure import digest
from classical_conditioning.analysis.cohort_outcomes import load_cohort_trial_outcomes
from classical_conditioning.analysis.figure4 import verify_expected_us
from classical_conditioning.analysis.movement_state import resolve_candidate_metric_source
from classical_conditioning.cohort import load_cohort_manifest
from classical_conditioning.figures.cohort_response import SelectedBlock, summarize_selected_block_ratios, summarize_trial_ratios
from classical_conditioning.figures.example_traces import METRIC_COLUMNS
from classical_conditioning.figures.signed_bout_heatmap import calculate_fish_heatmaps, summarize_equal_fish_signed_log_vigor
from render_legacy_ssd_figure2_delay import _verify_cohort, _read_verified_recording
from render_legacy_ssd_example_heatmaps import _read_selected_windows
from render_legacy_ssd_example_traces import _verified_paths

REPO = Path(__file__).resolve().parents[1]
LAYOUT = REPO / "configs/paper-figures/figure2-assembly.json"
METRIC = "legacy_distal_angular_speed"
COLORS = {"control": "#29abe2", "delay": "#e90e8b", "trace": "#f15b2d"}
LABELS = {"control": "Control", "delay": "Delay", "trace": "3sTrace"}
BLOCKS = (SelectedBlock("Pre 10-14", 10, 14), SelectedBlock("Early Test 65-69", 65, 69), SelectedBlock("Late Test 90-94", 90, 94))
OLD_BLOCKS = (SelectedBlock("Pre 5-9", 5, 9), *BLOCKS[1:])


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def artifact(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": digest(path)}


def load_delay(project: Path):
    cohort, cohort_hash, inputs = _verify_cohort(project)
    rows = []
    for _, row in cohort.iterrows():
        _, outcomes, fish_inputs = _read_verified_recording(project, row)
        rid = str(row.recording_id)
        summary_path = project / "Quality checks" / rid / "candidate-trial-outcomes-corrected-v1_summary.json"
        marker_path = project / "Metadata" / f"{rid}_candidate-trial-outcomes-corrected-v1_complete.json"
        summary = json.loads(summary_path.read_text())
        marker = json.loads(marker_path.read_text())
        assert digest(summary_path) == marker["summary_sha256"], rid
        assert summary["config"] == {"baseline_window_s": [-15., 0.], "response_window_s": [0., 9.], "interval_closure": "left"}, rid
        rows.append(outcomes)
        inputs.extend([*fish_inputs, artifact(summary_path)])
    return pd.concat(rows, ignore_index=True), cohort, cohort_hash, inputs


def load_trace(project: Path):
    outcomes, summary = load_cohort_trial_outcomes(project, "all3sTrace-full-exploratory")
    cohort = load_cohort_manifest(project, "all3sTrace-full-exploratory")
    cohort = cohort.loc[cohort.primary_included.eq(True)].copy()
    inputs = [artifact(project / "Processed data/Cohorts/all3sTrace-full-exploratory/cohort-trial-outcomes.parquet")]
    # The standard loader authenticates every source summary and its assay settings.
    for rid in cohort.recording_id:
        inputs.append(artifact(project / "Quality checks" / str(rid) / "candidate-trial-outcomes-corrected_summary.json"))
    assert len(cohort) == 59 and cohort.groupby("condition_id").size().to_dict() == {"control": 19, "trace": 40}
    return outcomes, cohort, summary["cohort_hash"], inputs


def block_plot(fish: pd.DataFrame, *, paired: bool = False, boxes: bool = False):
    conditions = [c for c in ("control", "delay", "trace") if c in set(fish.condition_id)]
    fig, axes = plt.subplots(1, 2 if paired else 1, figsize=(5.8, 3.7), sharey=True, layout="constrained", squeeze=False)
    labels = fish[["Selected block order", "Selected block"]].drop_duplicates().sort_values("Selected block order")
    rng = np.random.default_rng(10)
    for ci, condition in enumerate(conditions):
        ax = axes[0, ci if paired else 0]
        d = fish.loc[fish.condition_id.eq(condition) & fish.Eligible]
        if paired:
            wide = d.pivot(index="fish_id", columns="Selected block order", values="Fish median response / baseline").reindex(columns=[0, 1, 2])
            for _, row in wide.iterrows():
                ax.plot(range(3), row, "o-", color=COLORS[condition], alpha=.2, lw=.6, ms=2)
        for order in range(3):
            values = d.loc[d["Selected block order"].eq(order), "Fish median response / baseline"].dropna()
            x = order if paired else order + [-.16, .16][ci]
            if boxes:
                ax.boxplot(values, positions=[x], widths=.25, patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=COLORS[condition], alpha=.25), medianprops=dict(color="black"), manage_ticks=False)
            if not paired:
                ax.scatter(x + rng.uniform(-.035, .035, len(values)), values, s=10, alpha=.5, color=COLORS[condition], linewidths=0)
            q25, median, q75 = values.quantile([.25, .5, .75])
            ax.errorbar(x, median, yerr=[[median-q25], [q75-median]], fmt="o", color=COLORS[condition], capsize=3, ms=4)
        ax.set_xticks(range(3), labels["Selected block"], fontsize=8)
        if paired:
            ax.set_title(f"{LABELS[condition]} (n={d.fish_id.nunique()})")
    for ax in axes.ravel():
        ax.axhline(1, color=".4", lw=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Global CS trial blocks")
    axes[0, 0].set_ylabel("Response / pre-CS baseline")
    if not paired:
        axes[0, 0].legend([Line2D([], [], marker="o", color=COLORS[c], ls="") for c in conditions],
                         [f"{LABELS[c]} (n={fish.loc[fish.condition_id.eq(c) & fish.Eligible, 'fish_id'].nunique()})" for c in conditions], frameon=False)
    return fig


def trial_plot(fish: pd.DataFrame, *, bootstrap: bool = False, split: bool = False):
    conditions = [c for c in ("control", "delay", "trace") if c in set(fish.condition_id)]
    fig, axes = plt.subplots(2 if split else 1, 1, figsize=(5.8, 4.3 if split else 3.6), sharex=True, sharey=True, layout="constrained", squeeze=False)
    for ci, condition in enumerate(conditions):
        ax = axes[ci if split else 0, 0]
        d = fish.loc[fish.condition_id.eq(condition) & fish["Fish median response / baseline"].notna()]
        label = f"{LABELS[condition]} (n={d.fish_id.nunique()})"
        if bootstrap:
            sns.lineplot(data=d, x="trial_number", y="Fish median response / baseline", estimator="median",
                         errorbar=("ci", 95), n_boot=100, seed=10, color=COLORS[condition], lw=1, label=label, ax=ax, err_kws={"alpha": .2})
        else:
            grouped = d.groupby("trial_number")["Fish median response / baseline"]
            median, low, high = grouped.median(), grouped.quantile(.25), grouped.quantile(.75)
            ax.plot(median.index, median, color=COLORS[condition], lw=1.2, label=label)
            ax.fill_between(median.index, low, high, color=COLORS[condition], alpha=.2, lw=0)
    for ax in axes.ravel():
        ax.axhline(1, color=".4", lw=.7)
        for boundary in (14.5, 64.5):
            ax.axvline(boundary, color=".5", ls=":", lw=.7)
        ax.set_xlim(4, 95)
        ax.set_ylabel("Response / pre-CS baseline")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[-1, 0].set_xlabel("Global CS trial")
    return fig


def heatmap_plot(data: pd.DataFrame, counts: dict):
    conditions = [c for c in ("control", "delay", "trace") if c in set(data.condition_id)]
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 4.8), sharey=True)
    fig.subplots_adjust(left=.10, right=.83, bottom=.13, top=.91, wspace=.22)
    palette = plt.get_cmap("managua_r").copy()
    palette.set_bad("black")
    for ax, condition in zip(axes, conditions):
        matrix = data.loc[data.condition_id.eq(condition)].pivot(index="Trial number", columns="Time bin center (s)", values="Mean signed log vigor").reindex(index=np.arange(5,95), columns=np.arange(-19.75,20,.5))
        # Each 0.5-s cell remains an editable vector polygon. NaNs remain black.
        main = ax.pcolormesh(np.arange(-20,20.5,.5), np.arange(4.5,95.5), matrix.to_numpy(),
                             cmap=palette, vmin=-.25, vmax=.25, rasterized=False, edgecolors="none", linewidth=0)
        ax.set_ylim(94.5,4.5); ax.set_xlim(-20,20)
        ax.set_title(f"{LABELS[condition]} (n={counts[condition]})", color=COLORS[condition])
        for t in (0,10): ax.axvline(t,color="#168241",lw=.65)
        for b in (14.5,64.5): ax.axhline(b,color="white",lw=.8)
        ax.set_xticks([-20,0,20]); ax.set_xlabel("Time from CS onset (s)")
    axes[0].set_ylabel("Global CS trial")
    cax=fig.add_axes([.87,.2,.024,.65])
    bar=fig.colorbar(main,cax=cax); bar.set_label("Mean signed log vigor",fontsize=8)
    bar.solids.set_rasterized(False)
    return fig


def save_panel(fig, target: Path, data: pd.DataFrame, identity: dict, inputs: list) -> dict:
    data_path=target.with_suffix(".parquet")
    data.to_parquet(data_path,index=False)
    fig.savefig(target,format="svg",facecolor="white")
    fig.savefig(target.with_suffix(".png"),dpi=170,facecolor="white")
    plt.close(fig)
    tree=ET.parse(target).getroot()
    assert not any(e.tag.endswith("}image") for e in tree.iter()), target
    sidecar=target.with_suffix(".figure.json")
    deps=[Path(__file__),REPO/'src/classical_conditioning/figures/cohort_response.py',REPO/'src/classical_conditioning/figures/signed_bout_heatmap.py',REPO/'src/classical_conditioning/analysis/temporal_profiles.py']
    payload={"analysis_identity":identity,"scientific_status":"provisional descriptive review; no scientific approval inferred",
             "inputs":inputs,"code_dependencies":[artifact(p) for p in deps],"panel_data":artifact(data_path),
             "outputs":[artifact(target),artifact(target.with_suffix('.png'))],
             "reproduction":".venv-trace/Scripts/python.exe scripts/populate_figure2_available.py --heatmaps"}
    write_json(sidecar,payload)
    return {**identity,"svg_sha256":digest(target),"panel_data":str(data_path),"panel_data_sha256":digest(data_path),
            "sidecar":str(sidecar),"sidecar_sha256":digest(sidecar)}


def identity(cohort_hash: str, response_end: int, **extra) -> dict:
    return {"baseline_s":[-15,0],"baseline_interval":"[-15,0)","metric_id":METRIC,"cohort_hash":cohort_hash,
            "significance_marks":"none","response_window_s":[0,response_end], **extra}


def verified_heatmap_paths(project: Path, rid: str, experiment: str):
    """Authenticate the processed inputs actually used by this presentation.

    Intake camera/tracking files are recorded lineage, not inputs recalculated
    here. Authenticating these figures does not resolve the upstream audit.
    """
    if experiment=='allDelay':
        _,metric,protocol=_verified_paths(project,rid,verify_corrected=False)
        movement=project/'Processed data'/rid/'movement_state_candidates-corrected-v2.parquet'
        marker_path=project/'Metadata'/f'{rid}_movement-candidate-corrected-v2_complete.json'
        marker=json.loads(marker_path.read_text())
        assert marker['status']=='complete' and marker['recording_id']==rid
        assert marker['movement_sha256']==digest(movement)
        records=[artifact(marker_path)]
    else:
        route=resolve_candidate_metric_source(metric_recipe='tail-candidate-corrected')
        metric=project/'Processed data'/rid/route.metrics_name
        movement=project/'Processed data'/rid/route.movement_artifact_name
        protocol=project/'Processed data'/rid/'stimulus_events.parquet'
        candidate_marker_path=project/'Metadata'/f'{rid}_{route.metric_marker_suffix}'
        movement_marker_path=project/'Metadata'/f'{rid}_{route.movement_marker_suffix}'
        candidate_summary_path=project/'Quality checks'/rid/route.metric_summary_name
        movement_summary_path=project/'Quality checks'/rid/route.movement_summary_name
        candidate_marker=json.loads(candidate_marker_path.read_text())
        movement_marker=json.loads(movement_marker_path.read_text())
        candidate_summary=json.loads(candidate_summary_path.read_text())
        movement_summary=json.loads(movement_summary_path.read_text())
        metric_hash=digest(metric);movement_hash=digest(movement)
        for marker in (candidate_marker,movement_marker):
            assert marker['status']=='complete' and marker['recording_id']==rid,rid
        assert candidate_marker['recipe']==route.metric_recipe
        assert movement_marker['recipe']==route.movement_recipe
        assert candidate_marker['metrics_sha256']==candidate_summary['artifact']['sha256']==metric_hash
        assert movement_marker['movement_sha256']==movement_summary['artifact']['sha256']==movement_hash
        assert candidate_marker['summary_sha256']==digest(candidate_summary_path)
        assert movement_marker['summary_sha256']==digest(movement_summary_path)
        assert movement_summary['inputs']['candidate_metrics']['sha256']==metric_hash
        source_path=project/'Metadata'/f'{rid}_source_manifest.json'
        source=json.loads(source_path.read_text())
        assert source['recording_id']==rid
        assert source['artifacts']['protocol']['sha256']==digest(protocol)
        records=[artifact(p) for p in (candidate_marker_path,movement_marker_path,candidate_summary_path,movement_summary_path,source_path)]
    paths=(metric,movement,protocol)
    records.extend(artifact(p) for p in paths)
    return paths,records


def fish_heatmaps(project, cohort, experiment, cache):
    frames=[]; inputs=[]
    for n,row in enumerate(cohort.itertuples(index=False),1):
        rid=str(row.recording_id);target=cache/f'{rid}.parquet'; manifest=target.with_suffix('.json')
        paths,source_records=verified_heatmap_paths(project,rid,experiment)
        metric_path,movement_path,protocol_path=paths
        protocol=pq.read_table(protocol_path).to_pandas()
        cycles=protocol.loc[protocol.Type.eq('Cycle')].sort_values('Beg').reset_index(drop=True)
        assert len(cycles)>=94,rid
        if row.condition_id!='control': verify_expected_us(protocol,experiment)
        inputs.extend(source_records)
        signature={"inputs":[artifact(p) for p in paths],"baseline_s":[-15,0],"metric_id":METRIC,
                   "signed_code_sha256":digest(REPO/'src/classical_conditioning/analysis/temporal_profiles.py'),
                   "bin_code_sha256":digest(REPO/'src/classical_conditioning/figures/signed_bout_heatmap.py')}
        if target.is_file() and manifest.is_file() and json.loads(manifest.read_text())==signature:
            bins=pd.read_parquet(target)
        else:
            intervals=[(int(cycles.iloc[t-1].Beg-20000),int(cycles.iloc[t-1].Beg+20000)) for t in range(5,95)]
            metrics=_read_selected_windows(metric_path,['FrameID','AbsoluteTime',METRIC_COLUMNS[METRIC]],intervals)
            movement=_read_selected_windows(movement_path,['FrameID','AbsoluteTime','valid','moving','bout_id'],intervals)
            bins=calculate_fish_heatmaps(metrics,movement,cycles,recording_id=rid,baseline_start_s=-15.,metric_ids=(METRIC,))
            bins.to_parquet(target,index=False);write_json(manifest,signature)
            del metrics,movement
        frames.append(bins)
        gc.collect()
        print(f'{experiment} heatmap {n}/{len(cohort)}: {rid}',flush=True)
    fish=pd.concat(frames,ignore_index=True)
    return fish,summarize_equal_fish_signed_log_vigor(fish,metric_id=METRIC,condition_by_recording=cohort.set_index('recording_id').condition_id.to_dict()),inputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heatmaps',action='store_true')
    parser.add_argument('--delay-project',type=Path,default=Path('J:/Digested Data/allDelay-full-v1'))
    parser.add_argument('--trace-project',type=Path,default=Path('F:/Digested Data/all3sTrace-full-v1'))
    args=parser.parse_args()
    layout=json.loads(LAYOUT.read_text())
    base=Path(layout['storage_root']);stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=base/'sources'/stamp;output.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'})
    selected={}
    for exp,project,loader,letters,end,min_trials in [('allDelay',args.delay_project,load_delay,'DG',9,3),('all3sTrace',args.trace_project,load_trace,'EH',13,1)]:
        outcomes,cohort,cohort_hash,inputs=loader(project)
        counts=cohort.groupby('condition_id').size().to_dict()
        print(f'Authenticated {exp}: {counts}; baseline [-15,0), response [0,{end})',flush=True)
        trials,_=summarize_trial_ratios(outcomes,metric_id=METRIC)
        trials=trials.loc[trials.trial_number.between(5,94)].copy()
        fish,_=summarize_selected_block_ratios(outcomes,metric_id=METRIC,selected_blocks=BLOCKS,min_trials_per_fish_block=min_trials)
        block_id=identity(cohort_hash,end,block_trials=[[10,14],[65,69],[90,94]],min_trials_per_fish_block=min_trials,
                          ratio_formula='response_total_activity / baseline_total_activity; median of eligible trial ratios within fish/block',cohort_counts=counts,summary='fish points and median [fish IQR]')
        trial_id=identity(cohort_hash,end,ratio_formula='response_total_activity / baseline_total_activity; fish trial ratio then equal-fish condition median',cohort_counts=counts,summary='condition median [fish IQR]')
        for letter,fig,data,ident in [(letters[0],block_plot(fish),fish,block_id),(letters[1],trial_plot(trials),trials,trial_id)]:
            target=output/f'Fig2_Panel{letter}_{exp}_pre15.svg'
            selected[letter]=(target,save_panel(fig,target,data,ident,inputs))
        if exp=='all3sTrace':
            old,_=summarize_selected_block_ratios(outcomes,metric_id=METRIC,selected_blocks=OLD_BLOCKS,min_trials_per_fish_block=1)
            old=old.loc[old.Eligible].copy()
            historical=identity(cohort_hash,end,block_trials=[[5,9],[65,69],[90,94]],min_trials_per_fish_block=1,
                                cohort_counts=counts,selection='historical 5-9 block comparison; excluded from main E')
            for style,fig in [('boxplot',block_plot(old,boxes=True)),('median-iqr',block_plot(old)),('paired',block_plot(old,paired=True))]:
                save_panel(fig,output/f'Fig2_PanelE_historical5-9_{style}.svg',old,historical,inputs)
            for style,fig in [('bootstrap',trial_plot(trials,bootstrap=True)),('split-bootstrap',trial_plot(trials,bootstrap=True,split=True))]:
                save_panel(fig,output/f'Fig2_PanelH_{style}.svg',trials,{**trial_id,'summary':'median [95% bootstrap CI], 100 resamples, seed 10'},inputs)
            # Verify historical E/H reconstruction against the saved 59-fish dataset.
            old_root=Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/trace-legacy-full-59/figures/figure2-3strace-window13-legacy-59fish')
            for data,name,keys in [(old,'figure-2E_fish-block-data.csv',['fish_id','condition_id','Selected block order']),(trials,'figure-2H_fish-trial-data.csv',['fish_id','condition_id','trial_number'])]:
                prior=pd.read_csv(old_root/name);value='Fish median response / baseline'
                joined=data[keys+[value]].merge(prior[keys+[value]],on=keys,validate='one_to_one',suffixes=('_new','_old'))
                assert len(joined)==len(data)==len(prior),name
                np.testing.assert_allclose(joined[value+'_new'],joined[value+'_old'],rtol=1e-12,atol=1e-12,equal_nan=True)
            print('Verified historical 59-fish E/H numeric reproduction',flush=True)
        if args.heatmaps:
            cache=base/'heatmap-cache'/exp;cache.mkdir(parents=True,exist_ok=True)
            fish_bins,pooled,heat_inputs=fish_heatmaps(project,cohort,exp,cache)
            fish_path=output/f'{exp}_signed-fish-bins-pre15.parquet';fish_bins.to_parquet(fish_path,index=False)
            letter='A' if exp=='allDelay' else 'B'
            ident=identity(cohort_hash,end,cohort_counts=counts,signal='equal-fish mean of shared signed bout-log-vigor 0.5-s bins; no P10/P90 scaling',
                           display={'cmap':'managua_r','limits':[-.25,.25],'missing':'black'},upstream_audit='vigor and heatmap alignment remain unverified')
            target=output/f'Fig2_Panel{letter}_{exp}_signed-pre15.svg'
            selected[letter]=(target,save_panel(heatmap_plot(pooled,counts),target,pooled,ident,[*inputs,*heat_inputs,artifact(fish_path)]))
    for panel in layout['panels']:
        if panel['id'] not in selected: continue
        target,provenance=selected[panel['id']]
        panel['source']=str(target);panel['source_provenance']=provenance
        panel['selection_status']='provisional descriptive review; scientific approval pending'
        panel['pending_inputs']=['scientific outcome/cohort review','upstream vigor and alignment audit']
        panel['scientific_definition'].update(provenance)
    # Add short, visible panel-status notes in the assembly, outside each source.
    layout['annotations']=[a for a in layout['annotations'] if not a.get('panel_note')]
    for panel in layout['panels']:
        if panel.get('source'):
            x,y,w,h=panel['box']
            text='Provisional signed-bout review; upstream audit open' if panel['id'] in 'AB' else 'Provisional descriptive ratio; no inferential marks'
            layout['annotations'].append({'x':x+w/2,'y':y+h-9,'size':17,'anchor':'middle','text':text,'panel_note':True})
    layout['annotations'][1]['text']='Available-data review - populated sources remain provisional; 10sTrace remains inconclusive'
    layout['scientific_status']='populated available-data review; scientific approval remains separate'
    write_json(LAYOUT,layout)
    write_json(output/'recovery.json',{'source_chats':[{'id':'01a0d4b6-87d9-7703-a771-e6e91c1c896a','title':'Data processing'},
                                                    {'id':'01a0d967-3f9c-7bc3-b97b-0451f556de3d','title':'Compare panel layouts in Figures 1–2'}],
                                     'selected':{k:{'svg':str(v[0]),'provenance':v[1]} for k,v in selected.items()},
                                     'uploaded_reference':'C:/Users/joaquim/Desktop/Asset 7.svg',
                                     'uploaded_reference_sha256':digest(Path('C:/Users/joaquim/Desktop/Asset 7.svg'))})
    print('Updated layout sources: '+','.join(sorted(selected)),flush=True)
    print(output,flush=True)


if __name__=='__main__':main()
