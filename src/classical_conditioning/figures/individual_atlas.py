"""Native SVG fish cards and a self-contained exploratory classifier atlas."""
from __future__ import annotations

import base64
import html
import io
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd

from classical_conditioning.analysis.review.individual_atlas import VARIANTS, TRIALS
from classical_conditioning.figures.layout_frames import embed_spec

NS = "{http://www.w3.org/2000/svg}"
NAMES = {"legacy-nominal": "Nominal", "legacy-new": "New",
         "legacy-improved": "Improved", "legacy-wip": "WIP"}
COLOURS = {"delay": "#ec008c", "control": "#00aeef"}
HEATMAP_LIMIT = .25
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .6,
    "xtick.major.width": .6, "ytick.major.width": .6, "svg.hashsalt": "figure3-atlas"})


def results_for(output):
    results = {}
    for variant in VARIANTS:
        path = Path(output) / "classifiers" / variant / "results.parquet"
        if path.is_file():
            frame = pd.read_parquet(path)
            results[variant] = frame.set_index(["Condition", "Fish_ID"], verify_integrity=True)
    return results


def decision(results, variant, condition, fish):
    table = results.get(variant)
    if table is None:
        return "Failed", None
    if (condition, fish) not in table.index:
        return "Unclassified", None
    row = table.loc[(condition, fish)]
    flagged = bool(row.learner_primary)
    return ("Flagged" if condition == "control" and flagged else "Learner" if flagged
            else "Reference" if condition == "control" else "Nonlearner"), row


def score_value(row):
    if row is None:
        return np.nan
    for key in ("T_joint", "Mahalanobis_Distance", "Distance", "Mahalanobis_distance"):
        if key in row:
            return float(row[key])
    return np.nan


def feature_value(row, feature):
    if row is None:
        return np.nan
    for key in (f"Feature_{feature}_BLUP", f"Feature_{feature}", f"{feature}_BLUP", f"{feature}_blup"):
        if key in row:
            return float(row[key])
    return np.nan


def select_examples(inventory, results):
    """Transparent illustrative choices; agreement is never used as a truth label."""
    candidates = []
    for row in inventory.loc[inventory.condition_id.eq("delay")].itertuples(index=False):
        states = [decision(results, v, "delay", row.fish_id)[0] for v in VARIANTS]
        _, w = decision(results, "legacy-wip", "delay", row.fish_id)
        margin = abs(score_value(w)-float(w.decision_threshold)) if w is not None else np.inf
        candidates.append(dict(fish=row.fish_id, states=states, margin=margin,
            score=score_value(w), protocol_flag=bool(row.protocol_review_flag)))
    selected = []
    rules = [
        ("Clear agreement: all four rules flag this fish", lambda c: all(s=="Learner" for s in c["states"]), lambda c: -c["score"]),
        ("Agreement: all four rules leave this fish unflagged", lambda c: all(s=="Nonlearner" for s in c["states"]), lambda c: c["score"]),
        ("Disagreement across legacy rules", lambda c: "Learner" in c["states"] and "Nonlearner" in c["states"], lambda c: c["margin"]),
        ("Close to the WIP score threshold; all direction criteria still apply", lambda c: np.isfinite(c["margin"]), lambda c: c["margin"]),
    ]
    for reason, eligible, order in rules:
        pool = [c for c in candidates if not c["protocol_flag"] and eligible(c)
                and c["fish"] not in {s["fish_id"] for s in selected}]
        if pool:
            choice = sorted(pool, key=lambda c: (order(c), c["fish"]))[0]
            selected.append({"condition_id": "delay", "fish_id": choice["fish"], "reason": reason})
    for choice in candidates:
        if len(selected)==4:break
        if choice['fish'] not in {s['fish_id'] for s in selected} and not choice['protocol_flag']:
            selected.append({'condition_id':'delay','fish_id':choice['fish'],
                'reason':'Additional observed case; no class agreement is implied'})
    return selected


def svg_bytes(fig, identity, width, height):
    """Assign every artist to an invisible physical allocation, with explicit clips."""
    stream = io.BytesIO()
    fig.savefig(stream, format="svg", transparent=False, metadata={"Date":None})
    plt.close(fig)
    root = ET.fromstring(stream.getvalue())
    prefix = re.sub(r"[^a-zA-Z0-9_]", "_", identity)+"__"
    ids = {n.get("id"): prefix+n.get("id") for n in root.iter() if n.get("id")}
    text = ET.tostring(root, encoding="unicode")
    for old, new in ids.items():
        text = text.replace('id="'+old+'"', 'id="'+new+'"').replace('#'+old+'"', '#'+new+'"').replace('#'+old+')', '#'+new+')')
    parsed = ET.fromstring(text)
    spec = {"scope": "panel", "final_size_mm": [width, height], "inset_mm": 1.,
        "allocations": [{"id": identity, "kind": "panel", "box_mm": [0, 0, width, height],
                         "group_ids": [ids["figure_1"]]}],
        "background_ids": [ids["patch_1"]],
        "scientific_clips": [{"id": n.get("id"), "evidence": "Exact data-axis rectangle in this generated trial-data panel"}
                             for n in parsed.iter(NS+"clipPath")]}
    return embed_spec(text.encode(), spec)


def light_axes(ax):
    ax.tick_params(labelsize=7, pad=2, length=2)
    for boundary in (14.5, 64.5):
        ax.axvline(boundary, color="#a5abb0", lw=.6)
    ax.set_xlim(4.5, 94.5)
    ax.set_xticks([5, 15, 40, 65, 85, 94])


def g_trial_outcome(data):
    """Figure 2 G's observed outcome, without a second log or mean-ratio substitution."""
    return data.response_logmedian.to_numpy(float)-data.baseline_logmedian.to_numpy(float)


def trial_points(ax,data,values,colour,compact=False):
    """Observed trial values and descriptive five-trial medians; preserve missingness."""
    ax.scatter(data.trial_number,values,s=3.5 if compact else 8,color=colour,alpha=.55,linewidths=0)
    for start in range(5,95,5):
        mask=data.trial_number.between(start,start+4)&np.isfinite(values)
        if mask.any():ax.plot([start,start+4],[np.median(values[mask])]*2,color=colour,lw=1.2)
    ax.axhline(0,color="#666666",lw=.5,ls=":")
    ax.axvspan(40,59.5,color="#4b769b",alpha=.09,zorder=-1)
    ax.axvspan(60,69.5,color="#4b769b",alpha=.18,zorder=-1)
    light_axes(ax)


def fish_card(data, arrays, inventory_row, results, compact=False):
    fish, condition = inventory_row.fish_id, inventory_row.condition_id
    width, height = (91.5, 91.) if compact else (183., 190.)
    fig = plt.figure(figsize=(width/25.4, height/25.4))
    font = 7 if compact else 9
    colour = COLOURS[condition]
    def text(x,y,value,**kw):
        fig.text(x,y,value,fontsize=kw.pop("fontsize",font),**kw)
    text(.075,.965,fish+" · "+condition.capitalize(),weight="bold",color=colour)
    states = [NAMES[v]+": "+decision(results,v,condition,fish)[0] for v in VARIANTS]
    text(.075,.932,"  |  ".join(states[:2]), fontsize=font-1)
    text(.075,.905,"  |  ".join(states[2:]), fontsize=font-1)
    heat = arrays[fish+"__heat"]
    moving = arrays[fish+"__moving"]
    coverage = arrays[fish+"__coverage"]
    # In main cards retain heatmap + time-moving raster + exact WIP input.
    heat_box = [.13,.58,.72,.285] if compact else [.085,.66,.49,.205]
    ax = fig.add_axes(heat_box)
    cmap = plt.get_cmap("managua_r").copy(); cmap.set_bad("black")
    ax.imshow(heat,origin="upper",aspect="auto",extent=[-15,25,94.5,4.5],
              cmap=cmap,vmin=-HEATMAP_LIMIT,vmax=HEATMAP_LIMIT,interpolation="nearest",rasterized=True)
    # Detector availability is not raw tracking coverage. Never cover measured
    # vigor simply because a detector was eligible for less than 80% of a bin.
    overlay = np.zeros((*coverage.shape,4));overlay[coverage==0]=[.25,.28,.31,1]
    for boundary in (14.5,64.5): ax.axhline(boundary,color="#667078",lw=.55)
    for event in (0,10):ax.axvline(event,color="#111111",lw=.65)
    ax.axvline(9,color="#847043",lw=.6,ls="--")
    for trial in (25,39,53,59,65):ax.plot([-14],[trial],marker="<",color="#111111",ms=2.6,clip_on=False)
    # Actual US onsets, including unpaired controls, are event data rather than inferred guides.
    for row in data.itertuples(index=False):
        for event in json.loads(row.actual_us_s):
            ax.plot([event],[row.trial_number],"|",color="#986b19",ms=2.2,mew=.6)
    ax.set_yticks([5,15,40,65,85,94]);ax.set_xticks([-15,0,9,25])
    ax.tick_params(labelsize=font-1,pad=1,length=2)
    ax.set_ylabel("CS trial",fontsize=font);ax.set_title("Bout vigor · 1 s centred log median",fontsize=font,pad=3)
    if not compact:
        cax=fig.add_axes([.595,.695,.015,.135])
        bar=fig.colorbar(ScalarMappable(norm=Normalize(-HEATMAP_LIMIT,HEATMAP_LIMIT),cmap=cmap),cax=cax,extend="both",ticks=[-.25,0,.25])
        bar.set_label("Δ log vigor",fontsize=8);bar.ax.tick_params(labelsize=7)
        ax2=fig.add_axes([.70,.66,.235,.205])
        ax2.imshow(moving,origin="upper",aspect="auto",extent=[-15,25,94.5,4.5],
                   cmap="Greys",vmin=0,vmax=1,interpolation="nearest",rasterized=True)
        ax2.imshow(overlay,origin="upper",aspect="auto",extent=[-15,25,94.5,4.5],interpolation="nearest",rasterized=True)
        for event in (0,9,10):ax2.axvline(event,color="#757575",lw=.5)
        ax2.set_title("Fraction of time moving\nwhite 0 → black 1",fontsize=8,pad=3)
        ax2.set_xticks([-15,0,25]);ax2.set_yticks([]);ax2.tick_params(labelsize=7,length=2)
    ax.set_xlabel("Time from CS onset (s)",fontsize=font,labelpad=1)
    # A per-trial context strip preserves immobility rather than dropping no-bout outcomes.
    strip=fig.add_axes([.13,.485,.72,.045] if compact else [.085,.555,.85,.06])
    occ=np.vstack([data.baseline_fraction_moving,data.response_fraction_moving])
    labels=['Move B','Move CR']
    if not compact:
        occ=np.vstack([occ,data.baseline_coverage,data.response_coverage]);labels+=['Obs B','Obs CR']
    strip.imshow(occ,cmap="Greys",vmin=0,vmax=1,extent=[4.5,94.5,len(labels)-.5,-.5],aspect="auto",interpolation="nearest",rasterized=True)
    strip.set_yticks(range(len(labels)),labels,fontsize=font-1 if not compact else 5.5);strip.set_xticks([]);strip.tick_params(length=0,pad=2)
    scalar=fig.add_axes([.175,.235,.675,.19] if compact else [.105,.375,.83,.13])
    outcome=g_trial_outcome(data)
    trial_points(scalar,data,outcome,colour,compact)
    ratio=data.classifier_ratio.to_numpy(float)
    logratio=np.full_like(ratio,np.nan);ok=np.isfinite(ratio)&(ratio>0);logratio[ok]=np.log(ratio[ok])
    # Show full observed range, with common minimum context around no change.
    finite=outcome[np.isfinite(outcome)];lo=min(-.25,float(finite.min())-.05) if len(finite) else -.25
    hi=max(.25,float(finite.max())+.05) if len(finite) else .25
    scalar.set_ylim(lo,hi)
    scalar.set_ylabel("Median ln vigor\n(CR − baseline)",fontsize=font-1);scalar.set_xlabel("Scheduled CS trial",fontsize=font,labelpad=1)
    scalar.set_title("Figure 2 G outcome; bars = 5-trial medians",fontsize=font,pad=2)
    if not compact:
        legacy=fig.add_axes([.085,.205,.85,.11])
        trial_points(legacy,data,logratio,colour)
        legacy.set_ylabel("ln(CR/base)",fontsize=8)
        legacy.set_xlabel("Scheduled CS trial",fontsize=8,labelpad=1)
        legacy.set_title("Exact legacy WIP input · arithmetic bout means",fontsize=8,pad=2)
        # Reserve room for absolute log-axis ticks and the physical-unit label.
        context=fig.add_axes([.13,.065,.805,.075])
        context.plot(data.trial_number,data.baseline_bout_mean,color="#337f57",lw=.8,label="Baseline")
        context.plot(data.trial_number,data.response_bout_mean,color="#bf5b78",lw=.8,label="CR window")
        context.set_yscale("log");light_axes(context)
        context.set_ylabel("Bout mean\n(rad/ms)",fontsize=8)
        context.set_xlabel("Scheduled CS trial",fontsize=8,labelpad=1)
        context.legend(fontsize=7,frameon=False,ncol=2,loc="upper right")
        context.set_title("Conditional bout means on the same absolute scale",fontsize=8,pad=2)
    text(.075,.065 if compact else .013,
        f"G outcome {int(np.isfinite(outcome).sum())}/90 · WIP ratios {int(np.isfinite(logratio).sum())}/90 · "
        +("protocol flag" if inventory_row.protocol_review_flag else "no protocol flag"),fontsize=font-1)
    if compact:
        text(.075,.025,f"Detector observed: base {data.baseline_coverage.mean():.0%} · CR {data.response_coverage.mean():.0%}",fontsize=6)
    return svg_bytes(fig,"fish_"+condition+"_"+fish+("_compact" if compact else ""),width,height)


def overview_svg(trials, arrays, inventory, results, selected):
    """A four-example paper-layout trial; cards retain their own physical scale."""
    width,height=183.,220.
    root=ET.Element(NS+"svg",{"width":"183mm","height":"220mm","viewBox":"0 0 183 220"})
    head=ET.SubElement(root,NS+"g",{"id":"overview_heading"})
    def label(x,y,text,size=3,weight="normal"):
        n=ET.SubElement(head,NS+"text",{"x":str(x),"y":str(y),"font-family":"Arial","font-size":str(size),"font-weight":weight});n.text=text
    label(3,5,"Figure 3 · exploratory individual classification",3.6,"bold")
    label(3,10,"57 fish · tail bend angular speed · four historical rules on bout-only outcomes",2.5)
    label(3,15,"Illustrations explain rule-selected behavior. Method and individual uncertainty remain unapproved.",2.5)
    label(3,18,"Heatmaps: managua_r, -0.25 to +0.25; undefined intensity black; no numerical clipping",2.1)
    label(3,21,"Rule                 Delay flagged / eligible           Controls flagged / eligible",2.7,"bold")
    for i,variant in enumerate(VARIANTS):
        parts=[]
        for cond in ("delay","control"):
            table=results.get(variant)
            sub=table.loc[table.index.get_level_values(0)==cond] if table is not None else pd.DataFrame()
            parts.append(f"{int(sub.learner_primary.sum())}/{len(sub)}" if len(sub) else "failed")
        for x,value in [(3,NAMES[variant]),(62,parts[0]),(132,parts[1])]:
            label(x,26+i*3.2,value,2.6)
    allocations=[{"id":"A_B","kind":"panel","box_mm":[0,0,183,38],"group_ids":["overview_heading"]}]
    backgrounds=[];clips=[]
    for i,choice in enumerate(selected):
        row=inventory.loc[inventory.fish_id.eq(choice["fish_id"])].iloc[0]
        data=trials.loc[trials.fish_id.eq(row.fish_id)].sort_values("trial_number")
        card=ET.fromstring(fish_card(data,arrays,row,results,compact=True))
        x,y=(i%2)*91.5,38+(i//2)*91
        node=ET.SubElement(root,NS+"svg",{"x":str(x),"y":str(y),"width":"91.5","height":"91", "viewBox":card.get("viewBox")})
        metadata=card.find(NS+"metadata[@id='layout-frames']")
        spec=json.loads(metadata.text)
        for child in card:
            if child is not metadata:node.append(child)
        group=spec['allocations'][0]['group_ids'][0]
        allocations.append({"id":f"example_{i+1}","kind":"panel","box_mm":[x,y,91.5,91],"group_ids":[group]})
        backgrounds.extend(spec['background_ids']);clips.extend(spec['scientific_clips'])
    if len(selected)!=4:raise ValueError("Four distinct examples needed for this overview layout")
    spec={"scope":"figure","final_size_mm":[width,height],"inset_mm":1.,"allocations":allocations,
        "background_ids":backgrounds,"scientific_clips":clips}
    return embed_spec(ET.tostring(root),spec)


def decision_table(results, condition, fish):
    rows=[]
    for variant in VARIANTS:
        state,row=decision(results,variant,condition,fish)
        score=score_value(row);threshold=float(row.decision_threshold) if row is not None else np.nan
        def fmt(v):return f"{v:+.4g}" if np.isfinite(v) else "—"
        vote_key={'legacy-nominal':'Vote_Conservative','legacy-new':'Vote_Probabilistic',
                  'legacy-improved':'votes_dir','legacy-wip':'votes_dir'}[variant]
        votes=int(row[vote_key]) if row is not None else None
        failed=[]
        if row is not None:
            if score<=threshold:failed.append('score ≤ threshold')
            if votes<2:failed.append('fewer than 2 direction votes')
        reason='; '.join(failed) or ('criteria met' if row is not None else 'insufficient input/model information')
        rows.append(f"<tr><td>{NAMES[variant]}</td><td>{state}</td><td>{fmt(feature_value(row,'acquisition'))}</td>"
            f"<td>{fmt(feature_value(row,'extinction'))}</td><td>{fmt(score)}</td><td>{fmt(threshold)}</td>"
            f"<td>{str(votes)+'/2' if votes is not None else '—'}</td><td>{reason}</td></tr>")
    return '<table><thead><tr><th>Rule</th><th>Decision</th><th>Acquisition</th><th>Recovery</th><th>Score</th><th>Threshold</th><th>Votes</th><th>Rule explanation</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>'


def review_html(output, overview, cards, inventory, results, selected):
    """All cards and vectors embedded; controls filter without discarding any rows."""
    esc=html.escape
    select_rows=''.join(f'<li><a href="#fish-{r["fish_id"]}">{r["fish_id"]}</a> — {esc(r["reason"])}</li>' for r in selected)
    manifest=pd.read_csv(Path(output)/'classification-manifest.csv')
    warnings=[]
    for variant in VARIANTS:
        log=(Path(output)/'classifiers'/variant/'run.log').read_text()
        messages=re.findall(r'ConvergenceWarning: ([^\n]+)',log)
        warnings.append({'variant':variant,'warning_count':len(messages),'messages':sorted(set(messages))})
    warning_table='<table><thead><tr><th>Rule</th><th>Fit warnings</th><th>Reported warning</th></tr></thead><tbody>'+''.join(
        '<tr><td>'+NAMES[r['variant']]+'</td><td>'+str(r['warning_count'])+'</td><td>'+
        esc('; '.join(r['messages']) or 'None emitted')+'</td></tr>' for r in warnings)+'</tbody></table>'
    matrix=[]
    for row in inventory.sort_values(['condition_id','fish_id']).itertuples(index=False):
        states=[decision(results,v,row.condition_id,row.fish_id)[0] for v in VARIANTS]
        matrix.append(f'<tr><td><a href="#fish-{row.fish_id}">{row.fish_id}</a></td><td>{row.condition_id}</td>'
            +''.join('<td>'+s+'</td>' for s in states)+'</tr>')
    agreement='<table><thead><tr><th>Fish</th><th>Condition</th>'+''.join('<th>'+NAMES[v]+'</th>' for v in VARIANTS)+'</tr></thead><tbody>'+''.join(matrix)+'</tbody></table>'
    chunks=[]
    for row in inventory.sort_values(["condition_id","fish_id"]).itertuples(index=False):
        states={v:decision(results,v,row.condition_id,row.fish_id)[0] for v in VARIANTS}
        attributes=' '.join(f'data-{v}="{s.lower()}"' for v,s in states.items())
        flags=[name for name in ("baseline_3","cr_3","train_us","all_blocks_6") if not bool(getattr(row,name))]
        caution=("Protocol review flag. " if row.protocol_review_flag else "")+('Legacy sensitivity failures: '+', '.join(flags) if flags else 'No failures in the displayed legacy sensitivity checks.')
        ineligible=manifest.loc[manifest.fish_id.eq(row.fish_id)&~manifest.eligibility_reason.eq('classified')]
        reasons=''.join('<li>'+esc(NAMES[r.variant]+': '+r.eligibility_reason)+'</li>' for r in ineligible.itertuples(index=False))
        chunks.append(f'<article class="fish" id="fish-{row.fish_id}" data-condition="{row.condition_id}" {attributes}>'
            f'<h2>{row.fish_id} · {row.condition_id.capitalize()}</h2><p class="flag">{esc(caution)} '
            'All inventory fish are retained; these flags do not change membership.</p>'
            +cards[row.fish_id].decode().split('?>')[-1]+decision_table(results,row.condition_id,row.fish_id)
            +('<ul class="small">'+reasons+'</ul>' if reasons else '')
            +'<p class="small">Features and scores are legacy model outputs, not validated individual confidence statements. '
            'A nonlearner decision means the rule was not met. Control decisions are reference flags.</p></article>')
    payload=overview.decode().split('?>')[-1]
    return '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Figure 3 · individual fish exploration</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f6;color:#26343a;font:15px/1.5 system-ui,sans-serif}
header,main{max-width:1100px;margin:auto;padding:22px 28px}header{padding-bottom:0}h1{font-size:27px;line-height:1.2}h2{font-size:19px}h3{font-size:17px}
section,article{background:white;padding:22px;margin:18px 0;border-radius:9px}svg{display:block;max-width:100%;height:auto;margin:auto}
.note,.flag{background:#f5eee0;padding:10px 14px;border-radius:5px}.small{font-size:12px;color:#52646c}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:7px;text-align:left;border-bottom:1px solid #e0e5e7}
nav{display:flex;gap:18px;flex-wrap:wrap}a{color:#00738b}label{display:inline-flex;gap:8px;align-items:center;margin-right:12px}select,input{padding:8px;border:1px solid #b7c3c8;border-radius:5px;font:inherit}.toolbar{position:sticky;top:0;z-index:10;background:#e8eef0;padding:14px;border-radius:7px}
.fish[hidden]{display:none}.fish{scroll-margin-top:100px}.legend{max-width:900px}.count{font-weight:600}.overview svg{width:183mm;max-width:100%}
@media print{header,.toolbar,nav{display:none}body{background:white}main{padding:0;max-width:none}article{break-before:page;break-inside:avoid;padding:0;margin:0;border-radius:0}article svg{width:183mm}section{break-inside:avoid}}
</style><header><h1>Individual fish · exploratory Figure 3 review</h1>
<p>57 Delay/control fish · legacy tail bend angular speed · all four historical classifier recipes</p>
<nav><a href="#examples">Illustrative layout</a><a href="#methods">How to read the data</a><a href="#atlas">Complete fish atlas</a></nav>
<p class="note">Exploratory rule comparison. No canonical learner definition is selected. Same-data illustrations explain classifications; they do not independently validate learning.</p></header><main>
<section id="examples"><h2>Illustrative layout · first candidates</h2><p>Four automatically proposed, distinct examples, selected after computing the complete inventory. These choices are review candidates.</p><ul>'''+select_rows+'''</ul><div class="overview">'''+payload+'''</div></section>
<section id="methods" class="legend"><h2>Legend and interpretation</h2><h3>Legacy fit warnings</h3>'''+warning_table+'''
<p>Each heatmap row is one scheduled CS trial, 5–94. Columns span −15 to +25 s from CS onset. Color is the median natural log of positive, valid bout-frame angular speed in each 1 s bin, minus the median of defined baseline-bin medians [−15,0). All heatmaps use native managua_r with shared limits −0.25 to +0.25 and undefined intensity in black, matching the Figure 2 A/Figure 1 F–H heatmap analysis and scale. Saturation is visual only; numerical values remain unclipped in the data export. Condition marks and trial points use the selected Figure 2 G colors (Control cyan, Delay magenta). Measured bout vigor remains visible even when detector availability is low; no arbitrary coverage threshold masks these heatmaps.</p>
<p>Solid guides mark CS onset and offset (0 and 10 s), the dashed guide marks expected Delay US at 9 s, and ochre ticks mark actual US onsets, including unpaired control events. Left-edge arrowheads mark catches 25,39,53,59,65. The movement raster/strip uses white=0, black=1; dark gray raster cells indicate no detector-eligible time. Move B/CR are duration-weighted fractions moving among detector-eligible time in [−15,0) and [0,9). Obs B/CR show how much of each window was detector-eligible. Low detector availability is not a diagnosis of missing tracking: the independent source audit records complete coordinates for these fish.</p>
<p>The main trial dots use the Figure 2 G outcome: median ln positive response-bout vigor [0,9) minus median ln positive baseline-bout vigor [−15,0), computed directly from the saved eligible frame-log medians, with no second log or ratio-of-means substitution. Horizontal bars are five-trial medians for description; individual fish are not population summaries. The main trial axis includes at least −0.25 to +0.25 and expands to retain every observed value. Each full card separately shows ln(response/base), the exact legacy WIP input using arithmetic bout means before model/control adjustment. These quantities are distinct; the original classifiers and labels have not been recomputed. Green/pink context traces show conditional arithmetic bout means on one common absolute log axis. Missing bout windows remain undefined. No fabricated model trajectory is overlaid.</p>
<p>The light middle band marks trials 40–59 and the darker band 60–69. New/improved/WIP use middle trials 40–69; nominal uses 60–69. All use Pre5–14 and LateTest85–94. All default recipes consume late-Test data, so none of these Test displays is held-out validation.</p>
<p>Nominal/new fit log(response+1) adjusted for log(baseline+1); improved fits log(response) adjusted for log(baseline); WIP fits log(response/base). Their feature scales and thresholds differ. The nominal conservative vote uses 0.5 SE while historical reporting intervals use 1.96 SE. Joint-score thresholds, direction votes and empirical tail ranks are distinct criteria. Fit warnings are retained in each classifier log; the original fallback behavior is preserved.</p>
<p>The atlas retains the full 28-control/29-Delay source inventory. Technical and legacy behavioral-screen flags are context, not new exclusions. An unavailable/classification-ineligible fish would remain visible. Trace raw recordings are present, but an authenticated processed Trace frame/outcome bundle was not available in the checked mounted roots; no Trace labels or heatmaps have been invented.</p>
<p>Numerical data and provenance are saved beside this review: trial-data.csv/parquet, fish-inventory.csv, heatmaps.npz, classifiers/, selected-examples.json, input-lineage.json and data-checks.json. The matching editable main-layout artifact is overview.svg. The pilot uses the selected 1 mm invisible-frame policy.</p></section>
<section id="atlas"><h2>Complete atlas · all 57 fish</h2><p>Every fish has a full card and all four decisions. Choose a rule to filter its labels; this changes only what is visible.</p><details><summary>Show the complete cross-method decision matrix</summary>'''+agreement+'''</details></section>
<div class="toolbar"><label>Rule <select id="rule">'''+''.join(f'<option value="{v}">{NAMES[v]}</option>' for v in VARIANTS)+'''</select></label>
<label>Condition <select id="condition"><option value="all">All</option><option value="delay">Delay</option><option value="control">Control</option></select></label>
<label>Decision <select id="decision"><option value="all">All</option><option value="learner">Learner</option><option value="nonlearner">Nonlearner</option><option value="flagged">Flagged control</option><option value="reference">Reference control</option><option value="unclassified">Unclassified</option><option value="failed">Failed method</option></select></label>
<label>Fish <input id="search" placeholder="Fish ID" size="12"></label><span id="count" class="count"></span></div>
<div id="cards">'''+''.join(chunks)+'''</div></main>
<script>
const controls=['rule','condition','decision','search'].map(id=>document.getElementById(id));
function filter(){const [rule,condition,decision,search]=controls.map(x=>x.value.toLowerCase());let n=0;document.querySelectorAll('.fish').forEach(card=>{const visible=(condition==='all'||card.dataset.condition===condition)&&(decision==='all'||card.getAttribute('data-'+rule)===decision)&&card.id.includes(search);card.hidden=!visible;if(visible)n++});document.getElementById('count').textContent=n+' / 57 fish';}
controls.forEach(x=>x.addEventListener('input',filter));filter();
document.querySelectorAll('a[href^="#fish-"]').forEach(a=>a.addEventListener('click',()=>{controls[1].value='all';controls[2].value='all';controls[3].value='';filter()}));
</script></html>'''


def render(output):
    output=Path(output)
    trials=pd.read_parquet(output/"trial-data.parquet")
    inventory=pd.read_csv(output/"fish-inventory.csv")
    results=results_for(output)
    selected=select_examples(inventory,results)
    (output/"selected-examples.json").write_text(json.dumps(selected,indent=2)+'\n')
    cards={}
    with np.load(output/"heatmaps.npz") as arrays:
        for row in inventory.itertuples(index=False):
            cards[row.fish_id]=fish_card(trials.loc[trials.fish_id.eq(row.fish_id)].sort_values("trial_number"),arrays,row,results)
        overview=overview_svg(trials,arrays,inventory,results,selected)
    (output/"overview.svg").write_bytes(overview)
    (output/"review.html").write_text(review_html(output,overview,cards,inventory,results,selected))
    gdata=trials[['experiment_id','condition_id','fish_id','trial_number','baseline_logmedian','response_logmedian']].copy()
    gdata['response_minus_baseline_logmedian']=g_trial_outcome(trials)
    gdata.to_csv(output/'g-logmedian-outcomes.csv',index=False)
    (output/"legend.md").write_text("# Figure 3 exploratory individual atlas\n\n"+
        "All 57 Delay/control fish; legacy tail bend angular speed; four unapproved historical rules on reconstructed bout-only arithmetic means. "
        "Heatmap: direct 1 s median log positive bout vigor centred by median baseline-bin medians. "
        "Native managua_r, limits -0.25 to +0.25, undefined intensity black, no numerical clipping. "
        "Condition colors match the selected Figure 2 G (Control #00aeef, Delay #ec008c). "
        "Main trial plots use median ln response-bout vigor minus median ln baseline-bout vigor, as in Figure 2 G; axes expand beyond +/-0.25 when needed. "
        "Full cards retain the exact legacy WIP arithmetic-mean input separately; classifier results are unchanged. "
        "No-bout intensity is undefined; movement and detector availability are separate. "
        "Detector availability is not independent raw tracking completeness. "
        "The HTML interpretation section contains the complete event, epoch, transform, scope and eligibility legend. "
        "Selected examples illustrate same-data decisions and do not constitute independent validation.\n")
    # Exact embedded card bytes are also the standalone measurement inputs.
    encoded={fish:base64.b64encode(data).decode() for fish,data in cards.items()}
    (output/"card-svg-data.json").write_text(json.dumps(encoded))
    print(f"Rendered {len(cards)} complete fish cards and one illustrative SVG",flush=True)
