"""Occurrence context and schedule/event audits; neither alters vigor inputs."""
import hashlib
import html
from pathlib import Path
import numpy as np
import pandas as pd
from .panel_g_review_visuals import svg_text


def companion_views(out):
    import matplotlib.pyplot as plt
    windows=pd.read_parquet(out / "scheduled-window-outcomes.parquet")
    columns=["movement_probability","fraction_time_moving"]
    summaries=windows.groupby(["condition_id","trial_number"],observed=True)[columns].mean().reset_index()
    summaries.to_csv(out / "movement-occurrence-companion.csv",index=False)
    fig,axes=plt.subplots(2,1,figsize=(9,5),layout="constrained",sharex=True)
    for condition,color in [("control","#009fd5"),("delay","#d83983")]:
        g=summaries.loc[summaries.condition_id==condition]
        for axis,column in zip(axes,columns):axis.plot(g.trial_number,g[column],color=color,lw=1,label=condition.title())
    for axis in axes:
        axis.axvline(14.5,color=".5",ls=":",lw=.7);axis.axvline(64.5,color=".5",ls=":",lw=.7)
        axis.spines[["top","right"]].set_visible(False);axis.legend(frameon=False)
    axes[0].set_ylabel("Fish with response movement\n(proportion)")
    axes[1].set_ylabel("Analyzable response time moving\n(mean fraction)");axes[1].set_xlabel("Scheduled CS trial")
    fig.suptitle("Movement occurrence context — includes valid no-bout trials")
    image=svg_text(fig,"occurrence-companion");plt.close(fig)
    (out / "movement-occurrence-companion.svg").write_text(image)
    # Count prior schedule-declared pairings, with actual nearest-US timing
    # evidence retained separately. No arbitrary matching cutoff declares a pair.
    import json
    lineage=json.loads((out / "input-lineage.json").read_text())
    protocols={Path(x['path']).parent.name:x for x in lineage
               if Path(x['path']).name=='stimulus_events.parquet'}
    audit=[]
    for fish,g in windows.loc[windows.condition_id=='delay'].groupby('fish_id',observed=True):
        record=protocols[fish];p=Path(record['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Changed event source')
        events=pd.read_parquet(p);us=events.loc[events.Type.astype(str)=='Reinforcer'].sort_values('Beg')
        prior=0
        for row in g.sort_values('trial_number').itertuples():
            scheduled=15<=row.trial_number<=64 and not row.is_catch
            differences=us.Beg.to_numpy()-row.event_start_absolute_time_ms
            nearest=int(np.argmin(abs(differences-9000)))
            audit.append({'fish_id':fish,'trial_number':row.trial_number,
                'schedule_declared_pairing':scheduled,'schedule_declared_prior_pairings':prior,
                'nearest_US_latency_ms':int(differences[nearest]),
                'nearest_error_to_nominal_9s_ms':int(abs(differences[nearest]-9000)),
                'actual_pairing_status':'timing_evidence_only_no_new_pairing_classification'})
            prior+=int(scheduled)
    pd.DataFrame(audit).to_csv(out / 'pairing-exposure-audit.csv',index=False)
    return ('<h2>Movement occurrence context</h2>'+image+
        '<p>These descriptive source outcomes include valid no-bout trials. Time fractions refer to detector-analyzable response time. Each trial has one vote per fish. '
        'They contextualize conditional vigor and are not additional inferential claims. '
        'Pairing exposure is retained in the schedule/event audit, with actual timing evidence separate from schedule declarations.</p>')
