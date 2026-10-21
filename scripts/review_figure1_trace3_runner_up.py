"""Source-bound comparison of the current 3sTrace example and next candidates.

Descriptive analysis only: no learning-onset inference or Figure 1 publication.
"""
from __future__ import annotations
import base64
import html
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from classical_conditioning.figure_workspace import Snapshot, sha, publish_state
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme

REVIEW = Path(r'J:\ClassicalConditioning Outputs\current\reviews')
PRIOR = REVIEW / 'figure-01-trace3-learner-selection-20261019'
BASE = REVIEW / 'figure-02-3strace'
OUT = REVIEW / 'figure-01-trace3-runner-up-review-20261020'
TICKET = ROOT / 'records/figure-reviews/figure-01/trace3-second-example-start-20261020.json'
FISH = [('20230802_08', 'Current example; rank 2'),
        ('20230228_06', 'Next by score; rank 3'),
        ('20230419_10', 'Other shortlisted candidate; rank 5')]
EPOCHS = [('Pre',5,14), ('Train1-5',15,19), ('Train6-10',20,24),
          ('Train11-25',25,39), ('Train26-50',40,64), ('Test1-15',65,79), ('Test16-30',80,94)]


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    ticket = read(TICKET)
    for path, expected in ticket['starting_hashes'].items():
        if sha(Path(path).read_bytes()) != expected: raise ValueError('Changed starting source: ' + path)
    prior = read(PRIOR / 'selection-report.json')
    windows_path = BASE / 'matched-delay-statistics/windows.parquet'
    inputs = [TICKET, Path(__file__), windows_path, PRIOR / 'selection-report.json',
              PRIOR / 'full-score-coverage-exclusion-table.csv', ROOT / 'configs/paper-figures/nature-main.json',
              ROOT / 'configs/paper-figures/figure-elements.json', ROOT / 'configs/paper-figures/layout-frames.json']
    for fish, _ in FISH:
        inputs.extend([BASE/'fish'/(fish+'.json'), BASE/'fish'/(fish+'-bins.parquet')])
    existing = list(OUT.iterdir()) if OUT.is_dir() else []
    if existing and sha((OUT/'comparison.svg').read_bytes()) != read(OUT/'review.json')['artifact_sha256']:
        raise ValueError('Preserve manually edited comparison before regeneration')
    snapshot = Snapshot([*inputs, *map(Path, ticket['starting_hashes']), *existing])
    for path in [windows_path, *[BASE/'fish'/(fish+s) for fish,_ in FISH for s in ['.json','-bins.parquet']]]:
        if sha(path.read_bytes()) != prior['sources_sha256'][str(path)]:
            raise ValueError('Candidate input changed since original review: '+str(path))
    windows = pd.read_parquet(windows_path)
    summary, series, matrices, source_records = [], [], {}, {}
    for fish, description in FISH:
        record = read(BASE/'fish'/(fish+'.json'))
        bins_path = BASE/'fish'/(fish+'-bins.parquet')
        expected = next(x['sha256'] for x in record['outputs'] if x['path'] == str(bins_path))
        if sha(bins_path.read_bytes()) != expected: raise ValueError('Cache hash mismatch')
        cadence = record['heatmap_provenance']['cadence']
        if cadence['buffer_capacity_exceeded'] or cadence['missing_frame_ids']: raise ValueError('Timing gate failed')
        values = pd.read_parquet(bins_path).pivot(index='display_trial', columns='bin_index', values='delta_log_vigor').sort_index().to_numpy()
        if values.shape != (90,40): raise ValueError('Unexpected heatmap shape')
        matrices[fish] = values
        source_records[fish] = record
        g = windows[windows.fish_id == fish].sort_values('trial_number')
        for name, a, b in EPOCHS:
            x = g[g.trial_number.between(a,b)]
            summary.append(dict(fish_id=fish, epoch=name, acquisition_start=a, acquisition_end=b,
                n=int(x.legacy_log_median_difference.notna().sum()), scheduled=b-a+1,
                median=float(x.legacy_log_median_difference.median()),
                baseline_log_median=float(x.baseline_log_median.median()),
                response_log_median=float(x.response_log_median.median()),
                response_moving_fraction=float(x.response_positive_moving_samples.sum()/x.response_measured_frames.sum())))
        series.append(g.assign(display_trial=g.trial_number-4))
    table = pd.DataFrame(summary)
    profile = load_presentation_profile('nature-main')
    with mpl.rc_context():
        theme = apply_theme(profile=profile)
        mpl.rcParams.update({'figure.autolayout':False,'figure.constrained_layout.use':False})
        fig,axes = plt.subplots(3,2,figsize=(7.2,7.5),gridspec_kw={'width_ratios':[1,1.2]})
        fig.subplots_adjust(left=.095,right=.84,bottom=.075,top=.95,hspace=.40,wspace=.55)
        cmap=mpl.colormaps[theme.intensity_cmap].copy(); cmap.set_bad(theme.heatmap_bad_color)
        for i,(fish,description) in enumerate(FISH):
            ax,tx=axes[i]
            im=ax.imshow(matrices[fish],origin='upper',extent=(-20,20,90.5,.5),aspect='auto',
                interpolation='nearest',cmap=cmap,vmin=-.25,vmax=.25)
            ax.axvline(0,color=theme.cs_color,lw=.6)
            ax.axvline(10,color=theme.cs_color,lw=.6,ls='--')
            ax.axvline(13,color=theme.us_color,lw=.6,ls=':')
            for boundary in [10.5,60.5]:ax.axhline(boundary,color='white',lw=.5)
            ax.set(xlim=(-20,20),ylim=(90.5,.5),xticks=[-20,0,10,20],yticks=[1,10,30,60,90],ylabel='Displayed trial')
            ax.set_title(fish+'  |  '+description,loc='left',fontsize=6.5)
            g=series[i]
            tx.axvspan(.5,10.5,color='#777777',alpha=.10)
            tx.axvspan(60.5,90.5,color='#777777',alpha=.10)
            tx.axhline(0,color='#777777',lw=.5)
            tx.axvline(10.5,color='#777777',lw=.5,ls=':')
            tx.axvline(60.5,color='#777777',lw=.5,ls=':')
            tx.plot(g.display_trial,g.legacy_log_median_difference,'o',ms=2.1,color='#333333',clip_on=True)
            tx.set(xlim=(.5,90.5),ylim=(-.7,.7),xticks=[1,10,30,60,90],yticks=[-.5,0,.5],
                   ylabel='Response − baseline\nlog vigor')
            tx.set_title('Individual trial values; no smoothing',loc='left',fontsize=6.5)
        axes[-1,0].set_xlabel('Time from CS onset (s)')
        axes[-1,1].set_xlabel('Displayed trial')
        cb=fig.colorbar(im,cax=fig.add_axes([.905,.20,.015,.60]),ticks=[-.25,0,.25])
        cb.set_label('Log vigor relative to baseline')
        buf=__import__('io').BytesIO(); fig.savefig(buf,format='svg'); plt.close(fig)
    svg=buf.getvalue()
    method=('Heatmaps: direct 1 s medians of eligible log-frame vigor, centred by each trial’s median finite baseline-bin value over [−15,0) s; '
        'managua_r ±0.25, black missing. Same scales for all fish. Trial points: measured-time median log positive moving-frame response [0,13) minus '
        'baseline [−15,0); these are a distinct estimator from the heatmap bins. Display trials 1–10 / 11–60 / 61–90 are Pre / Train / Test. '
        'No learning-onset model or binary classification is applied. Rapid changes in these descriptive values alone do not establish single-trial learning.')
    notes=('20230228_06 is next after the selected fish in the original descriptive score ranking (rank 3), but its first two training values are already approximately −0.17 '
        'against a Pre median of +0.144. 20230419_10 (rank 5) is a previously shortlisted alternative whose first ten training values remain closer to its Pre level, '
        'with later training suppression. This is an illustrative trajectory comparison, not independent validation of learning or its onset. '
        'The earlier score leader was rejected because its contrast was driven by an unusually elevated Pre state. Figure 1 has not been changed by this review.')
    report=dict(scope='Figure 1 3sTrace next-example comparison; no replacement',
        fish=[f for f,_ in FISH], epochs=summary, methods=method, interpretation=notes,
        source_records=source_records, starting_ticket=str(TICKET), inputs=snapshot.hashes,
        profile_inputs=profile.inputs, artifact_sha256=sha(svg))
    legend=method+' '+notes
    page=('<!doctype html><html lang="en"><meta charset="utf-8"><title>3sTrace candidate comparison</title>'
        '<style>body{margin:24px auto;max-width:1050px;padding:0 20px;font:15px/1.5 Arial;color:#111}svg{width:100%;height:auto}.legend{border-top:1px solid #ddd;padding-top:16px}</style>'
        +svg.decode()+'<p class="legend">'+html.escape(legend)+'</p></html>').encode()
    snapshot.check()
    payloads=[('comparison.svg',svg),('comparison.html',page),
              ('epoch-summary.csv',table.to_csv(index=False).encode()),('trial-values.csv',pd.concat(series).to_csv(index=False).encode())]
    report['outputs']={name:sha(data) for name,data in payloads}
    if existing:
        # This comparison has its own guarded JSON commit point; Figure 1's
        # scoped presentation state is never written by the comparison tool.
        publish_state(OUT/'review.json',report,snapshot,
            evidence_updates=[(OUT/name,data) for name,data in payloads])
    else:
        OUT.mkdir(exist_ok=False)
        for name,data in [*payloads,('review.json',(json.dumps(report,indent=2)+'\n').encode())]:
            with (OUT/name).open('xb') as stream:stream.write(data)
        snapshot.check()
    print(table[['fish_id','epoch','n','median','response_moving_fraction']].to_string(index=False))
    print('Saved '+str(OUT))


if __name__=='__main__':main()
