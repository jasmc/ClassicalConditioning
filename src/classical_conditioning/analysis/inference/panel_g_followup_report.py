"""Review repair evidence, historical cohort effects and estimator comparison."""
import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd


def technical_section(out):
    from .panel_g_run import sha,verify_stage
    binding=out.parents[1]/'review-supplements.json'
    if not binding.exists():return ''
    starting=sha(binding);record=json.loads(binding.read_text())
    source=out.parents[1]/record['technical']['directory']
    expected=record['technical']['manifest_sha256'];verify_stage(source,expected)
    names={'technical-recording-audit.csv':'technical-recording-audit.csv',
        'technical-protocol-rows.csv':'technical-protocol-rows.csv',
        'legacy-cohort-phase-results.csv':'technical-phase-results.csv',
        'legacy-cohort-timing-results.csv':'technical-timing-results.csv',
        'technical-policy-before-fitting.json':'technical-policy-before-fitting.json'}
    for old,new in names.items():shutil.copyfile(source/old,out/new)
    trace=Path(record['raw_trace']['path'])
    if sha(trace)!=record['raw_trace']['sha256']:raise ValueError('Raw protocol trace changed')
    shutil.copyfile(trace,out/'raw-protocol-trace.json')
    audit=pd.read_csv(out/'technical-recording-audit.csv')
    flags=audit.loc[audit.protocol_review_flag]
    phase=pd.read_csv(out/'technical-phase-results.csv')
    if sha(binding)!=starting:raise ValueError('Supplement binding changed while rendering')
    verify_stage(source,expected)
    (out/'review-supplements-binding.json').write_text(json.dumps({**record,
        'selection_file':str(binding),'selection_sha256':starting},indent=2)+'\n')
    return ('<h2>Technical acquisition review</h2>'
        '<p>No frame-ID gaps, duplicates or reversals were recorded in either stream across these 57 fish. '
        'Camera/tracking boundary offsets range from zero to five frames. '
        'One Delay fish, 20221122_03, has precision-lost timestamps (1.66911E+12) in its initial two US records. '
        'Its selected CS windows are available. Intended malformed timestamps have not been reconstructed.</p>'
        +flags[['fish_id','condition_id','acquisition_status','invalid_protocol_durations','protocol_events_outside_acquisition']].to_html(index=False,border=0)
        +'<p>A separate conservative sensitivity excludes this fish and retains 28 controls and 28 Delay fish. '
        'It is independent of vigor/learner strength and does not change primary membership.</p>'
        +phase.round(4).to_html(index=False,border=0))


def followup_sections(out):
    parts=[]
    failures=[]
    for target in ['difference','adjusted_response']:
        directory=out/target
        if (directory/'leave-one-fish-out.csv').exists():
            f=pd.read_csv(directory/'leave-one-fish-out.csv')
            failures.append({'target':target,'check':'leave-one-fish-out','attempted':len(f),
                'failed':int(f.status.ne('ok').sum())})
        if (directory/'random-effects-sensitivity.json').exists():
            for row in json.loads((directory/'random-effects-sensitivity.json').read_text()):
                failures.append({'target':target,'check':row.get('sensitivity','random intercept'),
                    'attempted':1,'failed':int(row['diagnostic']['diagnostic_status']!='ok')})
    if failures:
        parts.extend(['<h2>Convergence and robustness checks</h2>',pd.DataFrame(failures).to_html(index=False,border=0),
            '<p>Numerical retries preserve the fixed/random model structure. A failed or singular sensitivity is not evidence of robustness.</p>'])
    if (out/'legacy-sensitivity-membership.csv').exists():
        m=pd.read_csv(out/'legacy-sensitivity-membership.csv')
        counts=m.groupby(['recipe','condition_id']).included.agg(['size','sum']).reset_index()
        counts=counts.rename(columns={'size':'original_fish','sum':'retained_fish'})
        counts['discarded_fish']=counts.original_fish-counts.retained_fish
        parts.extend(['<h2>February/March discarding experiments</h2>',
            '<p>Rules were read from Git revisions 2f63ef4 (February 14) and bf46bf7 (March 24). '
            'They are applied as named sensitivities to authenticated corrected detector/event data. '
            'The original segmented historical Bout/700 FPS pickle is unavailable, so these are projections, not exact historical cohorts. '
            'The combined recipe uses an explicitly labelled raw-US-duration proxy for the historical frame-annotated last-US end. '
            'No selected G or primary paper cohort changes.</p>',counts.to_html(index=False,border=0),
            '<details><summary>Fish identities and failed rules</summary>',m.to_html(index=False,border=0),'</details>'])
        if (out/'legacy-cohort-phase-results.csv').exists():
            results=pd.read_csv(out/'legacy-cohort-phase-results.csv')
            parts.extend(['<h3>Condition differences by sensitivity cohort</h3>',results.round(4).to_html(index=False,border=0),
                '<p>Bands cover 90 trial contrasts and three phase summaries within each candidate target and cohort. '
                'They do not control selection across recipes, and no recipe is chosen by its p-value. '
                'Behavior-based discarding changes the target population and can bias an apparent learning effect.</p>'])
    if (out/'unsmoothed-fish-change-trajectory.csv').exists():
        import matplotlib.pyplot as plt
        from .panel_g_review_visuals import svg_text
        g=pd.read_csv(out/'unsmoothed-fish-change-trajectory.csv')
        fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
        ax.plot(g.trial_number,g.estimate,color='#ca7530',lw=1,label='Unsmoothed fish change')
        ax.fill_between(g.trial_number,g.simultaneous_lower,g.simultaneous_upper,color='#ca7530',alpha=.16)
        p=out/'difference/population-trajectory.csv'
        if p.exists():
            model=pd.read_csv(p)
            for i,(phase,part) in enumerate(model.groupby('fit_phase',sort=False)):
                ax.plot(part.trial_number,part.estimate,color='#7543a1',lw=1.2,label='Phase-aware LMM' if i==0 else None)
        ax.axhline(0,color='.5',lw=.7);ax.axvline(14.5,color='.5',ls=':',lw=.7);ax.axvline(64.5,color='.5',ls=':',lw=.7)
        ax.set(xlabel='Scheduled CS trial',ylabel='Control − Delay change (log units)',xlim=(4,95))
        ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False)
        image=svg_text(fig,'unsmoothed-comparator');plt.close(fig)
        (out/'unsmoothed-fish-change.svg').write_text(image)
        parts.extend(['<h2>Trajectory shape comparison</h2>',image,
            '<p>The orange comparator gives each contributing fish one vote and anchors it by its own available Pre mean. '
            'It avoids a smooth mean curve, but has a different estimand from the LMM. Missing bout vigor stays undefined. '
            'Its uncertainty cannot validate the adjusted-response or individual models.</p>',
            '<p>Phase contributor counts reflect fish with an available Pre reference and at least one phase outcome.</p>'])
        from .panel_g_marginal import fish_matrix
        index,values=fish_matrix(pd.read_parquet(out/'model-input.parquet'))
        phase=pd.read_csv(out/'unsmoothed-fish-change-phases.csv')
        for condition in ['control','delay']:
            chosen=np.asarray(index.get_level_values('condition_id').astype(str))==condition
            phase[condition+'_contributing_fish']=np.isfinite(values[chosen,90:]).sum(axis=0)
        phase.to_csv(out/'unsmoothed-phase-contributors.csv',index=False)
        parts.append(phase.round(4).to_html(index=False,border=0))
    if (out/'unsmoothed-calibration-summary.csv').exists():
        parts.extend(['<h3>Repeated unsmoothed calibration</h3>',
            pd.read_csv(out/'unsmoothed-calibration-summary.csv',keep_default_na=False,na_values=['']).round(4).to_html(index=False,border=0),
            '<p>Rows count independent simulated experiments and show binomial uncertainty for coverage and false supported runs. '
            'Informative availability is a deliberately adverse selection scenario with a defined simulated latent outcome; '
            'it does not impute a real bout intensity where no bout occurred. Calibration limitations remain estimator-specific.</p>'])
    if (out/'individual-covariance-sensitivity.csv').exists():
        f=pd.read_csv(out/'individual-covariance-sensitivity.csv')
        summary=f.groupby(['method','variant']).maximum_mean_shift.max().rename('maximum_expression_shift').reset_index()
        parts.extend(['<h2>Individual covariance sensitivity</h2>',summary.round(4).to_html(index=False,border=0),
            '<p>Variance components are perturbed by factors 0.5 and 2; AR1 by ±0.1. '
            'These checks do not integrate covariance uncertainty or approve individual acquisition intervals. '
            'Full fits, excluded-control validation and simulations now use the same scheduled-grid timing and information rule.</p>'])
    parts.append(technical_section(out))
    return ''.join(parts)
