"""Add fish-level effect intervals and the scientific critique to a D/E outcome review."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    out = args.review.resolve()
    root = Path(__file__).resolve().parents[1]
    tests = pd.read_csv(out/"all_statistics.csv")
    pure = pd.read_csv(out/"sensitivity_log_existing_block_ratio_tests.csv")
    keys = ["panel", "family", "condition", "left", "right"]
    invariant = tests.loc[tests.version.eq("ratio") & tests.family.eq("between")].merge(pure.loc[pure.family.eq("between")], on=keys, suffixes=("_ratio", "_log"), validate="one_to_one")
    assert len(invariant) == 6
    np.testing.assert_allclose(invariant.p_raw_ratio, invariant.p_raw_log, rtol=1e-12, atol=1e-12)
    summary = pd.read_csv(out/"summary.csv")
    comparison = pd.read_csv(out/"paired_version_comparison.csv")
    diagnostic = pd.concat([pd.read_csv(out/f"{v}_paired_diagnostics.csv") for v in ["ratio", "log"]])
    effects = []
    for version in ["ratio", "log"]:
        changes = pd.read_csv(out/f"{version}_fish_changes.csv")
        for panel in ["D", "E"]:
            conditioned = "delay" if panel == "D" else "trace"
            for lo,hi in [(0,1),(1,2),(0,2)]:
                d=changes.loc[changes.panel.eq(panel)&changes.left.eq(lo)&changes.right.eq(hi)]
                t=d.loc[d.condition.eq(conditioned),"change"].to_numpy()
                c=d.loc[d.condition.eq("control"),"change"].to_numpy()
                # Resample fish changes, preserving both measurements of each pair.
                rng=np.random.default_rng(10)
                ti=rng.integers(0,len(t),size=(5000,len(t)))
                ci=rng.integers(0,len(c),size=(5000,len(c)))
                tb,cb=t[ti],c[ci]
                difference=np.median(tb,axis=1)-np.median(cb,axis=1)
                probability=((tb[:,:,None]>cb[:,None,:]).mean(axis=(1,2))+.5*(tb[:,:,None]==cb[:,None,:]).mean(axis=(1,2)))
                rank=2*probability-1
                point=2*((t[:,None]>c[None,:]).mean()+.5*(t[:,None]==c[None,:]).mean())-1
                lower,upper=np.quantile(difference,[.025,.975]); rlower,rupper=np.quantile(rank,[.025,.975])
                effects.append(dict(version=version,panel=panel,left=lo,right=hi,n_conditioned=len(t),n_control=len(c),
                                    difference_of_median_changes=np.median(t)-np.median(c),median_difference_ci_low=lower,median_difference_ci_high=upper,
                                    rank_biserial=point,rank_biserial_ci_low=rlower,rank_biserial_ci_high=rupper,
                                    bootstrap_draws=5000,seed=10,interval="pointwise percentile95; descriptive; no multiplicity correction"))
    effect=pd.DataFrame(effects)
    effect.to_csv(out/"direct_change_effects_bootstrap.csv",index=False)
    counts=[]
    for panel in ["D","E"]:
        trials=pd.read_parquet(out/f"{panel}_log_trials.parquet")
        for condition,d in trials.groupby("condition_id"):
            counts.append(dict(panel=panel,condition=condition,trials=len(d),current_eligible=int(d.current_trial_eligible.sum()),
                               log_eligible=int(d.log_trial_eligible.sum()),extra_missing=int((d.current_trial_eligible&~d.log_trial_eligible).sum()),
                               missing_baseline_positive_samples=int(d.baseline_positive_moving_samples.eq(0).sum()),
                               missing_response_positive_samples=int(d.response_positive_moving_samples.eq(0).sum())))
    flow=pd.DataFrame(counts);flow.to_csv(out/"trial_sample_flow.csv",index=False)
    sig=tests.loc[tests.p_holm24.lt(.05)]
    sig_lines=[]
    labels=["PT","ET","LT"]
    for r in sig.itertuples():
        contrast=labels[r.left]+(" versus "+labels[r.right] if r.left!=r.right else "")
        sig_lines.append(f"- {r.version}, {r.panel}, {r.family}, {r.condition}, {contrast}: raw p={r.p_raw:.6g}, Holm24 p={r.p_holm24:.6g}.")
    text=f'''# Figure 2 D/E: ratio versus legacy log-median outcome

Review date: 2026-10-09. These are alternative exploratory outcomes. The selected ratio panels and their selection/freeze records are preserved. No scientific choice or new freeze is made here.

## Recommendation

**Keep A, the current mean-activity ratio, as the primary D/E outcome for population conditioning/suppression; use B, the legacy log-median outcome, as a secondary analysis of movement intensity.** This recommendation follows the intended behavioral question, not which version produces more stars. A incorporates valid stationary frames, so it can reflect less movement as well as weaker movement. B conditions on positive detected movement and therefore asks what happens to typical vigor when a fish moves. It cannot by itself measure suppression through stillness. If typical within-bout intensity is the desired primary question, B is the relevant outcome and the claim/caption should be changed accordingly.

## What was held fixed and what changed

- Paper metric: `legacy_distal_angular_speed`, corrected frame column `legacy_distal_angular_speed_rad_per_ms`.
- Authenticated source cohorts: 29 Delay + 28 controls for D; 40 Trace + 19 controls for E. Identical source cohorts do not imply identical eligible fish/block counts after the outcome changes.
- CS alignment; measured-time half-open baseline [-15,0) s; response [0,9) s for D and [0,13) s for E; PT 10–14, ET 65–69, LT 90–94; minimum eligible trials D=3, E=1.
- A is the median of each fish/block's eligible trial ratios of **mean raw response activity / mean raw baseline activity**, including valid stationary frames. No-change reference=1.
- B computes the **median natural log of positive valid moving-frame vigor in response minus the corresponding median in baseline**, then the median of eligible trial differences within fish/block. No-change reference=0. Logs have no additive pseudocount. A multiplicative unit conversion cancels in the log difference.
- B reproduces the historical **outcome operation** on current corrected data. It does not restore historical rolling-median smoothing, row downsampling, interpolation, detector, inclusive endpoint overlap, or optional >90% NaN exclusion. It is not an exact reproduction of old generated figures. The historical files and SHA-256 records are in `../figure2_def_legacy_log_audit_20261009/`.
- B also requires current trial-ratio eligibility, plus at least one finite positive valid moving sample in each window. A trial with no eligible movement in either window has an undefined intensity comparison, not zero/no change. This extra eligibility is recorded per trial. The old conditional-vigor recipe's missingness cannot silently be interpreted as successful suppression or no response.

## Criticism of A: current ratio

1. It answers the total mean-activity question clearly and retains stillness, but combines movement frequency/duty and intensity. A ratio below one does not identify which component changed. Independent movement-probability and conditional-intensity outcomes are the remedy for a mechanistic claim.
2. Mean window values are sensitive to unusually large valid movements. Fish/block medians make the trial aggregation robust, but do not repair an influential frame within each window mean.
3. Dividing by a small/noisy baseline can amplify noise. The scale is asymmetric around one and paired ratio differences can be skewed. Positivity alone is not a precision or coverage guarantee. Inspect baseline level, measured coverage and influential fish before final inference; do not choose an arbitrary pseudocount to recover stars.
4. Current D and E minimum-trial rules differ. E can be represented by a single trial; per-block sample sizes and contributing-trial counts must accompany the plot. Ratios do not eliminate missingness bias or provide an extinction/onset estimate.

## Criticism of B: legacy log median

1. The zero-centered scale expresses multiplicative changes symmetrically, and the median within each window is less sensitive to extreme positive values. It is directly useful for typical movement-intensity modulation.
2. Dropping stationary/nonpositive samples changes the biological target. A fish can suppress the number of bouts without changing its retained moving-frame vigor, producing strong total-activity suppression and little B change. Median log differences are not simply log(A); changing the estimator and movement mask can change sample membership and results.
3. No movement makes intensity undefined. Extra exclusions are potentially informative: fish with the strongest suppression may disappear from B. Even retained B blocks may use a different set of trials than A. The comparison table records both outcomes by fish/block.
4. It weights moving samples, not bouts equally; long bouts contribute more samples. Short bouts, detector thresholds and the positive-value mask affect it. The median's robustness does not authenticate the detector or upstream metric.
5. Logs are not automatically required by rank tests and do not automatically validate signed-rank assumptions. The earlier legacy LME's additional log(x+1) is not copied into B's rank tests; applying it to already logged window values would be a separate, problematic transformation.

## Statistics and adequacy

Both versions have all 24 comparisons: six paired Wilcoxon within-condition block comparisons, three independent Mann–Whitney block comparisons, and three independent Mann–Whitney comparisons of matched within-fish changes per panel. Pairing is explicitly by fish. Data are not clipped before testing. Exact Wilcoxon is used only without zeros/absolute-difference ties; otherwise the recorded approximation is used. Recording day is not loaded into tests and no LMM is fitted. All 24 original ratio raw and Holm-adjusted p-values reproduced to tolerance 1e-12.

Holm24 is applied separately per version for consistency with the existing review family. This does not control a family selected across two alternative outcomes. These versions are neither independent replications nor an opportunity to select the smaller p-value. Paper primary/secondary estimands and families remain a scientific decision.

Wilcoxon concerns symmetric within-fish differences under its null; Mann–Whitney compares independent distributions/ranks, not automatically medians. Pairwise change distributions include asymmetry (see skewness and paired sign-test sensitivities in `*_paired_diagnostics.csv`). These descriptive sample diagnostics do not establish or categorically refute assumptions. The separate exact sign-test/Holm12 sensitivity asks about direction of paired change, discards its magnitude, and does not replace the displayed main tests or their Holm24 family.

Six direct condition-by-change contrasts across D/E are shown on companion plots for each version. Comparing a conditioned group's stars with control's ns does not test different changes. Neither version has {len(sig[sig.family.eq('change')])} statistically significant direct-change results combined under its respective Holm24 family; inspect the saved results rather than equating ns with equivalence or no conditioning.

`direct_change_effects_bootstrap.csv` adds rank-biserial effects and differences of median changes with 5,000 fish resamples, seed10. Each resampled fish brings its paired change intact. Percentile95 intervals are pointwise descriptive intervals, not simultaneous intervals; they need not agree with multiplicity-adjusted stars. Differences of medians are descriptive location effects and are not the Mann–Whitney test's universal estimand. The resampling preserves observed missingness and cannot remove informative-missingness bias.

`sensitivity_log_existing_block_ratio_tests.csv` isolates a log transform of the *existing fish/block ratios*. Monotone logging leaves same-block Mann–Whitney raw ranks/p-values unchanged; paired absolute-difference ranks and ranks of between-group changes can change. Therefore one must recompute paired/change tests, even for a pure transformation. This auxiliary sensitivity is not B's historical estimator.

### Significant main comparisons

{chr(10).join(sig_lines) if sig_lines else 'None under Holm24.'}

### Trial sample flow

```
{flow.to_string(index=False)}
```

### Eligible fish/block summaries

```
{summary.to_string(index=False)}
```

## Artifacts and verification

- `comparison.html`: A/B side-by-side D/E, all-test toggle, direct-change companions and complete results.
- `*_fish_blocks.parquet`, `*_log_trials.parquet`, `paired_version_comparison.csv`: plotted values, window sample counts and eligibility.
- `all_statistics.csv`, `*_tests.csv`, `*_fish_changes.csv`, `direct_change_effects_bootstrap.csv`: all tests, pairing, effects and uncertainty.
- `inputs.json`, `method.json`, figure sidecars and `verification.json`: authenticated provenance, recipe and checks. F remains placeholder/inconclusive; no missing inputs or tests are invented.
- Calculation checks cover exact window exclusion at response end, positive-only logarithms, unit invariance and undefined no-movement response. Geometry checks and PNG readback check legibility, annotation/headings, zero/one references and lack of point clipping.

References: [SciPy Wilcoxon](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html), [SciPy Mann–Whitney](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html), and [Gelman and Stern, 2006](https://doi.org/10.1198/000313006X152649).
'''
    # Avoid the awkward computed-count sentence when no adjusted direct effects survive.
    change_count=int(sig.family.eq('change').sum())
    text=text.replace(f"Neither version has {change_count} statistically significant direct-change results combined under its respective Holm24 family", f"There are {change_count} direct-change results below .05 across the two versions under their respective Holm24 families")
    (out/"critique.md").write_text(text,encoding="utf-8")
    notes=root/"Plans"/"PANEL_REVIEW_COMMENTS.md"
    current=notes.read_text(encoding="utf-8")
    start="#### Ratio versus legacy log-median outcome review — 2026-10-09"
    end="### F — 10sTrace/control block ratios"
    if start in current:
        a=current.index(start);b=current.index(end,a);current=current[:a]+current[b:]
    flow_lines="; ".join(f"{r.panel} {r.condition}: {r.extra_missing}/{r.current_eligible} otherwise-eligible trials additionally undefined" for r in flow.itertuples())
    section=f'''{start}

**Intended claim:** population conditioning/suppression through mean activity, with typical movement intensity as a distinct supporting question. **Author request:** create a separate legacy-log candidate, include updated statistics, criticize both versions and recommend the better primary outcome. No author selection of the new outcome is inferred.

**Current handling:** [side-by-side comparison](../reviews/figure2_ratio_vs_log_20261009/comparison.html) keeps the selected A ratios and adds B: median ln(positive valid moving vigor) in response minus baseline, followed by fish/block trial median. Both use the frozen metric, corrected measured-time sources, [-15,0) baseline, assay 9/13 s response windows, blocks 10–14/65–69/90–94 and minimum trial rules D=3/E=1. B implements the historical outcome; historical rolling smoothing/downsampling, interpolation, inclusive endpoints and detector are not restored. Reference is 0 for B and 1 for A; both are opaque black behind data. No day adjustment or LMM is used. Both have fresh 24 rank tests, separate Holm24, all-test tables and direct-change companions. The 24 A statistics reproduce the selected result. Extra B missingness: {flow_lines}.

| ID | Concern and evidence | Required remedy / current handling |
| --- | --- | --- |
| Fig2-DE-12 | A mixes movement frequency and intensity, can amplify a noisy small baseline, and uses frame-sensitive window means. B excludes stillness and uses robust positive moving-frame medians; this changes the biological target and can hide reduced movement probability. | **Recommendation, not decision:** A primary for overall conditioning/suppression; B supporting movement-intensity analysis. Choose by claim, not star count. Decompose duty/probability and intensity before making mechanism claims. |
| Fig2-DE-13 | B is undefined without positive movement in either window and may drop strongly suppressed trials/fish. Source cohorts are identical, eligible samples can differ. | Saved `trial_sample_flow.csv`, per-trial counts and `paired_version_comparison.csv` expose exclusions and trial availability. Review informative missingness and D/E thresholds under Plans02/07 before scientific freeze. No missing response is replaced with zero. |
| Fig2-DE-14 | Even a pure log transform can change paired Wilcoxon absolute-difference ranks and tests of between-group changes. B additionally changes masks/window aggregation. | Recompute all tests; pure-log-of-existing-block-ratio sensitivity saved separately. All same-block MW raw p-values remain invariant for that sensitivity. No borrowed legacy stars or old LME transformations. |
| Fig2-DE-15 | Outcome comparison adds analytic choices; separate Holm24 families do not authorize selecting the smaller adjusted p-value across versions. Pairwise changes show sample asymmetry; signed-rank adequacy remains unapproved. | Preserve every result, use paired sign-test/Holm12 only as a separately labeled direction sensitivity, and define paper primary/secondary estimands and families before final inference (Plans02; DECISIONS register when author selects). |
| Fig2-DE-16 | IQR is not effect uncertainty, and within-group significance does not demonstrate different changes. | Fresh direct-change tests and rank-biserial/difference-of-median-change estimates with 5,000 fish-bootstrap pointwise 95% intervals are saved. These descriptive intervals are not multiplicity-adjusted. There are {change_count} adjusted significant direct-change results across both versions; ns is not equivalence. |

**Presentation:** same paired style, connected black medians/fish IQR, significant-only main annotations and an all-test toggle. B is an unselected scientific candidate; no freeze command or current-selection overwrite. F remains placeholder/inconclusive. [Full critique, results and recommendation](../reviews/figure2_ratio_vs_log_20261009/critique.md) records exact sources, formulas, deviations and unresolved validation. Outstanding biological/coverage/assumption choices remain under Plans02/07 and require a decision entry if selected.

'''
    assert end in current
    notes.write_text(current.replace(end,section+end),encoding="utf-8")
    print(flow.to_string(index=False))
    print(effect.to_string(index=False))
    print("updated",notes)


if __name__=="__main__":
    raise SystemExit("This writer used the superseded all-frame interpretation. Run scripts/review_figure2_bout_only.py for the active comparison and critique.")
