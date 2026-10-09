"""Rebuild both active D/E outcomes under the standing bout-only vigor policy.

Reuse hash-authenticated log summaries; never duplicate the frame recordings.
"""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import review_figure2_block_log_median as review
from populate_figure2_available import load_delay, load_trace, BLOCKS, METRIC
from classical_conditioning.figures.cohort_response import summarize_selected_block_ratios
from classical_conditioning.analysis.bout_vigor import VIGOR_SAMPLE_POLICY

OLD = ROOT / "reviews/figure2_ratio_vs_log_20261009"
OUT = ROOT / "reviews/figure2_bout_only_ratio_vs_log_20261009"


def change_effects(changes, version):
    rows = []
    for panel in ("D", "E"):
        condition = "delay" if panel == "D" else "trace"
        for lo, hi in review.PAIRS:
            sub = changes.loc[changes.panel.eq(panel) & changes.left.eq(lo) & changes.right.eq(hi)]
            t = sub.loc[sub.condition.eq(condition), "change"].to_numpy()
            c = sub.loc[sub.condition.eq("control"), "change"].to_numpy()
            rng = np.random.default_rng(10)
            tb = t[rng.integers(0, len(t), size=(5000, len(t)))]
            cb = c[rng.integers(0, len(c), size=(5000, len(c)))]
            difference = np.median(tb, axis=1) - np.median(cb, axis=1)
            ranks = 2*((tb[:, :, None] > cb[:, None, :]).mean(axis=(1,2)) + .5*(tb[:, :, None] == cb[:, None, :]).mean(axis=(1,2))) - 1
            point = 2*((t[:,None] > c[None,:]).mean() + .5*(t[:,None] == c[None,:]).mean()) - 1
            low, high = np.quantile(difference, [.025, .975])
            rlow, rhigh = np.quantile(ranks, [.025, .975])
            rows.append(dict(version=version, panel=panel, left=lo, right=hi, n_conditioned=len(t), n_control=len(c), difference_of_median_changes=np.median(t)-np.median(c), median_difference_ci_low=low, median_difference_ci_high=high, rank_biserial=point, rank_biserial_ci_low=rlow, rank_biserial_ci_high=rhigh, bootstrap_draws=5000, seed=10, interval="pointwise percentile95; descriptive; no multiplicity correction"))
    return pd.DataFrame(rows)


def main():
    if OUT.resolve() == OLD.resolve():
        raise ValueError("The superseded historical review is preserved; use a new output directory")
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((OLD / "manifest.json").read_text())
    hashes = {Path(x["path"]).name: x["sha256"] for x in manifest["files"]}
    data = {"ratio": {}, "log": {}}
    inputs, trial_counts = [review.artifact(__file__), review.artifact(review.__file__), review.artifact(OLD / "manifest.json")], []
    inputs.extend(review.artifact(ROOT / path) for path in [
        "src/classical_conditioning/analysis/bout_vigor.py",
        "src/classical_conditioning/analysis/cohort_outcomes.py",
        "src/classical_conditioning/figures/cohort_response.py",
        "scripts/populate_figure2_available.py",
        "scripts/render_legacy_ssd_figure2_delay.py",
    ])
    for panel, loader, project, minimum in [
        ("D", load_delay, Path("J:/Digested Data/allDelay-full-v1"), 3),
        ("E", load_trace, Path("F:/Digested Data/all3sTrace-full-v1"), 1),
    ]:
        outcomes, cohort, cohort_hash, evidence = loader(project)
        assert outcomes.vigor_sample_policy.eq(VIGOR_SAMPLE_POLICY).all()
        inputs.extend(evidence)
        ratio, _ = summarize_selected_block_ratios(outcomes, metric_id=METRIC, selected_blocks=BLOCKS, min_trials_per_fish_block=minimum)
        data["ratio"][panel] = ratio
        cache = OLD / f"{panel}_log_trials.parquet"
        assert review.sha(str(cache)) == hashes[cache.name]
        inputs.append(review.artifact(cache))
        trials = pd.read_parquet(cache).drop(columns=["current_trial_eligible", "log_trial_eligible"])
        identity = ["recording_id", "fish_id", "condition_id"]
        assert set(map(tuple, trials[identity].drop_duplicates().astype(str).to_numpy())) == set(map(tuple, cohort[identity].astype(str).to_numpy()))
        selected = outcomes.loc[outcomes.metric_id.eq(METRIC) & outcomes.alignment.eq("CS")].copy()
        selected["current_trial_eligible"] = np.isfinite(selected.response_total_activity) & np.isfinite(selected.baseline_total_activity) & selected.baseline_total_activity.gt(0) & selected.response_total_activity.ge(0)
        trials = trials.merge(selected[["recording_id", "trial_number", "current_trial_eligible"]], on=["recording_id", "trial_number"], validate="one_to_one", how="left")
        assert trials.current_trial_eligible.notna().all()
        trials["log_trial_eligible"] = trials.current_trial_eligible & np.isfinite(trials.legacy_log_median_difference)
        trials["vigor_sample_policy"] = VIGOR_SAMPLE_POLICY
        trials.to_parquet(OUT / f"{panel}_log_trials.parquet", index=False)
        blocks = []
        for (fish, condition), group in trials.groupby(["fish_id", "condition_id"]):
            for order, (lo, hi) in enumerate(review.BLOCKS):
                sub = group.loc[group.trial_number.between(lo, hi)]
                valid = sub.loc[sub.log_trial_eligible]
                ok = len(valid) >= minimum
                blocks.append({"fish_id": fish, "condition_id": condition, "Selected block order": order,
                    "Selected block": review.LABELS[order], review.VALUE: float(valid.legacy_log_median_difference.median()) if ok else np.nan,
                    "Eligible": ok, "eligible_log_trials": len(valid), "eligible_current_trials": int(sub.current_trial_eligible.sum()), "minimum_trials": minimum})
        data["log"][panel] = pd.DataFrame(blocks)
        trial_counts.append({"panel": panel, "source_cohort_hash": cohort_hash, "fish": len(cohort), "selected_trials": len(trials), "ratio_eligible_trials": int(trials.current_trial_eligible.sum()), "log_eligible_trials": int(trials.log_trial_eligible.sum())})
        print(panel, trial_counts[-1], flush=True)
    all_tests, summaries, effects = [], [], []
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "axes.unicode_minus": False})
    for version, panels in data.items():
        tests, changes, diagnostics = review.block_statistics(panels, version)
        tests.to_csv(OUT / f"{version}_tests.csv", index=False)
        changes.to_csv(OUT / f"{version}_fish_changes.csv", index=False)
        diagnostics.to_csv(OUT / f"{version}_paired_diagnostics.csv", index=False)
        effects.append(change_effects(changes, version))
        all_tests.append(tests)
        for panel, fish in panels.items():
            fish["vigor_sample_policy"] = VIGOR_SAMPLE_POLICY
            fish.to_parquet(OUT / f"{panel}_{version}_fish_blocks.parquet", index=False)
            for (condition, block), group in fish.loc[fish.Eligible].groupby(["condition_id", "Selected block order"]):
                q = group[review.VALUE].quantile([.25, .5, .75]).to_numpy()
                summaries.append(dict(version=version, panel=panel, condition=condition, block=review.LABELS[block], n_fish=len(group), q25=q[0], median=q[1], q75=q[2]))
            review.render_panel(panel, fish, tests, version, OUT)
            review.render_panel(panel, fish, tests, version, OUT, True)
        review.render_changes(changes, tests, version, OUT)
    stats = pd.concat(all_tests, ignore_index=True)
    stats.to_csv(OUT / "all_statistics.csv", index=False)
    pd.concat(effects, ignore_index=True).to_csv(OUT / "direct_change_effects_bootstrap.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(trial_counts).to_csv(OUT / "trial_eligibility.csv", index=False)
    review.write(OUT / "inputs.json", inputs)
    review.write(OUT / "method.json", {"policy": VIGOR_SAMPLE_POLICY, "A": "Median across eligible trials of response mean bout vigor / baseline mean bout vigor; reference 1", "B": "Median across eligible trials of median(log positive bout vigor) response - baseline; reference 0", "no_bout": "NaN; excluded before all aggregation/scaling/logging; empty windows undefined", "statistics": "24 fish-level tests per version; paired Wilcoxon, Mann-Whitney at blocks and on paired fish changes; Holm24 separately per version; bootstrap effects use 5000 resamples, seed 10, pointwise 95% intervals", "windows": {"baseline": [-15, 0], "D": [0, 9], "E": [0, 13]}, "blocks": review.BLOCKS, "minimum_trials": {"D": 3, "E": 1}, "F": "No authenticated inputs; placeholder", "historical_preprocessing": "No historical rolling median/downsampling added", "historical_sources_preserved": True, "freeze": False})
    critique = """# Updated comparison: both outcomes are bout-only

The author instruction of 2026-10-09 supersedes the earlier overall-activity interpretation. No-bout frames are NaN in both active versions; they contribute neither zeros nor observations. An empty baseline or response makes the trial undefined. Exclusion means these plots describe intensity conditional on movement; neither measures movement frequency or total movement.

**A: mean bout-vigor ratio (reference 1).** Easy to interpret: 1.2 means response mean bout vigor is 20% above baseline within a trial. Means retain strong movements, but peaks and small baseline means can dominate a ratio. Fish/block medians reduce trial influence but do not remove within-trial peak sensitivity. Increases and decreases are asymmetric on the raw ratio scale.

**B: legacy-style log-median difference (reference 0).** Describes typical positive bout-frame vigor, reduces peak influence and represents proportional changes symmetrically. Natural-log units are less immediately intuitive (ln(2) means approximately a doubling of typical vigor). Zeros within detected bouts cannot enter a logarithm; therefore finite zero-valued bout samples are excluded by B, whereas A retains them. The outcome is robust but can miss changes driven specifically by extreme movements. Current corrected preprocessing is used; this is not an exact reconstruction of February/March smoothing/downsampling.

**Recommendation:** B as the main panel when the intended question is change in typical bout intensity; A as a sensitivity comparison. Choose A instead if the scientific target is average bout-frame intensity including vigorous peaks. Choose the outcome on that definition, not which p-value is smaller. Keep movement probability/bout frequency as separately named outcomes if needed.

Both versions require bouts in both windows. The remaining fish and trial counts must be reported: absence of movement is biologically meaningful even though it is excluded from vigor. Within-group significance alone is insufficient evidence of a conditioned-versus-control change; use the direct change tests. Holm correction covers 24 D/E tests separately per outcome; considering both outcomes is exploratory, not two independent confirmations. Whiskers on main panels are fish IQR, not confidence intervals. Bootstrap change intervals are pointwise, not simultaneous.
"""
    (OUT / "critique.md").write_text(critique, encoding="utf-8")
    grid = []
    for panel in ("D", "E"):
        for version, label in [("ratio", "A · Mean bout-vigor ratio"), ("log", "B · Legacy-style log-median difference")]:
            grid.append(f'<article><h2>{panel} · {label}</h2><img class="main" src="Fig2_{panel}_{version}.png"><img class="all" src="Fig2_{panel}_{version}_all_tests.png"><p><a href="Fig2_{panel}_{version}.svg">SVG</a> · <a href="{version}_tests.csv">Statistics</a></p></article>')
    page = '''<!doctype html><html><meta charset="utf-8"><title>Two bout-only versions</title><style>body{max-width:1400px;margin:25px auto;padding:20px;background:#f3f5f7;font:16px/1.5 system-ui;color:#20242a}.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}article,section{background:white;padding:20px;border-radius:10px;margin:15px 0}img{width:100%}.all{display:none}.show-all .all{display:block}.show-all .main{display:none}table{font-size:12px;border-collapse:collapse}td,th{padding:5px;border-bottom:1px solid #ddd}.scroll{overflow:auto}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style>
    <h1>Two versions · both ignore no-bout periods</h1><p><strong>Choose between A and B.</strong> D = Delay; E = 3 s Trace. These are two outcomes, shown for two panels. The checkbox changes statistical annotations only; it does not create another version.</p><section><h2>What does “no movement becomes missing” mean?</h2><p>No bout → vigor is NaN → skip that frame in every calculation. Example: bout values [2, 4] with a stationary period between them have mean 3. The stationary period contributes no zero and no observation. If a whole baseline or response has no bout, its trial ratio/log difference cannot be calculated and is excluded.</p><p>A compares mean bout vigor (unchanged = 1). B compares median natural-log bout vigor (unchanged = 0). Neither measures how often the fish moves. PT 10–14; ET 65–69; LT 90–94. Minimum eligible trials per fish/block: D 3; E 1.</p></section><label><input type="checkbox" onchange="document.body.classList.toggle('show-all',this.checked)"> Show all block comparisons, including ns</label><div class="grid">''' + "".join(grid) + '''</div><section><h2>Critique and recommendation</h2><p>A is intuitive but more sensitive to peaks and small baselines. B is robust and treats proportional increases/decreases symmetrically, but requires positive vigor and uses less intuitive units.</p><p><strong>Suggested primary: B for typical bout intensity; A as sensitivity.</strong> Choose A if average intensity including peaks is the intended outcome. Do not choose by significance. <a href="critique.md">Full critique</a> · <a href="summary.csv">Medians and fish counts</a> · <a href="trial_eligibility.csv">Excluded trial counts</a></p></section><section><h2>Conditioned-versus-control changes</h2><div class="grid"><img src="ratio_direct_changes.png"><img src="log_direct_changes.png"></div></section><section><h2>48 results: 24 per outcome, Holm24 separately</h2><p>Exploratory alternatives. Main whiskers = fish IQR. Change effect intervals = pointwise 95% bootstrap. <a href="all_statistics.csv">CSV</a></p><div class="scroll">''' + stats.to_html(index=False, float_format=lambda x:f"{x:.5g}", na_rep="—") + '''</div></section><section><h2>F · 10 s Trace</h2><p>Placeholder: no authenticated cohort inputs for this comparison.</p></section></html>'''
    page = page.replace("Change effect intervals = pointwise 95% bootstrap.", 'Change plots show fish median/IQR. <a href="direct_change_effects_bootstrap.csv">Effect estimates and pointwise 95% bootstrap intervals</a> are provided separately.')
    (OUT / "comparison.html").write_text(page, encoding="utf-8")
    review.write(OUT / "manifest.json", {"scope": "Both active outcomes obey bout-only policy; older reviews superseded, preserved", "files": [review.artifact(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name != "manifest.json"]})
    print(stats.loc[stats.p_holm24.lt(.05), ["version", "panel", "family", "condition", "left", "right", "p_holm24"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
