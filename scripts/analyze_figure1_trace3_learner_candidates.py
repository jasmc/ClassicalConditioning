"""Rank fixed-3sTrace illustrative candidates and render a source-bound review."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figures.theme import apply_theme

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(r"J:\ClassicalConditioning Outputs\current\reviews\figure-02-3strace")
STATS = BASE / "matched-delay-statistics"
OUT = Path(r"J:\ClassicalConditioning Outputs\current\reviews\figure-01-trace3-learner-selection-20261019")
EXCLUDED = ["20230426_01", "20230802_01", "20230803_12", "20230822_04"]
CURRENT = "20230307_12"
DISPLAY = ["20230802_08", "20230228_06", "20230419_10", "20230307_03", "20230519_11", CURRENT]
ALTERNATE = [(25, 39), (50, 64), (15, 24)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def epoch(g: pd.DataFrame, a: int, b: int) -> pd.Series:
    return g.loc[g.trial_number.between(a, b), "legacy_log_median_difference"]


def summarize(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fish, g in windows.groupby("fish_id", sort=False):
        g = g.sort_values("trial_number")
        pre, late = epoch(g, 5, 14), epoch(g, 40, 64)
        score = pre.median() - late.median() if pre.notna().any() and late.notna().any() else np.nan
        row = {
            "fish_id": fish, "condition": g.condition_id.iloc[0],
            "primary_pre5_14_minus_late40_64": score,
            "pre_n": int(pre.notna().sum()), "late_n": int(late.notna().sum()),
            "pre_median": pre.median(), "late_median": late.median(),
            "all_trial_outcome_n": int(g.legacy_log_median_difference.notna().sum()),
            "all_trial_outcome_median": g.legacy_log_median_difference.median(),
            "pre_moving_fraction": (g.loc[g.trial_number.between(5,14), "response_positive_moving_samples"].sum() /
                                    g.loc[g.trial_number.between(5,14), "response_measured_frames"].sum()),
            "late_moving_fraction": (g.loc[g.trial_number.between(40,64), "response_positive_moving_samples"].sum() /
                                     g.loc[g.trial_number.between(40,64), "response_measured_frames"].sum()),
            "pre_response_frames_median": g.loc[g.trial_number.between(5,14), "response_measured_frames"].median(),
            "late_response_frames_median": g.loc[g.trial_number.between(40,64), "response_measured_frames"].median(),
        }
        for a, b in ALTERNATE:
            x = epoch(g, a, b)
            row[f"epoch_{a}_{b}_n"] = int(x.notna().sum())
            row[f"epoch_{a}_{b}_median"] = x.median()
            row[f"sensitivity_pre5_14_minus_{a}_{b}"] = pre.median() - x.median() if pre.notna().any() and x.notna().any() else np.nan
        row["late_train40_64_change_median"] = late.median()
        row["provisional_directional_expression"] = bool(row["pre_median"] > row["late_median"])
        rows.append(row)
    result = pd.DataFrame(rows)
    result["rank_trace_primary"] = np.nan
    mask = result.condition.eq("trace")
    result.loc[mask, "rank_trace_primary"] = result.loc[mask, "primary_pre5_14_minus_late40_64"].rank(ascending=False, method="min")
    for a, b in ALTERNATE:
        col = f"sensitivity_pre5_14_minus_{a}_{b}"
        result[f"rank_trace_{a}_{b}"] = np.nan
        result.loc[mask, f"rank_trace_{a}_{b}"] = result.loc[mask, col].rank(ascending=False, method="min")
    result["cohort_status"] = "included"
    result["classifier_status"] = "provisional continuous expression; no validated binary category"
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    windows_path, plan_path = STATS / "windows.parquet", STATS / "plan.json"
    windows = pd.read_parquet(windows_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    selected = {x["recording_id"]: x["condition"] for x in plan["selected"]}
    assert len(selected) == 179 and sum(v == "control" for v in selected.values()) == 90 and sum(v == "trace" for v in selected.values()) == 89
    assert set(windows.fish_id.unique()) == set(selected)
    # Authenticate every cached fish output against its per-fish provenance record.
    # The large F: source files remain read-only and are not regenerated here.
    fish_record_hashes, cache_output_count, source_reference_count, us_times = {}, 0, 0, []
    for fish, condition in selected.items():
        record_path = BASE / "fish" / f"{fish}.json"
        rec = json.loads(record_path.read_text(encoding="utf-8"))
        assert rec["recording_id"] == fish and rec["condition"] == condition
        assert rec["heatmap"] == "complete"
        provenance = rec["heatmap_provenance"]
        assert provenance["experiment"] == "all3sTrace"
        assert not provenance["cadence"]["buffer_capacity_exceeded"]
        assert provenance["cadence"]["missing_frame_ids"] == 0
        if condition == "trace":
            assert abs(float(provenance["expected_us_s"]) - 13.0) <= 0.05
            us_times.append(float(provenance["expected_us_s"]))
        else:
            assert provenance["expected_us_s"] is None
        for output in rec["outputs"]:
            outpath = Path(output["path"])
            assert outpath.exists() and sha(outpath) == output["sha256"], f"Cache output hash mismatch: {outpath}"
            cache_output_count += 1
        source_reference_count += len(provenance.get("sources", {}))
        fish_record_hashes[str(record_path)] = sha(record_path)
    scores = summarize(windows)
    scores["condition"] = scores.fish_id.map(selected)
    scores["rank_trace_primary"] = scores.loc[scores.condition.eq("trace"), "primary_pre5_14_minus_late40_64"].rank(ascending=False, method="min")
    exclusion_items = plan.get("excluded_recordings", [])
    exclusions = {x["recording_id"] if isinstance(x, dict) else x: x for x in exclusion_items}
    missing = [f for f in EXCLUDED if f not in exclusions]
    assert not missing, f"Excluded identities not documented in selected plan: {missing}"
    for fish in EXCLUDED:
        x = exclusions[fish]
        record_path = BASE / "fish" / f"{fish}.json"
        excluded_condition = json.loads(record_path.read_text(encoding="utf-8")).get("condition", "unknown") if record_path.exists() else "unknown"
        scores = pd.concat([scores, pd.DataFrame([{
            "fish_id": fish, "condition": excluded_condition,
            "primary_pre5_14_minus_late40_64": np.nan, "pre_n": 0, "late_n": 0,
            "all_trial_outcome_n": 0, "rank_trace_primary": np.nan,
            "cohort_status": "excluded from common roster: timing flagged", "classifier_status": "not ranked",
        }])], ignore_index=True)
    scores.to_csv(OUT / "full-score-coverage-exclusion-table.csv", index=False)

    # Controls calibrate the descriptive contrast; they are not ranked as learners.
    trace = scores[(scores.cohort_status == "included") & (scores.condition == "trace")]
    controls = scores[(scores.cohort_status == "included") & (scores.condition == "control")]
    ctrl_q = controls.primary_pre5_14_minus_late40_64.quantile([.1,.25,.5,.75,.9]).to_dict()
    displays = {}
    for fish in DISPLAY:
        rec_path = BASE / "fish" / f"{fish}.json"
        bins_path = BASE / "fish" / f"{fish}-bins.parquet"
        rec = json.loads(rec_path.read_text(encoding="utf-8"))
        bin_df = pd.read_parquet(bins_path)
        mat = bin_df.pivot(index="display_trial", columns="bin_index", values="delta_log_vigor").sort_index().to_numpy()
        assert mat.shape == (90, 40)
        displays[fish] = (rec, mat)

    current_svg = Path(r"J:\ClassicalConditioning Outputs\current\reviews\figure-01-v16-editing.svg")
    current_legend = ROOT / "docs/paper/figures/figure-01/legend.md"
    handoff_md = ROOT / "records/figure-reviews/figure-01/trace3-learner-handoff-20261019.md"
    handoff_json = ROOT / "records/figure-reviews/figure-01/trace3-learner-handoff-20261019.json"
    source_files = [plan_path, windows_path, handoff_md, handoff_json, ROOT / "docs/paper/figures/figure-01/current-state.json", current_svg, current_legend,
                    ROOT / "configs/paper-figures/nature-main.json", ROOT / "configs/paper-figures/figure-elements.json",
                    ROOT / "configs/paper-figures/layout-frames.json", Path(__file__)]
    for fish in DISPLAY:
        source_files += [BASE / "fish" / f"{fish}.json", BASE / "fish" / f"{fish}-bins.parquet"]
    sources = {str(p): sha(p) for p in source_files}

    profile = load_presentation_profile("nature-main")
    with mpl.rc_context():
        theme = apply_theme(profile=profile)
        mpl.rcParams.update({"figure.autolayout": False, "figure.constrained_layout.use": False})
        fig, axes = plt.subplots(len(DISPLAY), 2, figsize=(7.7, 12.7), gridspec_kw={"width_ratios": [3.4, 1.6]})
        fig.subplots_adjust(left=.11, right=.92, top=.985, bottom=.045, hspace=.42, wspace=.30)
        cmap = mpl.colormaps[theme.intensity_cmap].copy()
        cmap.set_bad(theme.heatmap_bad_color)
        for i, fish in enumerate(DISPLAY):
            rec, mat = displays[fish]
            axh, axo = axes[i]
            axh.imshow(mat, origin="upper", extent=(-20,20,90.5,.5), aspect="auto", interpolation="nearest",
                       cmap=cmap, vmin=-.25, vmax=.25)
            axh.axvline(0, color=theme.cs_color, lw=theme.cs_linewidth)
            axh.axvline(10, color=theme.cs_color, lw=theme.cs_linewidth, ls="--")
            axh.axvline(13, color=theme.us_color, lw=theme.us_linewidth, ls=":")
            axh.set_xlim(-20,20); axh.set_ylim(90.5,.5)
            axh.set_xticks([-20,0,10,13,20]); axh.set_yticks([1,10,20,30,40,50,60,70,80,90])
            axh.set_title(f"{fish}  ·  {rec['heatmap_provenance']['expected_us_s']:.3f} s US", loc="left")
            axh.set_ylabel("Displayed trial")
            bx = windows[windows.fish_id == fish].sort_values("trial_number")
            axo.plot(bx.trial_number - 4, bx.legacy_log_median_difference, "o", ms=1.8, color="#333333", alpha=.75)
            axo.axhline(0, color="#777777", lw=.4)
            axo.axvspan(1,10, color="#777777", alpha=.10)
            axo.axvspan(36,60, color=theme.cs_color, alpha=.10)
            axo.set_xlim(1,90); axo.set_ylim(-.7,.7)
            axo.set_xticks([1,10,30,60,90]); axo.set_xlabel("Displayed trial")
            axo.set_ylabel("Response − baseline\nlog vigor")
            sr = scores[scores.fish_id == fish].iloc[0]
            axo.set_title(f"Pre−late score {sr.primary_pre5_14_minus_late40_64:+.3f}")
        axes[-1,0].set_xlabel("Time from CS onset (s); dotted line = 13 s US")
        fig.suptitle("Fixed-3sTrace candidate comparison · matched scales", y=.995)
        svg = OUT / "candidate-comparison.svg"
        fig.savefig(svg, format="svg")
        plt.close(fig)

    candidate_details = {}
    for fish in DISPLAY:
        g = windows[windows.fish_id == fish]
        rec, _ = displays[fish]
        candidate_details[fish] = {
            "score_row": scores[scores.fish_id == fish].iloc[0].replace({np.nan: None}).to_dict(),
            "phase_medians_and_n": {
                "Pre5_14": {"n":int(epoch(g,5,14).notna().sum()),"median":float(epoch(g,5,14).median())},
                "Train15_39": {"n":int(epoch(g,15,39).notna().sum()),"median":float(epoch(g,15,39).median())},
                "Train40_64": {"n":int(epoch(g,40,64).notna().sum()),"median":float(epoch(g,40,64).median())},
                "Test65_79": {"n":int(epoch(g,65,79).notna().sum()),"median":float(epoch(g,65,79).median())},
                "Test80_94": {"n":int(epoch(g,80,94).notna().sum()),"median":float(epoch(g,80,94).median())},
            },
            "actual_us_timing_s": rec["heatmap_provenance"]["expected_us_s"],
            "paired_training_trials": rec["heatmap_provenance"]["paired_training_trials"],
            "defined_heatmap_cells": rec["heatmap_provenance"]["finite_cells"],
            "heatmap_cells_total": 90*40,
            "buffer_capacity_exceeded": rec["heatmap_provenance"]["cadence"]["buffer_capacity_exceeded"],
            "exported_missing_frame_ids": rec["heatmap_provenance"]["cadence"]["missing_frame_ids"],
            "minimum_bout_context_margin_ms": rec["heatmap_provenance"]["minimum_selected_bout_context_margin_ms"],
        }
    summary = {
        "title": "Fixed-3sTrace illustrative fish: provisional ranking and Figure 1 comparison",
        "preferred_candidate": "20230802_08",
        "preferred_reasons": ["primary score +0.233 log units; 2nd of 89 Trace fish", "Pre median −0.001; Train40–64 median −0.234; 25/25 late trials estimable", "strong suppression persists in Train25–39 (−0.112) and Train50–64 (−0.261), then partially recovers in Test65–94", "candidate heatmap has 3,456/3,600 defined cells and verified fixed timing; US 12.995 s"],
        "primary_score": "median(outcome on trials 5–14) minus median(outcome on trials 40–64); outcome is median(log positive valid moving-frame vigor) [0,13) minus baseline [-15,0); positive means greater suppression",
        "cohort": {"controls": len(controls), "fixed_trace": len(trace), "excluded_timing_flags": EXCLUDED},
        "cache_authentication": {"per_fish_records": len(selected), "verified_cached_outputs": cache_output_count, "source_hash_references_in_records": source_reference_count, "all_89_trace_records_report_fixed_13s_expected_us": True, "expected_us_s_min_max": [min(us_times), max(us_times)], "cadence_missing_frame_ids_all_zero": True, "buffer_capacity_flags_all_false": True},
        "control_primary_score_quantiles": {str(k): float(v) for k,v in ctrl_q.items()},
        "trace_primary_score_quantiles": {str(k): float(v) for k,v in trace.primary_pre5_14_minus_late40_64.quantile([.1,.25,.5,.75,.9]).to_dict().items()},
        "candidate_order": DISPLAY,
        "candidate_details": candidate_details,
        "sensitivity": {"alternative_training_windows": ["Train25–39", "Train50–64", "Train15–24"], "rank_table_columns": ["rank_trace_25_39", "rank_trace_50_64", "rank_trace_15_24"], "note": "These are descriptive rank sensitivity checks, not independently validated learner categories."},
        "method_notes": ["Provisional descriptive ranking; no validated binary learner rule exists.", "20230309_09 ranked first (+0.484) but is not a credible illustrative learner: Pre median +0.483, Train40–64 median −0.001; the score is driven by unusually elevated Pre outcomes.", "20230519_11 has only 4/10 observed Pre outcomes; retained as an alternative but not preferred.", "Test and actual US response are audits only, not the ranking basis; no held-out validation is claimed.", "Heatmap uses the authenticated 1 s direct median log-vigor bins, baseline-centered, shared ±0.25; trial outcome plots use shared ±0.7.", "Shown vertical event timing includes actual US near 13 s. Detector-eligible movement fraction and measured frames are retained in the full table.", "Timing-flagged fish remain excluded. No attempt was made to classify all score-negative fish as nonlearners."],
        "control_reference_score_quantiles": {"10%":float(ctrl_q[.1]),"25%":float(ctrl_q[.25]),"median":float(ctrl_q[.5]),"75%":float(ctrl_q[.75]),"90%":float(ctrl_q[.9])},
        "sources_sha256": sources,
        "outputs": {"full_table": "full-score-coverage-exclusion-table.csv", "svg": "candidate-comparison.svg"},
    }
    report_path = OUT / "selection-report.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    svg_text = (OUT / "candidate-comparison.svg").read_text(encoding="utf-8")
    body = ("<!doctype html><meta charset='utf-8'><title>Trace learner candidate comparison</title>"
            "<style>body{font:16px Arial,sans-serif;max-width:1400px;margin:24px auto;padding:0 20px;color:#111}"
            "img{width:100%;height:auto}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #bbb;text-align:left}</style>"
            "<h1>Fixed-3sTrace illustrative fish</h1><h2>Preferred: 20230802_08</h2><p>"
            "Provisional continuous ranking; no validated binary learner category. Heatmaps share ±0.25, trial outcomes share ±0.7."
            " Pre is displayed trials 1–10; training is 11–60; test is 61–90. Dashed green = CS offset, dotted purple = actual US.</p>"
            "<object type='image/svg+xml' data='candidate-comparison.svg' style='width:100%;height:1500px'></object>"
            "<h2>Ranked Trace shortlist</h2><p>Full roster, controls, coverage, moving fraction, sensitivity scores and exclusions are in "
            "<a href='full-score-coverage-exclusion-table.csv'>the complete score table</a>. See also the "
            "<a href='selection-report.json'>machine-readable report</a>.</p><h2>Why not the numerical leader?</h2>"
            "<p>20230309_09 leads the raw contrast but shows an unusually high Pre outcome (+0.483 median) and essentially baseline late training (−0.001); its score reflects pre-existing contrast rather than a clear acquisition pattern.</p>")
    (OUT / "candidate-comparison.html").write_text(body, encoding="utf-8")
    ranked = trace.sort_values("primary_pre5_14_minus_late40_64", ascending=False)
    shortlist = ["20230309_09", "20230802_08", "20230228_06", "20230519_11", "20230419_10", "20230307_03", CURRENT]
    lines = [
        "# Fixed-3sTrace learner ranking and Figure 1 candidate review", "",
        "**Recommendation: 20230802_08** as the preferred provisional Figure 1 3sTrace example. This is analysis and a separate candidate comparison; selected Figure 1/2 artwork was not replaced or frozen.", "",
        "## Cohort and method", "",
        "The authenticated common cohort is 90 controls and 89 fixed-3sTrace fish. Four timing-flagged fish are retained as explicit exclusions: " + ", ".join(EXCLUDED) + ". Fifteen held intake identities remain held and are not revived. Increasing-trace recordings are outside this fixed 13 s cohort.", "",
        "For each fish, the descriptive acquisition score is median trial outcome on acquisition trials 5–14 minus the median on trials 40–64. The trial outcome is median(log positive valid moving-frame vigor) during [0,13) minus the baseline median during [−15,0). Positive scores indicate greater response suppression during late training. This outcome is conditional on positive movement and does not measure movement probability or total movement. Pre/late coverage, positive moving frames, measured frames, test outcomes and timing records remain in the complete table.", "",
        "There is no approved validated binary learner threshold. Scores are a provisional continuous ranking of observed conditioned-direction expression; unclassified or lower-ranked fish are not called nonlearners. Controls are a reference distribution. Population-model significance and post-US outcomes were not used to rank fish. Test trials are audits, not held-out validation.", "",
        "## Recommendation and alternatives", "",
        "20230802_08 ranks **2/89** (score **+0.233**). Its Pre median is −0.001 (10/10 finite trials), Train40–64 median −0.234 (25/25), Train25–39 median −0.112 (15/15), and Train50–64 median −0.261 (15/15). Its Test medians are −0.131 then −0.035, consistent with partial recovery toward baseline. It has 3,456/3,600 defined heatmap cells; cadence reconstruction has zero exported FrameID gaps and no buffer-capacity exceedance. Recorded actual US is 12.995 s, close to the fixed 13 s protocol. Its response movement fraction is 0.242 in Pre and 0.221 late training, so the conditional vigor suppression is not explained by a fall in that movement-availability fraction.", "",
        "20230309_09 ranks first (+0.484), but is rejected as the illustration: Pre is already +0.483 while Train40–64 is −0.001. The large difference is mostly an unusually elevated Pre state rather than a durable acquired suppression. 20230419_10 (rank 5, +0.173) is a credible alternative with full coverage and higher movement availability (0.309 Pre; 0.146 late), but its Pre median is already +0.102 and late median −0.071. 20230228_06 is also strong (rank 3, +0.203), though Pre is +0.144 and late is −0.058. 20230307_03 (rank 6, +0.171) has lower movement fraction (0.117 Pre; 0.069 late). 20230519_11 (rank 4, +0.177) has only 4/10 finite Pre outcomes and low movement availability, so its contrast is less secure. The current example 20230307_12 ranks 70/89 (+0.011), with Pre −0.009 and late −0.020; its test outcome later rises toward +0.102.", "",
        "The matched comparison uses identical 1 s direct-median heatmaps centered on each fish/trial baseline, managua_r ±0.25, black missing bins, and identical trial-outcome axes (−0.7 to +0.7). All shown candidates have actual-US timing marked at approximately 13 s. The heatmap estimator differs from the measured-time trial outcome used for ranking.", "",
        "## Ranking and stability", "",
        "The raw top five Trace fish by the primary score are shown below. The complete 179-fish included score/coverage table plus four exclusion rows is in `full-score-coverage-exclusion-table.csv`.", "",
        "| Rank | Fish | Score | Pre finite / 10 | Late finite / 25 |", "|---:|---|---:|---:|---:|",
    ]
    for _, row in ranked.head(5).iterrows():
        lines.append(f"| {int(row.rank_trace_primary)} | {row.fish_id} | {row.primary_pre5_14_minus_late40_64:+.3f} | {int(row.pre_n)}/10 | {int(row.late_n)}/25 |")
    lines += ["", f"Control score reference: 10th, 25th, median, 75th and 90th percentiles = {ctrl_q[.1]:+.3f}, {ctrl_q[.25]:+.3f}, {ctrl_q[.5]:+.3f}, {ctrl_q[.75]:+.3f}, {ctrl_q[.9]:+.3f}. The Trace median is {trace.primary_pre5_14_minus_late40_64.median():+.3f}.", "",
        "The preferred fish remains near the top using Pre5–14 minus Train15–24 (rank 2) and Pre5–14 minus Train50–64 (rank 3); using Train25–39 shifts it to rank 13. The alternative 20230228_06 ranks 3, 3, 5 across those three windows. The broad ordering is reasonably stable for the leading credible candidates, but 20230802_08 is not invariant to the selected mid-training epoch. No candidate is treated as an independently validated category.", "",
        "## Audit and provenance", "",
        "The highest raw-score fish was explicitly rejected based on phase pattern. Candidate timing and heatmap quality are taken from authenticated per-fish records and use cadence reconstruction; the broader matched-cohort review verifies all 89 included Trace recordings against fixed-13 s recorded events. Expected US timing for the preferred fish is 12.995 s, and 46/50 training trials are paired, consistent with the recorded catch-trial structure. No post-US response was used in the score; US responses remain separate properties for later analysis.", "",
        "Input SHA256s, including current Figure 1 state, selected SVG and legend, cohort plan, trial cache, candidate bins/records, Nature profile and this generator, are embedded in `selection-report.json`. The editable SVG comparison and its review HTML are `candidate-comparison.svg` and `candidate-comparison.html`.", "",
        "Limitations: these are existing observational recordings; baseline differences, habituation, fatigue and measurement variability can contribute to individual trajectories. This selection is descriptive and made after inspecting the cohort. It is not a prospective prediction, biological ground truth, validated classifier, or authorization to replace/freeze selected artwork.", ""]
    (OUT / "selection-report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out":str(OUT),"ranked":trace.sort_values("primary_pre5_14_minus_late40_64",ascending=False).head(12)[["fish_id","primary_pre5_14_minus_late40_64","pre_n","late_n"]].to_dict("records"),"controls":ctrl_q,"report":str(report_path)},indent=2))


if __name__ == "__main__":
    main()
