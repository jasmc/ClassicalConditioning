"""Finish exploratory 3sTrace Figures 2–4 for all 59 complete fish.

Reuses the corrected frozen trial outcomes; only the metric-specific
assessment, classifier comparison, and downstream figures are computed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from classical_conditioning.analysis.discarding import assess_discarding
from classical_conditioning.cohort import load_cohort_manifest


PROJECT = Path(r"F:\Digested Data\all3sTrace-full-v1")
RAW = Path(r"J:\Raw Data\all3sTtrace")
COHORT = "all3sTrace-full-exploratory"
METRIC = "legacy_distal_angular_speed"
FIG2 = "figure2-3strace-window13-legacy-59fish"
FIG3 = "figure3-3strace-window13-legacy-59fish"
FIG4 = "figure4-3strace-window13-legacy-59fish"
REPO = Path(__file__).resolve().parents[1]
STATE = REPO / "outputs" / "trace-legacy-full-59" / "status.json"


def status(stage: str, detail: str = "") -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"stage": stage, "detail": detail, "cohort": COHORT,
                                 "metric": METRIC, "fish": 59}, indent=2), encoding="utf-8")
    print(stage, detail, flush=True)


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=REPO, check=True)


def main() -> None:
    try:
        cohort = load_cohort_manifest(PROJECT, COHORT)
        selected = cohort.loc[cohort["primary_included"].astype(bool)]
        ids = selected["recording_id"].astype(str).tolist()
        if (len(ids) != 59 or len(set(ids)) != 59 or
                selected["condition_id"].value_counts().to_dict() != {"trace": 40, "control": 19}):
            raise ValueError("Full 3sTrace cohort identity is not 40 trace + 19 control")
        inventory = json.loads((PROJECT / "Metadata" / "recording_inventory.json").read_text(encoding="utf-8"))
        if not inventory.get("source_hashes_included"):
            raise ValueError("Saved raw inventory lacks hashes")
        assessment = PROJECT / "Processed data" / "Discarding" / "all3sTrace-window13-legacy-59fish" / "assessment-summary.json"
        if not assessment.exists():
            status("assessment", "Checking legacy metric in the full cohort")
            assess_discarding(RAW, PROJECT, analysis_id="all3sTrace-window13-legacy-59fish",
                             experiment="all3sTrace", metric_id=METRIC,
                             metric_recipe="tail-candidate-corrected", recording_ids=ids,
                             inventory=inventory)
        cohort_dir = PROJECT / "Processed data" / "Cohorts" / COHORT
        comparison = PROJECT / "Processed data" / "Analyses" / FIG3 / "legacy-distal"
        if not (comparison / "comparison.json").exists():
            status("comparison", "Archived learner variants, full 59 fish")
            run("-m", "classical_conditioning", "compare-legacy-learners",
                "--cohort", str(cohort_dir / "cohort-manifest.parquet"),
                "--trial-outcomes", str(cohort_dir / "cohort-trial-outcomes.parquet"),
                "--output-dir", str(comparison), "--metric", METRIC)
        output_root = PROJECT / "Figures" / "PNG" / "Analyses"
        if not (output_root / FIG3 / f"figure-3-3strace-review_{METRIC}.png").exists():
            status("figure3")
            run(str(REPO / "scripts" / "render_figure3_3strace_review.py"),
                "--project-dir", str(PROJECT), "--cohort-id", COHORT,
                "--comparison-dir", str(comparison), "--metric", METRIC,
                "--output-dir", str(output_root / FIG3))
        labels = PROJECT / "Processed data" / "Analyses" / FIG4 / "learner-labels.csv"
        if not labels.exists():
            status("labels")
            run(str(REPO / "scripts" / "finalize_3strace_exploratory.py"), "classify",
                "--project-dir", str(PROJECT), "--cohort-id", COHORT,
                "--comparison-dir", str(comparison), "--analysis-id", FIG4,
                "--assessment-summary", str(assessment), "--metric", METRIC,
                "--classifier-execution-id", "legacy-wip-3strace-distal-window13-59fish")
        analysis = PROJECT / "Processed data" / "Analyses" / FIG4 / "figure4" / "analysis.json"
        if not analysis.exists():
            status("figure4_analysis")
            run("-m", "classical_conditioning", "figure4-analyze", "--project-dir", str(PROJECT),
                "--analysis-id", FIG4, "--metric", METRIC, "--trace3-cohort-id", COHORT,
                "--learner-manifest", str(labels))
        if not (output_root / FIG4 / "all3sTrace" / f"figure-4_{METRIC}.png").exists():
            status("figure4_render")
            run("-m", "classical_conditioning", "figure4-render", "--analysis-summary", str(analysis),
                "--output-dir", str(output_root / FIG4), "--overwrite")
        status("figure2")
        run(str(REPO / "scripts" / "render_figure2_3strace_legacy_layout.py"),
            "--project-dir", str(PROJECT), "--cohort-id", COHORT,
            "--output-dir", str(output_root / FIG2))
        status("complete", "Legacy-metric Figures 2E/H, 3 and 4 use all 59 complete fish")
    except Exception as exc:
        status("failed", repr(exc))
        raise


if __name__ == "__main__":
    main()
