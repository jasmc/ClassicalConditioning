"""Run the supported manuscript-panel renderers with one recorded recipe.

The paper panel registry remains the authority for readiness. This command
produces review artifacts; it does not turn blocked panels into approved ones.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.example_traces import METRIC_COLUMNS


REPO = Path(__file__).resolve().parents[3]
REGISTRY = REPO / "configs/paper-figures/behavior-paper.json"
DEFAULT_TRIALS = (9, 17, 63, 66, 93)
DEFAULT_FISH = ("20221115_07", "20221115_09")


@dataclass(frozen=True)
class RenderStep:
    name: str
    panels: tuple[str, ...]
    argv: tuple[str, ...]


def build_render_plan(
    project_dir: Path,
    output_dir: Path,
    *,
    figure_set: str = "available",
    metric_id: str = "tail_length_weighted_angular_l1",
    delay_fish: str = DEFAULT_FISH[0],
    control_fish: str = DEFAULT_FISH[1],
    trials: tuple[int, ...] = DEFAULT_TRIALS,
    mode: str = "static",
    inference_review: bool = False,
    overwrite: bool = False,
    figure4_cohort_ids: dict[str, str] | None = None,
    figure4_manifest: Path | None = None,
    figure4_project_dirs: dict[str, Path] | None = None,
) -> list[RenderStep]:
    """Declare commands before modifying data or producing any image."""
    if figure_set not in {"available", "figure1", "figure2-delay", "figure4"}:
        raise ValueError(f"Unknown figure set: {figure_set}")
    if metric_id not in METRIC_COLUMNS:
        raise ValueError(f"Unknown vigor metric: {metric_id}")
    if mode not in {"static", "publication"}:
        raise ValueError(f"Unknown figure mode: {mode}")
    if not trials or len(trials) != len(set(trials)) or min(trials) < 1:
        raise ValueError("Trials must be distinct positive global CS trial numbers")
    if delay_fish == control_fish:
        raise ValueError("Delay and control examples must be different recordings")
    if inference_review and figure_set == "figure1":
        raise ValueError("Inference review is only available for Figure 2 Delay")
    if inference_review and metric_id != "tail_length_weighted_angular_l1":
        raise ValueError("The saved Figure 2G LME only supports tail_length_weighted_angular_l1")
    project_dir = project_dir.resolve()
    output_dir = output_dir.resolve()
    common = ("--project-dir", str(project_dir), "--mode", mode)
    force = ("--overwrite",) if overwrite else ()
    steps: list[RenderStep] = []
    if figure_set in {"available", "figure1"}:
        for fish in (delay_fish, control_fish):
            steps.append(RenderStep(
                f"figure1-cd-{fish}", ("fig-1C", "fig-1D"),
                ("scripts/render_legacy_ssd_example_traces.py", *common,
                 "--output-dir", str(output_dir / "figure1/traces"),
                 "--recording-id", fish, "--metric", metric_id,
                 *(item for trial in trials for item in ("--trial", str(trial))),
                 "--window-start", "-20", "--window-end", "20", *force),
            ))
        if (delay_fish, control_fish) != DEFAULT_FISH:
            raise ValueError(
                "The signed Figure 1 heatmap renderer currently supports only "
                "20221115_07 Delay and 20221115_09 control; choose those fish "
                "or run the paired C/D traces directly."
            )
        steps.append(RenderStep(
            "figure1-signed-heatmap", ("fig-1E",),
            ("scripts/render_legacy_ssd_example_heatmaps.py", *common,
             "--output-dir", str(output_dir / "figure1/heatmaps"),
             "--baseline-end-s", "0", "--metric", metric_id, *force),
        ))
    if figure_set in {"available", "figure2-delay"}:
        figure2_dir = output_dir / "figure2/delay"
        steps.append(RenderStep(
            "figure2-delay-descriptive", ("fig-2A", "fig-2D", "fig-2G", "sup-5"),
            ("scripts/render_legacy_ssd_figure2_delay.py", *common,
             "--output-dir", str(figure2_dir), "--metric", metric_id, *force),
        ))
        if inference_review:
            steps.append(RenderStep(
                "figure2-delay-inference-review", ("fig-2D", "fig-2G"),
                ("scripts/render_figure2_legacy_stats_lme_review.py",
                 "--project-dir", str(project_dir), "--input-dir", str(figure2_dir),
                 "--output-dir", str(figure2_dir / "inference-review"),
                 "--metric", metric_id, *force),
            ))
    if figure_set == "figure4":
        experiments = ("allDelay", "all3sTrace", "all10sTrace")
        if not figure4_cohort_ids or set(figure4_cohort_ids) != set(experiments) or figure4_manifest is None:
            raise ValueError("Figure 4 requires three cohort IDs and a frozen learner manifest.")
        analysis_id = output_dir.name
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", analysis_id):
            raise ValueError("Figure 4 output directory name must be a safe analysis ID.")
        source_args = tuple(
            item for experiment, flag in (("allDelay", "delay"), ("all3sTrace", "trace3"), ("all10sTrace", "trace10"))
            for item in (f"--{flag}-cohort-id", figure4_cohort_ids[experiment])
        )
        directory_args = tuple(
            item for experiment, flag in (("allDelay", "delay"), ("all3sTrace", "trace3"), ("all10sTrace", "trace10"))
            if figure4_project_dirs and experiment in figure4_project_dirs
            for item in (f"--{flag}-project-dir", str(figure4_project_dirs[experiment]))
        )
        summary = project_dir / "Processed data" / "Analyses" / analysis_id / "figure4" / "analysis.json"
        steps.extend((
            RenderStep("figure4-analysis", ("fig-4A", "fig-4B", "fig-4C"),
                       ("-m", "classical_conditioning", "figure4-analyze",
                        "--project-dir", str(project_dir), "--analysis-id", analysis_id,
                        "--metric", metric_id, "--learner-manifest", str(figure4_manifest),
                        *source_args, *directory_args, *force)),
            RenderStep("figure4-render", ("fig-4A", "fig-4B", "fig-4C"),
                       ("-m", "classical_conditioning", "figure4-render",
                        "--analysis-summary", str(summary), "--output-dir", str(output_dir / "figure4"),
                        "--mode", mode, *force)),
        ))
    return steps


def run_paper_panels(
    project_dir: Path,
    output_dir: Path,
    *,
    figure_set: str = "available",
    metric_id: str = "tail_length_weighted_angular_l1",
    delay_fish: str = DEFAULT_FISH[0],
    control_fish: str = DEFAULT_FISH[1],
    trials: tuple[int, ...] = DEFAULT_TRIALS,
    mode: str = "static",
    inference_review: bool = False,
    overwrite: bool = False,
    plan_only: bool = False,
    figure4_cohort_ids: dict[str, str] | None = None,
    figure4_manifest: Path | None = None,
    figure4_project_dirs: dict[str, Path] | None = None,
) -> dict:
    """Render selected families and write a manifest with each step's outcome."""
    steps = build_render_plan(
        project_dir, output_dir, figure_set=figure_set, metric_id=metric_id,
        delay_fish=delay_fish, control_fish=control_fish, trials=trials,
        mode=mode, inference_review=inference_review, overwrite=overwrite,
        figure4_cohort_ids=figure4_cohort_ids, figure4_manifest=figure4_manifest,
        figure4_project_dirs=figure4_project_dirs,
    )
    registry = json.loads(REGISTRY.read_text())
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_id": registry["paper_id"],
        "registry_path": str(REGISTRY),
        "registry_sha256": sha256_file(REGISTRY),
        "project_dir": str(project_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "metric_id": metric_id,
        "example_fish": {"delay": delay_fish, "control": control_fish},
        "global_cs_trials": list(trials),
        "mode": mode,
        "scientific_status": "review_only_not_approved",
        "steps": [{"name": step.name, "panels": list(step.panels),
                   "command": [sys.executable, *step.argv], "status": "planned"}
                  for step in steps],
        "blocked_registry_panels": {
            panel: item["reason"] for panel, item in registry["panels"].items()
            if item["status"] == "blocked"
        },
    }
    if plan_only:
        return report
    if not project_dir.is_dir():
        raise FileNotFoundError(project_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "paper-panel-run.json"
    if manifest.exists() and not overwrite:
        raise FileExistsError(manifest)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(REPO / "src"), env.get("PYTHONPATH", ""))))
    for index, step in enumerate(steps):
        report["steps"][index]["status"] = "running"
        manifest.write_text(json.dumps(report, indent=2) + "\n")
        try:
            subprocess.run([sys.executable, *step.argv], cwd=REPO, env=env, check=True)
        except subprocess.CalledProcessError as error:
            report["steps"][index]["status"] = "failed"
            report["steps"][index]["exit_code"] = error.returncode
            manifest.write_text(json.dumps(report, indent=2) + "\n")
            raise
        report["steps"][index]["status"] = "rendered_review"
        manifest.write_text(json.dumps(report, indent=2) + "\n")
    report["artifacts"] = sorted(
        str(path.relative_to(output_dir)) for path in output_dir.rglob("*.figure.json")
    )
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    return report
