"""Isolated executor for one unmodified historical learner algorithm.

This module is launched in a fresh process by legacy_learners. Historical
module globals and plotting defaults therefore cannot leak into other runs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd


SCRIPTS = {
    "legacy-nominal": "6_LearnersQuantification.py",
    "legacy-new": "6_LearnersQuantification_new.py",
    "legacy-improved": "6_LearnersQuantification_improved.py",
    "legacy-wip": "6_LearnersQuantification_WIP.py",
}


def execute(
    variant: str, input_path: Path, output_path: Path,
    *, figures: bool = False, individuals: bool = False,
) -> None:
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "legacy" / "helpers"))
    script = root / "legacy" / "scripts" / SCRIPTS[variant]
    spec = importlib.util.spec_from_file_location("isolated_legacy_learner", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    frame = pd.read_parquet(input_path)
    # The historical entry point creates folders, discovers input files, and
    # writes optional figures. Replace only those side effects in this process.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    module.file_utils.create_folders = lambda _path: (output_path.parent,) * 18
    if variant in {"legacy-nominal", "legacy-new"}:
        module.load_data = lambda _config, _path: frame.copy()
    else:
        module.load_pooled_data = lambda _config, _path: frame.copy()
    for flag in (
        "RUN_DIAGNOSTICS", "RUN_PLOT_DIAGNOSTICS", "RUN_PLOT_TRAJECTORIES",
        "RUN_PLOT_FEATURE_SPACE", "RUN_PLOT_BLUP_CATERPILLAR",
        "RUN_PLOT_INDIVIDUALS", "RUN_PLOT_INDIVIDUALS_AND_GRID",
        "RUN_PLOT_BLUP_OVERLAY", "RUN_PLOT_HEATMAP_GRID",
        "RUN_EXPORT_DETAILED_SUMMARY", "RUN_EXPORT_RESULTS",
    ):
        if hasattr(module, flag):
            setattr(module, flag, False)
    if figures:
        for flag in ("RUN_PLOT_TRAJECTORIES", "RUN_PLOT_FEATURE_SPACE",
                     "RUN_PLOT_BLUP_OVERLAY", "RUN_PLOT_BLUP_CATERPILLAR"):
            if hasattr(module, flag):
                setattr(module, flag, True)
    if individuals:
        for flag in ("RUN_PLOT_INDIVIDUALS", "RUN_PLOT_INDIVIDUALS_AND_GRID"):
            if hasattr(module, flag):
                setattr(module, flag, True)

    result = module.run_multivariate_lme_pipeline(module.analysis_cfg)
    if isinstance(result, tuple):
        classification, columns = result
        if figures:
            # Nominal/new define legacy plotting functions but have commented
            # out save calls in some of them. Persist their returned Figures.
            import matplotlib.pyplot as plt

            trajectory_path = output_path.parent / "Learner_Classification_Trajectories.png"
            feature_path = output_path.parent / "Feature_Space_Scatter.png"
            if not trajectory_path.exists():
                prepared = module.prepare_data(frame.copy(), module.analysis_cfg)
                fig = module.plot_behavioral_trajectories(prepared, classification, module.analysis_cfg)
                fig.savefig(trajectory_path, dpi=200, bbox_inches="tight")
                plt.close(fig)
            if not feature_path.exists():
                fig = module.plot_feature_space(classification, module.analysis_cfg)
                if fig is not None:
                    fig.savefig(feature_path, dpi=200, bbox_inches="tight")
                    plt.close(fig)
        out = pd.DataFrame(columns)
        out["decision_threshold"] = float(classification.threshold)
        out["decision_threshold_ci_low"] = float(classification.threshold_ci[0])
        out["decision_threshold_ci_high"] = float(classification.threshold_ci[1])
        out["learner_point"] = classification.is_learner_point
        if variant == "legacy-nominal":
            out["learner_primary"] = classification.is_learner_conservative
        else:
            out["learner_primary"] = classification.is_learner_probabilistic
    else:
        out = result.copy()
        # The original improved/WIP result is the primary rule; the returned
        # table already contains all feature evidence, T, votes, and p-values.
        out["learner_primary"] = out["Is_Learner"]
        # The threshold is absent from these legacy export frames. Recompute
        # exactly the empirical control quantile from their recorded T scores.
        control = out.loc[out["Condition"] == module.analysis_cfg.cond_types[0], "T_joint"]
        out["decision_threshold"] = float(control.quantile(1 - module.ALPHA_TARGET))

    if out.duplicated(["Fish_ID", "Condition"]).any():
        raise ValueError("Historical algorithm returned duplicate fish keys")
    out.to_parquet(output_path, index=False)
    settings = {"analysis_cfg": asdict(module.analysis_cfg)}
    for name in ("ALPHA_TARGET", "USE_PER_FISH_SE_IN_SCORING", "USE_LOG_TRANSFORM",
                 "MIN_FISH_WITH_ALL_FEATURES"):
        if hasattr(module, name):
            settings[name] = getattr(module, name)
    output_path.with_suffix(".config.json").write_text(
        json.dumps(settings, sort_keys=True, indent=2, default=str) + "\n"
    )


# Parse the isolated worker contract and dispatch the selected learner variant.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=SCRIPTS)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--individuals", action="store_true")
    args = parser.parse_args()
    execute(args.variant, args.input, args.output, figures=args.figures, individuals=args.individuals)


if __name__ == "__main__":
    main()
