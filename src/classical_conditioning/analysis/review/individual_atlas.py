"""Authenticated, descriptive individual-fish data; no cohort/label approval."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

from classical_conditioning.figures.pooled_log_heatmap import read_windows

METRIC = "legacy_distal_angular_speed"
VARIANTS = ("legacy-nominal", "legacy-new", "legacy-improved", "legacy-wip")
EDGES = np.arange(-15., 26.)
TRIALS = np.arange(5, 95)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stage_inputs(root):
    """Resolve the recorded G stage, authenticating only the inputs consumed here."""
    root = Path(root)
    state = json.loads((root / "analysis-state.json").read_text())
    stage = root / state["latest"]["directory"]
    manifest = stage / "artifact-manifest.json"
    if sha(manifest) != state["latest"]["manifest_sha256"]:
        raise ValueError("Changed source-stage manifest")
    expected = {e["path"]: e["sha256"] for e in json.loads(manifest.read_text())["artifacts"]}
    names = ["scheduled-window-outcomes.parquet", "input-lineage.json",
             "recording-inventory.json", "legacy-discard-rule-matrix.csv",
             "technical-recording-audit.csv"]
    inputs = [{"path": str(manifest), "sha256": sha(manifest)}]
    for name in names:
        p = stage / name
        if sha(p) != expected[name]:
            raise ValueError("Changed source-stage input: " + name)
        inputs.append({"path": str(p), "sha256": expected[name]})
    return stage, inputs


def bin_trial(times, values, valid, moving, duration_ms, onset_ms):
    """Separate bout intensity, detector-eligible movement time and availability."""
    seconds = (np.asarray(times) - onset_ms) / 1000
    idx = np.searchsorted(EDGES, seconds, side="right") - 1
    selected = (idx >= 0) & (idx < len(EDGES) - 1)
    # Durations belong to the frame derivative's actual observed interval.
    duration = np.asarray(duration_ms, float)
    physical = np.asarray(valid, bool) & np.isfinite(duration) & (duration > 0)
    bout = physical & np.asarray(moving, bool)
    positive = bout & np.isfinite(values) & (np.asarray(values) > 0)
    log_bins = np.full(len(EDGES)-1, np.nan)
    occurrence = np.full_like(log_bins, np.nan)
    coverage = np.zeros_like(log_bins)
    counts = np.zeros(len(log_bins), int)
    for b in range(len(log_bins)):
        take = selected & (idx == b)
        good = take & physical
        weight = duration[good].sum()
        coverage[b] = min(1., weight / 1000)
        if weight > 0:
            occurrence[b] = duration[take & bout].sum() / weight
        intensity = np.asarray(values)[take & positive]
        counts[b] = len(intensity)
        if len(intensity):
            log_bins[b] = np.median(np.log(intensity))
    baseline = log_bins[:15]
    baseline = baseline[np.isfinite(baseline)]
    centre = float(np.median(baseline)) if len(baseline) else np.nan
    return log_bins-centre, occurrence, coverage, counts, centre


def prepare(g_root, output):
    """Build one card-data row per scheduled fish/trial from immutable frame inputs."""
    output = Path(output)
    stage, inputs = stage_inputs(g_root)
    windows = pd.read_parquet(stage / "scheduled-window-outcomes.parquet")
    if set(windows.metric_id) != {METRIC} or windows.duplicated(
            ["experiment_id", "condition_id", "fish_id", "trial_number"]).any():
        raise ValueError("Metric or fish/trial identity mismatch")
    windows = windows.sort_values(["condition_id", "fish_id", "trial_number"]).copy()
    lineage = json.loads((stage / "input-lineage.json").read_text())
    by_path = {r["path"]: r["sha256"] for r in lineage}
    bound = {Path(r["path"]).parent.name: {} for r in lineage
             if Path(r["path"]).name.startswith("frame_activity_")}
    for r in lineage:
        p = Path(r["path"])
        if p.parent.name in bound and p.suffix == ".parquet":
            bound[p.parent.name][p.name] = r
    matrices, details, numerical_checks = {}, [], []
    for number, (fish, data) in enumerate(windows.groupby("fish_id", sort=True), 1):
        data = data.sort_values("trial_number")
        if not np.array_equal(data.trial_number, TRIALS):
            raise ValueError("Incomplete scheduled trial map: " + fish)
        refs = bound[fish]
        frame = next(r for n, r in refs.items() if n.startswith("frame_activity_"))
        movement = next(r for n, r in refs.items() if n.startswith("movement_state_"))
        protocol = refs["stimulus_events.parquet"]
        for r in (frame, movement, protocol):
            if sha(r["path"]) != r["sha256"]:
                raise ValueError("Changed frame input: " + r["path"])
            inputs.append(r)
        intervals = [(int(t)-15001, int(t)+25001)
                     for t in data.event_start_absolute_time_ms]
        frames = read_windows(Path(frame["path"]), ["FrameID", "AbsoluteTime", "FrameStep",
            "DeltaTimeMs", "legacy_distal_angular_speed_rad_per_ms"], intervals)
        moves = read_windows(Path(movement["path"]), ["FrameID", "AbsoluteTime", "valid", "moving"], intervals)
        np.testing.assert_array_equal(frames[["FrameID", "AbsoluteTime"]], moves[["FrameID", "AbsoluteTime"]])
        time = frames.AbsoluteTime.to_numpy(np.int64)
        if np.any(np.diff(time) < 0):
            raise ValueError("Nonchronological time: " + fish)
        value = frames.legacy_distal_angular_speed_rad_per_ms.to_numpy(float)
        valid = moves.valid.to_numpy(bool) & frames.FrameStep.eq(1).to_numpy() & np.isfinite(value)
        moving = moves.moving.to_numpy(bool)
        duration = frames.DeltaTimeMs.to_numpy(float)
        events = pd.read_parquet(protocol["path"])
        us = events.loc[events.Type.eq("Reinforcer")]
        heat, occurrence, coverage, counts = [], [], [], []
        fish_max_error = 0.
        for trial in data.itertuples(index=False):
            onset = int(trial.event_start_absolute_time_ms)
            a, b = np.searchsorted(time, [onset-15000, onset+25000])
            h, o, c, n, centre = bin_trial(time[a:b], value[a:b], valid[a:b], moving[a:b], duration[a:b], onset)
            heat.append(h); occurrence.append(o); coverage.append(c); counts.append(n)
            row = trial._asdict()
            row["heatmap_baseline_log"] = centre
            for prefix, lower, upper, expected in [
                ("baseline", -15000, 0, trial.baseline_conditional_intensity),
                ("response", 0, 9000, trial.conditional_intensity),
            ]:
                left, right = np.searchsorted(time, [onset+lower, onset+upper])
                v, m = valid[left:right], moving[left:right]
                w = duration[left:right]
                bout_values = value[left:right][v & m]
                mean = float(np.mean(bout_values)) if len(bout_values) else np.nan
                np.testing.assert_allclose(mean, expected, rtol=1e-10, atol=1e-12, equal_nan=True)
                if np.isfinite(mean): fish_max_error = max(fish_max_error, abs(mean-expected))
                denominator = w[v].sum()
                row[prefix+"_fraction_moving"] = float(w[v & m].sum()/denominator) if denominator > 0 else np.nan
                row[prefix+"_coverage"] = min(1., float(denominator/(upper-lower)))
                row[prefix+"_bout_mean"] = mean
            row["classifier_ratio"] = (row["response_bout_mean"] / row["baseline_bout_mean"]
                if np.isfinite(row["baseline_bout_mean"]) and row["baseline_bout_mean"] > 0 else np.nan)
            row["actual_us_s"] = json.dumps([round(float(x-onset)/1000, 4)
                for x in us.Beg if onset-15000 <= x < onset+25000])
            details.append(row)
        for key, array in [("heat", heat), ("moving", occurrence), ("coverage", coverage), ("counts", counts)]:
            matrices[fish+"__"+key] = np.asarray(array)
        numerical_checks.append({"fish_id": fish, "scheduled_trials": len(data),
            "conditional_mean_maximum_error": fish_max_error})
        del frames, moves, time, value, valid, moving, duration
        gc.collect()
        print(f"Card data {number}/57: {fish}", flush=True)
    trials = pd.DataFrame(details).sort_values(["condition_id", "fish_id", "trial_number"])
    trials.to_parquet(output / "trial-data.parquet", index=False)
    trials.to_csv(output / "trial-data.csv", index=False)
    np.savez_compressed(output / "heatmaps.npz", **matrices)
    fish = trials[["experiment_id", "condition_id", "fish_id"]].drop_duplicates()
    technical = pd.read_csv(stage / "technical-recording-audit.csv")
    rules = pd.read_csv(stage / "legacy-discard-rule-matrix.csv")
    fish = fish.merge(technical, on=["condition_id", "fish_id"], validate="one_to_one")
    fish = fish.merge(rules, on=["condition_id", "fish_id"], validate="one_to_one")
    fish["scope_status"] = "selected G inventory; paper-wide technical cohort not approved"
    fish.to_csv(output / "fish-inventory.csv", index=False)
    inventory = json.loads((stage / "recording-inventory.json").read_text())
    if {r["recording_id"] for r in inventory["records"]} != set(fish.fish_id):
        raise ValueError("Expected recording inventory does not reconcile")
    legacy = pd.DataFrame({"Exp.": trials.condition_id, "Fish": trials.fish_id,
        "Trial number": trials.trial_number, "Mean CR": trials.response_bout_mean,
        "Mean 9s before": trials.baseline_bout_mean, "Normalized vigor": trials.classifier_ratio})
    legacy.to_parquet(output / "classifier-input.parquet", index=False)
    unique_inputs = {r["path"]: r["sha256"] for r in inputs}
    save_json(output / "input-lineage.json", [{"path": p, "sha256": h} for p, h in unique_inputs.items()])
    save_json(output / "data-checks.json", {"fish": len(fish), "scheduled_trials": len(trials),
        "expected_inventory_reconciled": True, "all_bout_means_reconstructed": True,
        "per_fish": numerical_checks, "source_stage": str(stage),
        "metric": METRIC, "vigor_sample_policy": "valid-bout-frames-only-v1",
        "heatmap": "Direct 1 s median log positive bout vigor, minus median of defined 1 s baseline bins [-15,0); no-bout intensity undefined",
        "coverage_semantics": "Stored movement.valid and metric-adjacent frames define detector-eligible time, not physical tracking completeness",
        "scope": "Delay/control complete; Trace authenticated frame/outcome bundle not available in mounted processed roots"})


def classify(output):
    """Run original algorithms in isolated processes; retain failure logs and all fish."""
    output = Path(output)
    inventory = pd.read_csv(output / "fish-inventory.csv")
    summary = []
    for variant in VARIANTS:
        dest = output / "classifiers" / variant
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "results.parquet"
        if path.exists():
            raise FileExistsError("Preserve existing classifier run: " + str(path))
        run = subprocess.run([sys.executable, "-m", "classical_conditioning.analysis.legacy_learner_worker",
            variant, str(output / "classifier-input.parquet"), str(path)], capture_output=True, text=True)
        (dest / "run.log").write_text(run.stdout + "\n[stderr]\n" + run.stderr)
        if run.returncode:
            summary.append({"variant": variant, "status": "failed", "exit_code": run.returncode})
        else:
            result = pd.read_parquet(path)
            if result.duplicated(["Condition", "Fish_ID"]).any():
                raise ValueError("Duplicate classifier key")
            for condition in ("control", "delay"):
                sub = result.loc[result.Condition.eq(condition)]
                summary.append({"variant": variant, "condition": condition, "status": "completed",
                    "eligible": len(sub), "flagged": int(sub.learner_primary.sum()),
                    "unclassified": int(inventory.condition_id.eq(condition).sum()-len(sub))})
        save_json(output / "classification-summary.json", summary)
        print("Classifier completed: " + variant + f" (exit {run.returncode})", flush=True)
