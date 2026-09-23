"""Selected-fish tail-angle and vigor traces for manuscript Figure 1 C/D.

The two columns share one fish, the same global CS trial numbers, and the
same measured-time frame selection. This is a display of a chosen candidate
metric; selecting it here does not approve that metric for population analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from classical_conditioning.analysis.candidate_runner import (
    _verify_corrected_preprocess,
    _verify_metrics,
)
from classical_conditioning.analysis.movement_state import (
    METRIC_IDS,
    resolve_candidate_metric_source,
)
from classical_conditioning.artifacts import load_and_verify_source_manifest, sha256_file
from classical_conditioning.config.experiments import get_experiment_spec
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.figures.export import (
    FigureExportResult,
    FigureMode,
    FigureProvenance,
    export_matplotlib_figure,
)
from classical_conditioning.figures.temporal_profiles import METRIC_UNITS
from classical_conditioning.figures.theme import (
    DOUBLE_COLUMN_MM,
    apply_theme,
    mm_to_in,
    style_axes,
)

METRIC_COLUMNS = {metric_id: column for column, metric_id in METRIC_IDS.items()}
METRIC_DISPLAY_NAMES = {
    "tail_length_weighted_angular_l1": "tail length weighted angular L1",
    "whole_tail_xy_mean_speed_normalized": "whole tail XY speed",
    "legacy_distal_angular_speed": "distal angular speed",
}


def prepare_example_trace_data(
    corrected: pd.DataFrame,
    metrics: pd.DataFrame,
    protocol: pd.DataFrame,
    *,
    trial_numbers: Sequence[int],
    metric_id: str,
    tail_point: int = 15,
    window_s: tuple[float, float] = (-20.0, 20.0),
    cs_duration_s: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair distal tail angle and one vigor metric on identical measured frames."""
    if metric_id not in METRIC_COLUMNS:
        raise ConfigurationError(f"Unknown vigor metric: {metric_id!r}.")
    if tail_point < 0:
        raise ConfigurationError("Tail point must be non-negative.")
    trials = tuple(int(number) for number in trial_numbers)
    if not trials or any(number < 1 for number in trials) or len(trials) != len(set(trials)):
        raise ConfigurationError("Select one or more distinct positive CS trial numbers.")
    if not np.isfinite(window_s).all() or window_s[0] >= 0:
        raise ConfigurationError("Trace window must include the pre-CS period and CS offset.")
    if cs_duration_s <= 0 or window_s[1] <= cs_duration_s:
        raise ConfigurationError("Trace window must include the complete CS.")
    angle_columns = [f"angle{index}" for index in range(tail_point + 1)]
    required_corrected = {"FrameID", "AbsoluteTime", *angle_columns}
    required_metrics = {"FrameID", "AbsoluteTime", METRIC_COLUMNS[metric_id]}
    required_protocol = {"Type", "Beg", "End"}
    for name, data, required in (
        ("Corrected frames", corrected, required_corrected),
        ("Candidate metrics", metrics, required_metrics),
        ("Protocol", protocol, required_protocol),
    ):
        absent = required.difference(data.columns)
        if absent:
            raise ConfigurationError(f"{name} lack columns: {sorted(absent)}.")
    if len(corrected) != len(metrics) or not np.array_equal(
        corrected["FrameID"].to_numpy(), metrics["FrameID"].to_numpy()
    ) or not np.array_equal(
        corrected["AbsoluteTime"].to_numpy(), metrics["AbsoluteTime"].to_numpy()
    ):
        raise ConfigurationError("Corrected and vigor frames do not align exactly.")
    absolute = corrected["AbsoluteTime"].to_numpy(dtype=np.int64)
    # The camera can record multiple frames within one millisecond; the intake
    # clock stores integer milliseconds, so equal adjacent timestamps are valid.
    if np.any(np.diff(absolute) < 0):
        raise ConfigurationError("Corrected frame times must be chronological.")
    angles = corrected[angle_columns].to_numpy(dtype=float)
    distal_angle = np.where(np.isfinite(angles).all(axis=1), angles.sum(axis=1), np.nan)
    vigor = pd.to_numeric(metrics[METRIC_COLUMNS[metric_id]], errors="coerce").to_numpy(dtype=float)
    cycles = protocol.loc[protocol["Type"].astype(str).eq("Cycle")].sort_values(
        "Beg", kind="stable"
    ).reset_index(drop=True)
    if not cycles["Beg"].is_monotonic_increasing:
        raise ConfigurationError("CS events are not chronological.")
    if any(number > len(cycles) for number in trials):
        raise ConfigurationError(
            f"Requested CS trial outside the recording's {len(cycles)} Cycle events."
        )
    us_events = protocol.loc[protocol["Type"].astype(str).eq("Reinforcer")]
    frames: list[pd.DataFrame] = []
    events: list[dict[str, float | int | str]] = []
    for trial in trials:
        onset_ms = int(cycles.iloc[trial - 1]["Beg"])
        start = int(np.searchsorted(absolute, onset_ms + window_s[0] * 1000, side="left"))
        stop = int(np.searchsorted(absolute, onset_ms + window_s[1] * 1000, side="right"))
        if stop <= start:
            raise ConfigurationError(f"CS trial {trial} has no frames in the selected window.")
        time_s = (absolute[start:stop] - onset_ms) / 1000.0
        angle = distal_angle[start:stop].copy()
        baseline = angle[(time_s < 0) & np.isfinite(angle)]
        if baseline.size:
            angle -= float(np.median(baseline))
        frames.append(pd.DataFrame({
            "Trial number": trial,
            "FrameID": corrected["FrameID"].to_numpy()[start:stop],
            "Time relative to CS onset (s)": time_s,
            "Tail angle (rad)": angle,
            "Vigor": vigor[start:stop],
        }))
        events.extend((
            {"Trial number": trial, "Event": "CS onset", "Time (s)": 0.0},
            {"Trial number": trial, "Event": "CS offset", "Time (s)": cs_duration_s},
        ))
        for us in us_events.itertuples(index=False):
            relative = (int(us.Beg) - onset_ms) / 1000.0
            if window_s[0] <= relative <= window_s[1]:
                events.append({
                    "Trial number": trial, "Event": "actual US onset",
                    "Time (s)": relative,
                })
    return pd.concat(frames, ignore_index=True), pd.DataFrame(events)


def render_example_trace_figure(
    frames: pd.DataFrame,
    events: pd.DataFrame,
    *,
    trial_numbers: Sequence[int],
    metric_id: str,
    window_s: tuple[float, float],
    trial_labels: Sequence[str] | None = None,
) -> tuple[plt.Figure, list[str], dict[str, dict[str, str]]]:
    """Render the two Figure 1 trace columns with matched trial rows."""
    theme = apply_theme()
    trials = tuple(trial_numbers)
    if trial_labels is not None and len(trial_labels) != len(trials):
        raise ConfigurationError("Trial labels must match the number of trials.")
    figure, axes = plt.subplots(
        len(trials), 2,
        figsize=mm_to_in(DOUBLE_COLUMN_MM, max(45.0, 22.0 * len(trials))),
        squeeze=False, sharex=True, sharey="col", constrained_layout=True,
    )
    panel_ids: list[str] = []
    mappings: dict[str, dict[str, str]] = {}
    for index, trial in enumerate(trials):
        subset = frames.loc[frames["Trial number"].eq(trial)]
        trial_events = events.loc[events["Trial number"].eq(trial)]
        for column, (value_column, title, units) in enumerate((
            ("Tail angle (rad)", "Tail angle", "rad"),
            ("Vigor", "Vigor", METRIC_UNITS[metric_id]),
        )):
            axis = axes[index, column]
            panel = f"{'C' if column == 0 else 'D'}_trial_{trial}"
            panel_ids.append(panel)
            line, = axis.plot(
                subset["Time relative to CS onset (s)"], subset[value_column],
                color="black", linewidth=0.45, rasterized=True,
            )
            line_id = f"trace__{panel.lower()}"
            line.set_gid(line_id)
            mappings[line_id] = {
                "x_field": "Time relative to CS onset (s)",
                "y_field": value_column,
                "trial_number": str(trial),
                "metric_id": metric_id if column else "distal_cumulative_tail_angle",
                "units": units,
                "baseline_centering": "pre-CS median per trial" if column == 0 else "none",
            }
            for event_index, event in enumerate(trial_events.itertuples(index=False)):
                kind = str(event.Event)
                color = theme.us_color if kind == "actual US onset" else theme.cs_color
                linestyle = "--" if kind == "CS offset" else "-"
                marker = axis.axvline(
                    float(event[2]), color=color, linestyle=linestyle, linewidth=0.7
                )
                marker_id = f"stimulus__{panel.lower()}__{event_index}"
                marker.set_gid(marker_id)
                mappings[marker_id] = {
                    "source": "experiment definition" if kind == "CS offset" else "authenticated protocol",
                    "event": kind,
                    "time_s": str(float(event[2])),
                    "trial_number": str(trial),
                }
            axis.set_xlim(*window_s)
            if column == 1:
                axis.set_ylim(bottom=0)
            style_axes(
                axis, theme=theme, show_xticks=index == len(trials) - 1,
                xlabel="Time relative to CS onset (s)" if index == len(trials) - 1 else None,
                ylabel=(trial_labels[index] if trial_labels is not None else f"Trial {trial}")
                if column == 0 else None,
            )
            if index == 0:
                display_title = (
                    title if column == 0 else f"Vigor: {METRIC_DISPLAY_NAMES[metric_id]}"
                )
                axis.set_title(f"{display_title} ({units})")
    return figure, panel_ids, mappings


def build_example_trace_figure(
    project_dir: Path,
    recording_id: str,
    *,
    trial_numbers: Sequence[int],
    metric_id: str,
    experiment: str,
    mode: FigureMode = FigureMode.STATIC,
    tail_point: int = 15,
    window_s: tuple[float, float] = (-20.0, 20.0),
    overwrite: bool = False,
) -> FigureExportResult:
    """Read authenticated corrected artifacts and export paired C/D traces."""
    if mode == FigureMode.INTERACTIVE:
        raise ConfigurationError("Example traces support static or publication mode.")
    experiment_spec = get_experiment_spec(experiment)
    if metric_id not in METRIC_COLUMNS:
        raise ConfigurationError(f"Unknown vigor metric: {metric_id!r}.")
    if tail_point < 0 or tail_point > 15:
        raise ConfigurationError("Tail point must be between 0 and 15.")
    project_dir = project_dir.resolve()
    source = resolve_candidate_metric_source(metric_recipe="tail-candidate-corrected")
    _verify_corrected_preprocess(project_dir, recording_id)
    _verify_metrics(project_dir, recording_id, source)
    _, intake, _ = load_and_verify_source_manifest(project_dir, recording_id)
    root = project_dir / "Processed data" / recording_id
    corrected_path = root / "frame_preprocessed_corrected.parquet"
    metrics_path = root / source.metrics_name
    protocol_path = root / "stimulus_events.parquet"
    if sha256_file(protocol_path) != intake["protocol"]["sha256"]:
        raise ConfigurationError("Protocol differs from the authenticated intake.")
    angle_columns = [f"angle{index}" for index in range(tail_point + 1)]
    corrected = pq.read_table(
        corrected_path, columns=["FrameID", "AbsoluteTime", *angle_columns]
    ).to_pandas()
    metrics = pq.read_table(
        metrics_path, columns=["FrameID", "AbsoluteTime", METRIC_COLUMNS[metric_id]]
    ).to_pandas()
    protocol = pq.read_table(protocol_path).to_pandas()
    frames, events = prepare_example_trace_data(
        corrected, metrics, protocol, trial_numbers=trial_numbers,
        metric_id=metric_id, tail_point=tail_point, window_s=window_s,
        cs_duration_s=experiment_spec.cs_duration_s,
    )
    figure, panel_ids, mappings = render_example_trace_figure(
        frames, events, trial_numbers=trial_numbers,
        metric_id=metric_id, window_s=window_s,
    )
    output_base = (
        project_dir / "Figures"
        / ("Publication" if mode == FigureMode.PUBLICATION else "PNG")
        / recording_id / f"example-traces_{metric_id}_{'-'.join(map(str, trial_numbers))}"
    )
    inputs = tuple({
        "path": str(path), "sha256": sha256_file(path)
    } for path in (corrected_path, metrics_path, protocol_path))
    source_file = Path(__file__).resolve()
    try:
        return export_matplotlib_figure(
            figure, output_base,
            FigureProvenance(
                figure_id="figure-1-CD-example-traces",
                analysis_recipe="selected-fish-corrected-example-traces",
                source_file=str(source_file),
                source_symbol="build_example_trace_figure",
                source_hash=sha256_file(source_file),
                reproduction_snippet=(
                    "classical-conditioning figure-example-traces "
                    f"--project-dir {project_dir} --recording-id {recording_id} "
                    f"--experiment {experiment} --metric {metric_id} "
                    + " ".join(f"--trial {number}" for number in trial_numbers)
                    + f" --tail-point {tail_point} --window-start {window_s[0]} "
                    f"--window-end {window_s[1]} --mode {mode.value}"
                ),
                input_artifacts=inputs,
                artist_mappings=mappings,
            ),
            mode=mode, panel_ids=panel_ids, overwrite=overwrite,
        )
    finally:
        plt.close(figure)
