"""Compact review figure for frozen legacy stage-1 samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    write_json_atomic,
)
from classical_conditioning.figures.export import (
    FigureMode,
    FigureProvenance,
    export_matplotlib_figure,
)
from classical_conditioning.preprocessing.legacy_v1 import (
    SCALED_VIGOR_COLUMN,
    TIME_COLUMN,
    VIGOR_COLUMN,
)


@dataclass(frozen=True)
class LegacyReviewFigureResult:
    recording_id: str
    outputs: tuple[Path, ...]
    sidecar_path: Path
    summary_path: Path
    trial_counts: dict[str, int]


def _load_legacy_samples(project_dir: Path, recording_id: str) -> pd.DataFrame:
    path = project_dir / "Processed data" / recording_id / "samples_legacy-v1.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing legacy samples artifact: {path}")
    columns = [
        TIME_COLUMN,
        VIGOR_COLUMN,
        "Trial type",
        "Trial number",
        "Block name",
        "Bout",
    ]
    available = set(pq.ParquetFile(path).schema_arrow.names)
    if SCALED_VIGOR_COLUMN in available:
        columns.append(SCALED_VIGOR_COLUMN)
    missing = [column for column in columns if column not in available]
    if missing:
        raise KeyError(f"Legacy samples missing columns: {missing}")
    return pq.read_table(path, columns=columns).to_pandas()


def _downsample(series: pd.Series, maximum_points: int = 4_000) -> pd.Series:
    if len(series) <= maximum_points:
        return series
    step = int(np.ceil(len(series) / maximum_points))
    return series.iloc[::step]


def build_legacy_preprocessing_review_figure(
    project_dir: Path,
    recording_id: str,
    *,
    mode: FigureMode = FigureMode.STATIC,
    overwrite: bool = False,
) -> LegacyReviewFigureResult:
    """Render a small legacy QC figure from authenticated stage-1 samples."""
    project_dir = project_dir.resolve()
    samples = _load_legacy_samples(project_dir, recording_id)
    quality_dir = project_dir / "Quality checks" / recording_id
    quality_dir.mkdir(parents=True, exist_ok=True)
    figure_stem = quality_dir / "legacy-v1_preprocessing_review"
    summary_path = quality_dir / "legacy-v1_preprocessing_review_summary.json"
    sidecar_path = Path(str(figure_stem) + ".json")

    existing = [
        path
        for path in (
            Path(str(figure_stem) + ".png"),
            Path(str(figure_stem) + ".svg"),
            Path(str(figure_stem) + ".pdf"),
            sidecar_path,
            summary_path,
        )
        if path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"Legacy review figure outputs exist; pass overwrite to replace: {existing}"
        )

    trial_counts = {
        str(key): int(value)
        for key, value in samples["Trial type"].astype(str).value_counts().items()
    }
    cs = samples.loc[samples["Trial type"].astype(str).eq("CS")]
    example = None
    if not cs.empty:
        first_trial = int(cs["Trial number"].iloc[0])
        example = cs.loc[cs["Trial number"] == first_trial]

    figure, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), constrained_layout=True)

    vigor = _downsample(samples[VIGOR_COLUMN].reset_index(drop=True))
    axes[0].plot(vigor.to_numpy(), color="#222222", linewidth=0.6)
    axes[0].set_title(f"Legacy vigor overview ({recording_id})")
    axes[0].set_ylabel(VIGOR_COLUMN)
    axes[0].set_xlabel("Downsampled sample index")

    block_counts = (
        samples.groupby(["Trial type", "Block name"], observed=True)
        .size()
        .reset_index(name="rows")
    )
    if block_counts.empty:
        axes[1].text(0.5, 0.5, "No trial blocks", ha="center", va="center")
    else:
        labels = [
            f"{row['Trial type']}:{row['Block name']}"
            for _, row in block_counts.iterrows()
        ]
        axes[1].bar(range(len(labels)), block_counts["rows"], color="#4C78A8")
        axes[1].set_xticks(range(len(labels)))
        axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Sample rows")
    axes[1].set_title("Rows by trial type and block")

    if example is None or example.empty:
        axes[2].text(0.5, 0.5, "No CS trial available", ha="center", va="center")
    else:
        time = example[TIME_COLUMN].to_numpy()
        axes[2].plot(
            time,
            example[VIGOR_COLUMN].to_numpy(),
            color="#E45756",
            linewidth=0.8,
            label=VIGOR_COLUMN,
        )
        if SCALED_VIGOR_COLUMN in example.columns:
            axes[2].plot(
                time,
                example[SCALED_VIGOR_COLUMN].to_numpy(),
                color="#54A24B",
                linewidth=0.8,
                label=SCALED_VIGOR_COLUMN,
            )
        bout = example["Bout"].to_numpy(dtype=bool)
        if bout.any():
            axes[2].fill_between(
                time,
                0,
                1,
                where=bout,
                transform=axes[2].get_xaxis_transform(),
                color="#F58518",
                alpha=0.2,
                label="Bout",
            )
        axes[2].legend(loc="upper right", fontsize=8)
        axes[2].set_title(
            f"Example CS trial {int(example['Trial number'].iloc[0])}"
        )
        axes[2].set_xlabel(TIME_COLUMN)
        axes[2].set_ylabel("Vigor")

    for axis, panel_id in zip(axes, ("A", "B", "C"), strict=True):
        axis.set_gid(f"axes__{panel_id.lower()}__main")

    provenance = FigureProvenance(
        figure_id=f"legacy-v1-review-{recording_id}",
        analysis_recipe="legacy-paper-v1",
        source_file=__file__,
        source_symbol="build_legacy_preprocessing_review_figure",
        source_hash=sha256_file(Path(__file__).resolve()),
        reproduction_snippet=(
            "build_legacy_preprocessing_review_figure("
            f"project_dir, {recording_id!r}, mode={mode.value!r})"
        ),
        input_artifacts=(
            {
                "path": str(
                    (
                        project_dir
                        / "Processed data"
                        / recording_id
                        / "samples_legacy-v1.parquet"
                    ).resolve()
                ),
                "sha256": sha256_file(
                    project_dir
                    / "Processed data"
                    / recording_id
                    / "samples_legacy-v1.parquet"
                ),
            },
        ),
    )

    with artifact_staging(quality_dir, prefix=".legacy-v1-review-") as staging:
        staged_stem = staging / "legacy-v1_preprocessing_review"
        export = export_matplotlib_figure(
            figure,
            staged_stem,
            provenance,
            mode=mode,
            panel_ids=["A", "B", "C"],
            allow_dirty_publication=True,
        )
        plt.close(figure)
        summary: dict[str, Any] = {
            "artifact_kind": "legacy-preprocessing-review-v1",
            "recording_id": recording_id,
            "mode": mode.value,
            "trial_counts": trial_counts,
            "row_count": int(len(samples)),
            "outputs": [str(path) for path in export.outputs],
        }
        staged_summary = staging / summary_path.name
        write_json_atomic(staged_summary, summary)
        publish_units = tuple(
            (path, quality_dir / path.name) for path in export.outputs
        ) + (
            (export.sidecar, sidecar_path),
            (staged_summary, summary_path),
        )
        publish_transaction(
            publish_units,
            staging,
            overwrite=overwrite,
        )

    return LegacyReviewFigureResult(
        recording_id=recording_id,
        outputs=tuple(quality_dir / path.name for path in export.outputs),
        sidecar_path=sidecar_path,
        summary_path=summary_path,
        trial_counts=trial_counts,
    )
