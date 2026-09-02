"""Frame-sequence diagnostics for camera timing tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from classical_conditioning.exceptions import SchemaValidationError


@dataclass(frozen=True)
class FrameSequenceReport:
    row_count: int
    first_frame_id: int
    last_frame_id: int
    unique_frame_id_count: int
    duplicate_frame_id_count: int
    gap_event_count: int
    missing_frame_count: int
    reverse_event_count: int
    nonmonotonic_elapsed_count: int
    non_finite_elapsed_count: int
    median_elapsed_interval_ms: float | None
    mean_elapsed_interval_ms: float | None
    elapsed_interval_jitter_ms: float | None
    frame_id_span: int
    expected_rows_if_contiguous: int
    examples: tuple[dict[str, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "first_frame_id": self.first_frame_id,
            "last_frame_id": self.last_frame_id,
            "unique_frame_id_count": self.unique_frame_id_count,
            "duplicate_frame_id_count": self.duplicate_frame_id_count,
            "gap_event_count": self.gap_event_count,
            "missing_frame_count": self.missing_frame_count,
            "reverse_event_count": self.reverse_event_count,
            "nonmonotonic_elapsed_count": self.nonmonotonic_elapsed_count,
            "non_finite_elapsed_count": self.non_finite_elapsed_count,
            "median_elapsed_interval_ms": self.median_elapsed_interval_ms,
            "mean_elapsed_interval_ms": self.mean_elapsed_interval_ms,
            "elapsed_interval_jitter_ms": self.elapsed_interval_jitter_ms,
            "frame_id_span": self.frame_id_span,
            "expected_rows_if_contiguous": self.expected_rows_if_contiguous,
            "examples": list(self.examples),
        }


def validate_frame_sequence(camera: pd.DataFrame) -> FrameSequenceReport:
    """Compute exact FrameID/timestamp diagnostics without changing inclusion."""
    required = {"FrameID", "ElapsedTime"}
    missing = required.difference(camera.columns)
    if missing:
        raise SchemaValidationError(
            f"Frame-sequence validation requires columns: {sorted(missing)}"
        )
    if camera.empty:
        raise SchemaValidationError("Cannot validate an empty camera table.")

    frame_ids = pd.to_numeric(camera["FrameID"], errors="raise").to_numpy(dtype=np.int64)
    elapsed = pd.to_numeric(camera["ElapsedTime"], errors="raise").to_numpy(
        dtype=np.float64
    )
    differences = np.diff(frame_ids)
    gap_events = int(np.count_nonzero(differences > 1))
    missing_frames = int(np.sum(differences[differences > 1] - 1))
    reverse_events = int(np.count_nonzero(differences < 0))
    duplicate_steps = int(np.count_nonzero(differences == 0))

    elapsed_diff = np.diff(elapsed)
    finite_elapsed = np.isfinite(elapsed)
    non_finite_elapsed = int(np.count_nonzero(~finite_elapsed))
    finite_steps = elapsed_diff[np.isfinite(elapsed_diff)]
    nonmonotonic_elapsed = int(np.count_nonzero(finite_steps < 0))

    median_interval = float(np.median(finite_steps)) if finite_steps.size else None
    mean_interval = float(np.mean(finite_steps)) if finite_steps.size else None
    jitter = float(np.std(finite_steps)) if finite_steps.size else None

    examples: list[dict[str, int]] = []
    anomaly_indices = np.flatnonzero(differences != 1)
    for index in anomaly_indices[:50]:
        examples.append(
            {
                "previous_frame_id": int(frame_ids[index]),
                "frame_id": int(frame_ids[index + 1]),
                "difference": int(differences[index]),
            }
        )

    first = int(frame_ids[0])
    last = int(frame_ids[-1])
    span = last - first
    return FrameSequenceReport(
        row_count=int(len(frame_ids)),
        first_frame_id=first,
        last_frame_id=last,
        unique_frame_id_count=int(len(np.unique(frame_ids))),
        duplicate_frame_id_count=duplicate_steps,
        gap_event_count=gap_events,
        missing_frame_count=missing_frames,
        reverse_event_count=reverse_events,
        nonmonotonic_elapsed_count=nonmonotonic_elapsed,
        non_finite_elapsed_count=non_finite_elapsed,
        median_elapsed_interval_ms=median_interval,
        mean_elapsed_interval_ms=mean_interval,
        elapsed_interval_jitter_ms=jitter,
        frame_id_span=span,
        expected_rows_if_contiguous=span + 1 if span >= 0 else 0,
        examples=tuple(examples),
    )
