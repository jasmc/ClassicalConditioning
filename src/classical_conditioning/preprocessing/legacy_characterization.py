"""Synthetic characterization fixtures for legacy preprocessing equivalence."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from classical_conditioning.preprocessing.legacy_v1 import TIME_COLUMN

CharacterizationCase = Literal[
    "constant",
    "one_moving_point",
    "all_moving_together",
    "opposing_local_points",
    "no_movement",
    "short_bout_pulse",
    "two_close_pulses",
]


def make_legacy_angle_frame(
    case: CharacterizationCase,
    *,
    row_count: int = 60,
    point_count: int = 4,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a compact angle table for step-by-step legacy equivalence checks."""
    if row_count < 8:
        raise ValueError("Characterization fixtures need at least 8 rows.")
    if point_count < 2:
        raise ValueError("Characterization fixtures need at least 2 angle points.")

    rng = np.random.default_rng(seed)
    angles = np.zeros((row_count, point_count), dtype=np.float64)

    if case in {"constant", "no_movement"}:
        angles[:] = 0.5
    elif case == "one_moving_point":
        angles[:, -1] = np.linspace(0.0, 2.0, row_count)
    elif case == "all_moving_together":
        shared = np.sin(np.linspace(0.0, 4.0, row_count))
        angles[:] = shared[:, None]
    elif case == "opposing_local_points":
        angles[:, 0] = np.linspace(0.0, 1.0, row_count)
        angles[:, -1] = np.linspace(0.0, -1.0, row_count)
    elif case == "short_bout_pulse":
        angles[:, -1] = 0.1
        mid = row_count // 2
        angles[mid : mid + 4, -1] = np.array([0.1, 3.0, 3.0, 0.1])
    elif case == "two_close_pulses":
        angles[:, -1] = 0.1
        mid = row_count // 3
        angles[mid : mid + 3, -1] = 2.5
        angles[mid + 5 : mid + 8, -1] = 2.5
    else:
        raise ValueError(f"Unknown characterization case: {case!r}")

    # Tiny jitter keeps floating pipelines realistic without changing case intent.
    if case not in {"constant", "no_movement"}:
        angles = angles + rng.normal(0.0, 1e-9, size=angles.shape)

    frame = pd.DataFrame(
        {
            TIME_COLUMN: np.arange(row_count, dtype=np.int64),
            "AbsoluteTime": np.arange(row_count, dtype=np.float64) * (1000.0 / 700.0),
            "Original frame number": np.arange(row_count, dtype=np.int32),
        }
    )
    for index in range(point_count):
        frame[f"Angle of point {index} (deg)"] = angles[:, index].astype(np.float32)
    return frame


def make_camera_tracking_sync_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Camera/tracking pair with a gap for synchronization checks."""
    tracking = pd.DataFrame(
        {
            "FrameID": [10, 11, 13, 14],
            "Angle of point 0 (deg)": [0.0, 0.1, 0.2, 0.3],
        }
    )
    camera = pd.DataFrame(
        {
            "FrameID": [11, 12, 13, 14, 15],
            "ElapsedTime": [0.0, 1.4, 2.8, 4.2, 5.6],
            "AbsoluteTime": [1000, 1001, 1002, 1003, 1004],
        }
    )
    return tracking, camera


def make_interpolation_fixture() -> pd.DataFrame:
    """Short synchronized table for interpolation equivalence."""
    return pd.DataFrame(
        {
            "FrameID": np.array([0.0, 1.0, 2.0, 3.0]),
            "AbsoluteTime": np.array([0.0, 1.5, 3.0, 4.5]),
            "Angle of point 0 (deg)": np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
            "Angle of point 1 (deg)": np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32),
        }
    )


def make_stimulus_timeline_fixture(
    *,
    row_count: int = 200,
    cs_begin_abs: float = 50.0,
    cs_end_abs: float = 60.0,
    us_begin_abs: float = 120.0,
    us_end_abs: float = 125.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Continuous AbsoluteTime series plus indexed CS/US protocol events."""
    frame = pd.DataFrame(
        {
            TIME_COLUMN: np.arange(row_count, dtype=np.int64),
            "AbsoluteTime": np.arange(row_count, dtype=np.float64),
            "Original frame number": np.arange(row_count, dtype=np.int32),
            "Angle of point 0 (deg)": np.zeros(row_count, dtype=np.float32),
            "Angle of point 1 (deg)": np.linspace(
                0.0,
                1.0,
                row_count,
                dtype=np.float32,
            ),
            "Vigor (deg/ms)": np.full(row_count, 0.2, dtype=np.float32),
        }
    )
    protocol = pd.DataFrame(
        {
            "beg (ms)": [cs_begin_abs, us_begin_abs],
            "end (ms)": [cs_end_abs, us_end_abs],
        },
        index=pd.Index(["Cycle", "Reinforcer"], name="Experiment type"),
    )
    return frame, protocol


def make_presegment_fixture(
    *,
    trial_start_frames: int = -10,
    trial_end_frames: int = 10,
) -> tuple[pd.DataFrame, "LegacyPreprocessingConfig"]:
    """Build an annotated series with known CS/US trial markers for segmentation.

    Uses allDelay-compatible trial numbers (CS=5 in Pre-train, US=18 in Train 1)
    so block assignment can be checked against the migrated experiment map.
    """
    from classical_conditioning.preprocessing.legacy_v1 import (
        LegacyPreprocessingConfig,
    )

    settings = LegacyPreprocessingConfig(
        angle_point_count=2,
        trial_start_frames=trial_start_frames,
        trial_end_frames=trial_end_frames,
        baseline_window_frames=5,
    )
    row_count = 80
    frame = pd.DataFrame(
        {
            TIME_COLUMN: np.arange(row_count, dtype=np.int64),
            "AbsoluteTime": np.arange(row_count, dtype=np.float64),
            "Original frame number": np.arange(row_count, dtype=np.int32),
            "CS beg": np.zeros(row_count, dtype=np.int32),
            "CS end": np.zeros(row_count, dtype=np.int32),
            "US beg": np.zeros(row_count, dtype=np.int32),
            "US end": np.zeros(row_count, dtype=np.int32),
            "Angle of point 0 (deg)": np.zeros(row_count, dtype=np.float32),
            "Angle of point 1 (deg)": np.linspace(
                0.0,
                1.0,
                row_count,
                dtype=np.float32,
            ),
            "Vigor (deg/ms)": np.linspace(0.1, 0.4, row_count, dtype=np.float32),
            "Bout": np.zeros(row_count, dtype=bool),
            "Bout beg": np.zeros(row_count, dtype=bool),
            "Bout end": np.zeros(row_count, dtype=bool),
        }
    )
    # CS trial 5 centered near frame 20; US trial 18 near frame 50.
    frame.loc[20, "CS beg"] = 5
    frame.loc[25, "CS end"] = 5
    frame.loc[50, "US beg"] = 18
    frame.loc[53, "US end"] = 18
    return frame, settings
