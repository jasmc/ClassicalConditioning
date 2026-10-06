"""Legacy cadence inference for buffered camera-arrival timestamps.

Presumed acquisition times are a constant-cadence model, not hardware exposure
timestamps. Missing exported FrameIDs and buffer-capacity exceedances are
separate evidence; a brief arrival delay is not itself a lost frame.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CameraCadence:
    interval_ms: float
    reference_position: int
    end_position: int
    reference_frame_id: int
    missing_frame_ids: int
    buffer_capacity_exceeded: bool
    maximum_delay_ms: float

    @property
    def framerate(self) -> float:
        return 1000.0 / self.interval_ms

    @property
    def has_frame_loss_evidence(self) -> bool:
        return self.missing_frame_ids > 0 or self.buffer_capacity_exceeded


def estimate_camera_cadence(camera: pd.DataFrame, *, tolerance_ms: float = .005,
                            buffer_size: int = 700) -> CameraCadence:
    """Estimate cadence between stable arrival runs using actual FrameID span.

    Three consecutive near-median intervals identify each stable anchor, as in
    the original legacy routine. Indexing uses row positions, never FrameID
    arithmetic. Detect missing IDs directly; the historical buffer test detects
    capacity exceedance, not a count of individual lost exposures.
    """
    if tolerance_ms < 0 or buffer_size < 1:
        raise ValueError('Timing tolerance must be nonnegative and buffer size positive')
    ids = camera['FrameID'].to_numpy(dtype=np.int64)
    elapsed = camera['ElapsedTime'].to_numpy(dtype=float)
    if len(ids) < 4 or not np.isfinite(elapsed).all():
        raise ValueError('At least four finite camera timing rows are required')
    steps = np.diff(ids)
    dt = np.diff(elapsed)
    if np.any(steps <= 0) or np.any(dt <= 0):
        raise ValueError('FrameIDs and camera elapsed times must strictly increase')
    # Divide out explicit missing IDs before the initial cadence estimate.
    first_interval = float(np.median(dt / steps))
    good = (steps == 1) & (np.abs(dt - first_interval) <= tolerance_ms)
    runs = np.flatnonzero(good[:-2] & good[1:-1] & good[2:])
    if not len(runs):
        raise ValueError('No three-interval stable camera run; cannot infer acquisition cadence')
    start, end = int(runs[0]), int(runs[-1] + 3)
    interval = float((elapsed[end] - elapsed[start]) / (ids[end] - ids[start]))
    delay = (elapsed - elapsed[start]) - (ids - ids[start]) * interval
    # Before the stable reference, startup timing is not part of the model.
    maximum_delay = float(max(0., np.max(delay[start:])))
    return CameraCadence(interval, start, end, int(ids[start]),
                         int(np.sum(steps - 1)),
                         maximum_delay >= interval * buffer_size, maximum_delay)


def presumed_acquisition_times(camera: pd.DataFrame, cadence: CameraCadence
                               ) -> tuple[np.ndarray, np.ndarray]:
    """Return elapsed and absolute times anchored to the stable reference.

    Do not smooth over frame-loss evidence or silently substitute arrival times.
    The reference's absolute arrival time fixes the epoch, retaining its
    unknown arrival latency and millisecond timestamp precision.
    """
    if cadence.has_frame_loss_evidence:
        raise ValueError('Frame-loss evidence prevents constant-cadence reconstruction')
    offset = (camera.FrameID.to_numpy(dtype=np.int64) - cadence.reference_frame_id) * cadence.interval_ms
    ref = camera.iloc[cadence.reference_position]
    return float(ref.ElapsedTime) + offset, float(ref.AbsoluteTime) + offset
