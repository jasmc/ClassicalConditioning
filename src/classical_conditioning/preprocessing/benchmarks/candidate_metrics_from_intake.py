"""Direct-intake benchmark for comparing against corrected-frame metrics.

This is deliberately not the routine pipeline path.  It preserves the
historical/development calculation that derives metrics straight from raw
intake, so its output can be compared with the corrected-frame analysis.

Review note: this thin wrapper fixes the input provenance to intake artifacts;
the shared kernel owns metric calculations and output authentication.
"""

from __future__ import annotations

from pathlib import Path

# Re-export shared definitions while exposing only the direct-intake writer here.
from classical_conditioning.preprocessing.candidate_metric_kernel import (
    CANDIDATE_COLUMNS,
    CandidateMetricConfig,
    CandidateMetricResult,
    build_direct_intake_candidate_metrics,
    calculate_candidate_metrics,
)

# Public benchmark API: routine callers should use corrected-frame preprocessing.
__all__ = [
    "CANDIDATE_COLUMNS",
    "CandidateMetricConfig",
    "CandidateMetricResult",
    "build_candidate_activity_metrics",
    "calculate_candidate_metrics",
]


def build_candidate_activity_metrics(
    project_dir: Path,
    recording_id: str,
    *,
    config: CandidateMetricConfig | None = None,
    batch_size: int = 250_000,
    overwrite: bool = False,
) -> CandidateMetricResult:
    """Write the direct-intake benchmark metrics for one recording."""
    # Delegate unchanged options to the explicitly named benchmark implementation.
    return build_direct_intake_candidate_metrics(
        project_dir,
        recording_id,
        config=config,
        batch_size=batch_size,
        overwrite=overwrite,
    )
