"""Paper activity metric frozen by the author on 2026-10-06.

Keep the storage identifier stable for authenticated existing artifacts.
Alternative metric kernels remain available for explicit sensitivity analyses.
"""

PAPER_METRIC_ID = "legacy_distal_angular_speed"
PAPER_METRIC_NAME = "Tail bend angular speed"
PAPER_METRIC_UNITS = "rad/ms"
PAPER_METRIC_DECISION = "tail-bend-angular-speed-2026-10-06"


def require_paper_metric(metric_id: str) -> None:
    """Prevent manuscript rendering from silently using a sensitivity metric."""
    if metric_id != PAPER_METRIC_ID:
        raise ValueError(
            f"The frozen paper metric is {PAPER_METRIC_NAME} ({PAPER_METRIC_ID}); "
            "use a dedicated comparison/sensitivity command for other metrics."
        )
