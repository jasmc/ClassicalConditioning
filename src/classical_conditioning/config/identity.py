"""Stable fish and recording identity helpers.

Review note: a recording ID identifies an acquisition; FishKey carries the
experiment plus the stable biological day/fish identity used across analyses.
"""

from __future__ import annotations

import re

from classical_conditioning.config.domain import FishKey
from classical_conditioning.exceptions import ConfigurationError

# Accept only the authenticated ID shape, excluding condition/alignment tokens.
_RECORDING_ID_PATTERN = re.compile(r"^(?P<day>\d{8})_(?P<fish_number>\d+)$")


def fish_key_from_recording_id(
    recording_id: str,
    *,
    experiment_id: str,
) -> FishKey:
    """Build the biological fish key from an authenticated recording ID.

    Condition and alignment are never part of the fish key. The recording ID is
    the date and fish number only (`YYYYMMDD_NN`).
    """
    # Reject missing identity components before parsing, producing clear errors.
    if not isinstance(recording_id, str) or not recording_id.strip():
        raise ConfigurationError("Recording ID cannot be empty.")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ConfigurationError("Experiment ID cannot be empty.")

    # Full matching prevents accepting a valid-looking fragment with extras.
    match = _RECORDING_ID_PATTERN.fullmatch(recording_id.strip())
    if match is None:
        raise ConfigurationError(
            "Recording ID must be YYYYMMDD_NN "
            f"(date and fish number only); got {recording_id!r}."
        )
    # Delegate final non-empty-field validation to the immutable domain object.
    return FishKey(
        experiment_id=experiment_id.strip(),
        day=match.group("day"),
        fish_number=match.group("fish_number"),
    )


def recording_id_from_fish_key(fish_key: FishKey) -> str:
    # This is the inverse naming convention used for derived per-recording paths.
    return f"{fish_key.day}_{fish_key.fish_number}"
