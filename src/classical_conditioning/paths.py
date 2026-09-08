"""Relocatable raw/project path helpers."""

from __future__ import annotations

from pathlib import Path

from classical_conditioning.exceptions import ConfigurationError

DERIVED_PROJECT_DIR_NAME = "Paper data"
RESERVED_DERIVED_DIR_NAMES = frozenset(
    {
        DERIVED_PROJECT_DIR_NAME,
        "Processed data",
        "Quality checks",
        "Metadata",
    }
)


def condition_from_recording_name(recording_name: str) -> str:
    """Return the filename condition token (third underscore field), lowercased."""
    parts = recording_name.split("_")
    if len(parts) < 3 or not parts[2].strip():
        raise ConfigurationError(
            "Recording name must include a condition token after date and fish "
            f"number: {recording_name}"
        )
    return parts[2].strip().lower()


def is_reserved_derived_path(path: Path, root: Path) -> bool:
    """True when path sits under a derived folder inside the raw tree."""
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return any(part in RESERVED_DERIVED_DIR_NAMES for part in relative.parts)


def assert_project_dir_allowed(input_dir: Path, project_dir: Path) -> None:
    """Forbid mixing outputs with raw files except a nested Paper data folder."""
    input_dir = input_dir.resolve()
    project_dir = project_dir.resolve()
    if project_dir == input_dir:
        raise ValueError("Project output directory must not be the raw input directory.")
    if input_dir in project_dir.parents:
        relative = project_dir.relative_to(input_dir)
        if relative.parts[0] != DERIVED_PROJECT_DIR_NAME:
            raise ValueError(
                "A project directory inside the raw tree must be named "
                f"{DERIVED_PROJECT_DIR_NAME!r}."
            )


def assert_output_outside_raw_or_in_paper_data(input_dir: Path, output: Path) -> None:
    """Inventory/JSON outputs may live only outside raw or under Paper data."""
    root = input_dir.resolve()
    destination = output.resolve()
    if destination == root:
        raise ConfigurationError(
            "Recording inventory output must not overwrite the raw-data root."
        )
    try:
        relative = destination.relative_to(root)
    except ValueError:
        return
    if relative.parts[0] != DERIVED_PROJECT_DIR_NAME:
        raise ConfigurationError(
            "Recording inventory output must be outside the immutable raw-data "
            f"tree or under {DERIVED_PROJECT_DIR_NAME!r}."
        )
