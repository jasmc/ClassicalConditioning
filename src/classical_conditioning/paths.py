"""Relocatable raw/project path helpers.

Review note: these guards implement the raw-data immutability boundary used by
commands that read a raw directory and write a separate derived-data project.
"""

from __future__ import annotations

from pathlib import Path

from classical_conditioning.exceptions import ConfigurationError

# The sole permitted derived folder when it must sit inside a raw-data tree.
DERIVED_PROJECT_DIR_NAME = "Paper data"
# These names identify directories containing generated files, never raw input.
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
    # Recording names encode date, fish number, then condition; retain only the
    # normalised condition token so filenames can drive coarse pipeline filters.
    parts = recording_name.split("_")
    if len(parts) < 3 or not parts[2].strip():
        raise ConfigurationError(
            "Recording name must include a condition token after date and fish "
            f"number: {recording_name}"
        )
    return parts[2].strip().lower()


def is_reserved_derived_path(path: Path, root: Path) -> bool:
    """True when path sits under a derived folder inside the raw tree."""
    # A path outside root cannot be a derived subdirectory of that root.
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    # Any reserved component, not only the first, makes this a generated path.
    return any(part in RESERVED_DERIVED_DIR_NAMES for part in relative.parts)


def assert_project_dir_allowed(input_dir: Path, project_dir: Path) -> None:
    """Forbid mixing outputs with raw files except a nested Paper data folder."""
    # Resolve aliases/relative paths before comparing containment relationships.
    input_dir = input_dir.resolve()
    project_dir = project_dir.resolve()
    # Writing directly into raw input would make raw and derived data indistinct.
    if project_dir == input_dir:
        raise ValueError("Project output directory must not be the raw input directory.")
    # A nested project is allowed only under the explicitly named safe folder.
    if input_dir in project_dir.parents:
        relative = project_dir.relative_to(input_dir)
        if relative.parts[0] != DERIVED_PROJECT_DIR_NAME:
            raise ValueError(
                "A project directory inside the raw tree must be named "
                f"{DERIVED_PROJECT_DIR_NAME!r}."
            )


def assert_output_outside_raw_or_in_paper_data(input_dir: Path, output: Path) -> None:
    """Inventory/JSON outputs may live only outside raw or under Paper data."""
    # This is the equivalent guard for one-off inventory/JSON output paths.
    root = input_dir.resolve()
    destination = output.resolve()
    if destination == root:
        raise ConfigurationError(
            "Recording inventory output must not overwrite the raw-data root."
        )
    # An output outside raw input is always safe with respect to this rule.
    try:
        relative = destination.relative_to(root)
    except ValueError:
        return
    # An in-tree result belongs only below the designated derived-data folder.
    if relative.parts[0] != DERIVED_PROJECT_DIR_NAME:
        raise ConfigurationError(
            "Recording inventory output must be outside the immutable raw-data "
            f"tree or under {DERIVED_PROJECT_DIR_NAME!r}."
        )
