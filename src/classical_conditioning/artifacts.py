"""Local artifact integrity and transactional publication helpers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from classical_conditioning.exceptions import (
    ArtifactIntegrityError,
    ArtifactNotFoundError,
)


@dataclass(frozen=True)
class VerifiedArtifact:
    data_path: Path
    summary_path: Path
    marker_path: Path
    marker: dict[str, Any]
    summary: dict[str, Any]
    data_state: tuple[int, int]


@dataclass(frozen=True)
class VerifiedArtifactSet:
    data_paths: dict[str, Path]
    summary_path: Path
    marker_path: Path
    marker: dict[str, Any]
    summary: dict[str, Any]
    data_states: dict[str, tuple[int, int]]


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".incomplete",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def artifact_staging(parent: Path, *, prefix: str = ".staging-") -> Iterator[Path]:
    """Yield a staging directory whose files inherit the destination ACL.

    Windows ``tempfile.mkdtemp`` hardens the directory with a non-inheritable
    DACL, and ``os.replace`` carries that DACL onto every published artifact,
    leaving artifacts readable only by the account that wrote them.
    """
    parent.mkdir(parents=True, exist_ok=True)
    staging_root = parent / f"{prefix}{uuid.uuid4().hex}"
    staging_root.mkdir()
    try:
        yield staging_root
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)


def _remove_artifact(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def publish_transaction(
    units: tuple[tuple[Path, Path], ...],
    staging_root: Path,
    *,
    overwrite: bool,
    removals: tuple[Path, ...] = (),
) -> None:
    """Publish related artifacts with rollback if any replacement fails."""
    if not units:
        raise ValueError("Artifact publication requires at least one unit.")
    staged_paths = [staged.resolve() for staged, _ in units]
    final_paths = [final.resolve() for _, final in units]
    removal_paths = [path.resolve() for path in removals]
    if len(staged_paths) != len(set(staged_paths)):
        raise ValueError("Artifact publication contains duplicate staged paths.")
    if len(final_paths) != len(set(final_paths)):
        raise ValueError("Artifact publication contains duplicate final paths.")
    if len(removal_paths) != len(set(removal_paths)):
        raise ValueError("Artifact publication contains duplicate removal paths.")
    final_removal_overlap = set(final_paths).intersection(removal_paths)
    if final_removal_overlap:
        raise ValueError(
            "Artifact paths cannot be both published and removed: "
            f"{sorted(final_removal_overlap)}"
        )
    overlapping = set(staged_paths).intersection(
        {*final_paths, *removal_paths}
    )
    if overlapping:
        raise ValueError(
            "Staged and final artifact paths must be distinct: "
            f"{sorted(overlapping)}"
        )
    missing = [path for path in staged_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Staged artifacts are missing: {missing}")

    backup_root = staging_root.with_name(f"{staging_root.name}-backups")
    backup_root.mkdir()
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        destinations = [final_path for _, final_path in units]
        destinations.extend(removals)
        for index, final_path in enumerate(destinations):
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if final_path.exists():
                if final_path in removals and not overwrite:
                    raise FileExistsError(
                        f"Artifact removal requires overwrite: {final_path}"
                    )
                if not overwrite:
                    raise FileExistsError(f"Derived artifact exists: {final_path}")
                backup_path = backup_root / f"{index}-{final_path.name}"
                os.replace(final_path, backup_path)
                backups.append((final_path, backup_path))

        for staged_path, final_path in units:
            os.replace(staged_path, final_path)
            published.append(final_path)
    except Exception as publish_error:
        rollback_errors: list[str] = []
        for final_path in reversed(published):
            try:
                _remove_artifact(final_path)
            except OSError as error:
                rollback_errors.append(f"remove {final_path}: {error}")
        for final_path, backup_path in reversed(backups):
            try:
                if final_path.exists():
                    _remove_artifact(final_path)
                os.replace(backup_path, final_path)
            except OSError as error:
                rollback_errors.append(
                    f"restore {backup_path} to {final_path}: {error}"
                )
        if rollback_errors:
            details = "; ".join(rollback_errors)
            raise RuntimeError(
                "Artifact publication failed and rollback was incomplete. "
                f"Backups are preserved at {backup_root}. Details: {details}"
            ) from publish_error
        shutil.rmtree(backup_root)
        raise
    else:
        shutil.rmtree(backup_root)


def load_and_verify_source_manifest(
    project_dir: Path,
    recording_id: str,
) -> tuple[str, dict[str, dict[str, Any]], dict[str, tuple[int, int]]]:
    manifest_path = project_dir / "Metadata" / f"{recording_id}_source_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("recording_id") != recording_id:
        raise ValueError(f"Source manifest identity mismatch: {manifest_path}")
    expected_paths = {
        "camera": project_dir / "Processed data" / recording_id / "camera.parquet",
        "tracking": project_dir / "Processed data" / recording_id / "tracking.parquet",
        "protocol": project_dir
        / "Processed data"
        / recording_id
        / "stimulus_events.parquet",
    }
    artifact_records: dict[str, dict[str, Any]] = {}
    file_state: dict[str, tuple[int, int]] = {}
    for kind, expected_path in expected_paths.items():
        record = manifest["artifacts"][kind]
        recorded_path = Path(record["path"]).resolve()
        expected_path = expected_path.resolve()
        if recorded_path != expected_path:
            raise ValueError(
                f"{kind} artifact path differs from the source manifest: "
                f"{recorded_path} != {expected_path}"
            )
        stat = expected_path.stat()
        if sha256_file(expected_path) != record["sha256"]:
            raise ValueError(f"{kind} artifact hash differs from the source manifest.")
        artifact_records[kind] = record
        file_state[kind] = (stat.st_size, stat.st_mtime_ns)
    return str(manifest["recording_name"]), artifact_records, file_state


def verify_completed_parquet(
    data_path: Path,
    summary_path: Path,
    marker_path: Path,
    *,
    recipe: str,
    recording_id: str,
) -> VerifiedArtifact:
    paths = (data_path, summary_path, marker_path)
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise ArtifactNotFoundError(f"Missing completed artifacts: {missing}")

    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactIntegrityError(
            f"Completed artifact metadata is invalid JSON: {error}"
        ) from error

    data_hash = sha256_file(data_path)
    if (
        marker.get("status") != "complete"
        or marker.get("recipe") != recipe
        or marker.get("recording_id") != recording_id
        or summary.get("recipe") != recipe
        or summary.get("recording_id") != recording_id
        or marker.get("samples_sha256") != data_hash
        or marker.get("summary_sha256") != sha256_file(summary_path)
        or summary.get("artifact", {}).get("sha256") != data_hash
    ):
        raise ArtifactIntegrityError(
            f"Completed {recipe} lineage is invalid for {recording_id}."
        )

    stat = data_path.stat()
    return VerifiedArtifact(
        data_path=data_path,
        summary_path=summary_path,
        marker_path=marker_path,
        marker=marker,
        summary=summary,
        data_state=(stat.st_size, stat.st_mtime_ns),
    )


def verify_completed_parquet_set(
    expected_paths: dict[str, Path],
    summary_path: Path,
    marker_path: Path,
    *,
    recipe: str,
    recording_id: str,
) -> VerifiedArtifactSet:
    metadata_paths = (summary_path, marker_path)
    missing_metadata = [path for path in metadata_paths if not path.is_file()]
    if missing_metadata:
        raise ArtifactNotFoundError(
            f"Missing completed artifact metadata: {missing_metadata}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactIntegrityError(
            f"Completed artifact metadata is invalid JSON: {error}"
        ) from error

    summary_artifacts = summary.get("artifacts")
    marker_hashes = marker.get("artifact_sha256")
    if not isinstance(summary_artifacts, dict) or not summary_artifacts:
        raise ArtifactIntegrityError(f"Completed {recipe} artifact set is empty.")
    if not isinstance(marker_hashes, dict):
        raise ArtifactIntegrityError(
            f"Completed {recipe} marker has no artifact hash map."
        )
    artifact_keys = set(summary_artifacts)
    if artifact_keys != set(marker_hashes) or not artifact_keys.issubset(
        expected_paths
    ):
        raise ArtifactIntegrityError(
            f"Completed {recipe} artifact identities are inconsistent."
        )
    if (
        marker.get("status") != "complete"
        or marker.get("recipe") != recipe
        or marker.get("recording_id") != recording_id
        or summary.get("recipe") != recipe
        or summary.get("recording_id") != recording_id
        or marker.get("summary_sha256") != sha256_file(summary_path)
    ):
        raise ArtifactIntegrityError(
            f"Completed {recipe} lineage is invalid for {recording_id}."
        )

    verified_paths: dict[str, Path] = {}
    states: dict[str, tuple[int, int]] = {}
    for key in sorted(artifact_keys):
        expected_path = expected_paths[key].resolve()
        if not expected_path.is_file():
            raise ArtifactNotFoundError(
                f"Missing completed {recipe} artifact: {expected_path}"
            )
        record = summary_artifacts[key]
        if Path(record.get("path", "")).resolve() != expected_path:
            raise ArtifactIntegrityError(
                f"Completed {recipe} artifact path mismatch for {key}."
            )
        digest = sha256_file(expected_path)
        if record.get("sha256") != digest or marker_hashes.get(key) != digest:
            raise ArtifactIntegrityError(
                f"Completed {recipe} artifact hash mismatch for {key}."
            )
        stat = expected_path.stat()
        verified_paths[key] = expected_path
        states[key] = (stat.st_size, stat.st_mtime_ns)

    return VerifiedArtifactSet(
        data_paths=verified_paths,
        summary_path=summary_path,
        marker_path=marker_path,
        marker=marker,
        summary=summary,
        data_states=states,
    )


def verify_completed_analysis_parquet_set(
    expected_paths: dict[str, Path],
    summary_path: Path,
    marker_path: Path,
    *,
    recipe: str,
    analysis_id: str,
    recording_ids: tuple[str, ...],
    alignment: str | None = None,
) -> VerifiedArtifactSet:
    """Verify a cohort artifact set identified by analysis rather than recording."""
    metadata_paths = (summary_path, marker_path)
    missing_metadata = [path for path in metadata_paths if not path.is_file()]
    if missing_metadata:
        raise ArtifactNotFoundError(
            f"Missing completed analysis metadata: {missing_metadata}"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ArtifactIntegrityError(
            f"Completed analysis metadata is invalid JSON: {error}"
        ) from error

    summary_artifacts = summary.get("artifacts")
    marker_hashes = marker.get("artifact_sha256")
    if not isinstance(summary_artifacts, dict) or not summary_artifacts:
        raise ArtifactIntegrityError(f"Completed {recipe} analysis artifact set is empty.")
    if not isinstance(marker_hashes, dict):
        raise ArtifactIntegrityError(
            f"Completed {recipe} analysis marker has no artifact hash map."
        )
    artifact_keys = set(summary_artifacts)
    if artifact_keys != set(marker_hashes) or not artifact_keys.issubset(
        expected_paths
    ):
        raise ArtifactIntegrityError(
            f"Completed {recipe} analysis artifact identities are inconsistent."
        )

    expected_recording_ids = list(recording_ids)
    identity_matches = (
        marker.get("status") == "complete"
        and marker.get("recipe") == recipe
        and marker.get("analysis_id") == analysis_id
        and summary.get("recipe") == recipe
        and summary.get("analysis_id") == analysis_id
        and marker.get("recording_ids") == expected_recording_ids
        and summary.get("recording_ids") == expected_recording_ids
        and marker.get("summary_sha256") == sha256_file(summary_path)
    )
    if alignment is not None:
        identity_matches = (
            identity_matches
            and marker.get("alignment") == alignment
            and summary.get("alignment") == alignment
        )
    if not identity_matches:
        raise ArtifactIntegrityError(
            f"Completed {recipe} analysis lineage is invalid for {analysis_id}."
        )

    verified_paths: dict[str, Path] = {}
    states: dict[str, tuple[int, int]] = {}
    for key in sorted(artifact_keys):
        expected_path = expected_paths[key].resolve()
        if not expected_path.is_file():
            raise ArtifactNotFoundError(
                f"Missing completed {recipe} analysis artifact: {expected_path}"
            )
        record = summary_artifacts[key]
        if Path(record.get("path", "")).resolve() != expected_path:
            raise ArtifactIntegrityError(
                f"Completed {recipe} analysis artifact path mismatch for {key}."
            )
        digest = sha256_file(expected_path)
        if record.get("sha256") != digest or marker_hashes.get(key) != digest:
            raise ArtifactIntegrityError(
                f"Completed {recipe} analysis artifact hash mismatch for {key}."
            )
        stat = expected_path.stat()
        verified_paths[key] = expected_path
        states[key] = (stat.st_size, stat.st_mtime_ns)

    return VerifiedArtifactSet(
        data_paths=verified_paths,
        summary_path=summary_path,
        marker_path=marker_path,
        marker=marker,
        summary=summary,
        data_states=states,
    )
