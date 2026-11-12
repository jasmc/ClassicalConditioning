"""Append-only evidence sidecars. These functions never return analysis inputs.

Only metadata and copies of tables are accepted; no membership policy, cohort
writer or filtering callback is exposed. A stable pointer selects an immutable
snapshot, so later assessments cannot destroy evidence cited by a saved run.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from classical_conditioning.artifacts import (
    artifact_staging, publish_transaction, sha256_file, write_json_atomic,
)
from classical_conditioning.exceptions import ArtifactIntegrityError


def publish_observation(
    project_dir: Path, output_dir: Path, *, tables: dict[str, pd.DataFrame],
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Publish an authenticated snapshot and return only its evidence reference."""
    with artifact_staging(project_dir, prefix=".observation-") as stage:
        hashes = {}
        for name, frame in tables.items():
            if not name.replace("-", "").replace("_", "").isalnum():
                raise ValueError("Invalid observation table name")
            frame.to_parquet(stage / f"{name}.parquet", index=False)
            frame.to_csv(stage / f"{name}.csv", index=False, lineterminator="\n")
            for suffix in ("parquet", "csv"):
                filename = f"{name}.{suffix}"
                hashes[filename] = sha256_file(stage / filename)
        content = {
            "schema": "observation-only/1", "mode": "observation_only",
            "membership_decisions_applied": False, "identity": identity,
            "implementation_sha256": sha256_file(Path(__file__)),
            "artifacts": hashes,
        }
        digest = hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False).encode()).hexdigest()
        destination = output_dir / "evidence" / digest
        summary_path = destination / "observation.json"
        write_json_atomic(stage / "observation.json", content)
        if destination.exists():
            if (not summary_path.is_file()
                    or json.loads(summary_path.read_text()) != content
                    or any(not (destination / name).is_file()
                           or sha256_file(destination / name) != value
                           for name, value in hashes.items())):
                raise ArtifactIntegrityError(f"Observation snapshot was changed: {destination}")
        else:
            publish_transaction(
                tuple((stage / name, destination / name) for name in (*hashes, "observation.json")),
                stage, overwrite=False,
            )
        reference = {"status": "complete", "mode": "observation_only",
                     "path": str(summary_path), "sha256": sha256_file(summary_path),
                     "evidence_id": digest}
        write_json_atomic(output_dir / "observation-current.json", reference)
    return reference


def try_observation(operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """An observer failure is visible but cannot change scientific execution."""
    try:
        return operation()
    except Exception as error:
        return {"status": "failed", "mode": "observation_only",
                "error": f"{type(error).__name__}: {error}"}


def observe_tables(project_dir, output_dir, *, tables, identity):
    """Record already-calculated eligibility or diagnostics without blocking it."""
    return try_observation(lambda: publish_observation(
        project_dir, output_dir, tables=tables, identity=identity,
    ))


def observe_contributions(
    project_dir: Path, output_dir: Path, *, candidates: pd.DataFrame,
    contributors: pd.DataFrame, keys: list[str], consumer: str,
    identity: dict[str, Any], reasons: pd.Series | None = None,
) -> dict[str, Any]:
    """Describe actual consumer membership by joining copies, never by refiltering.

    Callers supply the rows they actually used. If no precise reason is known,
    absence is reported as such rather than inventing a biological exclusion.
    """
    before = candidates.loc[:, keys].copy()
    if before.duplicated(keys).any() or contributors.duplicated(keys).any():
        raise ValueError("Contribution keys must identify unique observations")
    if reasons is not None:
        before["reason"] = reasons.to_numpy()
    after = contributors.loc[:, keys].assign(contributed=True)
    if not after.merge(before[keys], how="left", indicator=True)["_merge"].eq("both").all():
        raise ValueError("Consumer returned observations outside its input")
    rows = before.merge(after, on=keys, how="left", validate="one_to_one")
    rows["contributed"] = rows["contributed"].eq(True)
    if "reason" not in rows:
        rows["reason"] = "not_returned_by_existing_consumer"
    rows.loc[rows["contributed"], "reason"] = ""
    rows["consumer"] = consumer
    fish_keys = [key for key in ("experiment_id", "recording_id", "fish_id", "condition_id") if key in rows]
    counts = rows.groupby(fish_keys, dropna=False, observed=True)["contributed"].agg(
        input_observations="size", contributing_observations="sum",
    ).reset_index()
    return publish_observation(project_dir, output_dir,
                               tables={"contributions": rows, "fish-counts": counts},
                               identity={**identity, "consumer": consumer})
