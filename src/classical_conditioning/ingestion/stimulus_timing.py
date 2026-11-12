"""Authenticated stimulus diagnostics and transactional publication, without behavior runs."""
from __future__ import annotations

from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from classical_conditioning.analysis.protocol_timing import (
    build_timing_evidence, read_stimulus_log, schedule_comparisons, summarize_lags,
)
from classical_conditioning.artifacts import artifact_staging, publish_transaction, sha256_file, write_json_atomic
from classical_conditioning.config.protocol_timing import default_timing_config, load_timing_config, timing_config_from_dict
from classical_conditioning.exceptions import ArtifactIntegrityError, ConfigurationError
from classical_conditioning.paths import assert_project_dir_allowed, is_reserved_derived_path


def _safe_id(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ConfigurationError("Timing output IDs must be path-safe identifiers")
    return value


def _raw_logs(root: Path) -> dict[str, list[Path]]:
    from classical_conditioning.intake import recording_id_from_name
    found = {}
    for path in root.rglob("*stim control.txt"):
        if path.name.startswith(".") or is_reserved_derived_path(path, root):
            continue
        name = path.name.removesuffix("_stim control.txt")
        rid = recording_id_from_name(name)
        found.setdefault(rid, []).append(path.resolve())
    return found


def _camera_evidence(chunks):
    """Observed frame-ID gaps, not arrival-time jitter or inferred physical exposure."""
    bounds, gaps, previous, issues = None, [], None, []
    for chunk in chunks:
        ids = pd.to_numeric(chunk.FrameID, errors="raise").to_numpy(dtype=np.int64)
        times = pd.to_numeric(chunk.AbsoluteTime, errors="raise").to_numpy(dtype=np.int64)
        if not len(ids):
            continue
        bounds = (min(bounds[0], int(times.min())), max(bounds[1], int(times.max()))) if bounds else (int(times.min()), int(times.max()))
        if previous is not None:
            ids = np.r_[previous[0], ids]
            times = np.r_[previous[1], times]
        for i in np.flatnonzero(np.diff(ids) > 1):
            if times[i+1] > times[i]:
                gaps.append((int(times[i]), int(times[i+1])))
        if np.any(np.diff(ids) <= 0) or np.any(np.diff(times) < 0):
            issues.append("camera_clock_or_frame_order_review")
        previous = (int(ids[-1]), int(times[-1]))
    if issues:
        # A min/max span across a reset is not continuous acquisition coverage.
        bounds = None
    return bounds, gaps, sorted(set(issues))


def _processed_markers(project, rid, metadata, raw_hash, protocol_hash, watch):
    """Compare explicitly supplied marker tables or the current corrected outcome family."""
    supplied = metadata.get("processed_markers")
    if supplied:
        path = Path(supplied["path"]).expanduser().resolve()
        if supplied["raw_protocol_sha256"] != raw_hash or watch(path) != supplied["sha256"]:
            raise ArtifactIntegrityError("Processed markers do not authenticate against this raw log")
        markers = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(markers, list):
            raise ValueError("Processed marker JSON must be a list")
        for marker in markers:
            if marker.get("type") not in {"CS", "US"} or not (marker.get("source_row") or marker.get("event_number")):
                raise ValueError("Processed markers require type and source_row or event_number")
            for field in ("beg_ms", "end_ms"):
                if marker.get(field) is not None and (isinstance(marker[field], bool) or not isinstance(marker[field], (int, float)) or not np.isfinite(marker[field])):
                    raise ValueError("Processed marker times must be finite milliseconds")
        return markers, {"label_status": "explicit marker table; labels retained separately", "limitation": "Correspondence uses declared row/number identity, not nearest-time matching."}
    from classical_conditioning.analysis.movement_state import resolve_candidate_metric_source
    source = resolve_candidate_metric_source(metric_recipe="tail-candidate-corrected")
    summary_path = project/"Quality checks"/rid/source.trial_summary_name
    if not summary_path.is_file():
        return None, {"limitation": "Current corrected trial outcomes unavailable; no historical filename fallback."}
    summary = json.loads(summary_path.read_text())
    watch(summary_path)
    marker_path = project/"Metadata"/f"{rid}_{source.trial_recipe}_complete.json"
    marker = json.loads(marker_path.read_text())
    watch(marker_path)
    if marker.get("summary_sha256") != sha256_file(summary_path) or marker.get("status") != "complete" or marker.get("recording_id") != rid or summary.get("recording_id") != rid:
        raise ArtifactIntegrityError("Processed outcome summary/marker identity mismatch")
    if summary.get("inputs", {}).get("protocol") != protocol_hash:
        raise ArtifactIntegrityError("Processed outcomes use a different stimulus table")
    path = project/"Processed data"/rid/source.trial_outcomes_name
    digest = watch(path)
    if summary.get("artifacts", {}).get("outcomes", {}).get("sha256") != digest or marker.get("artifact_sha256", {}).get("outcomes") != digest:
        raise ArtifactIntegrityError("Processed outcome hash mismatch")
    columns = ["alignment", "trial_number", "event_start_absolute_time_ms", "phase", "block_10_name"]
    frame = pq.read_table(path, columns=columns).to_pandas().drop_duplicates()
    markers = [{"type": str(row.alignment), "event_number": int(row.trial_number),
                "beg_ms": int(row.event_start_absolute_time_ms), "phase": row.phase if pd.notna(row.phase) else None,
                "block": row.block_10_name if pd.notna(row.block_10_name) else None} for row in frame.itertuples()]
    from classical_conditioning.config.trial_map import trial_map_sha256
    try:
        current = summary.get("trial_map_sha256") == trial_map_sha256(summary.get("experiment"))
    except ConfigurationError:
        current = False
    return markers, {"label_status": "current trial-map fingerprint" if current else "stale_or_missing_trial_map_fingerprint",
                     "limitation": "Saved outcome anchors originate in stimulus timestamps; equality does not test camera registration. Processed offsets unavailable."}


def _page(title: str, images: list[Path], *, prefix="") -> str:
    return ('<!doctype html><meta charset="utf-8"><title>'+html.escape(title)+'</title>'
            '<style>body{font:15px system-ui;margin:2rem}img{max-width:100%;display:block}figure{margin:2rem 0}</style>'
            '<h1>'+html.escape(title)+'</h1><p>Descriptive timing evidence; no cohort selection or randomness verdict.</p>'
            '<p><a href="evidence.json">Event identities, coverage, findings and provenance</a></p>'
            + ''.join('<figure><figcaption>'+html.escape(p.stem)+'</figcaption><img alt="'+html.escape(p.stem)+'" src="'+html.escape(prefix+p.name, quote=True)+'"></figure>' for p in images))


def run_protocol_timing(project_dir: Path, recording_ids=None, *, experiment_name=None,
                        config_path: Path | None = None, raw_dir: Path | None = None,
                        analysis_id="protocol-timing", expected_recording_ids=()) -> dict:
    """Run standalone or after analysis; raw logs suffice for future protocols."""
    from classical_conditioning.figures.presentation_profiles import load_presentation_profile
    from classical_conditioning.figures.protocol_timing import render_collection, render_recording
    from classical_conditioning.figure_workspace import state_lock
    analysis_id = _safe_id(analysis_id)
    project = Path(project_dir).expanduser().resolve()
    if raw_dir is not None:
        raw_dir = Path(raw_dir).expanduser().resolve()
        if not raw_dir.is_dir():
            raise FileNotFoundError(raw_dir)
        assert_project_dir_allowed(raw_dir, project)
    config_digest = sha256_file(config_path) if config_path else None
    config = load_timing_config(config_path) if config_path else default_timing_config(experiment_name) if experiment_name else timing_config_from_dict({"experiment_id": "unassigned", "phases": [], "conditions": {}})
    if experiment_name and config.experiment_id != experiment_name:
        raise ConfigurationError("Timing config experiment differs from requested experiment")
    profile = load_presentation_profile("nature-main")
    inputs = {}

    def watch(path):
        path = Path(path).resolve()
        digest = sha256_file(path)
        if str(path) in inputs and inputs[str(path)] != digest:
            raise ArtifactIntegrityError("Timing input changed during authentication")
        inputs[str(path)] = digest
        return digest

    if config_path:
        if watch(config_path) != config_digest:
            raise ArtifactIntegrityError("Timing configuration changed while loading")
    implementation = {}
    for module in (Path(__file__), Path(__file__).parents[1]/"analysis/protocol_timing.py", Path(__file__).parents[1]/"config/protocol_timing.py", Path(__file__).parents[1]/"figures/protocol_timing.py"):
        implementation[str(module)] = watch(module)
    for relative, digest in profile.inputs.items():
        from classical_conditioning.figures.presentation_profiles import ROOT
        path = ROOT/relative
        if watch(path) != digest:
            raise ArtifactIntegrityError("Presentation contract changed during resolution")
    raw = _raw_logs(raw_dir) if raw_dir else {}
    manifest_paths = {p.name.removesuffix("_source_manifest.json"): p for p in (project/"Metadata").glob("*_source_manifest.json")}
    selected = tuple(dict.fromkeys([*(recording_ids if recording_ids is not None else sorted(set(raw)|set(manifest_paths))), *expected_recording_ids]))
    if not selected:
        raise ConfigurationError("No recordings selected for stimulus timing")
    for rid in selected:
        _safe_id(rid)
    summary_path = project/"Metadata"/f"{analysis_id}_stimulus_timing.json"
    roots = [project/"Quality checks"/rid/"stimulus-timing"/analysis_id for rid in selected]
    collection_target = project/"Figures"/"Stimulus timing"/analysis_id
    roots.append(collection_target)
    starting = {str(p): sha256_file(p) for root in roots if root.exists() for p in root.rglob("*") if p.is_file()}
    previous = None
    if summary_path.exists():
        previous = json.loads(summary_path.read_text())
        starting[str(summary_path)] = sha256_file(summary_path)
        for name, digest in previous.get("outputs", {}).items():
            path = Path(name)
            if path.exists() and sha256_file(path) != digest:
                raise ArtifactIntegrityError(f"Preserving manually changed timing artifact: {path}")
    reports, errors, units = [], [], []
    project.mkdir(parents=True, exist_ok=True)
    with artifact_staging(project, prefix=".stimulus-timing-") as staging:
        for rid in selected:
            metadata = (config.recordings or {}).get(rid, {})
            condition, day = metadata.get("condition"), metadata.get("day")
            try:
                manifest = None
                if rid in manifest_paths:
                    watch(manifest_paths[rid])
                    manifest = json.loads(manifest_paths[rid].read_text())
                    if manifest.get("recording_id") != rid:
                        raise ArtifactIntegrityError("Intake recording identity differs")
                candidates = raw.get(rid, [])
                if len(candidates) > 1:
                    raise ConfigurationError(f"Ambiguous raw stimulus sources for {rid}; select an exact source directory")
                path = candidates[0] if candidates else None
                if path is None and manifest:
                    source = manifest["sources"]["protocol"]
                    path = Path(source["path"]).expanduser()
                    if not path.is_file():
                        from classical_conditioning.external_artifacts import resolve_artifact
                        path = resolve_artifact(source["path"], expected_sha256=source["sha256"])
                if path is None:
                    raise FileNotFoundError(f"Authentic stimulus log unavailable for {rid}")
                raw_hash = watch(path)
                if manifest and raw_hash != manifest["sources"]["protocol"]["sha256"]:
                    raise ArtifactIntegrityError("Raw stimulus source differs from intake manifest")
                rows = read_stimulus_log(path)
                name = manifest["recording_name"] if manifest else path.name.removesuffix("_stim control.txt")
                parts = name.split("_")
                day = day or (parts[0] if re.fullmatch(r"\d{8}", parts[0]) else None)
                token = parts[2].lower() if len(parts) > 2 else ""
                if condition is None:
                    condition = next((name for name, c in config.conditions.items() if token in {name.lower(), *(t.lower() for t in c.get("aliases", []))}), None)
                bounds, gaps, camera_issues, protocol_hash = None, [], [], None
                if manifest:
                    data = project/"Processed data"/rid
                    for kind, filename in (("protocol", "stimulus_events.parquet"), ("camera", "camera.parquet")):
                        entry = manifest["artifacts"][kind]
                        canonical = (data/filename).resolve()
                        if Path(entry["path"]).resolve() != canonical or watch(canonical) != entry["sha256"]:
                            raise ArtifactIntegrityError(f"Canonical {kind} intake artifact differs from manifest")
                    protocol_hash = manifest["artifacts"]["protocol"]["sha256"]
                    table = pq.read_table(data/"stimulus_events.parquet").to_pandas()
                    # Compare numeric rows separately from source order/text; malformed rows remain diagnostic evidence.
                    from classical_conditioning.analysis.protocol_timing import _milliseconds
                    numeric = sorted((r["raw_type"], _milliseconds(r["raw_beg"]), _milliseconds(r["raw_end"])) for r in rows if _milliseconds(r["raw_beg"]) is not None and _milliseconds(r["raw_end"]) is not None)
                    saved = sorted((str(r.Type), int(r.Beg), int(r.End)) for r in table.itertuples())
                    if len(numeric) != len(rows) or numeric != saved:
                        raise ArtifactIntegrityError("Raw and intake stimulus event values differ")
                    camera = pq.ParquetFile(data/"camera.parquet")
                    try:
                        bounds, gaps, camera_issues = _camera_evidence(batch.to_pandas() for batch in camera.iter_batches(columns=["FrameID", "AbsoluteTime"], batch_size=250000))
                    except (ValueError, KeyError, TypeError) as error:
                        camera_issues = [f"camera_coverage_unavailable: {error}"]
                elif raw_dir:
                    camera_path = path.with_name(name+"_cam.txt")
                    if camera_path.is_file():
                        watch(camera_path)
                        try:
                            bounds, gaps, camera_issues = _camera_evidence(pd.read_csv(camera_path, sep=r"\s+", usecols=["FrameID", "AbsoluteTime"], chunksize=250000))
                        except (ValueError, KeyError, TypeError) as error:
                            camera_issues = [f"camera_coverage_unavailable: {error}"]
                try:
                    markers, processed_status = _processed_markers(project, rid, metadata, raw_hash, protocol_hash, watch)
                except (ArtifactIntegrityError, ValueError, OSError, KeyError) as error:
                    markers, processed_status = None, {"status": "integrity_failure", "limitation": str(error)}
                report = build_timing_evidence(rows, config, recording_id=rid, condition=condition, day=day,
                    acquisition_bounds_ms=bounds, camera_gaps_ms=gaps, processed_markers=markers, processed_status=processed_status)
                report["source"] = {"path": str(path), "sha256": raw_hash, "intake_protocol_sha256": protocol_hash}
                report["camera_findings"] = camera_issues
                if camera_issues:
                    report["status"] = "review"
                if processed_status.get("status") == "integrity_failure":
                    report["status"] = "review"
                    report["findings"].append({"code": "processed_alignment_integrity_failure", "detail": processed_status["limitation"]})
                target = project/"Quality checks"/rid/"stimulus-timing"/analysis_id
                staged = staging/rid
                images = render_recording(report, config.to_dict(), staged, profile=profile)
                page = staged/"index.html"
                page.write_text(_page(rid+" stimulus timing", images), encoding="utf-8")
                report["outputs"] = {p.name: sha256_file(p) for p in [*images, page]}
                evidence = staged/"evidence.json"
                write_json_atomic(evidence, report)
                units.extend((p, target/p.name) for p in [*images, page, evidence])
                reports.append(report)
            except Exception as error:
                errors.append({"recording_id": rid, "error": f"{type(error).__name__}: {error}"})
                placeholder = build_timing_evidence([], config, recording_id=rid, condition=condition, day=day)
                placeholder.update(status="unavailable", unavailable_reason=errors[-1]["error"])
                reports.append(placeholder)
        successful = [r for r in reports if r["status"] != "unavailable"]
        comparisons = schedule_comparisons(reports)
        collection = staging/"collection"
        collection.mkdir()
        grouped = summarize_lags(reports)
        images = render_collection(grouped, comparisons, collection, profile=profile)
        for index, group in enumerate(grouped):
            group["figure"] = f"lags-{index:03d}.png"
        collection_evidence = {"groups": grouped, "schedule_comparisons": comparisons,
            "recordings": [{k: r[k] for k in ("recording_id", "condition_id", "day", "batch_id", "status")} for r in reports], "errors": errors}
        write_json_atomic(collection/"evidence.json", collection_evidence)
        (collection/"index.html").write_text(_page(config.experiment_id+" stimulus timing", images)
            + '<h2>Recording views</h2><ul>' + ''.join('<li>'+html.escape(r["recording_id"])+': '+html.escape(r["status"])+
                (' · <a href="../../../Quality%20checks/'+r["recording_id"]+'/stimulus-timing/'+analysis_id+'/index.html">plots and evidence</a>' if r["status"] != "unavailable" else ' · '+html.escape(r.get("unavailable_reason", "")))+'</li>' for r in reports)+'</ul>', encoding="utf-8")
        units.extend((p, collection_target/p.name) for p in [*images, collection/"index.html", collection/"evidence.json"])
        outputs = {str(target): sha256_file(staged) for staged, target in units}
        payload = {"schema_version": 1, "artifact_kind": "stimulus-timing-run", "analysis_id": analysis_id,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_id": config.experiment_id, "configuration": config.to_dict(),
            "status": "complete" if not errors else "partial" if successful else "failed",
            "recording_ids": list(selected), "completed": [r["recording_id"] for r in successful], "errors": errors,
            "inputs": inputs, "implementation": implementation, "presentation_contract": profile.inputs,
            "outputs": outputs, "collection": str(collection_target/"index.html"), "summary_path": str(summary_path)}
        staged_summary = staging/"summary.json"
        write_json_atomic(staged_summary, payload)
        summary_digest = sha256_file(staged_summary)
        units.append((staged_summary, summary_path))
        (project/"Metadata").mkdir(exist_ok=True)
        with state_lock(project/"Metadata"/"stimulus-timing-publication"):
            for name, digest in starting.items():
                if not Path(name).is_file() or sha256_file(Path(name)) != digest:
                    raise ArtifactIntegrityError("Timing output changed during rendering; preserving concurrent work")
            for _, target in units:
                if target.exists() and str(target) not in starting:
                    raise ArtifactIntegrityError(f"Timing output appeared during rendering: {target}")
                if target.exists() and previous is None:
                    raise ArtifactIntegrityError(f"Unmanaged timing artifact already exists: {target}")
                if target.exists() and target != summary_path and str(target) not in previous.get("outputs", {}):
                    raise ArtifactIntegrityError(f"Preserving unowned timing artifact: {target}")
            for name, digest in inputs.items():
                if not Path(name).is_file() or sha256_file(Path(name)) != digest:
                    raise ArtifactIntegrityError(f"Timing input changed before publication: {name}")
            def verify_published():
                for name, digest in {**inputs, **outputs, str(summary_path): summary_digest}.items():
                    if not Path(name).is_file() or sha256_file(Path(name)) != digest:
                        raise ArtifactIntegrityError(f"Timing publication verification failed: {name}")
            publish_transaction(tuple(units), staging, overwrite=True, verify_published=verify_published)
    return payload
