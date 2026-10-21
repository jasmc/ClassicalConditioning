"""Append reviewed, absent 3sTrace fish without replacing existing artifacts.

First create an immutable plan with --plan-only. Execution requires its exact
SHA-256, rechecks raw file states and protocol hashes, and uses supported stages
with overwrite=False. Conflicting fish remain in the reviewed inventory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from process_trace3_additions import PROJECT, existing_status, parse_protocol, sha
from classical_conditioning.analysis.trial_outcomes import (
    build_candidate_trial_outcomes, parse_recording_identity, verify_candidate_trial_outcomes,
)
from classical_conditioning.intake import RecordingSources, intake_recording
from classical_conditioning.preprocessing.corrected_frame_preprocessing import build_corrected_preprocessing
from classical_conditioning.preprocessing.candidate_metrics_from_corrected_frames import build_candidate_activity_metrics_from_corrected
from classical_conditioning.analysis.movement_state import build_candidate_movement_state
from classical_conditioning.analysis.temporal_profiles import build_candidate_temporal_profiles

REPO = Path(__file__).resolve().parents[1]
META = PROJECT / "Metadata"
REVIEW = META / "trace3-additions-20261018-integrity-review.json"
ALIAS_BLOCKER = "NoMK801 downstream alias mapping not implemented"


def now():
    return datetime.now(timezone.utc).isoformat()


def state(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def save_new(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)


def code_manifest():
    paths = sorted((REPO / "src" / "classical_conditioning").rglob("*.py"))
    paths += [Path(__file__).resolve(), REPO / "scripts" / "process_trace3_additions.py",
              REPO / "scripts" / "finalize_trace3_additions.py"]
    return {str(p): sha(p) for p in paths}


def make_plan(run_id):
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    if review["existing_counts"] != {"absent": 139, "complete-verified": 58}:
        raise ValueError("Reviewed existing-artifact coverage differs; reconcile before planning")
    if review["preserved_metadata_changed"] or review["D_scan_errors"]:
        raise ValueError("Review contains preservation or source-scan errors")
    selected, held = [], []
    for row in review["rows"]:
        if row["existing"]["state"] != "absent":
            continue
        issues = [b for b in row["blockers"] if b != ALIAS_BLOCKER]
        if issues:
            held.append({"recording_id": row["recording_id"], "issues": issues})
            continue
        if existing_status(row["recording_id"], False)["state"] != "absent":
            raise ValueError("Previously absent fish acquired artifacts: " + row["recording_id"])
        if len(row["complete_triplets"]) != 1 or len(row["protocols"]) != 1:
            raise ValueError("Ambiguous source passed review selection")
        sources = row["complete_triplets"][0]
        protocol_path = Path(sources["protocol"])
        name = protocol_path.name[:-len("_stim control.txt")]
        identity = parse_recording_identity(name, experiment_name="all3sTrace")
        if identity["recording_id"] != row["recording_id"] or identity["condition_id"] != row["condition"]:
            raise ValueError("Canonical source mapping mismatch")
        if row["condition"] == "control" and not row.get("matched_fixed_trace_day_and_directory"):
            raise ValueError("Control lacks reviewed acquisition provenance")
        protocol = parse_protocol(protocol_path.read_bytes())
        if protocol["sha256"] != row["protocols"][0]["sha256"]:
            raise ValueError("Protocol changed since review")
        selected.append({"recording_id": row["recording_id"], "condition": row["condition"],
                         "recording_name": name, "sources": sources,
                         "source_states": {k: state(Path(v)) for k, v in sources.items()},
                         "protocol_sha256": protocol["sha256"], "notes": row["notes"]})
    return {"run_id": run_id, "created_utc": now(), "project": str(PROJECT),
            "review_path": str(REVIEW), "review_sha256": sha(REVIEW),
            "code_sha256": code_manifest(), "selected": selected, "held": held,
            "policy": "absent fish only; scoped untreated source aliases; overwrite=False; no shared inventories or cohort outputs"}


def published_paths(rid):
    return list(META.glob(f"{rid}_*")) + [p for p in (
        PROJECT / "Processed data" / rid, PROJECT / "Quality checks" / rid) if p.exists()]


def staging_manifest(rid):
    return {str(p): sha(p) for directory in PROJECT.glob(f".{rid}-*")
            for p in directory.rglob("*") if p.is_file()}


def make_recovery_plan(run_id, previous_path, previous_sha256):
    if sha(previous_path) != previous_sha256:
        raise ValueError("Previous sealed plan hash mismatch")
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    if sha(REVIEW) != previous["review_sha256"]:
        raise ValueError("Original review changed")
    old_events = META / (previous["run_id"] + "-events.jsonl")
    completed = [json.loads(line) for line in old_events.read_text(encoding="utf-8").splitlines()
                 if json.loads(line).get("event") == "fish-result"]
    completed_by_id = {r["recording_id"]: r for r in completed}
    if len(completed_by_id) != len(completed):
        raise ValueError("Duplicate original fish results require reconciliation")
    from finalize_trace3_additions import verify_fish
    selected, verified = [], []
    for row in previous["selected"]:
        rid = row["recording_id"]
        status = existing_status(rid, False)
        if rid in completed_by_id:
            result = completed_by_id[rid]
            if result.get("status") != "completed" or existing_status(rid, True)["state"] != "complete-verified":
                raise ValueError("Previously completed fish failed full verification: " + rid)
            verify_fish(row, result)
            verified.append(result)
            continue
        if published_paths(rid):
            raise ValueError("Published partial fish requires stage reconciliation: " + rid)
        if status["state"] not in {"absent", "partial"}:
            raise ValueError("Unexpected recovery state: " + rid)
        for kind, source in row["sources"].items():
            if state(Path(source)) != row["source_states"][kind]:
                raise ValueError("Raw source state changed: " + rid)
        if sha(Path(row["sources"]["protocol"])) != row["protocol_sha256"]:
            raise ValueError("Raw protocol changed: " + rid)
        selected.append({**row, "preserved_staging": staging_manifest(rid)})
    preservation_path = META / "trace3-review-20261018-preservation-baseline.json"
    preservation = json.loads(preservation_path.read_text(encoding="utf-8"))
    changed = [r["path"] for r in preservation["files"]
               if not Path(r["path"]).exists() or sha(Path(r["path"])) != r["sha256"]]
    if changed:
        raise ValueError("Original preserved metadata changed: " + repr(changed))
    return {**previous, "run_id": run_id, "created_utc": now(), "code_sha256": code_manifest(),
            "selected": selected, "previous_plan": str(previous_path),
            "previous_plan_sha256": previous_sha256, "previous_events_sha256": sha(old_events),
            "previously_completed_and_reverified": verified, "preserved_metadata_changed": changed,
            "lock_path": str(META / (run_id + "-active.lock")),
            "policy": "recovery: reverified completed fish skipped; abandoned unpublished staging preserved; fresh intake overwrite=False"}


def execute(plan, prefix):
    if Path(plan["project"]).resolve() != PROJECT.resolve() or sha(REVIEW) != plan["review_sha256"]:
        raise ValueError("Project or reviewed evidence changed")
    if code_manifest() != plan["code_sha256"]:
        raise ValueError("Code changed after planning; review and create a new plan")
    events_path = Path(str(prefix) + "-events.jsonl")
    completion_path = Path(str(prefix) + "-completion.json")
    if events_path.exists() or completion_path.exists():
        raise FileExistsError("Existing run evidence preserved; do not restart this run")
    # A Windows named mutex excludes simultaneous recovered workers across run IDs.
    mutex = None
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
        kernel.CreateMutexW.restype = ctypes.c_void_p
        mutex = kernel.CreateMutexW(None, True, "Local\\ClassicalConditioningTrace3Additions")
        if not mutex or ctypes.get_last_error() == 183:
            raise RuntimeError("Another 3sTrace recovery worker holds the processing mutex")
    lock = Path(plan.get("lock_path", str(META / "trace3-additions-active.lock")))
    token = uuid.uuid4().hex
    save_new(lock, {"run_id": plan["run_id"], "pid": os.getpid(), "token": token, "created_utc": now()})
    eligible = {r["recording_id"] for r in plan["selected"]}
    # Detached stdout/stderr are created by the launcher, then owned by this run.
    owned_logs = {Path(str(prefix) + s) for s in ("-stdout.log", "-stderr.log")}
    baseline = {str(p.resolve()).casefold(): (str(p), state(p)) for p in PROJECT.rglob("*")
                if p.is_file() and p != lock and p not in owned_logs}
    directories = {str(p.resolve()).casefold() for p in PROJECT.rglob("*") if p.is_dir()}
    root = PROJECT.resolve()

    def guard(event, values):
        paths = []
        if event == "open" and values and isinstance(values[0], (str, bytes, os.PathLike)):
            mode, flags = values[1] or "", values[2] if len(values) > 2 else 0
            if any(c in mode for c in "wax+") or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
                paths = [values[0]]
        elif event in {"os.remove", "os.rmdir", "os.mkdir", "os.chmod", "os.utime"}:
            paths = [values[0]]
        elif event == "os.rename":
            paths = list(values[:2])
        for value in paths:
            if not isinstance(value, (str, bytes, os.PathLike)):
                continue
            path = Path(os.fsdecode(value)).resolve()
            key = str(path).casefold()
            if path.drive.lower() == "d:" or "results (paper)" in key or key in baseline:
                raise PermissionError(f"Preservation guard blocked {event}: {path}")
            if key in directories and event != "os.mkdir":
                raise PermissionError(f"Existing directory protected from {event}: {path}")
            if path == root:
                if event == "os.mkdir": continue
                raise PermissionError("Project root is protected")
            if root not in path.parents:
                if path.drive.lower() == "f:": raise PermissionError("F: writes must stay in the existing project")
                continue  # Runtime caches outside data storage; D: remains prohibited.
            parts = path.relative_to(root).parts
            allowed = False
            if parts[0] in {"Processed data", "Quality checks"}:
                allowed = (len(parts) == 1 and event == "os.mkdir") or (len(parts) >= 2 and parts[1] in eligible)
            elif parts[0] == "Metadata":
                allowed = (len(parts) == 1 and event == "os.mkdir") or (len(parts) == 2 and
                           (parts[1].startswith(plan["run_id"] + "-") or path == lock or
                            any(parts[1].startswith(rid + "_") for rid in eligible)))
            elif parts[0].startswith("."):
                allowed = any(parts[0].startswith("." + rid + "-") for rid in eligible)
            if not allowed:
                raise PermissionError(f"Output is outside selected per-fish scope: {path}")

    sys.addaudithook(guard)
    def event(payload):
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at_utc": now(), **payload}) + "\n")
    results, started = [], time.time()
    try:
        event({"event": "run-start", "selected_fish": len(plan["selected"]), "pid": os.getpid()})
        for index, row in enumerate(plan["selected"], 1):
            rid = row["recording_id"]
            if code_manifest() != plan["code_sha256"]:
                event({"event": "run-stop", "reason": "source code changed; preserve work and reconcile"})
                break
            result = {"recording_id": rid, "condition": row["condition"], "stages": []}
            print(f"START {rid} [{index}/{len(plan['selected'])}]", flush=True)
            try:
                if published_paths(rid) or (existing_status(rid, False)["state"] != "absent" and
                                            "preserved_staging" not in row):
                    raise ValueError("Fish acquired partial/existing artifacts; preserve for reconciliation")
                if "preserved_staging" in row and staging_manifest(rid) != row["preserved_staging"]:
                    raise ValueError("Abandoned staging changed after recovery planning")
                sources = {k: Path(v) for k,v in row["sources"].items()}
                if any(state(p) != row["source_states"][k] for k,p in sources.items()):
                    raise ValueError("Raw source state changed after planning")
                if sha(sources["protocol"]) != row["protocol_sha256"]:
                    raise ValueError("Raw protocol changed after planning")
                event({"event": "stage-start", "recording_id": rid, "stage": "intake"})
                intake_recording(sources["protocol"].parent, PROJECT, recording_id=rid,
                                 sources=RecordingSources(row["recording_name"], rid, **sources), overwrite=False)
                result["stages"].append("intake")
                event({"event": "stage-complete", "recording_id": rid, "stage": "intake"})
                stages = (
                    ("corrected-preprocess", build_corrected_preprocessing, {}),
                    ("candidate-metrics", build_candidate_activity_metrics_from_corrected, {}),
                    ("movement-state", build_candidate_movement_state, {"metric_recipe": "tail-candidate-corrected"}),
                    ("temporal-profiles", build_candidate_temporal_profiles, {"metric_recipe": "tail-candidate-corrected", "experiment_name": "all3sTrace"}),
                    ("trial-outcomes", build_candidate_trial_outcomes, {"metric_recipe": "tail-candidate-corrected", "experiment_name": "all3sTrace"}),
                )
                for name, function, kwargs in stages:
                    event({"event": "stage-start", "recording_id": rid, "stage": name})
                    function(PROJECT, rid, overwrite=False, **kwargs)
                    result["stages"].append(name)
                    event({"event": "stage-complete", "recording_id": rid, "stage": name})
                    print(f"STAGE {rid} {name}", flush=True)
                verified = verify_candidate_trial_outcomes(PROJECT, rid, metric_recipe="tail-candidate-corrected")
                if verified.summary["experiment"] != "all3sTrace" or verified.summary["identity"]["condition_id"] != row["condition"]:
                    raise ValueError("Final experiment/condition identity mismatch")
                manifest = json.loads((META / f"{rid}_source_manifest.json").read_text())
                if manifest["sources"]["protocol"]["sha256"] != row["protocol_sha256"]:
                    raise ValueError("Published intake protocol lineage mismatch")
                result.update(status="completed", lineage_verified=True,
                              raw_source_sha256={k:v["sha256"] for k,v in manifest["sources"].items()})
            except Exception as exc:
                traceback.print_exc()
                result.update(status="failed", error=repr(exc))
            results.append(result)
            event({"event": "fish-result", **result})
            print(f"DONE {rid} {result['status']}", flush=True)
        changed = [p for p,s in baseline.values() if not Path(p).exists() or state(Path(p)) != s]
        completion = {"run_id": plan["run_id"], "finished_utc": now(), "results": results,
                      "pending": [r["recording_id"] for r in plan["selected"] if r["recording_id"] not in {v["recording_id"] for v in results}],
                      "held": plan["held"], "existing_files_changed": changed, "elapsed_seconds": time.time()-started}
        save_new(completion_path, completion)
        if changed or completion["pending"] or any(r["status"] != "completed" for r in results):
            raise SystemExit(1)
    finally:
        if lock.exists() and json.loads(lock.read_text()).get("token") == token:
            lock.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="trace3-additions-20261018-verified-run")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--plan-sha256")
    ap.add_argument("--recover-plan", type=Path)
    ap.add_argument("--previous-plan-sha256")
    args = ap.parse_args()
    if not __import__("re").fullmatch(r"[a-z0-9-]+", args.run_id): raise ValueError("Invalid run ID")
    prefix = META / args.run_id
    plan_path = Path(str(prefix) + "-plan.json")
    if args.plan_only:
        plan = (make_recovery_plan(args.run_id, args.recover_plan, args.previous_plan_sha256)
                if args.recover_plan else make_plan(args.run_id))
        save_new(plan_path, plan)
        print(json.dumps({"plan": str(plan_path), "sha256": sha(plan_path), "selected": len(plan["selected"]), "held": len(plan["held"])}, indent=2))
        return
    if not args.plan_sha256 or sha(plan_path) != args.plan_sha256: raise ValueError("Exact reviewed plan SHA-256 required")
    execute(json.loads(plan_path.read_text(encoding="utf-8")), prefix)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    main()
