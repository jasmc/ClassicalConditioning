"""Verify an append run and publish new completion/anomaly evidence.

Optionally wait on the exact worker process; never restart or modify it.
The worker performs full per-fish lineage verification. This final check
authenticates current trial-output bytes and source/condition identities again.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from process_trace3_additions import PROJECT, sha
from classical_conditioning.artifacts import verify_completed_parquet_set


def verify_fish(row, result):
    rid = row["recording_id"]
    root = PROJECT / "Processed data" / rid
    verified = verify_completed_parquet_set(
        {"outcomes": root/"candidate-trial-outcomes-corrected.parquet", "coverage": root/"candidate-trial-outcomes-corrected_coverage.parquet"},
        PROJECT/"Quality checks"/rid/"candidate-trial-outcomes-corrected_summary.json",
        PROJECT/"Metadata"/f"{rid}_candidate-trial-outcomes-corrected_complete.json",
        recipe="candidate-trial-outcomes-corrected", recording_id=rid)
    summary = verified.summary
    if summary["experiment"] != "all3sTrace" or summary["identity"]["condition_id"] != row["condition"]:
        raise ValueError("Current outcome experiment/condition mismatch")
    manifest = json.loads((PROJECT/"Metadata"/f"{rid}_source_manifest.json").read_text())
    if manifest["recording_name"] != row["recording_name"] or manifest["recording_id"] != rid:
        raise ValueError("Current source identity mismatch")
    if manifest["sources"]["protocol"]["sha256"] != row["protocol_sha256"]:
        raise ValueError("Current source protocol differs from sealed plan")
    for kind, path in row["sources"].items():
        source = manifest["sources"][kind]
        if Path(source["path"]).resolve() != Path(path).resolve():
            raise ValueError("Current source path differs from sealed plan")
        if source["sha256"] != result.get("raw_source_sha256", {}).get(kind):
            raise ValueError("Worker raw-source hash and manifest disagree")


def check_run(plan, completion, verifier=verify_fish):
    expected = {r["recording_id"]: r for r in plan["selected"]}
    errors, results = [], completion.get("results", [])
    ids = [r["recording_id"] for r in results]
    if len(ids) != len(set(ids)): errors.append("duplicate per-fish results")
    if set(ids)-set(expected): errors.append("unexpected fish outside sealed plan")
    pending = sorted(set(expected)-set(ids))
    if sorted(completion.get("pending", [])) != pending: errors.append("worker pending list disagrees with results")
    if completion.get("existing_files_changed"): errors.append("pre-existing files changed")
    stages = ["intake","corrected-preprocess","candidate-metrics","movement-state","temporal-profiles","trial-outcomes"]
    checked = []
    for result in results:
        rid = result["recording_id"]
        row_errors = []
        if result.get("status") == "completed":
            if result.get("stages") != stages or not result.get("lineage_verified"):
                row_errors.append("worker did not verify every required stage")
            if rid in expected:
                try: verifier(expected[rid], result)
                except Exception as exc: row_errors.append(str(exc))
        else:
            row_errors.append(result.get("error", "worker did not complete this fish"))
        checked.append({**result,"final_verification_errors":row_errors})
    success = not errors and not pending and len(results)==len(expected) and all(not r["final_verification_errors"] for r in checked)
    return {"status":"completed-and-verified" if success else "incomplete-or-review-required",
            "errors":errors,"pending":pending,"results":checked,"held":plan["held"],
            "existing_files_changed":completion.get("existing_files_changed", []),
            "status_counts":dict(Counter(r["status"] for r in checked))}


def wait_for_worker(pid, completion):
    """Read-only Windows wait handle stays bound to this process if its PID is reused."""
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE; no termination/write rights
    if not handle:
        if completion.exists(): return
        raise RuntimeError("Worker is unavailable and has no completion record")
    try:
        while not completion.exists():
            state = kernel.WaitForSingleObject(handle, 30000)
            if state == 0 and not completion.exists():
                raise RuntimeError("Worker stopped without a completion record; preserve all partial data")
            if state not in (0,258): raise RuntimeError("Worker wait failed")
    finally:
        kernel.CloseHandle(handle)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id",default="trace3-additions-20261018-verified-run")
    ap.add_argument("--wait-for-pid",type=int)
    args = ap.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+",args.run_id): raise ValueError("Invalid run ID")
    prefix = PROJECT/"Metadata"/args.run_id
    outputs = [Path(str(prefix)+s) for s in ("-verification.json","-verification.md")]
    if any(p.exists() for p in outputs): raise FileExistsError("Existing verification evidence preserved")
    plan_path = Path(str(prefix)+"-plan.json")
    completion_path = Path(str(prefix)+"-completion.json")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    try:
        if args.wait_for_pid and not completion_path.exists():
            lock = json.loads(Path(plan.get("lock_path", str(PROJECT/"Metadata"/"trace3-additions-active.lock"))).read_text())
            if lock["run_id"] != args.run_id or lock["pid"] != args.wait_for_pid:
                raise ValueError("Worker lock identity mismatch")
            wait_for_worker(args.wait_for_pid, completion_path)
        completion = json.loads(completion_path.read_text())
        report = check_run(plan, completion)
        if plan.get("previous_plan"):
            previous_path = Path(plan["previous_plan"])
            if sha(previous_path) != plan["previous_plan_sha256"]:
                raise ValueError("Previous sealed plan changed")
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            previous_rows = {r["recording_id"]: r for r in previous["selected"]}
            for result in plan["previously_completed_and_reverified"]:
                verify_fish(previous_rows[result["recording_id"]], result)
            report["previously_completed_and_reverified"] = plan["previously_completed_and_reverified"]
            report["total_verified_additions"] = len(report["previously_completed_and_reverified"]) + sum(
                r["status"] == "completed" and not r["final_verification_errors"] for r in report["results"])
        report["worker_completion_sha256"] = sha(completion_path)
    except Exception as exc:
        report = {"status":"incomplete-or-review-required","errors":[str(exc)],
                  "pending":[r["recording_id"] for r in plan["selected"]],"results":[],"held":plan["held"]}
    report.update(run_id=args.run_id,generated_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(plan_path))
    with outputs[0].open("x",encoding="utf-8") as stream: json.dump(report,stream,indent=2)
    with outputs[1].open("x",encoding="utf-8") as stream:
        stream.write(f"# 3sTrace additions verification\n\nRun: {args.run_id}. Status: **{report['status']}**.\n\n")
        stream.write("The worker performed supported full per-fish lineage verification. This report rechecks current trial-output hashes, canonical outcome roles and original raw-source paths/hash lineage. It does not freeze or replace a cohort or paper selection.\n\n")
        for error in report["errors"]: stream.write('- Run issue: '+error+'\n')
        if report.get("pending"): stream.write('\nPending fish: '+', '.join(report['pending'])+'\n')
        stream.write('\n## Per-fish results\n\n')
        for r in report['results']:
            stream.write('- '+r['recording_id']+': '+r['status']+'; stages: '+', '.join(r['stages'])+'; final issues: '+('; '.join(r['final_verification_errors']) or 'none')+'\n')
        stream.write('\n## Held identities\n\n')
        for r in report['held']: stream.write('- '+r['recording_id']+': '+'; '.join(r['issues'])+'\n')
        stream.write('\nIncreasing raw-context fish and stimulus anomalies remain separately recorded in the existing Metadata review reports. Historical exclusion labels were ignored for intake.\n')
    print(json.dumps({"status":report['status'],"report":str(outputs[1])}),flush=True)
    if report['status']!='completed-and-verified': raise SystemExit(1)


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding='utf-8')
    main()
