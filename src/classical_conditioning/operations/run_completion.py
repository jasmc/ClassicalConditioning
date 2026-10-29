"""Completion decisions for the existing sealed 3sTrace append contract.

The caller must supply real lineage verification; this module never disables it.
"""
from collections import Counter

def check_run(plan, completion, verifier):
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

