"""Condition-independent protocol evidence; discrepancies never select a cohort."""
from __future__ import annotations

from dataclasses import asdict
import html
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from classical_conditioning.artifacts import (
    load_and_verify_source_manifest, sha256_file, write_json_atomic,
    artifact_staging, publish_transaction,
)
from classical_conditioning.config.experiments import (
    condition_aliases, get_experiment_spec, resolve_condition,
)
from classical_conditioning.config.domain import ConditionRole
from classical_conditioning.config.protocols import SESSION, VERIFICATION_POLICY, protocol_settings
from classical_conditioning.exceptions import ConfigurationError, ArtifactIntegrityError
from classical_conditioning.paths import condition_from_recording_name


REPORT_NAME = "protocol_verification.json"


def evaluate_protocol(protocol: pd.DataFrame, recording_name: str, *,
                      experiment_name: str | None = None,
                      acquisition_bounds_ms: tuple[int, int] | None = None) -> dict:
    """Retain event identities and evaluate every established session expectation."""
    findings = []

    def issue(code, **details):
        findings.append({"code": code, **details})

    token = None
    try:
        token = condition_from_recording_name(recording_name)
    except ConfigurationError:
        issue("unrecognized_filename", recording_name=recording_name)
    spec = get_experiment_spec(experiment_name) if experiment_name else None
    expected_condition = None
    if spec and token:
        try:
            expected_condition = resolve_condition(experiment_name, token)
        except ConfigurationError:
            issue("filename_experiment_mismatch", token=token, experiment=experiment_name)
    # A bare 'trace' or 'control' cannot establish which assay it belongs to.
    if spec is None and token:
        infer = {"delay": "allDelay", "3strace": "all3sTrace",
                 "10strace": "all10sTrace", "10sfixedtrace": "all10sTrace"}
        if token in infer:
            spec = get_experiment_spec(infer[token])
            expected_condition = resolve_condition(spec.experiment_id, token)
    events = protocol.copy().reset_index(drop=True)
    events["source_row"] = np.arange(1, len(events) + 1)
    if not {"Type", "Beg", "End"}.issubset(events.columns):
        raise ValueError("Protocol requires Type, Beg, End columns")
    for col in ("Beg", "End"):
        values = pd.to_numeric(events[col], errors="coerce")
        bad = ~np.isfinite(values) | (values % 1 != 0)
        if bad.any():
            issue("invalid_timestamps", column=col,
                  rows=events.loc[bad, "source_row"].tolist())
        events[col] = values
    good = np.isfinite(events.Beg) & np.isfinite(events.End)
    good &= (events.Beg % 1 == 0) & (events.End % 1 == 0)
    events = events.loc[good].copy()
    events[["Beg", "End"]] = events[["Beg", "End"]].astype("int64")
    for row in events.itertuples(index=False):
        if row.End <= row.Beg:
            issue("nonpositive_duration", row=int(row.source_row))
        if row.Type not in ("Cycle", "Reinforcer"):
            issue("unknown_event_type", row=int(row.source_row), event_type=str(row.Type))
        if acquisition_bounds_ms and (row.Beg < acquisition_bounds_ms[0] or row.End > acquisition_bounds_ms[1]):
            issue("outside_acquisition", row=int(row.source_row),
                  beg_ms=int(row.Beg), end_ms=int(row.End))
    # Ordering is evidence, not corruption: CS/US lines may interleave.
    original_ordered = bool(events.Beg.is_monotonic_increasing)
    events = events.sort_values("Beg", kind="stable")
    cs = events.loc[events.Type.eq("Cycle")].copy()
    us = events.loc[events.Type.eq("Reinforcer")].copy()
    cs["event_number"] = range(1, len(cs) + 1)
    us["event_number"] = range(1, len(us) + 1)
    for name, group in (("CS", cs), ("US", us)):
        duplicates = group.Beg.duplicated(keep=False)
        if duplicates.any():
            issue("duplicate_onset", event_type=name,
                  rows=group.loc[duplicates, "source_row"].tolist())
        intervals = group.Beg.to_numpy()[1:] - group.End.to_numpy()[:-1]
        if (intervals < 0).any():
            issue("same_type_overlap", event_type=name,
                  event_numbers=(np.flatnonzero(intervals < 0) + 2).tolist())
    expected_cs = spec.expected_cs_count if spec else SESSION.cs_count
    complete_us = spec.expected_us_count if spec else SESSION.complete_us_count
    if len(cs) != expected_cs:
        issue("cs_count", expected=expected_cs, observed=len(cs))
    if len(us) != complete_us:
        issue("us_count", expected=complete_us, regular=SESSION.regular_us_count,
              observed=len(us))
    tolerance = VERIFICATION_POLICY.timing_tolerance_ms
    bad_cs = abs(cs.End - cs.Beg - SESSION.cs_duration_s * 1000) > tolerance
    if bad_cs.any():
        issue("cs_duration", expected_ms=SESSION.cs_duration_s * 1000,
              events=cs.loc[bad_cs, ["event_number", "source_row", "Beg", "End"]].to_dict("records"))
    phase_rows, cs_offset, us_offset = [], 0, 0
    for phase in SESSION.phases:
        phase_us = us.iloc[us_offset:us_offset + phase.us_count]
        duration = phase_us.End - phase_us.Beg
        bad = abs(duration - phase.us_duration_ms) > VERIFICATION_POLICY.pulse_duration_tolerance_ms
        if bad.any():
            issue("us_pulse_duration", phase=phase.name, expected_ms=phase.us_duration_ms,
                  event_numbers=phase_us.loc[bad, "event_number"].tolist(),
                  observed_ms=duration.loc[bad].tolist())
        phase_rows.append({"phase": phase.name,
                           "cs_ids": [cs_offset + 1, cs_offset + phase.cs_count],
                           "us_ids": [us_offset + 1, us_offset + phase.us_count]})
        cs_offset += phase.cs_count
        us_offset += phase.us_count
    if len(us) >= complete_us:
        final = us.iloc[SESSION.regular_us_count]
        if abs(final.End - final.Beg - SESSION.final_check_duration_ms) > VERIFICATION_POLICY.pulse_duration_tolerance_ms:
            issue("final_check_duration", expected_ms=SESSION.final_check_duration_ms,
                  observed_ms=int(final.End - final.Beg))
        if len(cs) and final.Beg < cs.iloc[-1].End:
            issue("final_check_before_last_cs_end")
    # Pair by unique recorded onsets inside the full supported timing window.
    # Independent short pulses remain evidence; nothing is silently filtered.
    train_first = 1 + sum(p.cs_count for p in SESSION.phases[:2])
    train_last = train_first + SESSION.phases[2].cs_count - 1
    train = cs.loc[cs.event_number.between(train_first, train_last)]
    supported = {exp: get_experiment_spec(exp) for exp in ("allDelay", "all3sTrace", "all10sTrace")}
    max_latency_ms = max(max(c.us_latency_s) for assay in supported.values()
                         for c in assay.conditions if c.role is ConditionRole.CONDITIONED) * 1000
    pair_rows = []
    for c in train.itertuples(index=False):
        latency = us.Beg - c.Beg
        hits = us.loc[latency.between(0, max_latency_ms + tolerance)]
        pair_rows.append({"cs": int(c.event_number), "cs_row": int(c.source_row),
                          "catch": c.event_number in SESSION.training_catch_cs,
                          "us_ids": hits.event_number.tolist(),
                          "latencies_ms": (hits.Beg - c.Beg).tolist(),
                          "gaps_ms": (hits.Beg - c.End).tolist()})
    paired = [p for p in pair_rows if not p["catch"]]
    matched_ids = [p["us_ids"][0] for p in paired if len(p["us_ids"]) == 1]
    complete_pairs = (len(paired) == SESSION.paired_count
                      and len(matched_ids) == SESSION.paired_count
                      and len(set(matched_ids)) == SESSION.paired_count)
    observed = "unknown" if len(train) < SESSION.phases[2].cs_count else "variable_or_unpaired"
    signatures = {}
    for exp, other in supported.items():
        condition = next(c for c in other.conditions if c.role is ConditionRole.CONDITIONED)
        target = condition.us_latency_s[0] * 1000
        count = sum(len(p["us_ids"]) == 1 and abs(p["latencies_ms"][0] - target) <= tolerance
                    and abs(p["gaps_ms"][0] - (target - other.cs_duration_s * 1000)) <= tolerance
                    for p in paired)
        signatures[condition.display_name] = count
        if complete_pairs and count == SESSION.paired_count:
            observed = condition.display_name
    if spec and expected_condition:
        condition = next(c for c in spec.conditions if c.condition_id == expected_condition)
        if condition.role is ConditionRole.CONDITIONED:
            if observed != condition.display_name:
                issue("paired_timing_mismatch", expected=condition.display_name, observed=observed)
            expected_us_ids = list(range(SESSION.first_training_us,
                                         SESSION.first_training_us + SESSION.paired_count))
            if matched_ids != expected_us_ids:
                issue("training_us_sequence", expected=expected_us_ids, observed=matched_ids)
            for p in pair_rows:
                if p["catch"] and p["us_ids"]:
                    issue("us_at_catch", **p)
                elif not p["catch"] and len(p["us_ids"]) != 1:
                    issue("missing_or_ambiguous_pairing", **p)
        else:
            if observed in ("Delay", "3sTrace", "10sTrace"):
                issue("control_has_fixed_pairing", observed=observed)
            elif complete_pairs:
                issue("control_all_training_cs_have_unique_nearby_us",
                      note="Variable timing is not sufficient evidence of an unpaired control")
            diffs = np.diff(train.Beg)
            if (diffs < SESSION.control_min_cs_interval_s * 1000).any():
                issue("control_cs_spacing", minimum_s=SESSION.control_min_cs_interval_s)
            if len(train) and len(us):
                distances = np.abs(train.Beg.to_numpy()[:, None] - us.Beg.to_numpy()[None, :])
                if distances.min() < SESSION.control_min_cs_us_onset_separation_s * 1000:
                    issue("control_cs_us_onset_separation",
                          minimum_s=SESSION.control_min_cs_us_onset_separation_s)
    elif token in ("control", "controlnomk801") and observed in ("Delay", "3sTrace", "10sTrace"):
        issue("control_has_fixed_pairing", observed=observed)
    elif token in ("trace", "tracenomk801") and observed == "Delay":
        issue("filename_timing_mismatch", token=token, observed=observed)
    # Non-training CSs must not have a conditioning (100 ms) pulse nearby;
    # separate short maintenance pulses are legitimate in these phases.
    for c in cs.loc[~cs.event_number.between(train_first, train_last)].itertuples(index=False):
        hits = us.loc[(us.Beg - c.Beg).between(0, max_latency_ms + tolerance)
                      & ((us.End - us.Beg) == SESSION.phases[2].us_duration_ms)]
        if len(hits):
            issue("long_us_at_nontraining_cs", cs=int(c.event_number), us_ids=hits.event_number.tolist())
    records = []
    for label, group in (("CS", cs), ("US", us)):
        records.extend({"type": label, **r} for r in group[
            ["source_row", "event_number", "Beg", "End"]].to_dict("records"))
    return {
        "schema_version": 1, "artifact_kind": "protocol-verification",
        "recording_name": recording_name, "filename_token": token,
        "experiment": experiment_name,
        "resolved_experiment": spec.experiment_id if spec else None,
        "condition_id": expected_condition, "observed_schedule": observed,
        "status": "warn" if findings else ("pass" if spec else "not_checkable"),
        "context_status": "configured" if experiment_name else "filename_only",
        "control_assay_proof": "Timing alone does not establish assay membership or prove randomization",
        "settings": protocol_settings(), "counts": {"CS": len(cs), "US": len(us)},
        "count_checks": {
            "CS": {"expected": expected_cs, "observed": len(cs), "matches": len(cs) == expected_cs},
            "US": {"expected": complete_us, "observed": len(us), "matches": len(us) == complete_us},
        },
        "rows_chronological": original_ordered, "phases": phase_rows,
        "pairing_counts": signatures, "training_pairs": pair_rows,
        "events": records, "findings": findings,
        "intervals_ms": {name: {"onset_intervals": np.diff(group.Beg).tolist(),
                                  "durations": (group.End - group.Beg).tolist()}
                         for name, group in (("CS", cs), ("US", us))},
        "unchecked": (["assay assignment requires experiment context"] if spec is None else [])
                     + (["acquisition coverage unavailable"] if acquisition_bounds_ms is None else [])
                     + ["randomization cannot be proved from one event log"],
    }


def verify_recording_protocol(project_dir: Path, recording_id: str, *,
                              experiment_name: str | None = None,
                              emit_warnings: bool = True) -> dict:
    """Audit authenticated intake on every invocation, independent of frame caches."""
    project_dir = Path(project_dir).resolve()
    name, inputs, input_state = load_and_verify_source_manifest(project_dir, recording_id)
    data = project_dir / "Processed data" / recording_id
    camera = pq.read_table(data / "camera.parquet", columns=["AbsoluteTime"]).to_pandas()
    bounds = (int(camera.AbsoluteTime.min()), int(camera.AbsoluteTime.max())) if len(camera) else None
    report = evaluate_protocol(pq.read_table(data / "stimulus_events.parquet").to_pandas(),
                               name, experiment_name=experiment_name,
                               acquisition_bounds_ms=bounds)
    # Record evaluator and configuration identity without coupling frame caches
    # to diagnostics. Changed inputs/context always produce fresh evidence.
    report["inputs"] = {k: v["sha256"] for k, v in inputs.items()}
    report["evaluator_sha256"] = sha256_file(Path(__file__))
    report["recording_id"] = recording_id
    if experiment_name:
        report["experiment_settings"] = asdict(get_experiment_spec(experiment_name))
        report["filename_aliases"] = condition_aliases(experiment_name)
    # Recheck before publication: never attach evidence to changed source bytes.
    for kind, entry in inputs.items():
        stat = (data / {"camera": "camera.parquet", "tracking": "tracking.parquet",
                       "protocol": "stimulus_events.parquet"}[kind]).stat()
        if (stat.st_size, stat.st_mtime_ns) != input_state[kind]:
            raise ArtifactIntegrityError("Intake changed during protocol verification")
    target = project_dir / "Quality checks" / recording_id / REPORT_NAME
    payload = json.loads(json.dumps(report, default=lambda v: v.value))
    # HTML and machine evidence describe the same bytes; JSON is published last.
    with artifact_staging(project_dir, prefix=f".{recording_id}-protocol-") as staging:
        staged_json = staging / REPORT_NAME
        staged_html = staging / "protocol_verification.html"
        write_json_atomic(staged_json, payload)
        staged_html.write_text(
            '<!doctype html><meta charset="utf-8"><title>Protocol verification</title>'
            '<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto}'
            'pre{white-space:pre-wrap}</style><h1>Protocol verification</h1>'
            f'<p>{html.escape(name)}: <strong>{report["status"]}</strong>. '
            'Discrepancies warn; analysis continues.</p><pre>'
            + html.escape(json.dumps(payload, indent=2)) + '</pre>', encoding="utf-8")
        publish_transaction([(staged_html, target.with_suffix(".html")),
                             (staged_json, target)], staging, overwrite=True)
    if emit_warnings and report["status"] != "pass":
        codes = sorted({f["code"] for f in report["findings"]})
        warnings.warn(f"{recording_id}: protocol {report['status']}: {', '.join(codes) or 'incomplete context'}; "
                      f"continuing; see {target}", UserWarning, stacklevel=2)
    return payload
