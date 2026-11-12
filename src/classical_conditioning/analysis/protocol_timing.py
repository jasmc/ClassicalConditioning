"""Lossless, descriptive stimulus timing evidence; no cohort or randomness verdicts."""
from __future__ import annotations

from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np

from classical_conditioning.config.protocol_timing import ProtocolTimingConfig


def read_stimulus_log(path: Path) -> list[dict]:
    """Keep source lines and timestamp text, including malformed/coarse values."""
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    header = next((i for i, line in enumerate(lines) if line.strip()), None)
    if header is None or lines[header].split() != ["Type", "Beg", "End"]:
        raise ValueError("Stimulus log requires Type Beg End header")
    rows = []
    for line_number, line in enumerate(lines[header+1:], header+2):
        if not line.strip():
            continue
        cells = line.split()
        rows.append({"source_row": len(rows)+1, "source_line": line_number,
                     "raw_type": cells[0] if cells else "", "raw_text": line,
                     "raw_beg": cells[1] if len(cells) > 1 else "",
                     "raw_end": cells[2] if len(cells) > 2 else "",
                     "column_count_valid": len(cells) == 3})
    return rows


def _milliseconds(text: str) -> int | None:
    try:
        value = Decimal(str(text))
        if value.is_finite() and value == value.to_integral_value() and abs(value) < 2**63:
            return int(value)
    except (InvalidOperation, ValueError):
        pass
    return None


def _slots(config: ProtocolTimingConfig, kind: str) -> dict[int, dict]:
    slots, cursor = {}, 1
    for phase in config.phases:
        count = phase[f"{kind.lower()}_count"]
        counts = phase.get(f"{kind.lower()}_block_counts")
        if counts is None:
            size = phase.get("block_size", max(count, 1))
            counts = [min(size, count-i) for i in range(0, count, size)]
        position = 0
        for block, length in enumerate(counts, 1):
            for _ in range(length):
                slots[cursor+position] = {
                    "phase": phase["name"], "role": phase.get("us_role", "unspecified") if kind == "US" else "CS",
                    "block": f"{phase['name']} {block}",
                    "expected_duration_ms": phase.get("us_duration_ms") if kind == "US" else phase.get("cs_duration_ms", config.cs_duration_ms),
                }
                position += 1
        cursor += count
    return slots


def build_timing_evidence(rows: list[dict], config: ProtocolTimingConfig, *,
                          recording_id: str, condition: str | None = None,
                          day: str | None = None, acquisition_bounds_ms=None,
                          camera_gaps_ms=(), processed_markers: list[dict] | None = None,
                          processed_status: dict | None = None) -> dict:
    """Observe every row; expected correspondence remains conditional on evidence."""
    metadata = (config.recordings or {}).get(recording_id, {})
    condition = metadata.get("condition", condition)
    condition_settings = config.conditions.get(condition, {})
    inverse = {value: key for key, value in config.event_types.items()}
    findings, events = [], []
    if condition not in config.conditions:
        findings.append({"code": "condition_unresolved", "condition": condition})
    for row in rows:
        begin, end = _milliseconds(row["raw_beg"]), _milliseconds(row["raw_end"])
        event = {**row, "type": inverse.get(row["raw_type"], "unknown"),
                 "beg_ms": begin, "end_ms": end, "event_number": None,
                 "expected_slot": None, "phase": "unknown", "role": "unknown",
                 "block": None, "mapping_provenance": "unresolved", "is_catch": False,
                 "expected_duration_ms": None}
        codes = []
        if not row["column_count_valid"] or begin is None or end is None:
            codes.append("invalid_timestamp_or_row")
        if any("e" in str(row[key]).lower() for key in ("raw_beg", "raw_end")):
            codes.append("scientific_notation_timestamp_review")
        if begin is not None and end is not None and end <= begin:
            codes.append("nonpositive_duration")
        if event["type"] == "unknown":
            codes.append("unknown_event_type")
        if acquisition_bounds_ms and ((begin is not None and not acquisition_bounds_ms[0] <= begin <= acquisition_bounds_ms[1]) or (end is not None and not acquisition_bounds_ms[0] <= end <= acquisition_bounds_ms[1])):
            codes.append("outside_acquisition")
        event["issues"] = codes
        findings.extend({"code": code, "source_row": row["source_row"]} for code in codes)
        events.append(event)
    counts, mapping = {}, {}
    for kind in ("CS", "US"):
        group = [e for e in events if e["type"] == kind]
        ordered = sorted([e for e in group if e["beg_ms"] is not None], key=lambda e: e["beg_ms"])
        slots = _slots(config, kind)
        duplicates = Counter(e["beg_ms"] for e in ordered)
        safe = (bool(slots) and len(group) == len(slots) and len(ordered) == len(group)
                and not any(n > 1 for n in duplicates.values())
                and not any(e["issues"] for e in group))
        counts[kind] = {"observed": len(group), "usable_onsets": len(ordered),
                        "expected": len(slots) if config.phases else None,
                        "matches": len(group) == len(slots) if config.phases else None}
        mapping[kind] = "configured_ordinal" if safe else "unresolved"
        if config.phases and not counts[kind]["matches"]:
            findings.append({"code": "event_count_mismatch", "type": kind, **counts[kind]})
        previous = None
        for number, event in enumerate(ordered, 1):
            event["event_number"] = number
            event["event_id"] = f"{kind}:row-{event['source_row']}"
            event["duration_ms"] = event["end_ms"]-event["beg_ms"] if event["end_ms"] is not None else None
            event["onset_interval_ms"] = event["beg_ms"]-previous["beg_ms"] if previous else None
            event["offset_gap_ms"] = event["beg_ms"]-previous["end_ms"] if previous and previous["end_ms"] is not None else None
            if duplicates[event["beg_ms"]] > 1:
                findings.append({"code": "duplicate_onset", "source_row": event["source_row"]})
            if event["offset_gap_ms"] is not None and event["offset_gap_ms"] < 0:
                findings.append({"code": "same_type_overlap", "source_row": event["source_row"]})
            if safe:
                event.update(slots[number], expected_slot=number, mapping_provenance="configured_ordinal")
            previous = event
    by_row = {e["source_row"]: e for e in events}
    for annotation in metadata.get("event_annotations", []):
        if annotation["source_row"] not in by_row:
            raise ValueError("Annotated source row is absent from stimulus log")
        event = by_row[annotation["source_row"]]
        slot = annotation.get("slot")
        if slot is not None:
            definition = _slots(config, event["type"]).get(slot)
            if definition is None:
                raise ValueError("Annotated slot is not configured")
            event.update(definition, expected_slot=slot)
        for key in ("phase", "role", "block"):
            if key in annotation:
                event[key] = annotation[key]
        event["mapping_provenance"] = annotation["provenance"]
    keys = [(e["type"], e["expected_slot"]) for e in events if e["expected_slot"] is not None]
    if len(keys) != len(set(keys)):
        raise ValueError("Multiple observed events map to one expected slot")
    catch_slots = set(condition_settings.get("catch_cs", []))
    for event in events:
        event["is_catch"] = event["type"] == "CS" and event["expected_slot"] in catch_slots
        duration = event.get("duration_ms")
        expected = event["expected_duration_ms"]
        if expected is not None and duration is not None:
            tolerance = config.timing_tolerance_ms if event["type"] == "CS" else 0
            if abs(duration-expected) > tolerance:
                findings.append({"code": "duration_deviation", "source_row": event["source_row"], "expected_ms": expected, "observed_ms": duration})
    origin = acquisition_bounds_ms[0] if acquisition_bounds_ms else min((e["beg_ms"] for e in events if e["beg_ms"] is not None), default=0)
    recorded_bounds = {name: interval for name, interval in metadata.get("phase_bounds_s", {}).items()
                       if metadata.get("phase_bounds_provenance", {}).get(name, {}).get("kind") == "recorded"}
    for event in events:
        event["expected_phase"] = event["phase"]
        event["phase_provenance"] = event["mapping_provenance"]
        event["observed_phase"] = None
        # Acquisition-relative recorded boundaries cannot be used without acquisition origin.
        if event["beg_ms"] is not None and recorded_bounds and acquisition_bounds_ms:
            relative = (event["beg_ms"]-origin)/1000
            hits = [name for name, (left, right) in recorded_bounds.items() if left <= relative < right]
            if len(hits) > 1:
                raise ValueError("Recorded phase intervals overlap at an event")
            if hits:
                event["observed_phase"] = hits[0]
                if event["phase"] not in {"unknown", hits[0]}:
                    findings.append({"code": "recorded_configured_phase_disagreement", "source_row": event["source_row"], "configured": event["phase"], "recorded": hits[0]})
                event["phase"] = hits[0]
                event["phase_provenance"] = metadata["phase_bounds_provenance"][hits[0]]["source"]
            elif event["phase"] in recorded_bounds:
                findings.append({"code": "outside_recorded_phase", "source_row": event["source_row"], "configured": event["phase"]})
                event["phase"] = "unknown"
    last_cs_end = max((e["end_ms"] for e in events if e["type"] == "CS" and e["end_ms"] is not None), default=None)
    for event in events:
        if event["role"] == "viability" and last_cs_end is not None and event["beg_ms"] is not None and event["beg_ms"] < last_cs_end:
            findings.append({"code": "viability_before_last_cs_end", "source_row": event["source_row"]})
    mapped = {(e["type"], e["expected_slot"]): e for e in events if e["expected_slot"] is not None}
    pairs = []
    for pair in condition_settings.get("pairings", []):
        cs, us = mapped.get(("CS", pair["cs"])), mapped.get(("US", pair["us"]))
        valid = cs is not None and us is not None and cs["beg_ms"] is not None and us["beg_ms"] is not None
        lag = us["beg_ms"]-cs["beg_ms"] if valid else None
        gap = us["beg_ms"]-cs["end_ms"] if valid and cs["end_ms"] is not None else None
        pairs.append({**pair, "latency_ms": lag, "offset_gap_ms": gap,
                      "expected_cs_duration_ms": cs["expected_duration_ms"] if cs else None,
                      "cs_source_row": cs["source_row"] if cs else None,
                      "us_source_row": us["source_row"] if us else None,
                      "status": "observed" if valid else "uncheckable"})
        if valid and abs(lag-pair["latency_s"]*1000) > config.timing_tolerance_ms:
            findings.append({"code": "pair_latency_deviation", **pair, "observed_ms": lag})
    report = {
        "schema_version": 1, "artifact_kind": "stimulus-timing-diagnostics",
        "scientific_status": "descriptive; no cohort selection or randomness inference",
        "experiment_id": config.experiment_id, "recording_id": recording_id,
        "condition_id": condition or "unknown", "condition_role": condition_settings.get("role", "unspecified"),
        "day": metadata.get("day", day) or "unknown", "day_provenance": "configured" if metadata.get("day") else "acquisition_filename" if day else "unknown",
        "batch_id": metadata.get("batch_id") or "unknown", "metadata": metadata,
        "origin_ms": origin, "origin_provenance": "acquisition" if acquisition_bounds_ms else "first_logged_onset",
        "acquisition_bounds_ms": acquisition_bounds_ms, "camera_gaps_ms": list(camera_gaps_ms),
        "counts": counts, "mapping_status": mapping, "events": events, "expected_pairs": pairs,
        "findings": findings, "status": "review" if findings else "descriptive",
        "unplottable_source_rows": [e["source_row"] for e in events if e["beg_ms"] is None],
        "limits": ["Configured ordinal phases are not independent phase verification.",
                   "Observed event numbers do not prove expected-slot identity.",
                   "Event logs describe logged commands, not measured optical output.",
                   "An irregular lag plot does not prove randomness or absence of learning."],
    }
    window_lo, window_hi = (t*1000 for t in config.local_window_s)
    report["local_window_coverage"] = {}
    for kind in ("CS", "US"):
        anchors = [e for e in events if e["type"] == kind and e["beg_ms"] is not None]
        report["local_window_coverage"][kind] = {
            "anchor_count": len(anchors),
            "windows_outside_acquisition": sum(e["beg_ms"]+window_lo < acquisition_bounds_ms[0] or e["beg_ms"]+window_hi > acquisition_bounds_ms[1] for e in anchors) if acquisition_bounds_ms else None,
            "windows_crossing_camera_gaps": sum(any(e["beg_ms"]+window_lo < end and e["beg_ms"]+window_hi > begin for begin, end in camera_gaps_ms) for e in anchors) if acquisition_bounds_ms else None,
        }
    report["local_window_coverage"]["expected_pairs_outside_cs_window"] = sum(not window_lo <= p["latency_s"]*1000 <= window_hi for p in pairs)
    report["local_window_coverage"]["observed_pairs_outside_cs_window"] = sum(p["latency_ms"] is not None and not window_lo <= p["latency_ms"] <= window_hi for p in pairs)
    report["alignment"] = alignment_evidence(report, processed_markers, processed_status)
    report["lags"] = lag_evidence(report, config)
    return report


def alignment_evidence(report: dict, markers: list[dict] | None, status: dict | None = None) -> dict:
    """Match by raw row or explicit event number, never nearest-time remapping."""
    if markers is None:
        return {"status": "unavailable", "matches": [], **(status or {})}
    by_row = {e["source_row"]: e for e in report["events"]}
    by_number = {(e["type"], e["event_number"]): e for e in report["events"] if e["event_number"] is not None}
    matches, seen = [], set()
    for marker in markers:
        kind = marker.get("type")
        event = by_row.get(marker.get("source_row")) if marker.get("source_row") else (
            by_number.get((kind, marker.get("event_number")))
            if report["mapping_status"].get(kind) == "configured_ordinal" else None)
        if event is None or event["type"] != kind:
            matches.append({**marker, "status": "unmatched_processed"})
            continue
        seen.add(event["source_row"])
        row = {**marker, "source_row": event["source_row"], "status": "matched",
               "expected_slot": event["expected_slot"], "raw_phase": event["phase"]}
        normalized_phase = {"Pre": "pretraining", "Train": "training", "Test": "testing"}.get(marker.get("phase"), marker.get("phase"))
        row["phase_label_mismatch"] = normalized_phase is not None and event["phase"] != "unknown" and normalized_phase != event["phase"]
        for edge in ("beg", "end"):
            raw = event[f"{edge}_ms"]
            measured = marker.get(f"{edge}_ms")
            row[f"{edge}_residual_ms"] = measured-raw if measured is not None and raw is not None else None
        matches.append(row)
    matches.extend({"source_row": e["source_row"], "type": e["type"], "status": "unmatched_raw"} for e in report["events"] if e["type"] in {"CS", "US"} and e["source_row"] not in seen)
    return {"status": "available", "matches": matches,
            "phase_label_mismatch_count": sum(r.get("phase_label_mismatch", False) for r in matches), **(status or {})}


def _observed_segments(report: dict, phase: str) -> list[tuple[float, float]] | None:
    bounds = report["acquisition_bounds_ms"]
    if bounds is None:
        return None
    left, right = bounds
    recorded = report["metadata"].get("phase_bounds_provenance", {}).get(phase, {}).get("kind") == "recorded"
    declared = report["metadata"].get("phase_bounds_s", {}).get(phase) if recorded else None
    if declared:
        left, right = max(left, report["origin_ms"]+declared[0]*1000), min(right, report["origin_ms"]+declared[1]*1000)
    segments = [(left, right)] if left < right else []
    for gap_left, gap_right in report["camera_gaps_ms"]:
        segments = [(a, b) for start, end in segments for a, b in
                    ((start, min(end, gap_left)), (max(start, gap_right), end)) if a < b]
    return segments


def lag_evidence(report: dict, config: ProtocolTimingConfig) -> list[dict]:
    """All-pairs counts, exact target hits, half-bin sensitivity, and exposure."""
    events = [e for e in report["events"] if e["beg_ms"] is not None]
    phases = list(dict.fromkeys([p["name"] for p in config.phases] + [e["phase"] for e in events]))
    if config.phases and any(status == "unresolved" for status in report["mapping_status"].values()):
        phases.append("unresolved_recording")
    output = []
    for phase in phases:
        selected = events if phase == "unresolved_recording" else [e for e in events if e["phase"] == phase]
        cs = [e for e in selected if e["type"] == "CS"]
        roles = sorted({e["role"] for e in selected if e["type"] == "US"})
        if not cs:
            continue
        for role in roles:
            us = sorted([e for e in selected if e["type"] == "US" and e["role"] == role], key=lambda e: e["beg_ms"])
            starts = np.array([e["beg_ms"] for e in us], dtype=np.int64)
            max_horizon = max([v["horizon_s"] for v in config.lag_views] + [abs(t)+config.timing_tolerance_ms/1000 for t in config.target_latencies_s])
            lags = []
            overlaps = []
            targets = {str(t): 0 for t in config.target_latencies_s}
            for anchor in cs:
                a = anchor["beg_ms"]
                lo = np.searchsorted(starts, a-max_horizon*1000, side="left")
                hi = np.searchsorted(starts, a+max_horizon*1000, side="right")
                differences = starts[lo:hi]-a
                lags.extend((differences/1000).tolist())
                for target in config.target_latencies_s:
                    targets[str(target)] += int(np.count_nonzero(np.abs(differences-target*1000) <= config.timing_tolerance_ms))
                if anchor["end_ms"] is not None and anchor["end_ms"] > a:
                    first = np.searchsorted(starts, a, side="left")
                    last = np.searchsorted(starts, anchor["end_ms"], side="left")
                    overlaps.extend({"cs_source_row": anchor["source_row"], "us_source_row": u["source_row"], "lag_ms": u["beg_ms"]-a} for u in us[first:last])
            segments = _observed_segments(report, phase)
            for view_index, view in enumerate(config.lag_views):
                horizon, width = view["horizon_s"], view["bin_s"]
                edges = np.linspace(-horizon, horizon, round(2*horizon/width)+1)
                # Use full-width shifted bins only: variable edge-bin widths
                # would manufacture differences in a count-per-anchor plot.
                shifted_edges = edges[:-1]+width/2
                counts = np.histogram(lags, bins=edges)[0]
                shifted = np.histogram(lags, bins=shifted_edges)[0]
                exposure = np.zeros(len(counts))
                if segments is not None:
                    for anchor in cs:
                        left = anchor["beg_ms"]+edges[:-1]*1000
                        right = anchor["beg_ms"]+edges[1:]*1000
                        for begin, end in segments:
                            exposure += np.maximum(0, np.minimum(right, end)-np.maximum(left, begin))/1000
                output.append({"phase": phase, "us_role": role, "view_index": view_index,
                    "edges_s": edges.tolist(), "counts": counts.tolist(),
                    "counts_per_anchor": (counts/len(cs)).tolist(),
                    "shifted_edges_s": shifted_edges.tolist(), "shifted_counts": shifted.tolist(),
                    "anchor_count": len(cs), "us_count": len(us),
                    "lag_scope": "whole recording with unresolved correspondence" if phase == "unresolved_recording" else "CS and US within the same assigned phase; boundary neighbors remain in raw rasters",
                    "observable_exposure_s": exposure.tolist() if segments is not None else None,
                    "phase_time_verified": report["acquisition_bounds_ms"] is not None and report["metadata"].get("phase_bounds_provenance", {}).get(phase, {}).get("kind") == "recorded",
                    "exposure_scope": "recorded_phase_and_acquisition" if report["acquisition_bounds_ms"] is not None and report["metadata"].get("phase_bounds_provenance", {}).get(phase, {}).get("kind") == "recorded" else "acquisition_only; phase time unverified",
                    "target_coincidences": targets, "us_onsets_inside_cs": overlaps,
                    "bin_origin_sensitivity": "half-bin shifted equal-width bins; incomplete edge bins omitted",
                    "interpretation": "counts per CS anchor; coverage not rate-corrected; no randomization null"})
    return output


def schedule_comparisons(reports: list[dict]) -> list[dict]:
    """Compare equal-length phase vectors by translation only; never warp time."""
    output = []
    for experiment in sorted({r["experiment_id"] for r in reports}):
        group = sorted([r for r in reports if r["experiment_id"] == experiment], key=lambda r: (r["condition_id"], r["day"], r["batch_id"], r["recording_id"]))
        for phase in sorted({e["phase"] for r in group for e in r["events"]}):
            for kind in ("CS", "US"):
                vectors = [np.array(sorted(e["beg_ms"] for e in r["events"] if e["type"] == kind and e["phase"] == phase and e["beg_ms"] is not None), dtype=np.int64) for r in group]
                matrix = [[None]*len(group) for _ in group]
                pairs = []
                for i, a in enumerate(vectors):
                    if len(a) < 2:
                        continue
                    matrix[i][i] = 0.0
                    for j in range(i):
                        b = vectors[j]
                        if len(a) != len(b):
                            continue
                        delta = a-b
                        residual = (delta-np.median(delta)).astype(float)
                        distance = float(np.max(np.abs(residual)))
                        matrix[i][j] = matrix[j][i] = distance
                        pairs.append({"first": group[j]["recording_id"], "second": group[i]["recording_id"],
                                      "maximum_residual_ms": distance, "residuals_ms": residual.tolist(),
                                      "removed_offset_ms": float(np.median(delta))})
                output.append({"experiment_id": experiment, "phase": phase, "type": kind,
                    "recording_ids": [r["recording_id"] for r in group], "maximum_residual_ms": matrix,
                    "recording_labels": [f"{r['recording_id']} | {r['condition_id']} | {r['day']} | batch {r['batch_id']}" for r in group],
                    "closest_pairs": sorted(pairs, key=lambda p: p["maximum_residual_ms"])[:3],
                    "interpretation": "Descriptive similarity, not intended-schedule identity; unequal counts or <2 events uncheckable."})
    return output


def summarize_lags(reports: list[dict]) -> list[dict]:
    """Declared estimand: equal fish, plus equal known intended schedule pairs."""
    groups = defaultdict(list)
    for report in reports:
        for lag in report["lags"]:
            groups[(report["experiment_id"], report["condition_id"], lag["phase"], lag["us_role"], lag["view_index"])].append((report, lag))
    summaries = []
    for key, entries in sorted(groups.items()):
        experiment, condition, phase, role, view = key
        population = sorted([r for r in reports if r["experiment_id"] == experiment and r["condition_id"] == condition], key=lambda r: (r["day"], r["batch_id"], r["recording_id"]))
        lookup = {r["recording_id"]: lag for r, lag in entries}
        sample = entries[0][1]
        edges = np.array(sample["edges_s"])
        curves, identities, schedule_groups, strata = [], [], defaultdict(list), defaultdict(list)
        exposure, shifted, valid = [], [], []
        for report in population:
            lag = lookup.get(report["recording_id"])
            identities.append({**{k: report[k] for k in ("recording_id", "condition_id", "day", "batch_id", "status")},
                "anchor_count": lag["anchor_count"] if lag else None,
                "us_count": lag["us_count"] if lag else None,
                "target_coincidences": lag["target_coincidences"] if lag else None,
                "observable_exposure_s": lag["observable_exposure_s"] if lag else None,
                "exposure_scope": lag["exposure_scope"] if lag else "unavailable"})
            if lag is None:
                curves.append([None]*(len(edges)-1))
                continue
            curve = np.array(lag["counts_per_anchor"])
            curves.append(curve.tolist())
            valid.append(curve)
            shifted.append(np.array(lag["shifted_counts"])/lag["anchor_count"])
            strata[(report["day"], report["batch_id"])].append(curve)
            if lag["observable_exposure_s"] is not None:
                exposure.append(np.array(lag["observable_exposure_s"])/(lag["anchor_count"]*np.diff(edges)))
            metadata = report["metadata"]
            if metadata.get("cs_schedule_id") and metadata.get("us_schedule_id"):
                schedule_groups[(metadata["cs_schedule_id"], metadata["us_schedule_id"])].append(curve)
        schedule_means = [np.mean(values, axis=0) for values in schedule_groups.values()]
        summaries.append({"experiment_id": experiment, "condition_id": condition, "phase": phase, "us_role": role,
            "view_index": view, "recordings": identities, "curves": curves, "contributing_fish": len(valid),
            "edges_s": sample["edges_s"], "equal_fish_counts_per_anchor": np.mean(valid, axis=0).tolist(),
            "shifted_edges_s": sample["shifted_edges_s"], "shifted_equal_fish_counts_per_anchor": np.mean(shifted, axis=0).tolist(),
            "equal_intended_schedule_counts_per_anchor": np.mean(schedule_means, axis=0).tolist() if schedule_means else None,
            "known_intended_schedule_count": len(schedule_groups), "known_schedule_fish_count": sum(map(len, schedule_groups.values())),
            "strata": [{"day": day, "batch_id": batch, "fish_count": len(values), "counts_per_anchor": np.mean(values, axis=0).tolist()} for (day, batch), values in sorted(strata.items())],
            "observable_fraction_equal_fish": np.mean(exposure, axis=0).tolist() if exposure else None,
            "coverage_fish_count": len(exposure),
            "lag_scope": sample["lag_scope"],
            "interpretation": "Equal-fish descriptive counts per CS anchor; intended-schedule curve uses only explicitly identified schedules. No population uncertainty or randomness claim."})
    return summaries
