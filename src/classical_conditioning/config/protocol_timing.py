"""Declarative stimulus diagnostics, independent of behavioral experiment recipes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

from classical_conditioning.exceptions import ConfigurationError


@dataclass(frozen=True)
class ProtocolTimingConfig:
    experiment_id: str
    phases: tuple[dict, ...]
    conditions: dict[str, dict]
    event_types: dict[str, str]
    cs_duration_ms: float | None = None
    timing_tolerance_ms: float = 200
    local_window_s: tuple[float, float] = (-30, 30)
    lag_views: tuple[dict, ...] = (
        {"horizon_s": 30, "bin_s": 1}, {"horizon_s": 300, "bin_s": 5},
    )
    target_latencies_s: tuple[float, ...] = ()
    recordings: dict[str, dict] | None = None

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), allow_nan=False))


def _number(value: Any, name: str, *, minimum: float = 0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ConfigurationError(f"{name} must be finite and >= {minimum}")
    return value


def timing_config_from_dict(value: dict) -> ProtocolTimingConfig:
    """Validate JSON before any outputs are touched; reject misspelled settings."""
    fields = set(ProtocolTimingConfig.__dataclass_fields__)
    if not isinstance(value, dict) or set(value) - fields:
        raise ConfigurationError("Unknown protocol timing settings")
    payload = json.loads(json.dumps(value))
    if not isinstance(payload.get("experiment_id"), str) or not payload["experiment_id"].strip():
        raise ConfigurationError("experiment_id is required")
    phases = payload.get("phases", [])
    if not isinstance(phases, list):
        raise ConfigurationError("phases must be a list")
    names = set()
    for phase in phases:
        if not isinstance(phase, dict) or set(phase) - {"name", "cs_count", "us_count", "us_role", "cs_duration_ms", "us_duration_ms", "block_size", "cs_block_counts", "us_block_counts"}:
            raise ConfigurationError("Invalid phase settings")
        name = phase.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ConfigurationError("Phase names must be nonempty and unique")
        names.add(name)
        for field in ("cs_count", "us_count"):
            n = phase.get(field, 0)
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ConfigurationError(f"{field} must be a nonnegative integer")
            phase[field] = n
        for kind in ("cs", "us"):
            counts = phase.get(f"{kind}_block_counts")
            if counts is not None and (not isinstance(counts, list) or not counts or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts) or sum(counts) != phase[f"{kind}_count"]):
                raise ConfigurationError("Block counts must partition the phase event count")
        if "block_size" in phase:
            n = phase["block_size"]
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ConfigurationError("block_size must be a positive integer")
        for duration in ("cs_duration_ms", "us_duration_ms"):
            if duration in phase:
                _number(phase[duration], duration, minimum=0.001)
        if not isinstance(phase.get("us_role", "unspecified"), str):
            raise ConfigurationError("us_role must be a string")
    conditions = payload.get("conditions", {})
    if not isinstance(conditions, dict):
        raise ConfigurationError("conditions must be an object")
    aliases = set()
    for name, condition in conditions.items():
        if not name or not isinstance(condition, dict) or set(condition) - {"aliases", "role", "pairings", "catch_cs"}:
            raise ConfigurationError("Invalid condition settings")
        if condition.get("role", "unspecified") not in {"conditioned", "control", "unspecified"}:
            raise ConfigurationError("Condition role must be conditioned, control, or unspecified")
        tokens = condition.get("aliases", [])
        if not isinstance(tokens, list) or not all(isinstance(t, str) and t for t in tokens):
            raise ConfigurationError("aliases must be a list of nonempty strings")
        for token in {name.lower(), *(t.lower() for t in tokens)}:
            if token in aliases:
                raise ConfigurationError(f"Ambiguous condition alias: {token}")
            aliases.add(token)
        for n in condition.get("catch_cs", []):
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ConfigurationError("catch_cs must contain positive integers")
        pair_ids = {"cs": set(), "us": set()}
        for pair in condition.get("pairings", []):
            if not isinstance(pair, dict) or set(pair) != {"cs", "us", "latency_s"}:
                raise ConfigurationError("Each pairing requires cs, us, latency_s")
            for kind in ("cs", "us"):
                n = pair[kind]
                if isinstance(n, bool) or not isinstance(n, int) or n < 1 or n in pair_ids[kind]:
                    raise ConfigurationError("Pair slots must be positive and unique")
                pair_ids[kind].add(n)
            _number(pair["latency_s"], "latency_s")
        if phases:
            for kind in ("cs", "us"):
                total = sum(p[f"{kind}_count"] for p in phases)
                if any(n > total for n in pair_ids[kind]):
                    raise ConfigurationError("Pairing references an unconfigured slot")
            if any(n > sum(p["cs_count"] for p in phases) or n in pair_ids["cs"] for n in condition.get("catch_cs", [])):
                raise ConfigurationError("Catch slots must exist and cannot also be paired")
    event_types = payload.get("event_types", {"CS": "Cycle", "US": "Reinforcer"})
    if set(event_types) != {"CS", "US"} or not all(isinstance(v, str) and v for v in event_types.values()) or len(set(event_types.values())) != 2:
        raise ConfigurationError("event_types must map CS and US to distinct raw labels")
    if payload.get("cs_duration_ms") is not None:
        _number(payload["cs_duration_ms"], "cs_duration_ms", minimum=0.001)
    _number(payload.get("timing_tolerance_ms", 200), "timing_tolerance_ms")
    window = payload.get("local_window_s", [-30, 30])
    if not isinstance(window, list) or len(window) != 2:
        raise ConfigurationError("local_window_s requires two bounds")
    for n in window:
        _number(n, "local_window_s", minimum=-1e12)
    if not window[0] < 0 < window[1]:
        raise ConfigurationError("local_window_s must straddle zero")
    views = payload.get("lag_views", [{"horizon_s": 30, "bin_s": 1}, {"horizon_s": 300, "bin_s": 5}])
    if not isinstance(views, list) or not views:
        raise ConfigurationError("lag_views must be nonempty")
    for view in views:
        if not isinstance(view, dict) or set(view) != {"horizon_s", "bin_s"}:
            raise ConfigurationError("Lag views require horizon_s and bin_s")
        h = _number(view["horizon_s"], "horizon_s", minimum=.001)
        b = _number(view["bin_s"], "bin_s", minimum=.001)
        if not math.isclose(2*h/b, round(2*h/b)) or not 2 <= 2*h/b <= 2000:
            raise ConfigurationError("Lag views require 2–2000 equal bins spanning both bounds")
    for t in payload.get("target_latencies_s", []):
        _number(t, "target_latencies_s", minimum=-1e12)
    recordings = payload.get("recordings", {})
    if not isinstance(recordings, dict):
        raise ConfigurationError("recordings must be an object keyed by recording ID")
    for metadata in recordings.values():
        if not isinstance(metadata, dict) or set(metadata) - {"condition", "day", "batch_id", "batch_provenance", "cs_schedule_id", "us_schedule_id", "schedule_provenance", "phase_bounds_s", "phase_bounds_provenance", "event_annotations", "processed_markers"}:
            raise ConfigurationError("Invalid recording metadata")
        if metadata.get("condition") is not None and metadata["condition"] not in conditions:
            raise ConfigurationError("Recording condition is not configured")
        if metadata.get("batch_id") and not metadata.get("batch_provenance"):
            raise ConfigurationError("batch_id requires batch_provenance")
        if (metadata.get("cs_schedule_id") or metadata.get("us_schedule_id")) and not metadata.get("schedule_provenance"):
            raise ConfigurationError("Intended schedule IDs require schedule_provenance")
        markers = metadata.get("processed_markers")
        if markers is not None and (not isinstance(markers, dict) or set(markers) != {"path", "sha256", "raw_protocol_sha256"}):
            raise ConfigurationError("processed_markers requires path, sha256 and raw_protocol_sha256")
        for name, bounds in metadata.get("phase_bounds_s", {}).items():
            if name not in names or len(bounds) != 2 or not all(isinstance(n, (int, float)) and math.isfinite(n) for n in bounds) or bounds[0] >= bounds[1]:
                raise ConfigurationError("Invalid acquisition-relative phase_bounds_s")
        for phase, provenance in metadata.get("phase_bounds_provenance", {}).items():
            if phase not in metadata.get("phase_bounds_s", {}) or not isinstance(provenance, dict) or set(provenance) != {"kind", "source"} or provenance["kind"] not in {"configured", "recorded"} or not isinstance(provenance["source"], str) or not provenance["source"]:
                raise ConfigurationError("Phase-bound provenance requires configured/recorded kind and source")
        annotated_rows = set()
        for annotation in metadata.get("event_annotations", []):
            if not isinstance(annotation, dict) or set(annotation) - {"source_row", "slot", "phase", "role", "block", "provenance"} or not annotation.get("provenance"):
                raise ConfigurationError("Event annotations require source_row and provenance")
            for key in ("source_row", "slot"):
                n = annotation.get(key)
                if key == "source_row" or n is not None:
                    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                        raise ConfigurationError(f"Annotation {key} must be positive")
            if annotation.get("phase") is not None and annotation["phase"] not in names:
                raise ConfigurationError("Unknown annotated phase")
            if annotation["source_row"] in annotated_rows:
                raise ConfigurationError("Duplicate source-row annotations")
            annotated_rows.add(annotation["source_row"])
    return ProtocolTimingConfig(
        experiment_id=payload["experiment_id"], phases=tuple(phases), conditions=conditions,
        event_types=event_types, cs_duration_ms=payload.get("cs_duration_ms"),
        timing_tolerance_ms=payload.get("timing_tolerance_ms", 200),
        local_window_s=tuple(window), lag_views=tuple(views),
        target_latencies_s=tuple(payload.get("target_latencies_s", [])), recordings=recordings,
    )


def load_timing_config(path: Path) -> ProtocolTimingConfig:
    return timing_config_from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def default_timing_config(experiment: str) -> ProtocolTimingConfig:
    """Adapt existing authority without changing its counts, catches or tolerances."""
    from classical_conditioning.config.experiments import condition_aliases, get_experiment_spec
    from classical_conditioning.config.protocols import SESSION, VERIFICATION_POLICY
    from classical_conditioning.config.trial_map import get_experiment_trial_map
    from classical_conditioning.config.domain import Alignment
    spec = get_experiment_spec(experiment)
    aliases = condition_aliases(experiment)
    pairs = get_experiment_trial_map(experiment)["conditioned_pairings"]
    return timing_config_from_dict({
        "experiment_id": spec.experiment_id,
        "event_types": {"CS": "Cycle", "US": "Reinforcer"},
        "phases": [{"name": p.name, "cs_count": p.cs_count, "us_count": p.us_count,
                    "us_role": "training" if p.name == "training" else "maintenance",
                    "us_duration_ms": p.us_duration_ms, "block_size": 10,
                    **({"us_block_counts": [len(ids) for _, ids in spec.trial_blocks(Alignment.US)]} if p.name == "training" else {})}
                   for p in SESSION.phases] + [{"name": "final_check", "cs_count": 0,
                    "us_count": SESSION.final_check_count, "us_role": "viability",
                    "us_duration_ms": SESSION.final_check_duration_ms}],
        "conditions": {c.condition_id: {"role": c.role.value,
            "aliases": [a for a, target in aliases.items() if target == c.condition_id],
            "catch_cs": list(SESSION.analysis_catch_cs) if c.role.value == "conditioned" else [],
            "pairings": [{"cs": p["cs_trial"], "us": p["us_trial"], "latency_s": p["us_latency_s"]}
                         for p in pairs] if c.role.value == "conditioned" else []}
            for c in spec.conditions},
        "cs_duration_ms": spec.cs_duration_s * 1000,
        "timing_tolerance_ms": VERIFICATION_POLICY.timing_tolerance_ms,
        "target_latencies_s": sorted({p["us_latency_s"] for p in pairs}),
    })
