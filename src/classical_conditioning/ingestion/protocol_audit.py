"""Exact fixed-10s Trace source-log audit; not a replacement protocol definition."""
from classical_conditioning.artifacts import sha256_file as sha

def audit_fixed10_protocol(path):
    rows, errors = [], []
    for line, text in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line == 1 and text.split() == ["Type", "Beg", "End"]:
            continue
        if not text.strip():
            continue
        cells = text.split()
        if len(cells) != 3 or cells[0] not in {"Cycle", "Reinforcer"}:
            errors.append({"line": line, "text": text}); continue
        try:
            event = {"type": cells[0], "beg_ms": int(cells[1]), "end_ms": int(cells[2]), "line": line}
            if event["end_ms"] < event["beg_ms"]:
                errors.append({"line": line, "error": "negative duration"})
            rows.append(event)
        except ValueError:
            errors.append({"line": line, "text": text})
    # Unpaired CS/US log lines can interleave out of timestamp order. Keep the
    # original rows/lines as evidence and match the supported stages' stable
    # event-time ordering rather than treating logging order as corrupted data.
    cs = sorted((r for r in rows if r["type"] == "Cycle"), key=lambda r:r["beg_ms"])
    us = sorted((r for r in rows if r["type"] == "Reinforcer"), key=lambda r:r["beg_ms"])
    for kind, events in (("Cycle",cs),("Reinforcer",us)):
        if len({r["beg_ms"] for r in events}) != len(events):
            errors.append({"error":"duplicate onset within event type", "type":kind})
    training, catches = [], []
    duration_outliers = []
    for trial, e in enumerate(cs, 1):
        if abs(e["end_ms"] - e["beg_ms"] - 10000) > 200:
            duration_outliers.append({"trial": trial, **e})
        if 15 <= trial <= 64:
            hits = [{"us_trial": n, "latency_ms": u["beg_ms"]-e["beg_ms"], "us_line": u["line"]}
                    for n,u in enumerate(us,1) if abs(u["beg_ms"]-e["beg_ms"]-20000) <= 200]
            training.append({"cs_trial": trial, "cs_line": e["line"], "near20": hits})
            if not hits: catches.append(trial)
    ordered = all(a["beg_ms"] <= b["beg_ms"] for a,b in zip(rows,rows[1:]))
    return {"path": str(path), "sha256": sha(path), "counts": {"Cycle": len(cs), "Reinforcer": len(us)},
            "events": rows, "errors": errors, "ordered": ordered,
            "cs_duration_outliers": duration_outliers, "training": training,
            "unpaired_training_cs": catches,
            "near20_training_count": sum(bool(t["near20"]) for t in training),
            "standard_event_counts": len(cs)==94 and len(us)==79}

