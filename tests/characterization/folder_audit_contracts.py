"""Pre-refactor characterization oracles, captured on 28 October 2026.

Only the named original functions are retained. They are independent expected
behavior, not maintained production helpers; source identities are recorded below.
"""
from collections import Counter
from pathlib import Path
from typing import Any
import hashlib
import io
import re
import xml.etree.ElementTree as ET
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from classical_conditioning.figures.layout_frames import embed_spec

NS = "{http://www.w3.org/2000/svg}"
STAGES = ("intake", "corrected-preprocess", "candidate-metrics", "movement-state", "temporal-profiles", "trial-outcomes")
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
sha256_file = sha

def verify_fish(*_):
    raise RuntimeError('Characterization oracle requires an explicit synthetic verifier')

SOURCE_IDENTITIES = {'scripts/process_trace10_additions.py': '4bbddd7e0654ce82c8fd6ecb8cf1149726f01a29fab01a942306227b60d43034', 'scripts/resume_trace10_additions.py': '5dac0c8597dfc900845a3e0f912dbb51eb4f4f03718478e11e5d00e2c1804a42', 'scripts/finalize_trace3_additions.py': 'a4bdc9b0a989a801fba3fafbcd8b98132ef4348db776dd7361c4d73ef4921cd3', 'src/classical_conditioning/analysis/cohort_outcomes.py': '52ad14f1142b1c8dc6c1cc38512b58f988ce44d63dd07ce053c867567b708fd3', 'src/classical_conditioning/figures/individual_atlas.py': '688c87b75415a9cdef0c0177505010c0b6737974fd4f2288685c37e1c42c71c3'}
LEGACY_RC = {'font.family': 'Arial', 'font.size': 8, 'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'svg.hashsalt': 'figure3-atlas'}

def protocol(path):
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

def dispositions(exists):
    """Require a contiguous prefix of wholly published stages."""
    result, missing = {}, False
    for name in STAGES:
        flags = exists[name]
        if any(flags) and not all(flags):
            raise ValueError("Partially published stage must be preserved for reconciliation: "+name)
        present = all(flags)
        if missing and present:
            raise ValueError("Published downstream stage has missing upstream stage: "+name)
        if not present: missing = True
        result[name] = "verify-existing" if present else "build-missing"
    return result

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

def _write_parquet(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    # Write one lossless table and return path/hash metadata for its parent summary.
    table = pa.Table.from_pandas(frame, preserve_index=False, safe=True)
    pq.write_table(table, path, compression="zstd", write_statistics=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "columns": len(table.column_names),
        "size_bytes": path.stat().st_size,
        "compression": "zstd",
        "compression_lossless": True,
    }

def svg_bytes(fig, identity, width, height):
    """Assign every artist to an invisible physical allocation, with explicit clips."""
    stream = io.BytesIO()
    fig.savefig(stream, format="svg", transparent=False, metadata={"Date":None})
    plt.close(fig)
    root = ET.fromstring(stream.getvalue())
    prefix = re.sub(r"[^a-zA-Z0-9_]", "_", identity)+"__"
    ids = {n.get("id"): prefix+n.get("id") for n in root.iter() if n.get("id")}
    text = ET.tostring(root, encoding="unicode")
    for old, new in ids.items():
        text = text.replace('id="'+old+'"', 'id="'+new+'"').replace('#'+old+'"', '#'+new+'"').replace('#'+old+')', '#'+new+')')
    parsed = ET.fromstring(text)
    spec = {"scope": "panel", "final_size_mm": [width, height], "inset_mm": 1.,
        "allocations": [{"id": identity, "kind": "panel", "box_mm": [0, 0, width, height],
                         "group_ids": [ids["figure_1"]]}],
        "background_ids": [ids["patch_1"]],
        "scientific_clips": [{"id": n.get("id"), "evidence": "Exact data-axis rectangle in this generated trial-data panel"}
                             for n in parsed.iter(NS+"clipPath")]}
    return embed_spec(text.encode(), spec)
