"""Review 3sTrace intake and publish new evidence; this route audits only.

The earlier processing route was withdrawn during review. Keep observed timing,
source identity, untreated aliases and artifact integrity as separate findings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

PROJECT = Path(r"F:\Digested Data\all3sTrace-full-v1")
PKL_ROOT = Path(r"F:\Results (paper)\2025_3sTrace\Processed data\pkl files\1. Original")
SUFFIXES = {"camera": "_cam.txt", "tracking": "_mp tail tracking.txt", "protocol": "_stim control.txt"}
CATCHES = {25, 39, 53, 59}
ALIASES = {"trace": "trace", "control": "control", "tracenomk801": "trace", "controlnomk801": "control"}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fish_id(name: str) -> str | None:
    match = re.match(r"^(\d{8})_(\d+)(?:_|$)", name)
    return f"{match[1]}_{int(match[2]):02d}" if match else None


def parse_protocol(content: bytes) -> dict:
    """Retain every valid event and malformed row; use exact millisecond timing."""
    lines = content.decode("utf-8-sig").splitlines()
    events, errors = [], []
    if not lines or lines[0].split() != ["Type", "Beg", "End"]:
        errors.append("unexpected protocol header")
    for number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        fields = line.split()
        try:
            if len(fields) != 3 or fields[0] not in {"Cycle", "Reinforcer"}:
                raise ValueError("unknown event or unexpected column count")
            values = [Decimal(v) for v in fields[1:]]
            if any(not v.is_finite() or v != v.to_integral_value() for v in values):
                raise ValueError("timestamps must be finite integer milliseconds")
            begin, end = map(int, values)
            if end <= begin:
                raise ValueError("nonpositive duration")
            events.append({"type": fields[0], "beg_ms": begin, "end_ms": end,
                           "duration_ms": end-begin, "line": number})
        except (ValueError, ArithmeticError) as exc:
            errors.append({"line": number, "text": line, "error": str(exc)})
    cs = sorted((e for e in events if e["type"] == "Cycle"), key=lambda e: e["beg_ms"])
    us = sorted((e for e in events if e["type"] == "Reinforcer"), key=lambda e: e["beg_ms"])
    for label, group in (("CS", cs), ("US", us)):
        if len({e["beg_ms"] for e in group}) != len(group):
            errors.append(f"duplicate {label} onsets")
    training = []
    for trial, c in enumerate(cs, 1):
        if not 15 <= trial <= 64:
            continue
        following = next(((i,u) for i,u in enumerate(us,1) if u["beg_ms"] >= c["beg_ms"]), None)
        training.append({"cs_trial": trial, "cs_line": c["line"],
                         "us_trial": following[0] if following else None,
                         "us_line": following[1]["line"] if following else None,
                         "latency_ms": following[1]["beg_ms"]-c["beg_ms"] if following else None})
    pairs = [e for e in training if e["cs_trial"] not in CATCHES]
    values = [e["latency_ms"] for e in pairs]
    def fixed(milliseconds: int) -> bool:
        return len(values) == 46 and all(v is not None and abs(v-milliseconds) <= 200 for v in values)
    schedule = "nonfixed-or-unpaired"
    for milliseconds, label in ((13000,"fixed-3s-trace"),(20000,"fixed-10s-trace"),(9000,"delay")):
        if fixed(milliseconds):
            schedule = label
            break
    if schedule == "nonfixed-or-unpaired" and len(values) == 46 and all(v is not None and 10200 <= v <= 13300 for v in values):
        schedule = "increasing-trace" if values[-1]-values[0] > 1000 and all(a-b < 100 for a,b in zip(values,values[1:])) else "variable-trace-review"
    catch_rows = [e for e in training if e["cs_trial"] in CATCHES]
    overlaps = [{"cs_trial": i,"cs_line": c["line"],"us_trial": j,"us_line": u["line"],
                 "overlap_ms": min(c["end_ms"],u["end_ms"])-max(c["beg_ms"],u["beg_ms"])}
                for i,c in enumerate(cs,1) for j,u in enumerate(us,1)
                if min(c["end_ms"],u["end_ms"]) > max(c["beg_ms"],u["beg_ms"])]
    return {"sha256": hashlib.sha256(content).hexdigest(), "errors": errors, "events": events,
            "counts": {"Cycle":len(cs),"Reinforcer":len(us)}, "schedule": schedule, "training": training,
            "catch_schedule_matches": len(catch_rows)==4 and all(e["latency_ms"] is None or e["latency_ms"]>20200 for e in catch_rows),
            "paired_us_numbers": [e["us_trial"] for e in pairs],
            "paired_us_numbers_unique_and_consecutive": len(pairs)==46 and all(e["us_trial"] is not None for e in pairs) and all(b["us_trial"]-a["us_trial"]==1 for a,b in zip(pairs,pairs[1:])),
            "cs_duration_outliers": [dict(trial=i,**e) for i,e in enumerate(cs,1) if not 9800<=e["duration_ms"]<=10200],
            "short_us_screen_only": [dict(trial=i,**e) for i,e in enumerate(us,1) if e["duration_ms"]<20],
            "cs_us_intersections": overlaps}


def existing_status(rid: str, full: bool) -> dict:
    from classical_conditioning.artifacts import verify_completed_parquet_set
    meta, data, qc = PROJECT/"Metadata", PROJECT/"Processed data"/rid, PROJECT/"Quality checks"/rid
    found = list(meta.glob(f"{rid}_*")) + [p for p in (data,qc) if p.exists()]
    staging = list(PROJECT.glob(f".{rid}-*"))
    result = {"state":"partial" if found or staging else "absent", "staging":list(map(str,staging)),"errors":[],"upstream_hashes":"pending"}
    marker = meta/f"{rid}_candidate-trial-outcomes-corrected_complete.json"
    if not marker.exists():
        return result
    try:
        v = verify_completed_parquet_set({"outcomes":data/"candidate-trial-outcomes-corrected.parquet","coverage":data/"candidate-trial-outcomes-corrected_coverage.parquet"},qc/"candidate-trial-outcomes-corrected_summary.json",marker,recipe="candidate-trial-outcomes-corrected",recording_id=rid)
        if v.summary.get("experiment")!="all3sTrace" or v.summary.get("config",{}).get("response_window_s")!=[0.0,13.0]:
            raise ValueError("trial summary experiment/window mismatch")
        m=json.loads((meta/f"{rid}_source_manifest.json").read_text())
        if m.get("recording_id")!=rid or v.summary.get("inputs",{}).get("protocol")!=m["artifacts"]["protocol"]["sha256"]:
            raise ValueError("source-manifest identity/protocol lineage mismatch")
        result.update(state="trial-outputs-verified-upstream-pending",raw_sources=m.get("sources",{}),identity=v.summary.get("identity"))
        if full:
            from classical_conditioning.analysis.trial_outcomes import verify_candidate_trial_outcomes
            verify_candidate_trial_outcomes(PROJECT,rid,metric_recipe="tail-candidate-corrected")
            result.update(state="complete-verified",upstream_hashes="verified")
    except Exception as exc:
        result.update(state="invalid-or-incomplete",errors=[str(exc)])
    return result


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audit-only",action="store_true",help="Compatibility option; this route always audits only")
    ap.add_argument("--verify-existing",action="store_true",help="Also perform large supported upstream hash checks")
    ap.add_argument("--output-prefix",default="trace3-additions-20261018-reviewed-final")
    args=ap.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+",args.output_prefix): raise ValueError("Invalid evidence name")
    meta=PROJECT/"Metadata"
    output=[meta/(args.output_prefix+s) for s in (".json",".md")]
    if any(p.exists() for p in output): raise FileExistsError("Existing evidence preserved; choose a new output prefix")
    intake=json.loads((meta/"trace3-additions-20261018-reconciled.json").read_text())
    fish,raw,invalid,scan_errors=defaultdict(list),defaultdict(list),[],[]
    for p in sorted(PKL_ROOT.rglob("*")):
        if p.is_file() and p.suffix.lower()==".pkl":
            rid=fish_id(p.stem)
            if rid: fish[rid].append({"path":str(p),"size_bytes":p.stat().st_size,"condition_alias":p.stem.split("_")[2].lower()})
            else: invalid.append(str(p))
    for root,dirs,names in os.walk("D:/",onerror=lambda e:scan_errors.append(str(e))):
        dirs[:]=[d for d in dirs if d not in {"$RECYCLE.BIN","System Volume Information"}]
        for name in names:
            rid=fish_id(name)
            if rid in fish and any(name.lower().endswith(s) for s in SUFFIXES.values()): raw[rid].append(Path(root)/name)
    rows=[]
    for rid,aliases in sorted(fish.items()):
        blockers,notes,groups,protocols=[],[],defaultdict(dict),[]
        for p in sorted(raw[rid]):
            kind=next(k for k,s in SUFFIXES.items() if p.name.lower().endswith(s))
            stem=p.name[:-len(SUFFIXES[kind])]
            groups[(str(p.parent),stem)][kind]=str(p)
            if kind=="protocol": protocols.append({"path":str(p),**parse_protocol(p.read_bytes())})
        complete=[parts for parts in groups.values() if set(parts)==set(SUFFIXES)]
        labels={ALIASES.get(a["condition_alias"],"unsupported:"+a["condition_alias"]) for a in aliases}
        condition=next(iter(labels)) if len(labels)==1 else "conflict"
        if len(aliases)!=1 or len(labels)!=1: blockers.append("pickle aliases require identity/condition reconciliation")
        if len(complete)!=1 or len(protocols)!=1: blockers.append(f"source reconciliation: {len(complete)} complete triplets, {len(protocols)} protocol candidates")
        raw_condition=None
        if len(complete)==1:
            raw_stem=Path(complete[0]["protocol"]).name[:-len(SUFFIXES["protocol"])]
            raw_condition=raw_stem.split("_")[2].lower()
            if ALIASES.get(raw_condition)!=condition: blockers.append("raw/pickle condition conflict")
            normal=re.sub(r"(?i)_(trace|control)nomk801_",r"_\1_",raw_stem).lower()
            if normal!=Path(aliases[0]["path"]).stem.lower(): blockers.append("alias difference beyond the explicit untreated condition suffix")
            if raw_condition in {"tracenomk801","controlnomk801"}:
                notes.append("Explicit NoMK801 untreated alias; current downstream parser needs supported canonical mapping")
                blockers.append("NoMK801 downstream alias mapping not implemented")
        for p in protocols:
            if p["errors"]: blockers.append("malformed/invalid protocol rows")
            if p["counts"]!={"Cycle":94,"Reinforcer":79}: blockers.append("nonstandard full event counts")
            if p["cs_duration_outliers"]: blockers.append("CS duration outside 10 ± 0.2 s")
            if condition=="trace":
                if p["schedule"]!="fixed-3s-trace": blockers.append("separate nonfixed/increasing protocol from fixed 3sTrace")
                if not p["catch_schedule_matches"] or not p["paired_us_numbers_unique_and_consecutive"]: blockers.append("catch schedule or nonunique/nonconsecutive paired-US events")
            if condition=="control" and p["schedule"]!="nonfixed-or-unpaired": blockers.append("control alias conflicts with systematic paired timing")
            if p["short_us_screen_only"]: notes.append("Short recorded US durations retained as descriptive anomalies; no approved exclusion threshold was established")
        existing=existing_status(rid,args.verify_existing)
        if len(protocols)==1 and existing.get("raw_sources"):
            if existing["raw_sources"].get("protocol",{}).get("sha256")!=protocols[0]["sha256"]: blockers.append("D: protocol differs from existing source lineage")
        if existing["state"] in {"partial","invalid-or-incomplete"}: blockers.append("partial/conflicting existing artifacts require preservation and reconciliation")
        rows.append({"recording_id":rid,"condition":condition,"raw_condition_alias":raw_condition,"pickle_aliases":aliases,"raw_paths":list(map(str,sorted(raw[rid]))),"complete_triplets":complete,"protocols":protocols,"existing":existing,"blockers":blockers,"notes":notes})
        print(rid,existing["state"],flush=True)
    contexts={(r["recording_id"][:8],str(Path(r["protocols"][0]["path"]).parent)) for r in rows if r["condition"]=="trace" and len(r["protocols"])==1 and r["protocols"][0]["schedule"]=="fixed-3s-trace" and not r["protocols"][0]["errors"]}
    for r in rows:
        if r["condition"]=="control" and len(r["protocols"])==1:
            matched=(r["recording_id"][:8],str(Path(r["protocols"][0]["path"]).parent)) in contexts
            r["matched_fixed_trace_day_and_directory"]=matched
            if not matched: r["blockers"].append("matched fixed-3sTrace control provenance remains unresolved for this acquisition day/directory")
    old_paths={a["path"] for f in intake["fish"] for a in f["pkl_files"]}
    fresh_paths={a["path"] for group in fish.values() for a in group}
    baseline=json.loads((meta/"trace3-review-20261018-preservation-baseline.json").read_text())
    changed=[r["path"] for r in baseline["files"] if not Path(r["path"]).exists() or sha(Path(r["path"]))!=r["sha256"]]
    old_raw={p for f in intake["fish"] for p in f["raw_files"]}
    fresh_raw={str(p) for paths in raw.values() for p in paths}
    import csv
    with (meta/"trace3-additions-20261018-reconciled.csv").open(encoding="utf-8-sig",newline="") as f:
        csv_rows=list(csv.DictReader(f))
    csv_pickle_paths={p for r in csv_rows for p in r["pkl_paths"].split(" | ") if p}
    review={"generated_utc":datetime.now(timezone.utc).isoformat(),"mode":"audit-only; processing route withdrawn","audit_code_sha256":sha(Path(__file__)),"pickle_files":sum(map(len,fish.values())),"identities":len(fish),"invalid_names":invalid,"D_scan_errors":scan_errors,"new_pickle_paths":sorted(fresh_paths-old_paths),"missing_pickle_paths":sorted(old_paths-fresh_paths),"csv_pickle_path_disagreement":sorted(fresh_paths^csv_pickle_paths),"new_raw_paths":sorted(fresh_raw-old_raw),"missing_raw_paths":sorted(old_raw-fresh_raw),"existing_counts":dict(Counter(r["existing"]["state"] for r in rows)),"protocol_file_counts":dict(Counter(p["schedule"] for r in rows for p in r["protocols"])),"preserved_metadata_changed":changed,"rows":rows}
    with output[0].open("x",encoding="utf-8") as f: json.dump(review,f,indent=2,ensure_ascii=False)
    with output[1].open("x",encoding="utf-8") as f:
        f.write("# Reviewed 3sTrace intake\n\nEarlier run/v2/v3/v4 and initial reviewed reports are preliminary and superseded here. Their PASS/ready counts must not launch processing. No per-fish 3sTrace processing was launched. The unsafe execution route has been removed; its original source is preserved beside the review baseline.\n\n")
        f.write("The earlier route missed the actual marker hash fields, used weak control provenance, lacked NoMK801 downstream mapping, applied an unsupported US-duration exclusion threshold, ignored malformed rows and could combine source components across directories. It also blocked all unambiguous work when any unrelated fish needed review.\n\n")
        f.write(f"Fresh recursive inventory: {review['pickle_files']} pickle files, {len(fish)} date/number identities. Existing status: {json.dumps(review['existing_counts'])}. Protocol-file classes: {json.dumps(review['protocol_file_counts'])}. Existing baseline metadata changed: {len(changed)}. No readiness count is asserted.\n\n")
        f.write("The actual versionless trial markers, summary hashes and trial-output parquet hashes were checked using the supported artifact validator. Large upstream artifact hashes and raw camera/tracking hashes remain pending unless explicitly marked verified. Fixed 3sTrace timing is checked in exact milliseconds for all 46 paired training trials at 13 ± 0.2 s; catches and uniqueness/continuity of paired US events are separate checks. Raw paired-US event numbers are observed evidence, not inferred from the historically independent configured US analysis blocks (18–63). All event rows and timestamps are retained in the JSON.\n\n")
        f.write("## Increasing and other nonfixed trace fish\n\n")
        nonfixed=[r['recording_id'] for r in rows if r['condition']=='trace' and any(p['schedule']!='fixed-3s-trace' for p in r['protocols'])]
        f.write(', '.join(nonfixed) if nonfixed else "None observed among available trace-labeled protocol files. Missing/ambiguous sources remain unresolved.")
        f.write("\n\n## All fish and source evidence\n\n")
        for r in rows:
            f.write(f"### {r['recording_id']}\n\nCondition {r['condition']}; raw alias {r['raw_condition_alias']}; processing {r['existing']['state']}.\n\n")
            for a in r['pickle_aliases']: f.write(f"- Pickle ({a['size_bytes']} bytes): `{a['path']}`\n")
            for p in r['raw_paths']: f.write(f"- Raw candidate: `{p}`\n")
            for p in r['protocols']:
                f.write(f"- Protocol `{p['path']}`: SHA-256 `{p['sha256']}`, {p['schedule']}, counts {p['counts']}, catch check {p['catch_schedule_matches']}, unique/consecutive paired US numbering {p['paired_us_numbers_unique_and_consecutive']}; observed paired US events {p['paired_us_numbers']}.\n")
                for e in p['cs_duration_outliers']: f.write(f"- CS duration anomaly: trial {e['trial']}, raw line {e['line']}, {e['beg_ms']}–{e['end_ms']} ms, duration {e['duration_ms']} ms.\n")
                for e in p['short_us_screen_only']: f.write(f"- Short US: event {e['trial']}, raw line {e['line']}, {e['beg_ms']}–{e['end_ms']} ms, duration {e['duration_ms']} ms. Descriptive flag only.\n")
                if p['cs_us_intersections']: f.write(f"- CS/US intersections (events and raw lines): {json.dumps(p['cs_us_intersections'])}\n")
            for issue in r['blockers']: f.write(f"- Unresolved: {issue}\n")
            for note in r['notes']: f.write(f"- Note: {note}\n")
            for error in r['existing']['errors']: f.write(f"- Integrity error: {error}\n")
            f.write("\n")
    print(json.dumps({k:review[k] for k in ('pickle_files','identities','existing_counts','protocol_file_counts','preserved_metadata_changed')},indent=2))


if __name__=="__main__":
    if hasattr(sys.stdout,"reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    main()
