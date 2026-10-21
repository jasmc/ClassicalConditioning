"""Append authenticated fixed-10s Trace acquisitions to the authorized project.

No shared inventory/ledger is regenerated. All existing files and all D: writes
are protected. Raw aliases remain in source manifests; outcome conditions and
response windows are selected by all10sTrace. Shared controls keep one fish ID.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

PROJECT = Path(r"F:\Digested Data\all10sTrace-full-v1")
INVENTORY = Path(r"F:\Digested Data\trace10-additions-20261018-inventory.json")
TRACE3 = Path(r"F:\Digested Data\all3sTrace-full-v1")
SUFFIXES = {"camera": "cam.txt", "tracking": "mp tail tracking.txt", "protocol": "stim control.txt"}
ALIASES = {"10strace": "trace", "10sfixedtrace": "trace", "trace": "trace", "control": "control"}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save_new(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


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


def audit():
    data = json.loads(INVENTORY.read_text())
    fresh = sorted(Path(data["pkl_root"]).rglob("*.pkl"))
    saved = sorted(Path(a["path"]) for f in data["fish"] for a in f["pkl_files"])
    if fresh != saved: raise ValueError("Recursive pickle inventory changed; reconciliation required")
    ids = {f["recording_id"] for f in data["fish"]}
    raw, scan_errors = defaultdict(list), []
    for folder, dirs, names in os.walk("D:/", onerror=lambda e: scan_errors.append(str(e))):
        dirs[:] = [d for d in dirs if d not in {"$RECYCLE.BIN", "System Volume Information"}]
        for name in names:
            if name[:11] in ids and name.lower().endswith(".txt"):
                raw[name[:11]].append(Path(folder)/name)
    t3 = json.loads((TRACE3/"Metadata/trace3-additions-20261018-reconciled.json").read_text())
    t3ids = {f["recording_id"] for f in t3["fish"]}
    rows = []
    for fish in data["fish"]:
        rid = fish["recording_id"]
        groups, issues = defaultdict(dict), []
        aliases = fish["filename_conditions"]
        condition = ALIASES.get(aliases[0]) if len(aliases)==1 else None
        if len(fish["pkl_files"])!=1 or condition is None: issues.append("pickle identity/condition conflict")
        for p in raw[rid]:
            kind = next((k for k,s in SUFFIXES.items() if p.name.endswith(s)), None)
            if kind is None: issues.append("unrecognized same-ID raw file: "+str(p)); continue
            stem = p.name[:-len(SUFFIXES[kind])].rstrip("_")
            key = (str(p.parent), stem)
            if kind in groups[key]: issues.append("duplicate raw component")
            groups[key][kind] = str(p)
        complete = [(key,parts) for key,parts in groups.items() if set(parts)==set(SUFFIXES)]
        if len(complete)!=1 or len(raw[rid])!=3: issues.append("ambiguous/incomplete raw acquisition")
        row = {"recording_id": rid, "condition": condition, "pickle_aliases": fish["pkl_files"],
               "raw_candidates": list(map(str,raw[rid])), "shared_control_identity": rid in t3ids,
               "issues": issues, "notes": []}
        if len(complete)==1:
            (_,stem),parts = complete[0]
            row.update(recording_name=stem, sources=parts)
            if ALIASES.get(stem.split("_")[2].lower()) != condition: issues.append("raw/pickle condition disagreement")
            pkstem = Path(fish["pkl_files"][0]["path"]).stem
            def comparable(s):
                tokens=s.lower().split("_");tokens[2]=ALIASES.get(tokens[2],tokens[2]);return re.sub(r"[^a-z0-9]", "", "_".join(tokens))
            if comparable(stem)!=comparable(pkstem): issues.append("genotype/age/individual alias discrepancy")
            p = protocol(Path(parts["protocol"])); row["protocol"] = p
            if p["errors"]: issues.append("malformed/duplicate stimulus events")
            if not p["ordered"]: row["notes"].append("Raw log interleaves CS/US out of timestamp order; original rows retained, supported stages use stable onset order")
            if p["cs_duration_outliers"]: issues.append("non-10-second CS duration")
            if condition == "trace":
                expected = [t for t in range(15,min(p["counts"]["Cycle"],64)+1) if t not in {25,39,53,59}]
                paired = [t["cs_trial"] for t in p["training"] if len(t["near20"])==1]
                if paired != expected: issues.append("not fixed 20-second pairing on the observed training schedule")
            elif p["near20_training_count"] > 1: issues.append("control has repeated near-20 associations")
            if not p["standard_event_counts"]:
                if rid == "20250715_08" and p["counts"]=={"Cycle":53,"Reinforcer":54}:
                    row["notes"].append("INCOMPLETE stimulus schedule: supported stages may process observed events only; no standard-completion claim")
                else: issues.append("unreconciled event counts")
            if rid in t3ids:
                row["notes"].append("Shared historical control: same acquisition ID in 3s/10s inventories; one fish identity, all10sTrace outcomes [0,20)")
                other = list((TRACE3/"Metadata").glob(rid+"_*"))
                if other: issues.append("processing found in 3s project; reconcile/reuse before duplicating")
        existing = list((PROJECT/"Metadata").glob(rid+"_*"))
        existing += [p for p in (PROJECT/"Processed data"/rid,PROJECT/"Quality checks"/rid) if p.exists()]
        row["existing_paths"] = list(map(str,existing))
        if existing: issues.append("existing per-fish outputs require lineage verification")
        rows.append(row)
    contexts = {(r["recording_id"][:8],str(Path(r["sources"]["protocol"]).parent)) for r in rows
                if r["condition"]=="trace" and not r["issues"]}
    for r in rows:
        if r["condition"]=="control" and "sources" in r:
            if (r["recording_id"][:8],str(Path(r["sources"]["protocol"]).parent)) not in contexts:
                r["issues"].append("no authenticated same-day/directory 10sTrace context")
    return {"generated_utc": datetime.now(timezone.utc).isoformat(), "project": str(PROJECT),
            "inventory_path":str(INVENTORY), "inventory_sha256":sha(INVENTORY),
            "preserved_handoff_hashes":{str(INVENTORY.with_suffix(s)):sha(INVENTORY.with_suffix(s)) for s in (".json",".csv",".md")},
            "scan_errors":scan_errors,"rows":rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="trace10-additions-20261018-run")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--report-dir", type=Path,
                        help="Existing directory for audit evidence only; does not choose a processing project")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9-]+",args.run_id): raise ValueError("Invalid run ID")
    if args.report_dir is not None and not args.audit_only:
        raise ValueError("--report-dir is permitted only with --audit-only")
    if not args.audit_only and not (PROJECT/"Metadata").is_dir():
        raise FileNotFoundError("Authorized project/Metadata is unavailable; do not recreate a retired root implicitly")
    report_dir = args.report_dir or PROJECT/"Metadata"
    if not report_dir.is_dir():raise FileNotFoundError("Existing evidence directory required")
    prefix = report_dir/args.run_id
    if any(Path(str(prefix)+s).exists() for s in (".json",".md","-events.jsonl","-completion.json")):
        raise FileExistsError("Existing run evidence preserved")
    baseline = {str(p.resolve()):(p.stat().st_size,p.stat().st_mtime_ns) for p in PROJECT.rglob("*") if p.is_file()
                and p.name not in {args.run_id+".stdout.log",args.run_id+".stderr.log"}}
    data = audit()
    data["code_sha256"] = sha(Path(__file__))
    data["source_code_sha256"] = {str(p):sha(p) for p in
        (Path("src/classical_conditioning/intake.py"),Path("src/classical_conditioning/analysis/trial_outcomes.py"))}
    data["baseline"] = baseline
    save_new(prefix.with_suffix(".json"), data)
    with prefix.with_suffix(".md").open("x",encoding="utf-8") as f:
        f.write(f"# Fixed 10-second Trace additions\n\nDestination: {PROJECT}, explicitly selected by the author. Shared inventories, old results and D: are preserved. Per-fish stages use overwrite=False. Historical exclusions are ignored. Fixed Trace is tested using observed 20-second pairings and training catches 25/39/53/59. Controls retain acquisition IDs shared with historical 3s pickles and receive all10sTrace outcomes in [0,20); no independent-fish duplication is implied. Source condition aliases 10sTrace / 10sFixedTrace / trace map explicitly to canonical outcome condition trace; control remains control. The explicit-source intake API retains original D: paths and names, including missing-delimiter suffixes. Full camera/tracking raw SHA-256 values are generated by supported intake and recorded in each source manifest.\n\n")
        for r in data["rows"]:
            f.write(f"## {r['recording_id']} — {r['condition']}\n\n")
            for a in r['pickle_aliases']:f.write(f"- Pickle alias: `{a['path']}`\n")
            for p in r['raw_candidates']:f.write(f"- Raw: `{p}`\n")
            if 'protocol' in r:
                p=r['protocol'];f.write(f"- Stimulus SHA-256: `{p['sha256']}`; counts {p['counts']}; near-20 training associations {p['near20_training_count']}; observed unpaired training CSs {p['unpaired_training_cs']}. Full raw-line event mapping is in companion JSON.\n")
            for note in r['notes']:f.write('- Note: '+note+'\n')
            for issue in r['issues']:f.write('- Review required: '+issue+'\n')
            f.write('\n')
    print(json.dumps({"fish":len(data['rows']),"ready":sum(not r['issues'] for r in data['rows']),
                      "review":{r['recording_id']:r['issues'] for r in data['rows'] if r['issues']}},indent=2),flush=True)
    if args.audit_only:return
    def guard(event, values):
        candidates=[]
        if event=='open' and values and isinstance(values[0],(str,bytes,os.PathLike)):
            mode=values[1] or '';flags=values[2] if len(values)>2 else 0
            if any(c in mode for c in 'wax+') or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC):candidates=[values[0]]
        elif event in {'os.remove','os.rmdir','os.mkdir','os.chmod','os.utime'}:candidates=[values[0]]
        elif event=='os.rename':candidates=values[:2]
        for v in candidates:
            if isinstance(v,(str,bytes,os.PathLike)):
                p=Path(os.fsdecode(v)).resolve()
                if p.drive.lower()=='d:' or str(p) in baseline or 'Results (paper)' in p.parts:
                    raise PermissionError(f'Preservation guard blocked {event}: {p}')
    sys.addaudithook(guard)
    from classical_conditioning.intake import RecordingSources, intake_recording
    from classical_conditioning.preprocessing.corrected_frame_preprocessing import build_corrected_preprocessing
    from classical_conditioning.preprocessing.candidate_metrics_from_corrected_frames import build_candidate_activity_metrics_from_corrected
    from classical_conditioning.analysis.movement_state import build_candidate_movement_state
    from classical_conditioning.analysis.temporal_profiles import build_candidate_temporal_profiles
    from classical_conditioning.analysis.trial_outcomes import build_candidate_trial_outcomes, verify_candidate_trial_outcomes
    results=[];started=time.time()
    for index,row in enumerate(data['rows'],1):
        rid=row['recording_id'];result={'recording_id':rid,'condition':row['condition'],'stages':[],'incomplete_protocol':not row.get('protocol',{}).get('standard_event_counts',False)}
        if row['issues']:
            result.update(status='review-required',issues=row['issues']);results.append(result);continue
        print(f'START {rid} [{index}/{len(data["rows"])}]',flush=True)
        try:
            parts={k:Path(v) for k,v in row['sources'].items()}
            if sha(parts['protocol'])!=row['protocol']['sha256']:raise ValueError('Protocol changed after audit')
            sources=RecordingSources(row['recording_name'],rid,**parts)
            intake_recording(parts['protocol'].parent,PROJECT,recording_id=rid,sources=sources,overwrite=False)
            result['stages'].append('intake');print(f'STAGE {rid} intake',flush=True)
            for name,fn,kwargs in (
                ('corrected-preprocess',build_corrected_preprocessing,{}),
                ('candidate-metrics',build_candidate_activity_metrics_from_corrected,{}),
                ('movement-state',build_candidate_movement_state,{'metric_recipe':'tail-candidate-corrected'}),
                ('temporal-profiles',build_candidate_temporal_profiles,{'metric_recipe':'tail-candidate-corrected','experiment_name':'all10sTrace'}),
                ('trial-outcomes',build_candidate_trial_outcomes,{'metric_recipe':'tail-candidate-corrected','experiment_name':'all10sTrace'})):
                fn(PROJECT,rid,overwrite=False,**kwargs);result['stages'].append(name);print(f'STAGE {rid} {name}',flush=True)
            verified=verify_candidate_trial_outcomes(PROJECT,rid,metric_recipe='tail-candidate-corrected')
            if verified.summary['experiment']!='all10sTrace' or verified.summary['identity']['condition_id']!=row['condition']:raise ValueError('Final identity/experiment mismatch')
            result.update(status='completed',lineage_verified=True)
        except Exception as exc:
            traceback.print_exc();result.update(status='failed',error=repr(exc))
        results.append(result)
        with Path(str(prefix)+'-events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(result)+'\n')
        print(f'DONE {rid} {result["status"]} elapsed={time.time()-started:.1f}s',flush=True)
    changed=[p for p,s in baseline.items() if not Path(p).exists() or (Path(p).stat().st_size,Path(p).stat().st_mtime_ns)!=tuple(s)]
    changed_handoffs=[p for p,h in data['preserved_handoff_hashes'].items() if sha(p)!=h]
    completion={'results':results,'status_counts':dict(Counter(r['status'] for r in results)), 'existing_files_changed':changed,
                'preserved_handoffs_changed':changed_handoffs,'elapsed_seconds':time.time()-started}
    save_new(Path(str(prefix)+'-completion.json'),completion)
    print(json.dumps(completion,indent=2),flush=True)
    if changed or changed_handoffs or any(r['status']!='completed' for r in results):raise SystemExit(1)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    main()
