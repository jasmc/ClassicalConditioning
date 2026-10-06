"""Finalize report tables and code provenance without changing source artifacts."""
import json,sys
from pathlib import Path
from classical_conditioning.artifacts import sha256_file
ROOT=Path(__file__).resolve().parents[1]; OUT=Path(sys.argv[1])
a=json.loads((OUT/'audit.json').read_text()); archived=json.loads((OUT/'archival_routes.json').read_text()); hist=json.loads((OUT/'historical_and_endpoints.json').read_text()); retries=json.loads((OUT/'historical_compression_retry.json').read_text())
configured={r['route']:r.get('configured',False) for r in a['routes']}
text='\n## Every experiment route: explicit coverage\n\nCurrent helper configuration supports '+str(sum(configured.values()))+' of '+str(len(configured))+' declared routes. Archival source has '+str(len(archived))+' branches including deprecated routes. Counts below are inventory evidence, not acceptance/exclusion counts. Protocol samples are one available recording per archival branch; full camera/tracking hash checks above cover four recordings in Delay/3sTrace only.\n\n| Route | Current configured | Raw protocol files at archival path | Historical pickles | Raw protocol sample | Archival defect |\n|---|---|---:|---:|---|---|\n'
for r in archived:
    sample=r['protocol_sample']; status='parsed' if sample and 'event_counts' in sample else ('parse error' if sample else 'no sample / placeholder')
    defect='duplicate key: '+','.join(r['duplicate_condition_keys']) if r['duplicate_condition_keys'] else ''
    text+=f"| {r['route']} | {'yes' if configured.get(r['route']) else 'no'} | {r['raw_protocol_files']} | {r['historical_pickle_files']} | {status} | {defect} |\n"
text+='\nCondition-specific current-route samples are recorded in `route_and_filter_comparisons.json`. Many-delay raw files use `delayLong` naming whereas current configuration expects control/delay; neither matched. The `10sTrace` 3sfixedtrace condition had no matching filename in that folder. These are unresolved routing findings, not evidence of absent biology.\n'
read=len([r for r in hist['historical_tables'] if r['status']=='read'])+len([r for r in retries if r['status']=='read'])
text+=f'\nHistorical table inspection authenticated and read {read} distinct sampled saved tables (23 attempted), including gzip-pickled .pkl files. No files were rewritten. Schema, row count, bout support counts where available, and index names are in the two historical JSON files. These saved tables are not a full numerical reconstruction oracle because exact source/recipe provenance is unavailable.\n'
p=ROOT/'docs/analysis/PREPROCESSING_CONTRACT_AUDIT_2026-10-07.md'; p.write_text(p.read_text().split('\n## Every experiment route: explicit coverage')[0]+text)
paths=[*ROOT.glob('scripts/audit_preprocessing_*.py'),ROOT/'tests/test_legacy_preprocessing_parity.py',ROOT/'legacy/helpers/analysis_utils.py',ROOT/'legacy/helpers/data_io.py',ROOT/'legacy/scripts/1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py',ROOT/'src/classical_conditioning/preprocessing/acquisition_timing.py',ROOT/'src/classical_conditioning/preprocessing/corrected_frame_preprocessing.py',p]
(OUT/'final_code_provenance.json').write_text(json.dumps({str(q.relative_to(ROOT)):sha256_file(q) for q in paths},indent=2))
(OUT/p.name).write_text(p.read_text())
print(f'Current configured {sum(configured.values())}/{len(configured)}; historical tables read {read}/{len(hist["historical_tables"])}')
