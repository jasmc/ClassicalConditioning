"""Read trusted local historical tables and source endpoints; never rewrite them."""
import json,sys
from pathlib import Path
import pandas as pd
import numpy as np
from classical_conditioning.artifacts import sha256_file
OUT=Path(sys.argv[1]); routes=json.loads((OUT/'archival_routes.json').read_text()); result={'historical_tables':[], 'raw_endpoints':[]}; seen=set()
for route in routes:
    folder=Path(route.get('path_save',''))/'Processed data/pkl files/1. Original'
    files=sorted(folder.glob('*.pkl'))
    if not files or str(files[0]) in seen: continue
    p=files[0]; seen.add(str(p))
    row={'route':route['route'],'path':str(p),'sha256':sha256_file(p),'status':'unread'}
    try:
        d=pd.read_pickle(p); row.update(status='read',rows=len(d),columns=[str(c) for c in d.columns],index_names=list(d.index.names))
        for col in ['Vigor (deg/ms)','Bout','Bout beg','Bout end']:
            if col in d: row[col]={'nonmissing':int(d[col].notna().sum()),'nonzero':int((d[col].fillna(0)!=0).sum())}
        del d
    except Exception as e: row.update(status='read_error',error=str(e))
    result['historical_tables'].append(row);(OUT/'historical_and_endpoints.json').write_text(json.dumps(result,indent=2));print(route['route'],row['status'],flush=True)
for project,rid in [(Path('J:/Digested Data/allDelay-full-v1'),'20221115_07'),(Path('J:/Digested Data/allDelay-full-v1'),'20221115_09'),(Path('F:/Digested Data/all3sTrace-full-v1'),'20230227_03'),(Path('F:/Digested Data/all3sTrace-full-v1'),'20230306_01')]:
    m=json.loads((project/'Metadata'/f'{rid}_source_manifest.json').read_text())
    p=Path(m['sources']['tracking']['path']); row={'recording_id':rid,'path':str(p),'exists':p.exists()}
    if p.exists():
        with p.open('rb') as f:
            header=f.readline();first=f.readline();f.seek(max(0,p.stat().st_size-16384));last=f.read().splitlines()[-2:]
        row.update(header_fields=len(header.split()),first_frame_token=first.split()[0].decode(),last_frame_tokens=[line.split()[0].decode(errors='replace') for line in last],last_row_field_counts=[len(line.split()) for line in last])
    result['raw_endpoints'].append(row)
(OUT/'historical_and_endpoints.json').write_text(json.dumps(result,indent=2))
