"""Read gzip-pickled historical tables stored with .pkl extension."""
import json,sys
from pathlib import Path
import pandas as pd
OUT=Path(sys.argv[1]); d=json.loads((OUT/'historical_and_endpoints.json').read_text()); rows=[]
for r in d['historical_tables']:
    if r['status']!='read_error':continue
    p=Path(r['path'])
    with p.open('rb') as f: magic=f.read(2)
    row={'path':str(p),'route':r['route'],'magic_hex':magic.hex()}
    try:
        frame=pd.read_pickle(p,compression='gzip' if magic==b'\x1f\x8b' else 'infer'); row.update(status='read',rows=len(frame),columns=[str(c) for c in frame.columns],index_names=list(frame.index.names)); del frame
    except Exception as e:row.update(status='read_error',error=str(e))
    rows.append(row);(OUT/'historical_compression_retry.json').write_text(json.dumps(rows,indent=2));print(r['route'],row['status'],flush=True)
