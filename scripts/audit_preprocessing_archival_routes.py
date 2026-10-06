"""Static archival route audit plus read-only protocol samples; no legacy imports."""
import ast,json,sys
from pathlib import Path
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.ingestion.readers import read_protocol
ROOT=Path(__file__).resolve().parents[1]
tree=ast.parse((ROOT/'legacy/modules/my_experiment_specific_variables.py').read_text(encoding='utf-8-sig'))
rows=[]
for match in [n for n in tree.body if isinstance(n,ast.Match)]:
    for case in match.cases:
        if not isinstance(case.pattern,ast.MatchValue): continue
        route=ast.literal_eval(case.pattern.value); constants={}; paths={}; duplicate=[]
        for n in case.body:
            if not isinstance(n,ast.Assign): continue
            for target in n.targets:
                if not isinstance(target,ast.Name): continue
                key=target.id
                if key in ('path_home','path_save') and isinstance(n.value,ast.Call):
                    try: paths[key]=ast.literal_eval(n.value.args[0])
                    except Exception: pass
                try: constants[key]=ast.literal_eval(n.value)
                except Exception: pass
                if key=='cond_dict' and isinstance(n.value,ast.Dict):
                    keys=[]
                    for k in n.value.keys:
                        if isinstance(k,ast.Name): val=constants.get(k.id,k.id)
                        else:
                            try: val=ast.literal_eval(k)
                            except Exception: val=ast.unparse(k)
                        if val in keys: duplicate.append(val)
                        keys.append(val)
        p=Path(paths.get('path_home','')); placeholder=str(p)=='.'; protocols=[] if placeholder else sorted(p.glob('*stim control.txt'))
        saved=Path(paths.get('path_save',''))/'Processed data/pkl files/1. Original'
        row={'route':route,**paths,'raw_path_placeholder':placeholder,'raw_protocol_files':len(protocols),'duplicate_condition_keys':duplicate,'historical_pickle_files':len(list(saved.glob('*.pkl'))) if str(saved)!='.' else 0,'protocol_sample':None}
        if protocols:
            sample=protocols[0]
            try:
                d=read_protocol(sample).frame;row['protocol_sample']={'path':str(sample),'sha256':sha256_file(sample),'event_counts':{str(k):int(v) for k,v in d.Type.value_counts().items()}}
            except Exception as e:row['protocol_sample']={'path':str(sample),'error':str(e)}
        rows.append(row)
out=Path(sys.argv[1]);(out/'archival_routes.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
