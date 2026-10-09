from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
manifest=HERE/'data_manifest.json'
variants=[]
for kind in ['C_BoutSamples','D_BoutSamples','C_DirectBins']:
    path=HERE/f'{kind}_validation.json';record=json.loads(path.read_text())
    for output in record['outputs']:assert sha(Path(output['path']))==output['sha256']
    variants.append({'variant':kind,'validation':str(path),'validation_sha256':sha(path)})
config=REPO/'configs/paper-figures/figure1-fgh-full-bout-correction-20261009.json'
record={'figure':'Figure 1','panels':['F','G','H'],'status':'User-requested processing corrections implemented; previous freeze preserved',
        'fish':{'F':'20221115_07','G':'20230310_08','H':'20221115_09'},
        'current_primary_variant':'C_BoutSamples','data_manifest':str(manifest),'data_manifest_sha256':sha(manifest),
        'numeric_verification':str(HERE/'numeric_verification.json'),'variants':variants,
        'summary':str(REPO/'reviews/fgh_latest_summary_20261009/index.html'),
        'historical_freeze':str(REPO/'reviews/fgh_v1_trace20230310_08_20261009/frozen-version1/freeze.json'),
        'baseline':'same unbinned timepoints with complete-bout median values for P10/P50/P90 in every version',
        'scope':'F/G/H only; no change to A-D, E, or whole-figure assembly'}
config.write_text(json.dumps(record,indent=2)+'\n')
main=REPO/'configs/paper-figures/figure1-freeze.json';old=json.loads(main.read_text())
old['single_fish_heatmap_current_analysis']={'config':str(config),'status':record['status']}
main.write_text(json.dumps(old,indent=2)+'\n')
print(str(config))
