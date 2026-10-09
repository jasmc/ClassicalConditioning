from pathlib import Path
import json,hashlib
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
path=REPO/'configs/paper-figures/figure1-fgh-full-bout-correction-20261009.json'
config=json.loads(path.read_text())
validation=HERE/'C_DirectBins_validation.json'
for row in json.loads(validation.read_text())['outputs']:assert sha(Path(row['path']))==row['sha256']
for variant in config['variants']:
    if variant['variant']=='C_DirectBins':
        variant['validation']=str(validation);variant['validation_sha256']=sha(validation)
config['version2_data_manifest']=str(HERE/'data_manifest.json')
config['version2_data_manifest_sha256']=sha(HERE/'data_manifest.json')
config['version2_numeric_verification']=str(HERE/'numeric_verification.json')
config['baseline']='C/D samples: unbinned timepoints carrying complete-bout medians; Version2: finite unscaled direct baseline-bin means'
config['version2_approval']='Use baseline-bin means; guarantee displayed median zero (recommended)'
config['status']='Full-bout sample corrections retained; direct bins restored to own baseline-bin reference as approved'
path.write_text(json.dumps(config,indent=2)+'\n')
main=REPO/'configs/paper-figures/figure1-freeze.json';state=json.loads(main.read_text())
state['single_fish_heatmap_current_analysis']['status']=config['status']
main.write_text(json.dumps(state,indent=2)+'\n')
print(str(path))
