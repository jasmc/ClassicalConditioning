"""Read-only SSD data-coverage audit; write only a small handoff report."""
from collections import Counter
from pathlib import Path
import json
import hashlib

SSD = Path('J:/')
HANDOFF = SSD / 'ClassicalConditioning-Mac-20261009'


def main():
    report = {'scope':'code/figure transfer only; no fish data moved', 'raw_inventories':[], 'selected_artifacts':[]}
    for assay, project, raw in [
        ('allDelay', SSD/'Digested Data/allDelay-full-v1', SSD/'Raw Data/allDelay'),
        ('all3sTrace', Path('F:/Digested Data/all3sTrace-full-v1'), SSD/'Raw Data/all3sTtrace')]:
        inv = json.loads((project/'Metadata/recording_inventory.json').read_text())
        missing=[]; different=[]; count=0
        for record in inv['records']:
            for components in record['components'].values():
                for item in components:
                    path=raw/item['relative_path']; count+=1
                    if not path.is_file(): missing.append(str(path))
                    elif path.stat().st_size != item['size_bytes']: different.append(str(path))
        report['raw_inventories'].append({'assay':assay,'record_count':inv['record_count'],
            'statuses':dict(Counter(r['status'] for r in inv['records'])), 'checked_files':count,
            'missing_files':missing,'size_mismatches':different, 'method':'existence and size against historical inventory; raw bytes not rehashed'})
    repo=HANDOFF/'ClassicalConditioning'
    for name in ['figure1-fgh-version12-freeze-20261009.json','figure2-DE-B-freeze-20261009.json','figure2-G-logmedian-freeze-20261009.json']:
        obj=json.loads((repo/'configs/paper-figures'/name).read_text())
        def walk(value):
            if isinstance(value,dict):
                if 'path' in value and 'sha256' in value:
                    original=Path(value['path']); text=str(original).replace('\\','/')
                    if text.startswith('C:/Users/joaquim/Documents/ClassicalConditioning/'):
                        target=repo/text.split('ClassicalConditioning/',1)[1]
                    elif text.startswith('F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/'):
                        target=HANDOFF/'pc-outputs'/text.split('/outputs/',1)[1]
                    else: target=original
                    row={'selection':name,'source':str(original),'ssd_path':str(target),'exists':target.is_file()}
                    if target.is_file():
                        row['hash_matches_selection']=hashlib.sha256(target.read_bytes()).hexdigest()==value['sha256']
                    report['selected_artifacts'].append(row)
                for child in value.values():walk(child)
            elif isinstance(value,list):
                for child in value:walk(child)
        walk(obj)
    manifest=[json.loads(line) for line in (HANDOFF/'transfer-manifest.jsonl').read_text().splitlines()]
    report['transfer_groups']=dict(Counter(r['group'] for r in manifest))
    report['trace_processed_on_ssd']=(SSD/'Digested Data/all3sTrace-full-v1').exists()
    (HANDOFF/'ssd-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
