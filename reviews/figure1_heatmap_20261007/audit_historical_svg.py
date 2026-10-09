"""Read-only SVG/Git provenance comparison; preserve evidence in this review."""
from pathlib import Path
import subprocess
import hashlib
import json
import xml.etree.ElementTree as ET
from datetime import datetime

REPO=Path(__file__).resolve().parents[2]
HERE=Path(__file__).parent
SVG=Path('F:/Results (paper)/2025_delay/Processed data/20221115_07_delay_blue-1_mitfaminusminus,elavl3gff,10uasgcamp6fef05_6dpf_scaled vigor heatmap aligned to CS_cmap_managua_r_vlim_auto.svg')
root=ET.parse(SVG).getroot()
ns={'s':'http://www.w3.org/2000/svg','dc':'http://purl.org/dc/elements/1.1/'}
def git(*args):
    return subprocess.check_output(['git','-c','core.fsmonitor=false',*args],cwd=REPO)
records=[]
for commit,date in [('2f63ef4','2026-02-14'),('bf46bf7','2026-03-24')]:
    for name in ['2_ExampleFishPlotting.py','1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py',
                 'general_configuration.py','analysis_utils.py']:
        content=git('show',f'{commit}:{name}')
        snapshot=HERE/'history'/f'{date}_{name}'
        if snapshot.exists():
            assert snapshot.read_bytes()==content
        else:
            snapshot.write_bytes(content)
        records.append({'commit':git('rev-parse',commit).decode().strip(),'date':date,
                        'git_path':name,'snapshot':str(snapshot),
                        'sha256':hashlib.sha256(content).hexdigest()})
stat=SVG.stat()
report={'svg':str(SVG),'sha256':hashlib.sha256(SVG.read_bytes()).hexdigest(),'size_bytes':stat.st_size,
        'embedded_generation_time':root.find('.//dc:date',ns).text,
        'embedded_creator':[node.text for node in root.findall('.//dc:title',ns)],
        'filesystem_creation_time_host_local':datetime.fromtimestamp(stat.st_ctime).isoformat(),
        'filesystem_last_write_time_host_local':datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'visible_text':[''.join(node.itertext()).strip() for node in root.findall('.//s:text',ns)],
        'svg_axes_ids':[node.attrib['id'] for node in root.findall('.//s:g',ns) if node.attrib.get('id','').startswith('axes_')],
        'snapshots':records,
        'example_script_changes_february_to_march':git('diff','2f63ef4','bf46bf7','--','2_ExampleFishPlotting.py').decode(),
        'preprocessing_and_helpers_changes_february_to_march':git('diff','2f63ef4','bf46bf7','--',
          '1_Preprocessing_IndividualFishPlotting_ProtocolPlotting_Discarding.py','general_configuration.py','analysis_utils.py').decode(),
        'certainty':'SVG date and visible contents are direct evidence; generating worktree/commit and numeric colour limits are not recorded in the SVG.'}
(HERE/'historical_svg_20260216_evidence.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:report[k] for k in ['embedded_generation_time','embedded_creator','visible_text','sha256']},indent=2))
