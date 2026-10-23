"""Append an authenticated conversation checkpoint without rebuilding scientific panels."""
from pathlib import Path
import argparse, base64, hashlib, html, io, json, re, sys, zipfile
from html.parser import HTMLParser
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from classical_conditioning.figure_workspace import Snapshot, publish_state

def digest(data): return hashlib.sha256(data).hexdigest()
def packed(value): return (json.dumps(value, indent=2) + '\n').encode()
def link(name, data, mime, label):
    return f'<a download="{html.escape(name)}" href="data:{mime};base64,{base64.b64encode(data).decode()}">{html.escape(label)}</a>'

TOPICS = [
 ('Purpose; independent review before choosing/freeze', 'orientation', 'Author goal and unfinished decisions'),
 ('Figure 3 individual evidence; Figure 4 group comparisons', 'questions', 'Author correction on figure scope'),
 ('Legacy metric only; G outcome versus classifier inputs', 'quantities', 'Explicit outcome confirmation'),
 ('Heatmap ±0.25, managua_r, undefined black, condition colors', 'quantities', 'Display mappings and estimator distinction'),
 ('All 57 fish and illustrative main-paper examples', 'atlas', 'Atlas versus chosen illustrations'),
 ('Data availability, Trace gap and JOAQUIM disconnection', 'data', 'Authenticated pilot and missing inputs'),
 ('Four original legacy versions, windows and thresholds', 'legacy', 'Rules are not interchangeable'),
 ('Current and historical learner counts with denominators', 'counts', 'Separate runs/input policies'),
 ('Discarding, eligibility, undefined outcomes and retention', 'exclusions', 'Projections versus exact historical cohorts'),
 ('Old scalar grids, heatmap source and missing saved grids', 'legacy', 'Located outputs versus source-supported designs'),
 ('Initial mixed-method draft and author rejection', 'drafts', 'One selected analysis path in main paper'),
 ('Current a–e descriptive draft, examples and score', 'drafts', 'No binary G classification'),
 ('BLUP rationale, legacy defects and corrected prototype', 'blups', 'Implementation versus validation'),
 ('Was corrected BLUP work finished? No', 'blups', 'Full validation/comparison still outstanding'),
 ('Other analysis alternatives and Figure 2 G background', 'background', 'Discussed versus executed alternatives'),
 ('More learners/retention do not establish accuracy', 'choose', 'No demonstrated winning method'),
 ('Final main/supplement panels and freeze remain undecided', 'panels', 'Worksheet does not approve science'),
 ('Sources, logs, configurations and code downloads', 'sources', 'Original evidence retained'),
 ('Reusable skill creation and validation', 'conversation-record', 'Installed skill snapshot downloadable below'),
 ('Download-link repair and earlier browser checks', 'conversation-record', 'Blob handler and unique source filenames'),
 ('Durable HTML, browser notes and chat deletion', 'conversation-record', 'Export worksheet JSON separately'),
]

class Inventory(HTMLParser):
    def __init__(self): super().__init__(); self.ids=[]; self.anchors=[]; self.downloads=[]
    def handle_starttag(self, tag, attrs):
        a=dict(attrs)
        if 'id' in a: self.ids.append(a['id'])
        if tag=='a':
            if a.get('href','').startswith('#'): self.anchors.append(a['href'][1:])
            if 'download' in a: self.downloads.append((a['download'], a.get('href','')))

def main(out):
    ticketpath=out/'conversation-preservation-ticket-20261023.json'
    ticket=json.loads(ticketpath.read_text()); record=Path(ticket['record']); path=Path(ticket['html']['path'])
    checkpoint=ROOT/'records/figure-reviews/figure-03/conversation-checkpoint-2026-10-23.md'
    summary=ROOT/'records/figure-reviews/figure-03/conversation-summary-2026-10-16.md'
    skill=Path(ticket['skill_root'])
    skillfiles=sorted(p for p in skill.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='.DS_Store')
    inputs=[record,path,ticketpath,checkpoint,summary,Path(__file__),*skillfiles,*map(Path,ticket['protected'])]
    snapshot=Snapshot(inputs)
    expected={str(record):ticket['record_sha256'],str(path):ticket['html']['sha256'],**ticket['protected']}
    for p,h in expected.items():
        if digest(Path(p).read_bytes())!=h: raise ValueError('Changed input: '+p)
        snapshot.hashes[str(Path(p).resolve())]=h
    before=path.read_text(); state=json.loads(record.read_text())
    if 'id="conversation-record"' in before: raise ValueError('Checkpoint already present')
    checkpointbytes=checkpoint.read_bytes(); summarybytes=summary.read_bytes()
    skillbuf=io.BytesIO()
    with zipfile.ZipFile(skillbuf,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for p in skillfiles:
            info=zipfile.ZipInfo('analysis-review-guide/'+p.relative_to(skill).as_posix(),date_time=(2026,10,23,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,p.read_bytes())
    skillbytes=skillbuf.getvalue()
    rows=''.join('<tr><td>'+html.escape(topic)+'</td><td><a href="#'+anchor+'">'+html.escape(anchor)+'</a></td><td>'+html.escape(note)+'</td></tr>' for topic,anchor,note in TOPICS)
    chapter='''<section id="conversation-record"><h2>15. Conversation record and preservation</h2>
<p class="tag">Coverage checkpoint · 23 October 2026</p>
<p>This chapter preserves the substantive discussion, your instructions and corrections, completed outputs and unfinished work so you can review independently after deleting the chat. It extends the original 17 October scientific snapshot; it does not rerun analyses or select a method.</p>
<div class="callout warning"><b>Before deleting the chat, export your worksheet notes.</b> Notes are saved in this browser, not written into this HTML. Use <a href="#panels">chapter 13 → Export notes</a>, and keep the JSON with this file. Moving to a different browser or file origin can make existing local notes unavailable.</div>
<h3>The decisions that matter</h3><ul>
<li>Use the legacy physical metric only. Main trial plots use the Figure 2 G outcome; exact legacy classifier inputs remain separately available. Heatmaps use managua_r at ±0.25 and black undefined values.</li>
<li>The main Figure 3 must follow one selected analysis path. Control/Learner/Nonlearner group comparisons within and across experiments belong to Figure 4.</li>
<li>Use a few openly illustrative fish in the main paper and all fish in supplementary review. Retain availability, eligibility and uncertainty distinctions.</li>
<li>The current main draft is a continuous descriptive G acquisition contrast and three examples. It has no binary learner classification. It has not been established as superior.</li>
<li>Four legacy methods ran on the current pilot; WIP flags the most Delay fish. Counts alone do not establish accuracy or justify discarding fish.</li></ul>
<div class="callout warning"><b>The corrected BLUP request remains unfinished.</b> A prototype, initial fit, five-refit smoke check and eight implementation tests exist. Full bootstrap/calibration, independent simulation validation, recovery analysis, sensitivity comparison and comparison figures were not completed. No validated corrected-BLUP learner count exists. No final scientific winner or freeze was approved in this chat.</div>
<h3>Coverage index</h3><p>The detailed scientific explanations remain in chapters 1–14. This index connects the discussion to those chapters; the full dated checkpoint below records qualifications, numerical counts, paths and work status.</p>
<div class="table-wrap"><table><thead><tr><th>Discussion topic</th><th>Detailed chapter</th><th>Preserved distinction</th></tr></thead><tbody>'''+rows+'''</tbody></table></div>
<h3>Reusable skill and repaired links</h3><p>We created the personal <code>analysis-review-guide</code> skill for source-grounded explanations, alternatives, exclusions, outputs and independent review. The archive below captures the currently installed skill and its supporting files. Later refinements may come from other chats; the archive does not attribute every current feature to this conversation.</p>
<p>The earlier link check found that embedded data-URL downloads failed in the in-app browser. A guarded repair added Blob download handling and disambiguated duplicate source names. Fourteen chapter anchors, 17 embedded payloads and three external references were checked then; real CSV and Python downloads matched their sources. This revision preserves that handler and all earlier embedded payloads.</p>
<h3>Offline preservation package</h3><p>Save this HTML on accessible storage. All core data, fish cards, figures and source downloads remain embedded. Localhost URLs are temporary conveniences. External references still require a network; source paths are provenance, not bundled repositories.</p><ul><li>'''+link(checkpoint.name,checkpointbytes,'text/markdown;charset=utf-8','Download the full 23 October conversation checkpoint')+'''</li><li>'''+link('analysis-review-guide-installed-20261023.zip',skillbytes,'application/zip','Download the installed review-guide skill snapshot')+'''</li></ul>
<details><summary>Read the full dated conversation checkpoint</summary><pre>'''+html.escape(checkpointbytes.decode())+'''</pre></details>
<details><summary>Read the earlier 16 October summary</summary><pre>'''+html.escape(summarybytes.decode())+'''</pre></details>
<p>The earlier summary is also available through its original embedded download in <a href="#sources">chapter 14</a>. These summaries preserve substantive content, not a verbatim transcript or every tool log. Keep exported notes separately.</p>
<h3>What remains to do scientifically</h3><p>Complete and validate the corrected BLUP comparison; review control calibration, coverage, assumptions, retention and sensitivity analyses; resolve Trace/cohort choices as needed; then select the single analysis path and main/supplementary panels. Only a later explicit freeze request starts the freeze process.</p></section>'''
    notice='''<div class="callout"><b>Conversation preserved through 23 October 2026.</b> Read <a href="#conversation-record">chapter 15 for the coverage index, full checkpoint and skill archive</a>. Export browser-local worksheet notes separately before deleting the chat. The scientific outputs retain their original dated snapshot.</div>'''
    after=before.replace('<p class="reading">','<a href="#conversation-record">15. Conversation record and preservation</a><p class="reading">',1)
    marker='<section id="orientation"><h2>1. Start here</h2>'
    if after.count(marker)!=1 or after.count('</main>')!=1: raise ValueError('Unexpected guide structure')
    after=after.replace(marker,marker+notice,1).replace('</main>',chapter+'</main>',1)
    extract=lambda s:re.search(r'<script id="guide-data" type="application/json">(.*?)</script>',s,re.S)[1]
    assert extract(before)==extract(after)
    scripts=lambda s:re.findall(r'<script\b[^>]*>.*?</script>',s,re.S)
    assert scripts(before)==scripts(after)
    old=Inventory(); old.feed(before); new=Inventory(); new.feed(after)
    assert len(new.ids)==len(set(new.ids)), 'Duplicate IDs'
    assert set(new.anchors)<=set(new.ids), 'Broken section anchors'
    assert all(item in new.downloads for item in old.downloads), 'Original download changed'
    assert re.findall(r'<img\b[^>]*>',before)==re.findall(r'<img\b[^>]*>',after)
    data=json.loads(extract(after))
    counts={k:len(data[k]) for k in ['fish','cards','decisions','scores']}
    assert counts=={'fish':57,'cards':57,'decisions':228,'scores':57}
    for name,uri in new.downloads:
        if uri.startswith('data:') and ';base64,' in uri: base64.b64decode(uri.split(',',1)[1],validate=True)
    report={'date':'2026-10-23','scope':ticket['scope'],'html_sha256':digest(after.encode()),'before_sha256':digest(before.encode()),'counts':counts,'all_original_scripts_images_and_downloads_preserved':True,'note_schema_storage_key_and_values_not_modified':True,'anchors_resolve':True,'download_count':len(new.downloads),'coverage_topics':len(TOPICS),'analysis_recomputed':False,'protected_states':ticket['protected'],'source_hashes':{str(p):digest(p.read_bytes()) for p in [checkpoint,summary,*skillfiles,Path(__file__)]},'browser_verification':'Pending separate recorded check'}
    reportbytes=packed(report); reportpath=out/'conversation-preservation-20261023.json'
    history=out/'history'/'before-conversation-preservation-20261023';history.mkdir(parents=True,exist_ok=False)
    state['html']['sha256']=report['html_sha256'];state['conversation_preserved_through']='2026-10-23'
    state['conversation_checkpoint']={'path':str(out/checkpoint.name),'sha256':digest(checkpointbytes),'report':str(reportpath),'report_sha256':digest(reportbytes)}
    state['verification_before_conversation_preservation']=state.pop('verification',None)
    state['verification']={'path':str(reportpath),'sha256':digest(reportbytes)}
    publish_state(record,state,snapshot,artifacts=[(history/path.name,before.encode()),(history/'guide-record.json',record.read_bytes()),(out/checkpoint.name,checkpointbytes),(out/'analysis-review-guide-installed-20261023.zip',skillbytes),(reportpath,reportbytes),(out/'source-conversation-preservation.py',Path(__file__).read_bytes())],evidence_updates=[(path,after.encode())])
    print(json.dumps({k:report[k] for k in ['html_sha256','counts','download_count','coverage_topics','analysis_recomputed']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    main(parser.parse_args().output_dir.resolve())
