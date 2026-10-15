"""Match Figure 2 F/I pending placeholders to the current C presentation.

No scientific regeneration: edit the authenticated active SVG, preserve all other
subtrees, and publish its matching HTML/evidence before the state pointer.
"""
from pathlib import Path
import sys,json,copy,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from classical_conditioning.figure_workspace import Snapshot,tag,sha,file_hash,publish_state,measured_presentation
from build_figure_legend_review import svg_editing_page
STATE=ROOT/'docs/paper/figures/figure-02/current-state.json'
TICKET=Path('/tmp/fig2-placeholder-ticket.json')
TEMP=Path('/tmp/figure2-placeholders.svg')
def encode(d):return (json.dumps(d,indent=2,allow_nan=False)+'\n').encode()
def prepare():
    snapshot=Snapshot([]);snapshot.hashes=json.loads(TICKET.read_text());snapshot.hashes[str(Path(__file__).resolve())]=file_hash(__file__);snapshot.check()
    state=json.loads(STATE.read_text());draft=state['draft']
    for key in ('working_svg','review','legend','semantic_registry'):
        a=draft[key]
        if file_hash(a['path'])!=a['sha256']:raise ValueError('Preserve edited input: '+a['path'])
    svgpath=Path(draft['working_svg']['path']);root=ET.parse(svgpath).getroot()
    before={n.get('id'):sha(ET.tostring(n)) for n in root if n.get('id') not in ('panel-F','panel-I')}
    # C's named artists are the authenticated presentation reference.
    lookup={n.get('id'):n for n in root.iter() if n.get('id')}
    pending=lookup['fig2__c0__heatmap__pending'];ctext=next(pending.iter(tag('text')))
    units_per_pt=1800/(183*72/25.4)
    registry={};changes={}
    for pid in ('F','I'):
        group=lookup['panel-'+pid];rect=next(group.iter(tag('rect')))
        x,y,w,h=[float(rect.get(k)) for k in ('x','y','width','height')]
        for n in list(group):group.remove(n)
        frame=ET.SubElement(group,tag('rect'),dict(id=f'fig2__{pid.lower()}__placeholder__frame',x=str(x),y=str(y+65),width=str(w),height=str(h-65),fill='white',stroke='black',**{'stroke-width':str(.5*units_per_pt)}))
        label=ET.SubElement(group,tag('text'),dict(id=f'fig2__{pid.lower()}__placeholder__panel-letter',x=str(x),y=str(y+25),**{'font-family':'DejaVu Sans','font-size':str(10*units_per_pt),'font-weight':'bold'}));label.text=pid
        note=ET.SubElement(group,tag('text'),dict(id=f'fig2__{pid.lower()}__placeholder__pending',x=str(x+w/2),y=str(y+65+(h-65)/2),**{'font-family':'DejaVu Sans','font-size':str(7*units_per_pt),'font-weight':'normal','fill':'black','text-anchor':'middle','dominant-baseline':'central'}));note.text='Pending'
        for node,role in [(frame,'background.panel'),(label,'label.panel_letter'),(note,'annotation.note')]:
            key=node.get('id');registry[key]=dict(element_id=key,scientific_role=role,figure_id='fig2',panel_id=pid.lower(),artist_type='SVG.'+node.tag.split('}')[-1],coordinate_system='assembly user units',scientific_context={'quantity':'pending placeholder; no scientific results','units':'not applicable'},geometry=dict(node.attrib),presentation_reference={'scope':'Figure 2 C','element_id':'fig2__c0__heatmap__pending' if node is note else 'fig2__c0__heatmap__spine-left' if node is frame else 'fig2__c0__heading__panel-letter'},status='unfrozen presentation only')
        changes[pid]={'white_fill':True,'solid_black_border_pt':.5,'pending_font_pt':7,'panel_letter_font_pt':10,'details':'Explanations remain in the existing associated legend; no invented tick values or data for these different plot families.'}
    after={n.get('id'):sha(ET.tostring(n)) for n in root if n.get('id') not in ('panel-F','panel-I')}
    if before!=after:raise ValueError('Unrequested SVG subtree changed')
    svg=ET.tostring(root);TEMP.write_bytes(svg)
    html=svg_editing_page(svg,scope_label='Figure 2',legend=Path(draft['legend']['path']).read_text()).encode()
    TEMP.with_suffix('.html').write_bytes(html)
    evidence=json.loads(Path(draft['semantic_registry']['path']).read_text())
    evidence.setdefault('element_registry',{}).update(registry)
    evidence['placeholder_consistency']={'author_direction':'2026-10-15: all placeholders follow current Figure 2 C','scope':['C','F','I'],'reference_svg':draft['working_svg'],'C_unchanged':True,'changes':changes,'preserved_other_root_subtrees':before,'candidate_sha256':sha(svg),'renderer':{'path':str(Path(__file__).resolve()),'sha256':file_hash(__file__)},'effective_presentation':measured_presentation(svg,width_mm=183),'visual_review':{'status':'pending'}}
    TEMP.with_suffix('.json').write_bytes(encode(evidence))
    Path('/tmp/fig2-placeholder-publish-ticket.json').write_bytes(encode(snapshot.hashes))
    print(TEMP)
def publish():
    snapshot=Snapshot([]);snapshot.hashes=json.loads(Path('/tmp/fig2-placeholder-publish-ticket.json').read_text());snapshot.check()
    state=json.loads(STATE.read_text());draft=state['draft'];svg=TEMP.read_bytes();html=TEMP.with_suffix('.html').read_bytes();evidence=json.loads(TEMP.with_suffix('.json').read_text());change=evidence['placeholder_consistency']
    assert change['visual_review']['status']=='passed' and sha(svg)==change['candidate_sha256'] and svg in html
    assert file_hash(__file__)==change['renderer']['sha256']
    # Preserve the previous active pair externally before replacing it.
    history=Path(draft['working_svg']['path']).parent/'history/figure2-before-placeholder-match-20261015'
    history.mkdir(parents=True,exist_ok=True)
    for key in ('working_svg','review'):
        source=Path(draft[key]['path']);destination=history/source.name
        if destination.exists():assert destination.read_bytes()==source.read_bytes()
        else:
            with destination.open('xb') as f:f.write(source.read_bytes())
    prior=history/'current-state.json'
    if not prior.exists():prior.write_bytes(STATE.read_bytes())
    ebytes=encode(evidence);epath=ROOT/'records/figure-reviews/figure-02/placeholder-consistency-20261015.json';draft['semantic_registry']['path']=str(epath);draft['working_svg']['sha256']=sha(svg);draft['review']['sha256']=sha(html);draft['semantic_registry']['sha256']=sha(ebytes)
    draft['consistency_requests'].append({'id':'fig2-c-f-i-placeholder-presentation','scope':['C','F','I'],'author_direction':'2026-10-15: all placeholders follow current Figure 2 C','properties':['white fill','solid black 0.5 pt border','centred 7 pt Pending','10 pt bold panel letter','explanations in legend'],'status':'verified','evidence':{'path':str(epath),'sha256':sha(ebytes)}})
    draft['reproduction']={'generator':change['renderer'],'command':'.venv/bin/python scripts/style_figure2_placeholders.py; render and review temporary candidate; --publish with captured pre-edit ticket','source':'authenticated prior active assembly; all non-F/I subtrees retained'}
    publish_state(STATE,state,snapshot,artifacts=[(epath,ebytes)],evidence_updates=[(Path(draft['working_svg']['path']),svg),(Path(draft['review']['path']),html)])
    print(draft['review']['path'])
if __name__=='__main__':
    publish() if '--publish' in sys.argv else prepare()
