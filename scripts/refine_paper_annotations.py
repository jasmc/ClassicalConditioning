"""Stage author-requested legend/heading semantics on measured axis candidates."""
import argparse,copy,json,re,sys
from pathlib import Path
from xml.etree import ElementTree as ET
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import sha,tag
from main_paper_figures import encode,reference
from correct_panel_relationships import find
from register_figure_axes import build_inventory

def wording(text):return re.sub(r'\b(3|10)\s+s\s+Trace\b',r'\1sTrace',text)

def prepare(directory):
    for n in (1,2):
        path=directory/f'figure-{n:02}-layout.svg';raw=path.read_bytes();root=ET.fromstring(raw);paths={}
        def walk(node,p):
            paths[node]=p
            for i,c in enumerate(node):walk(c,p+[i])
        walk(root,[]);edits={};groups=[]
        def edit(node,role):
            if node not in edits:edits[node]={'xml_index_path':paths[node],'source_id':node.get('id'),'before_attributes':dict(node.attrib),'before_text':node.text,'roles':[]}
            edits[node]['roles'].append(role)
        for node in root.iter(tag('text')):
            if node.text and wording(node.text)!=node.text:edit(node,'condition-spelling');node.text=wording(node.text)
        if n==1:
            groups=[{'element_id':'edit-B-heading','panel':'B','semantic_tag':'plot.legend.stimulus-key','scientific_role':'legend.key','alignment_reference':{'element_id':'edit-B-time-axis-delay'}}, {'element_id':'edit-C-stimulus-key','panel':'C','semantic_tag':'plot.legend.stimulus-key','scientific_role':'legend.key','alignment_reference':{'local_right_edge':1585.}}]
        else:
            for node in find(root,'fig2-g-native-figure').iter(tag('text')):
                text=''.join(node.itertext())
                if text in ('Delay','Control n=28; Delay n=29')and node not in list(find(root,'legend_1').iter()):
                    edit(node,'remove-ambiguous-experiment-heading'if text=='Delay'else 'sample-count-to-caption');node.set('display','none');node.set('data-semantic-tag','experiment.heading.removed'if text=='Delay'else 'caption.sample-count')
            for node in root.iter(tag('text')):
                if re.fullmatch(r'n\s*=\s*\d+',node.text or ''):edit(node,'sample-count-to-caption');node.set('display','none');node.set('data-semantic-tag','caption.sample-count')
            for node in find(root,'edit-H-legend_1').iter(tag('text')):
                edit(node,'sample-count-to-caption');node.text=re.sub(r'\s*\(n=\d+\)','',node.text or '')
            groups=[{'element_id':'legend_1','panel':'G','semantic_tag':'plot.legend.condition-key','scientific_role':'legend.key','alignment_reference':{'element_id':'fig2__g__native__spine-bottom'}},{'element_id':'edit-H-legend_1','panel':'H','semantic_tag':'plot.legend.condition-key','scientific_role':'legend.key','alignment_reference':{'element_id':'edit-H-patch_2'}}]
            # Restore the E boundary and its shared-scale ticks. The matching
            # y coordinates come from the authored D/E common-scale axis,
            # never from inferred values or a new data transformation.
            source=find(root,'shared-DEF-edit-E-axis__e-trace__y');clone=copy.deepcopy(source);clone.set('id','fig2-E-visible-y');clone.set('data-semantic-tag','axis.shared-scale-local-ticks')
            for parent in list(clone.iter()):
                for child in list(parent):
                    if child.tag==tag('g')and any(s in child.get('id','')for s in ('tick-label','axis-title')):parent.remove(child)
            mapping={node.get('id'):'fig2-E-visible-'+node.get('id')for node in clone.iter()if node.get('id')and node is not clone}
            for node in clone.iter():
                if node.get('id')in mapping:node.set('id',mapping[node.get('id')])
                for key,value in list(node.attrib.items()):
                    for old,new in mapping.items():value=value.replace('#'+old,'#'+new)
                    node.set(key,value)
            parent=find(root,'edit-E-figure');parent.append(clone)
            spine=ET.SubElement(parent,tag('path'),{'id':'fig2-E-visible-y-spine','d':'M 40.461732 116.118425 L 40.461732 61.829291','fill':'none','stroke':'#000000','stroke-width':'0.5','data-semantic-tag':'axis.spine.local-shared-y'})
        for group in groups:
            node=find(root,group['element_id']);edit(node,group['semantic_tag']);node.set('data-semantic-tag',group['semantic_tag']);node.set('data-scientific-role','legend.key');node.set('data-align','right')
        headings=[]
        if n==2:
            for panel,assay in [('D','delay'),('E','trace')]:
                for condition in [assay,'control']:
                    identity=f'edit-{panel}-fig2__{panel.lower()}__{condition}__condition-title';node=find(root,identity);edit(node,'heading.condition');node.set('data-semantic-tag','heading.condition');node.set('data-scientific-role','title.condition')
                    headings.append({'element_id':identity,'panel':panel,'semantic_tag':'heading.condition','scientific_role':'title.condition','experiment':'Delay'if panel=='D'else '3sTrace','condition':'Control'if condition=='control'else 'Delay'if panel=='D'else '3sTrace','alignment_reference':f'edit-{panel}-fig2__{panel.lower()}__{condition}__background','alignment':'glyph center over own data-axis rectangle; common baseline within panel'})
        sp=ROOT/f'docs/paper/figures/figure-{n:02}/current-state.json';state=json.loads(sp.read_text());registry=state['draft']['semantic_registry'];registry=json.loads(Path(registry['path']).read_text())if 'path'in registry else registry
        legend=wording(Path(state['draft']['legend']['path']).read_text())
        if n==2:
            legend=legend.replace('(G) Historical LogMedian analysis for Delay and its control cohort.','(G) Historical LogMedian analysis for the Delay experiment (28 control fish; 29 Delay fish).')
            legend=legend.replace('(H) 3sTrace response/baseline trajectory:','(H) 3sTrace response/baseline trajectory (19 control fish; 40 3sTrace fish):')
        (directory/f'figure-{n:02}-legend.md').write_text(legend)
        changes=[{**record,'after_attributes':dict(node.attrib),'after_text':node.text}for node,record in edits.items()]
        added=['fig2-E-visible-y','fig2-E-visible-y-spine']if n==2 else []
        def payload(node):return sha(encode([(x.tag,sorted(x.attrib.items()),x.text)for x in node.iter()]))
        manifest={'source_candidate_sha256':sha(raw),'legend_groups':groups,'condition_headings':headings,'changes':changes,'added_axis_ids':added,'added_axis_payloads':{i:payload(find(root,i))for i in added},'axis_source':'Authored shared D/E coordinate system; clone ticks only; labels/title shared as in A/B','sample_counts_moved':n==2,'scientific_scope':{'experiment':'Assay including paired and matched-control cohorts','condition':'Individual cohort/treatment within its experiment','G_heading':'Removed; line legend identifies Control/Delay conditions, caption identifies Delay experiment'}}
        path.write_bytes(ET.tostring(root));(directory/f'figure-{n:02}-annotation-changes.json').write_bytes(encode(manifest))
        inv=build_inventory(path.read_bytes(),n,state['draft']['legend'],state['scientific_selection']['records'],registry)
        (directory/f'figure-{n:02}-axis-input.json').write_bytes(encode({'axes':inv['axes'],'svg':state['draft']['working_svg']}))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',required=True,type=Path);prepare(p.parse_args().directory)
