"""Give the requested two-panel Figure 4 PDF a shared signed-vigor y scale.

Consume authenticated saved artwork only. No model or summary is recalculated.
Curve/band vertices and observed marker centres are remapped affinely; glyphs,
strokes, timing guides, x coordinates, legends and other artwork stay unchanged.
"""
from pathlib import Path as _Path
import sys as _sys
ROOT=next(p for p in _Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
_sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import copy,json,re
from pathlib import Path
import xml.etree.ElementTree as ET
from classical_conditioning.figure_workspace import Snapshot,sha,tag
NUMBER=r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?'

def find(root,identity):
    nodes=[n for n in root.iter()if n.get('id')==identity]
    if len(nodes)!=1:raise ValueError('Missing/ambiguous '+identity)
    return nodes[0]

def remap_path(value,scale,shift):
    if re.sub(NUMBER,'',value).strip(' MLZmlz\t\r\n,'):raise ValueError('Unsupported path commands')
    numbers=list(re.finditer(NUMBER,value))
    if len(numbers)%2:raise ValueError('Non-paired path')
    index=0
    def mapped(match):
        nonlocal index
        index+=1
        return format(scale*float(match[0])+shift,'.14g')if index%2==0 else match[0]
    return re.sub(NUMBER,mapped,value)

def shared_panel(raw,limits=(-.30,.15)):
    root=ET.fromstring(raw);before=copy.deepcopy(root);prefix=next(n.get('id').split('--')[0]for n in root.iter()if n.get('id','').endswith('--panel-a-axes'))+'--';axis=find(root,prefix+'matplotlib.axis_2');original_axis=copy.deepcopy(axis);defs=next(n for n in root if n.tag==tag('defs'));original_defs=copy.deepcopy(defs);edits=[]
    ticks=[n for n in axis if re.search(r'ytick_\d+$',n.get('id',''))];pairs=[]
    for tick in ticks:
        t=next(tick.iter(tag('text')));u=next(tick.iter(tag('use')));pairs.append((float(t.text.replace('−','-')),float(u.get('y'))))
    v0,y0=pairs[0];v1,y1=pairs[-1];old_slope=(y1-y0)/(v1-v0);old_zero=y0-old_slope*v0
    top,bottom=59.527559,170.078740;lo,hi=limits;new_slope=(top-bottom)/(hi-lo);new_zero=bottom-new_slope*lo;scale=new_slope/old_slope;shift=new_zero-scale*old_zero
    # Bind all visible original tick coordinates, not merely their end values.
    if max(abs(y-(old_slope*v+old_zero))for v,y in pairs)>1e-5:raise ValueError('Nonlinear/inconsistent source y ticks')
    def set_attr(node,key,value):
        edits.append((node.get('id'),node,key,node.get(key)));node.set(key,value)
    # The horizontal zero line is the first saved line artist.
    zero=next(find(root,prefix+'line2d_1').iter(tag('path')));set_attr(zero,'d',remap_path(zero.get('d'),scale,shift))
    for idx in range(3):
        line=find(root,prefix+'response-'+str(idx))
        for node in line.iter(tag('path')):set_attr(node,'d',remap_path(node.get('d'),scale,shift))
        band=find(root,prefix+'response-band-'+str(idx));uses=list(band.iter(tag('use')))
        if len(uses)!=1:raise ValueError('Unexpected saved band structure')
        use=uses[0];href=use.get('{http://www.w3.org/1999/xlink}href');node=find(root,href[1:]);offset=float(use.get('y','0'));set_attr(node,'d',remap_path(node.get('d'),scale,shift+(scale-1)*offset))
        observed=find(root,prefix+'observed-'+str(idx))
        for node in observed.iter(tag('use')):set_attr(node,'y',format(scale*float(node.get('y'))+shift,'.14g'))
    # Rebuild only y ticks at a common 0.05 interval; retain font, tick mark and axis title.
    template=copy.deepcopy(ticks[0]);use=next(template.iter(tag('use')));text=next(template.iter(tag('text')));pad=float(text.get('y'))-float(use.get('y'))
    for d in list(template.iter(tag('defs'))):
        defs.append(copy.deepcopy(d));next(p for p in template.iter()if d in list(p)).remove(d)
    for tick in ticks:axis.remove(tick)
    for idx in range(round((hi-lo)/.05)+1):
        value=lo+idx*.05;y=new_slope*value+new_zero;node=copy.deepcopy(template)
        for k,n in enumerate(node.iter()):
            if n.get('id'):n.set('id',prefix+f'shared-y-tick-{idx}-{k}')
        u=next(node.iter(tag('use')));u.set('y',format(y,'.14g'));t=next(node.iter(tag('text')));t.set('y',format(y+pad,'.14g'));t.set('transform',f"rotate(-0 {t.get('x')} {y+pad:.14g})");t.text=('0.00'if abs(value)<1e-8 else f'{value:.2f}'.replace('-','−'));axis.append(node)
    # Undo exactly the declared presentation changes and compare all other XML.
    restored=copy.deepcopy(root);a=find(restored,prefix+'matplotlib.axis_2');p=next(p for p in restored.iter()if a in list(p));i=list(p).index(a);p.remove(a);p.insert(i,original_axis)
    d=next(n for n in restored if n.tag==tag('defs'));i=list(restored).index(d);restored.remove(d);restored.insert(i,original_defs)
    for identity,node,key,old in edits:
        # Anonymous paths/uses have stable child-index routes within their owners.
        route=[];current=node
        parents={c:p for p in root.iter()for c in p}
        while current is not root:parent=parents[current];route.append(list(parent).index(current));current=parent
        target=restored
        for index in reversed(route):target=list(target)[index]
        if old is None:target.attrib.pop(key,None)
        else:target.set(key,old)
    if ET.tostring(restored)!=ET.tostring(before):raise ValueError('Unapproved artwork change')
    output=ET.tostring(root,encoding='unicode');return output,{'panel':prefix[:-2],'limits':limits,'old_mapping_y_per_unit':[old_slope,old_zero],'new_mapping_y_per_unit':[new_slope,new_zero],'zero_y':new_zero,'plot_bounds_y':[top,bottom],'data_coordinate_edits':len(edits),'all_other_artwork_preserved':True,'remapping':[scale,shift],'source_sha256':sha(raw.encode()),'candidate_sha256':sha(output.encode())}

def prepare(ticket_path,directory):
    ticket=json.loads(ticket_path.read_bytes());snapshot=Snapshot([ticket_path,Path(__file__)]);snapshot.bind_hashes(ticket['starting_hashes']);snapshot.check();source=Path(ticket['source_html']).read_text();svgs=re.findall(r'<svg\b.*?</svg>',source,re.S)
    if len(svgs)!=2:raise ValueError('Exactly two panels required')
    result=source;reports=[]
    for raw in svgs:
        new,report=shared_panel(raw);result=result.replace(raw,new,1);reports.append(report)
    if abs(reports[0]['zero_y']-reports[1]['zero_y'])>1e-9:raise ValueError('Zero alignment failed')
    evidence={'scope':ticket['scope'],'source_hashes':snapshot.hashes,'panels':reports,'scientific_selection_changed':False,'no_analysis_refit':True,'freeze_created':False}
    result=result.replace('</body>','<script type="application/json" id="export-shared-y-provenance">'+json.dumps(evidence).replace('<','\\u003c')+'</script></body>');snapshot.check();directory.mkdir(exist_ok=False);(directory/'figure-04.html').write_text(result);(directory/'presentation-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(reports,indent=2))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ticket',type=Path,required=True);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();prepare(a.ticket,a.directory)
