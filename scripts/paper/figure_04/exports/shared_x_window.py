"""Extend the two saved Figure 4 export axes to -20..20 s without extrapolation."""
from pathlib import Path as _Path
import sys as _sys
ROOT=next(p for p in _Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
_sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import copy,json,re
from pathlib import Path
import xml.etree.ElementTree as ET
from classical_conditioning.figure_workspace import Snapshot,sha,tag
from classical_conditioning.command_provenance import command_sources
from scripts.paper.figure_04.exports.shared_y import find,NUMBER

def remap_x(value,scale,shift):
    if re.sub(NUMBER,'',value).strip(' MLZz\t\r\n,'):raise ValueError('Unsupported path commands')
    if len(re.findall(NUMBER,value))%2:raise ValueError('Non-paired path')
    index=0
    def mapped(match):
        nonlocal index
        index+=1
        return format(scale*float(match[0])+shift,'.14g')if index%2 else match[0]
    return re.sub(NUMBER,mapped,value)

def panel(raw):
    root=ET.fromstring(raw);before=copy.deepcopy(root);prefix=next(n.get('id').split('--')[0]for n in root.iter()if n.get('id','').endswith('--panel-a-axes'))+'--';axis=find(root,prefix+'matplotlib.axis_1');old_axis=copy.deepcopy(axis);defs=next(n for n in root if n.tag==tag('defs'));old_defs=copy.deepcopy(defs);edits=[]
    ticks=[n for n in axis if re.search(r'xtick_\d+$',n.get('id',''))];pairs=[]
    for tick in ticks:
        t=next(tick.iter(tag('text')));u=next(tick.iter(tag('use')));pairs.append((float(t.text.replace('−','-')),float(u.get('x'))))
    (t0,x0),(t1,x1)=pairs[0],pairs[-1];old_scale=(x1-x0)/(t1-t0);old_zero=x0-old_scale*t0
    if max(abs(x-old_scale*t-old_zero)for t,x in pairs)>1e-5:raise ValueError('Inconsistent saved x ticks')
    left,right=42.519685,161.574803;new_scale=(right-left)/40;new_zero=(left+right)/2;scale=new_scale/old_scale;shift=new_zero-scale*old_zero;maximum_error=0.;checked=0
    parents={c:p for p in root.iter()for c in p}
    def set_attr(node,key,value):
        route=[];current=node
        while current is not root:parent=parents[current];route.append(list(parent).index(current));current=parent
        edits.append((list(reversed(route)),key,node.get(key)));node.set(key,value)
    def mapped_path(node,offset=0):
        nonlocal maximum_error,checked
        old=node.get('d');new=remap_x(old,scale,shift+(scale-1)*offset);a=list(map(float,re.findall(NUMBER,old)));b=list(map(float,re.findall(NUMBER,new)))
        for i in range(0,len(a),2):
            if a[i+1]!=b[i+1]:raise ValueError('Y value changed')
            maximum_error=max(maximum_error,abs((a[i]+offset-old_zero)/old_scale-(b[i]+offset-new_zero)/new_scale));checked+=1
        set_attr(node,'d',new)
    # Full-width zero reference; temporal guides retain their original times.
    zero=next(find(root,prefix+'line2d_1').iter(tag('path')));values=list(map(float,re.findall(NUMBER,zero.get('d'))));set_attr(zero,'d',f'M {left} {values[1]} L {right} {values[3]}')
    axes=find(root,prefix+'panel-a-axes');events=[]
    for child in axes:
        if child.get('id','').startswith(prefix+'line2d_')and child.get('id')!=prefix+'line2d_1':
            node=next(child.iter(tag('path')));values=list(map(float,re.findall(NUMBER,node.get('d'))))
            if len(values)!=4 or values[0]!=values[2]:raise ValueError('Unexpected timing guide')
            events.append({'id':child.get('id'),'time_s':(values[0]-old_zero)/old_scale});mapped_path(node)
    for idx in range(3):
        for node in find(root,prefix+'response-'+str(idx)).iter(tag('path')):mapped_path(node)
        use=next(find(root,prefix+'response-band-'+str(idx)).iter(tag('use')));href=use.get('{http://www.w3.org/1999/xlink}href')[1:];mapped_path(find(root,href),float(use.get('x','0')))
        for node in find(root,prefix+'observed-'+str(idx)).iter(tag('use')):
            x=float(node.get('x'));new=scale*x+shift;maximum_error=max(maximum_error,abs((x-old_zero)/old_scale-(new-new_zero)/new_scale));checked+=1;set_attr(node,'x',format(new,'.14g'))
    template=copy.deepcopy(ticks[0]);u=next(template.iter(tag('use')));t=next(template.iter(tag('text')));pad=float(t.get('x'))-float(u.get('x'))
    for d in list(template.iter(tag('defs'))):defs.append(copy.deepcopy(d));next(p for p in template.iter()if d in list(p)).remove(d)
    for tick in ticks:axis.remove(tick)
    for idx,value in enumerate((-20,-10,0,10,20)):
        x=new_scale*value+new_zero;node=copy.deepcopy(template)
        for k,n in enumerate(node.iter()):
            if n.get('id'):n.set('id',prefix+f'shared-x-tick-{idx}-{k}')
        u=next(node.iter(tag('use')));u.set('x',format(x,'.14g'));t=next(node.iter(tag('text')));t.set('x',format(x+pad,'.14g'));t.set('transform',f"rotate(-0 {x+pad:.14g} {t.get('y')})");t.text=str(value).replace('-','−');axis.append(node)
    # Restore only declared edits to prove every other SVG attribute/text is exact.
    restored=copy.deepcopy(root);a=find(restored,prefix+'matplotlib.axis_1');parent=next(p for p in restored.iter()if a in list(p));i=list(parent).index(a);parent.remove(a);parent.insert(i,old_axis);d=next(n for n in restored if n.tag==tag('defs'));i=list(restored).index(d);restored.remove(d);restored.insert(i,old_defs)
    for route,key,old in edits:
        node=restored
        for i in route:node=list(node)[i]
        if old is None:node.attrib.pop(key,None)
        else:node.set(key,old)
    if ET.tostring(restored)!=ET.tostring(before):raise ValueError('Artwork outside declared x-axis changes altered')
    if maximum_error>1e-5:raise ValueError('Time coordinate preservation failed')
    out=ET.tostring(root,encoding='unicode');return out,{'panel':prefix[:-2],'x_limits_s':[-20,20],'old_x_map':[old_scale,old_zero],'new_x_map':[new_scale,new_zero],'temporal_guides':events,'data_coordinates_checked':checked,'maximum_time_roundtrip_error_s':maximum_error,'all_y_coordinates_and_other_artwork_preserved':True,'model_refit':False,'extrapolation':False,'svg_sha256':sha(out.encode())}

def prepare(ticket_path,directory):
    ticket=json.loads(ticket_path.read_bytes());snapshot=Snapshot([ticket_path,*command_sources(__file__)]);snapshot.bind_hashes(ticket['starting_hashes']);snapshot.check();source=Path(ticket['source_html']).read_text();svgs=re.findall(r'<svg\b.*?</svg>',source,re.S)
    if len(svgs)!=2:raise ValueError('Exactly two selected panel-a views required')
    page=source;reports=[];files={}
    for raw in svgs:
        new,report=panel(raw);page=page.replace(raw,new,1);reports.append(report);files['figure-04-'+report['panel']+'-panel-a.svg']=new.encode()
    # State the actual saved support so empty extended axes are unambiguous.
    anchor='All views are unfrozen.'
    if page.count(anchor)!=1:raise ValueError('Matching shared legend advanced')
    page=page.replace(anchor,'Saved fitted profiles cover −5 to 12 s (Delay) and −5 to 16 s (3sTrace). '+anchor)
    evidence={'source_hashes':snapshot.hashes,'source_state':{'path':ticket['state_path'],'sha256':ticket['state_sha256']},'previous_lock':ticket['previous_lock'],'author_authorization':ticket['authorization'],'panels':reports,'shared_y_limits':ticket['y_limits'],'scientific_selection_changed':False,'publication_freeze_completed':False}
    page=page.replace('</body>','<script type="application/json" id="export-shared-x-provenance">'+json.dumps(evidence).replace('<','\\u003c')+'</script></body>');files['figure-04.html']=page.encode();files['presentation-evidence.json']=(json.dumps(evidence,indent=2)+'\n').encode();snapshot.check();directory.mkdir(exist_ok=False)
    for name,data in files.items():
        with (directory/name).open('xb')as f:f.write(data)
    print(json.dumps(reports,indent=2))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ticket',type=Path,required=True);p.add_argument('--directory',type=Path,required=True);a=p.parse_args();prepare(a.ticket,a.directory)
