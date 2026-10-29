"""Preserve author-selected artwork/PDF bytes as provisional presentation locks.

This version checkpoint never publishes a scientific or publication freeze.
All prior/exploratory artwork and dependencies remain in place. Guarded control
publication creates immutable copies before updating scoped presentation locks.
"""
from pathlib import Path as _Path
import sys as _sys
ROOT=next(p for p in _Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
_sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import argparse,json,re
from pathlib import Path
from classical_conditioning.figure_workspace import Snapshot,sha
from classical_conditioning.command_provenance import command_sources
from scripts.maintenance.storage_migration import publish_controls

def packed(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
def ref(path,raw):return {'path':str(path),'sha256':sha(raw)}

def prepare(ticket_path,stage):
    ticket=json.loads(ticket_path.read_bytes())
    if ticket.get('kind')!='provisional-presentation-lock':raise ValueError('Explicit provisional-lock ticket required')
    snapshot=Snapshot([ticket_path,*command_sources(__file__)]);snapshot.bind_hashes({str(Path(p).resolve()):h for p,h in ticket['starting_hashes'].items()});snapshot.check();stage.mkdir(exist_ok=False);controls=[];results=[]
    def stage_control(target,raw,role='artifact'):
        target=Path(target).resolve();expected=snapshot.hashes.get(str(target))
        if role!='state'and target.exists():raise FileExistsError(target)
        src=stage/str(len(controls));src.write_bytes(raw);controls.append({'path':str(target),'source':str(src),'sha256':sha(raw),'bytes':len(raw),'expected_sha256':expected,'role':role})
    for figure in ticket['figures']:
        num=figure['figure'];target=Path(figure['target_dir']);target.mkdir(exist_ok=True);
        if any(target.iterdir()):raise FileExistsError(target)
        files={name:Path(source).read_bytes()for name,source in figure['files'].items()};sp=Path(figure['state_path'])
        if figure['state_exists']:
            state=json.loads(sp.read_bytes());draft=state['draft']
            if 'path'not in draft['semantic_registry']:files['semantic-registry.json']=packed(draft['semantic_registry'])
        else:
            page=files['figure-04.html'].decode();svgs=re.findall(r'<svg\b.*?</svg>',page,re.S)
            if len(svgs)!=2:raise ValueError('Figure 4 scope must retain exactly two saved panel-a SVGs')
            for label,raw in zip(('delay','3strace'),svgs):files[f'figure-04-{label}-panel-a.svg']=raw.encode()
            state={'schema_version':1,'figure_id':'fig4','scope':figure['scope'],'default_display':'draft','default_display_format':'html-svg-and-legend','whole_figure_frozen':False,'scientific_selection':{'status':'Unchanged exploratory Delay/3sTrace model pilots; no new scientific approval','records':[]},'draft':{'status':'Author-locked provisional presentation','review':ref(target/'figure-04.html',files['figure-04.html']),'legend':{**ref(target/'figure-04.html',files['figure-04.html']),'format':'matching legends embedded in self-contained HTML'},'panel_svgs':[ref(target/f'figure-04-{label}-panel-a.svg',files[f'figure-04-{label}-panel-a.svg'])for label in ('delay','3strace')],'exports':{kind:ref(target/f'figure-04{suffix}',files[f'figure-04{suffix}'])for kind,suffix in [('pdf','.pdf'),('png','-600dpi.png')]}}}
        file_refs={name:ref(target/name,raw)for name,raw in files.items()};old_versions=sorted(p.name for p in target.parent.iterdir()if p.is_dir()and p!=target)
        manifest={'schema_version':1,'kind':'provisional-presentation-lock','date':ticket['date'],'figure_id':'fig'+str(num),'scope':figure['scope'],'author_authorization':ticket['authorization'],'files':file_refs,'source_bindings':{figure['files'][name]:snapshot.hashes[str(Path(figure['files'][name]).resolve())]for name in figure['files']},'source_state':file_refs.get('source-state.json'),'source_guard_snapshot':snapshot.hashes,'retention':{'disposition':'keep all','instruction':figure['retention'],'earlier_assembly_version_directories':old_versions,'other_figure_scopes_and_dependencies':'keep in their existing homes','removed_files':[]},'verification':{'copies':'Exact saved bytes, authenticated before staging and rechecked under publication locks','artwork_restyled':False,'pdf_reencoded':False},'publication_freeze_completed':False,'scientific_selection_changed':False,'future_edits':'Preserve these locked files; start a separate version and update scoped state deliberately.'}
        mb=packed(manifest);lock={**ref(target/'version-lock.json',mb),'status':'locked','kind':ticket['kind'],'date':ticket['date'],'scope':figure['scope'],'retain_other_versions':True,'future_edits_require_new_version':True,'publication_freeze_completed':False}
        state['presentation_lock']=lock
        for name,raw in files.items():stage_control(target/name,raw)
        stage_control(target/'version-lock.json',mb);stage_control(sp,packed(state),'state');results.append({'figure':num,'manifest':lock,'pdf':file_refs[f'figure-{num:02}.pdf'],'review':file_refs[f'figure-{num:02}.html']})
    snapshot.check();plan={'lock':str(ROOT/'docs/paper/figures/presentation-lock-publication'),'guards':[{'path':p,'sha256':h,'bytes':Path(p).stat().st_size}for p,h in snapshot.hashes.items()if h is not None],'controls':controls,'results':results};(stage/'publication-plan.json').write_bytes(packed(plan));return plan

def publish(plan_path):
    plan=json.loads(plan_path.read_bytes());result=publish_controls(plan)
    for control in plan['controls']:
        if sha(Path(control['path']).read_bytes())!=control['sha256']:raise ValueError('Published byte mismatch')
    return {'status':'locked','figures':plan['results'],'publication':result,'removed_files':[],'publication_freeze_completed':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','publish']);p.add_argument('--ticket',type=Path);p.add_argument('--stage',type=Path);p.add_argument('--plan',type=Path);a=p.parse_args();result=prepare(a.ticket,a.stage)if a.command=='prepare'else publish(a.plan);print(json.dumps({'prepared_controls':len(result['controls']),'figures':result['results']}if a.command=='prepare'else result,indent=2))
if __name__=='__main__':main()
