"""Verify/publish presentation metadata for the standalone 10sTrace example."""

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / 'pyproject.toml').is_file() and (p / 'src/classical_conditioning').is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / 'src'), str(_COMMAND_REPOSITORY / 'scripts')]

from pathlib import Path
import argparse,json,sys,xml.etree.ElementTree as ET
REPO=_COMMAND_REPOSITORY
sys.path.insert(0,str(REPO/'src'))
from classical_conditioning.operations.candidate_paths import candidate_directory, evidence_file
from classical_conditioning.command_provenance import command_sources
from classical_conditioning.figure_workspace import Snapshot,publish_state,sha,tag
from classical_conditioning.figures.layout_frames import embed_spec,load_policy,assess,COLLECTOR
from classical_conditioning.artifacts import sha256_file
from scripts.render_trace10_example import SCOPE,OUT,PARENT,encoded,ref



def prepare(staging_dir):
    staging_dir = candidate_directory(staging_dir)
    state=json.loads(SCOPE.read_text());paths=[SCOPE,PARENT,Path(__file__),COLLECTOR]
    paths += [Path(r['path']) for r in state['draft'].values() if isinstance(r,dict) and 'path' in r]
    for r in state['draft'].values():
        if isinstance(r,dict) and 'path' in r and sha256_file(Path(r['path']))!=r['sha256']:raise ValueError('Changed standalone artifact')
    import ast, io, base64
    import pandas as pd
    import scripts.render_trace10_example as renderer
    paths += [Path(renderer.__file__),OUT/'bins.parquet']
    snap=Snapshot(paths)
    matrix=pd.read_parquet(OUT/'bins.parquet').pivot(index='display_trial',columns='bin_index',values='delta_log_vigor').to_numpy()
    # Reuse the original plotting statements with an explicit native canvas export.
    tree=ast.parse(Path(renderer.__file__).read_text(encoding='utf-8'))
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    drawing=next(n for n in main.body if isinstance(n,ast.With))
    env=dict(vars(renderer));profile=renderer.load_presentation_profile('nature-main')
    env.update(matrix=matrix,width=80.,height=96.,profile=profile)
    save=next(n for n in ast.walk(drawing) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='savefig')
    save.keywords.append(ast.keyword(arg='bbox_inches',value=ast.Constant(None)))
    # apply_theme selects tight export by default; restore the physical canvas at save time.
    save.keywords.append(ast.keyword(arg='pad_inches',value=ast.Constant(0)))
    env['mpl'].rcParams['savefig.bbox']=None
    # The theme is applied inside the drawing block, so scope export explicitly around savefig.
    drawing.body.insert(-2,ast.parse("mpl.rcParams['savefig.bbox']=None; fig.set_layout_engine(None)").body[0])
    drawing.body.insert(-2,ast.parse("fig.set_layout_engine(None)").body[0])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[drawing],type_ignores=[])),str(Path(renderer.__file__)),'exec'),env)
    root=ET.fromstring(env['svg'].getvalue())
    old=ET.fromstring(Path(state['draft']['working_svg']['path']).read_bytes())
    for node in old:
        if node.tag==tag('defs') and any(child.tag==tag('style') and '@font-face' in (child.text or '') for child in node):root.append(node)
        if node.tag==tag('metadata') and node.get('id')=='scientific-provenance':root.append(node)
    for node in root.iter(tag('image')):node.set('style',node.get('style','')+';image-rendering:pixelated')
    svg=ET.tostring(root,encoding='utf-8')
    policy,ph=load_policy()
    spec={'scope':'panel','final_size_mm':[80.,96.],'inset_mm':policy['selected_inset_mm'],
          'allocations':[{'id':'trace10-example','kind':'panel','box_mm':[0,0,80.,96.],'group_ids':['figure_1']}],
          'background_ids':['patch_1'],'policy_sha256':ph,
          'scientific_clips':[{'id':'heatmap-scientific-domain','evidence':'Image extent -20 to 20 s, acquisition trials 5-94; mapped numerical cells retained.'}]}
    svg=embed_spec(svg,spec)
    legend=Path(state['draft']['legend']['path']).read_text(encoding='utf-8')
    page=('<!doctype html><html><meta charset="utf-8"><style>body{margin:24px;background:white;color:black;font:14px Arial,sans-serif}main{max-width:800px;margin:auto}svg{display:block;width:min(100%,520px);height:auto;margin:auto}p{line-height:1.45}</style><main>'+svg.decode()+'<p>'+legend+'</p></main></html>').encode()
    staging_dir.mkdir(exist_ok=True)
    for p,b in [(staging_dir/'candidate.svg',svg),(staging_dir/'candidate.html',page),(staging_dir/'ticket.json',encoded({'hashes':snap.hashes,'state':state}))]:
        with p.open('xb') as f:f.write(b)
    print(staging_dir/'candidate.svg')


def publish(note, staging_dir, *, browser_evidence=None):
    staging_dir = candidate_directory(staging_dir,existing=True)
    browser_evidence = evidence_file(browser_evidence if browser_evidence else staging_dir/"browser.json")
    snap=Snapshot([*command_sources(__file__),staging_dir/'candidate.svg',staging_dir/'candidate.html',browser_evidence,staging_dir/'ticket.json'])
    ticket=json.loads((staging_dir/'ticket.json').read_text())
    snap.bind_hashes(ticket['hashes'])
    snap.check()
    svg=(staging_dir/'candidate.svg').read_bytes();page=(staging_dir/'candidate.html').read_bytes()
    report=json.loads(browser_evidence.read_text());evidence=assess(svg,report)
    if evidence['status']!='passed':raise ValueError(evidence['issues'])
    state=ticket['state'];d=state['draft']
    history=OUT/'history';history.mkdir(exist_ok=True)
    oldsvg=Path(d['working_svg']['path']);oldhtml=Path(d['review']['path'])
    new=[(history/'initial-unverified.svg',oldsvg.read_bytes()),(history/'initial-unverified.html',oldhtml.read_bytes()),
         (history/'initial-state.json',encoded(state)),(OUT/'layout-browser.json',encoded(report)),
         (OUT/'layout-evidence.json',encoded(evidence))]
    state['draft']['working_svg']['sha256']=sha(svg);state['draft']['review']['sha256']=sha(page)
    state['verification'].update({'visual_review':{'status':'performed','note':note},
        'layout_frames':{'path':str(OUT/'layout-evidence.json'),'sha256':sha(encoded(evidence))},
        'renderer':ref(Path(__file__))})
    publish_state(SCOPE,state,snap,artifacts=new,evidence_updates=[(oldsvg,svg),(oldhtml,page)])
    print(json.dumps({'review':d['review']['path'],'layout_status':evidence['status']}))


def cli():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepare',action='store_true')
    ap.add_argument('--publish-note')
    ap.add_argument('--staging-dir',type=Path,required=True)
    ap.add_argument('--browser-evidence',type=Path)
    a=ap.parse_args()
    if a.prepare and a.publish_note:ap.error('Choose one operation')
    if a.prepare:prepare(a.staging_dir)
    elif a.publish_note:publish(a.publish_note,a.staging_dir,browser_evidence=a.browser_evidence)
    else:ap.error('Choose --prepare or --publish-note')

if __name__=='__main__':cli()
