"""Integrate the selected compact A freeze and retained B into Figure 2.

Render --prepare, inspect the exact assembly, record visual review, then --publish.
Publication preserves prior working copies and checks the pre-edit ticket under
both the A scope and whole-figure locks, publishing the parent state last.
"""
from pathlib import Path
import sys,json,copy,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from classical_conditioning.figure_workspace import Snapshot,tag,sha,file_hash,editable_svg,measured_presentation,publish_state,state_lock
from classical_conditioning.artifacts import sha256_file
from build_figure_legend_review import svg_editing_page
from build_figure2_heatmap_row import STAGE,artifact,write,AUTH
STATE=ROOT/'docs/paper/figures/figure-02/current-state.json'
ASTATE=STATE.parent/'panel-a/current-state.json'
LEGEND=STATE.parent/'legend.md'
FREEZE=ROOT/'records/frozen-analyses/figure-02/panel-a/2026-10-14-pooled-six-heatmap-row/manifests/freeze.json'
SELECTION=ROOT/'configs/paper-figures/selections/figure2-A-pooled-freeze-20261014.json'
EVIDENCE=ROOT/'records/figure-reviews/figure-02/six-heatmap-row-evidence.json'
REVIEW=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/reviews/figure2-current-minimal.html')
SVG=REVIEW.with_suffix('.svg')
TICKET=Path('/tmp/figure2-a-freeze-start.json')
def encoded(x):return (json.dumps(x,indent=2,allow_nan=False)+'\n').encode()
def prepare():
    from update_latest_figure_assemblies import build
    raw,meta=build(2, prefer_current=False)
    native,inventory=editable_svg(raw,kind='Figure 2 with compact A/B/C row',source_records=meta['panels'])
    root=ET.fromstring(native)
    before={n.get('id'):sha(ET.tostring(n)) for n in root if n.get('id','').startswith('panel-') and n.get('id') not in ('panel-A','panel-B','panel-C')}
    for n in list(root):
        if n.tag in (tag('text'),tag('metadata')) or n.get('id') in ('panel-A','panel-B','panel-C'):root.remove(n)
    root.set('viewBox','0 0 1800 3020');root.set('width','183mm');root.set('height',str(3020/1800*183)+'mm')
    for rect in root.findall(tag('rect')):rect.set('height','3020')
    for n in root:
        if n.get('id','') in before:n.set('transform','translate(0 100)')
    after={}
    for n in root:
        if n.get('id') in before:
            clone=copy.deepcopy(n);clone.attrib.pop('transform',None)
            after[n.get('id')]=sha(ET.tostring(clone))
    assert before==after,'Unrequested panel content changed'
    row=ET.parse(STAGE/'row.svg').getroot()
    # Crisp cell edges reproduce the selected F/G/H colourbar rendering.
    ramp=next(n for n in row.iter() if n.get('id')=='fig2__shared__colorbar__colour-ramp');ramp.set('shape-rendering','crispEdges')
    row.set('x','0');row.set('y','0');row.set('width','1800');row.set('height',str(78/183*1800));row.set('overflow','visible')
    ET.SubElement(root,tag('g'),{'id':'panel-ABC'}).append(row)
    full=ET.tostring(root);(STAGE/'assembly-candidate.svg').write_bytes(full)
    report=measured_presentation(full,width_mm=183)
    a=measured_presentation((STAGE/'frozen-panel-a.svg').read_bytes(),width_mm=61)
    def address(item):
        keys=[k for k in item['context'] if k.startswith(('fig2__a0__','fig2__a1__'))]
        return keys[-1] if keys else None
    import numpy as np
    checks=[]
    for kind,props in {'texts':['font_height_pt','font_width_scale_pt','font_family','weight'],'strokes':['width_range_pt','normal_width_pt','effective_opacity','color'],'ticks':['length_pt','normal_width_pt','effective_opacity','color']}.items():
        refs={address(r):r for r in a[kind] if address(r)}
        for r in report[kind]:
            key=address(r)
            if not key:continue
            for prop in props:
                actual,expected=r.get(prop),refs[key].get(prop)
                if isinstance(actual,(float,int,list)):np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-7)
                else:assert actual==expected,(key,prop)
            checks.append(key)
    selected=ET.parse(STAGE/'frozen-panel-a.svg').getroot();frozenids={n.get('id'):n for n in selected.iter() if n.get('id','').startswith(('fig2__a0__','fig2__a1__'))}
    assembledids={n.get('id'):n for n in row.iter() if n.get('id') in frozenids}
    # Ignore standalone canvas crop: it does not encode scientific values.
    for key,node in frozenids.items():
        if 'canvas__background' not in key:assert ET.tostring(node)==ET.tostring(assembledids[key]),key
    legend=LEGEND.read_text().replace('This descriptive scientific draft is unfrozen.','Panel A is frozen. The six-heatmap row uses a single shared managua_r colourbar at the far right (−0.25 to +0.25 natural-log vigor difference); A and B retain their distinct estimators.').replace('It has not been updated by this Delay-only request.','Its earlier data are retained at the author’s request. B and C will later adopt A’s individual-fish pipeline and pooled estimator. B is unfrozen.')
    (STAGE/'legend-current.md').write_text(legend)
    (STAGE/'assembly-candidate.html').write_text(svg_editing_page(full,scope_label='Figure 2',legend=legend))
    alegend=legend[legend.index('(A)'):legend.index('\n\n(B)')]
    abytes=(STAGE/'frozen-panel-a.svg').read_bytes()
    (STAGE/'a-review.html').write_text(svg_editing_page(abytes,scope_label='Figure 2 A',legend=alegend).replace('width:183mm','width:61mm'))
    evidence=dict(scope='Figure 2 A/B/C row layout; only A newly frozen',freeze=artifact(FREEZE),candidate_svg=artifact(STAGE/'assembly-candidate.svg'),candidate_review=artifact(STAGE/'assembly-candidate.html'),generator=artifact(Path(__file__)),renderer=artifact(ROOT/'scripts/build_figure2_heatmap_row.py'),row=artifact(STAGE/'row.svg'),A_source=artifact(STAGE/'frozen-panel-a.svg'),A_final_scale=1,A_effective_style_comparisons=len(checks),unchanged_A_semantic_subtrees=True,preserved_other_panel_payload_hashes=before,source_panels=meta['panels'],shared_colourbar={'count':1,'limits':[-.25,.25],'units':'natural-log vigor difference','mappable_ids':[f'fig2__{p}__heatmap__vigor-cells' for p in ('a0','a1','b0','b1')],'C':'two pending slots; no data'},pixel_checks=json.loads((STAGE/'pixel-evidence.json').read_text()),element_registry=json.loads((STAGE/'row-registry.json').read_text()),measured_presentation=report,visual_review={'status':'pending'},future_requirements={'B_C_science':'A individual-fish direct log-frame 1 s pipeline followed by equal-fish mean','B_C_style':'Figure 1 F/G/H fonts/axes/ticks/guides; six-axis geometry and shared row colourbar','status':'pending full processed Trace inputs; old B retained'},whole_figure_frozen=False)
    write(STAGE/'assembly-evidence.json',evidence)
    print(STAGE/'assembly-candidate.html')
def publish():
    e=json.loads((STAGE/'assembly-evidence.json').read_text());assert e['visual_review']['status']=='passed'
    for key in ('candidate_svg','candidate_review','generator','renderer','row','A_source','freeze'):
        a=e[key];assert sha256_file(Path(a['path']))==a['sha256'],a
    f=json.loads(FREEZE.read_text())
    for a in f['source_artifacts']+f['data_artifacts']+f['exports']:assert sha256_file(Path(a['path']))==a['sha256'],a
    snapshot=Snapshot([]);snapshot.hashes=json.loads(TICKET.read_text())
    for name in ['scripts/update_figure2_panel_a_freeze.py','src/classical_conditioning/figure_freeze.py','tests/test_freeze_embedded_fonts.py','scripts/build_figure2_heatmap_row.py','scripts/publish_figure2_heatmap_row.py','scripts/update_latest_figure_assemblies.py']:
        p=ROOT/name;snapshot.hashes[str(p)]=file_hash(p)
    retained=json.loads((STAGE/'selection.json').read_text())['retained_B']
    snapshot.hashes[retained['path']]=retained['sha256']
    for panel in e['source_panels']:
        source=panel.get('source')
        if source and 'script_id' not in source:snapshot.hashes[source['path']]=source['sha256']
    snapshot.check()
    svg=(STAGE/'assembly-candidate.svg').read_bytes();html=(STAGE/'assembly-candidate.html').read_bytes();assert svg in html
    legend=(STAGE/'legend-current.md').read_bytes();abytes=(STAGE/'frozen-panel-a.svg').read_bytes();ahtml=(STAGE/'a-review.html').read_bytes();assert abytes in ahtml
    old=json.loads(ASTATE.read_text());active=old['draft'];asvg=Path(active['working_svg']['path']);areview=Path(active['review']['path'])
    for key in ('working_svg','review','legend','semantic_registry'):
        a=active[key];assert file_hash(a['path'])==a['sha256'],'Manual edit: '+a['path']
    # Preserve prior active bytes, authenticated under the initial ticket.
    history=STAGE/'prior-working-copies';history.mkdir(exist_ok=True)
    for p in (asvg,areview,REVIEW,ASTATE,LEGEND):
        destination=history/p.name
        if destination.exists():assert destination.read_bytes()==p.read_bytes(),'Preserve differing history'
        else:destination.write_bytes(p.read_bytes())
    selection=json.loads((STAGE/'selection.json').read_text());selection.update(freeze_manifest=artifact(FREEZE),svg=artifact(STAGE/'frozen-panel-a.svg'),freeze_scope=['A'],status='current accepted compact A freeze')
    request=dict(id='fig2-abc-pooled-pipeline-and-fgh-formatting',scope=['A','B','C'],author_direction=AUTH,reference=artifact(ROOT/'configs/paper-figures/selections/figure1-fgh-historical-one-second-managua-freeze-20261013.json'),status='partial',verified=['A scientific method and styles','six heatmap slots','single shared colourbar','retained earlier B values'],pending=['B and C scientific pipeline after full processed Trace inputs'],evidence={'path':str(EVIDENCE),'sha256':sha(encoded(e))})
    apair=dict(status='frozen A compact display',working_svg={'path':str(asvg),'sha256':sha(abytes)},review={'path':str(areview),'sha256':sha(ahtml)},legend={'path':str(LEGEND),'sha256':sha(legend),'panel_ids':['A'],'extraction':'A paragraph and shared colourbar definition'},semantic_registry={'path':str(EVIDENCE),'sha256':sha(encoded(e))},assembly_scale={'final_width_mm':61,'final_height_mm':78,'native_element_scale':1},consistency_requests=[request])
    astate=dict(schema_version=1,figure_id='fig2',panel_ids=['A'],default_display='frozen_display',default_display_format='html-svg-and-legend',scientific_selection={'records':[{'path':str(SELECTION),'sha256':sha(encoded(selection))},artifact(FREEZE)]},frozen_display=apair,draft={'status':'no active scientific draft; current working copy derives from accepted A freeze','previous_draft':{'path':str(history/ASTATE.name),'sha256':sha((history/ASTATE.name).read_bytes())}},whole_figure_frozen=False,assembly_integration=str(STATE))
    state=dict(schema_version=1,figure_id='fig2',default_display='draft',default_display_format='html-svg-and-legend',whole_figure_frozen=False,scientific_selection={'records':[{'path':str(SELECTION),'sha256':sha(encoded(selection))},artifact(ROOT/'configs/paper-figures/selections/figure2-DE-B-freeze-20261009.json'),artifact(ROOT/'configs/paper-figures/selections/figure2-G-logmedian-freeze-20261009.json')]},frozen_display={'status':'scoped A, D/E and G freezes; no whole-figure freeze'},draft=dict(status='compact A/B/C row; A frozen; B earlier exploratory data; C pending',working_svg={'path':str(SVG),'sha256':sha(svg)},review={'path':str(REVIEW),'sha256':sha(html)},legend={'path':str(LEGEND),'sha256':sha(legend),'panel_ids':list('ABCDEFGHI')},semantic_registry={'path':str(EVIDENCE),'sha256':sha(encoded(e))},consistency_requests=[request],reproduction={'generator':e['generator'],'command':'.venv/bin/python scripts/publish_figure2_heatmap_row.py --prepare; inspect and publish with guarded starting ticket'},assembly_scale={'final_width_mm':183,'final_height_mm':3020/1800*183,'A_native_scale':1}))
    updates=[(asvg,abytes),(areview,ahtml),(REVIEW,html),(ASTATE,encoded(astate)),(LEGEND,legend),(EVIDENCE,encoded(e)),(SELECTION,encoded(selection))]
    docs={STATE.parent/'README.md':(STATE.parent/'README.md').read_text().replace('D/E B and G LogMedian have accepted scoped freezes. A/B/C/F/H/I remain proposed.','A, D/E B and G LogMedian have accepted scoped freezes. B/C/F/H/I remain proposed.').split('Current A presentation draft:')[0]+'\nCurrent display: [Figure 2 state](current-state.json). [A scientific definition and scoped state](panel-a/README.md). The top row contains A/B/C with two heatmaps each and one shared colourbar at the far right. A is frozen; B retains earlier data by author direction; C has two pending slots. Future B/C must follow A’s pooled pipeline and the Figure 1 F/G/H element formatting.\n',ROOT/'docs/paper/figures/README.md':(ROOT/'docs/paper/figures/README.md').read_text().replace('D/E B and G LogMedian; other panels proposed','A pooled 1 s medians (2026-10-14), D/E B and G LogMedian; other panels proposed'),ROOT/'records/frozen-analyses/selections.md':(ROOT/'records/frozen-analyses/selections.md').read_text()+'\nCurrent Figure 2 A: [pooled one-second direct log-frame medians, compact six-heatmap row](../../configs/paper-figures/selections/figure2-A-pooled-freeze-20261014.json). A only; previous wider A freeze remains preserved. B retains earlier data; B/C pipeline adoption remains pending.\n',ROOT/'records/frozen-analyses/figure-02/README.md':(ROOT/'records/frozen-analyses/figure-02/README.md').read_text()+'\n[A scoped freeze and previous versions](panel-a/README.md): current compact pair within the six-heatmap row; no B/C or whole-figure freeze.\n',ASTATE.parent/'README.md':'# Figure 2 A\n\nCurrent accepted scope: [selection](../../../../../configs/paper-figures/selections/figure2-A-pooled-freeze-20261014.json); [presentation state](current-state.json).\n\nDelay 29 fish and controls 28 fish. Each fish/trial applies the selected Figure 1 F/G/H finite positive detected-bout frame log pipeline: direct 1 s log-frame medians; subtract median finite baseline-bin medians over [−15,0); no amplitude scaling. Pool centred values by arithmetic mean with equal fish weight. Display 1–90 = acquisition 5–94. NaN black, native managua_r limits ±0.25, no numerical clipping.\n\nCompact A has two boxed axes at native F/G/H fonts, ticks, strokes and guide styles. One xlabel spans the pair; row y labels are shared. The colourbar belongs to the full A/B/C row at far right. B and C will use this pipeline later; earlier B data are retained unchanged and remain exploratory.\n',ROOT/'records/frozen-analyses/figure-02/panel-a/README.md':'# Figure 2 A frozen records\n\nCurrent: [compact pooled one-second freeze](2026-10-14-pooled-six-heatmap-row/manifests/freeze.json). Previous [wide pair freeze](2026-10-14-pooled-one-second/manifests/freeze.json) is preserved with its numerical/reproducibility dependencies. [Disposition inventory](../../../figure-reviews/figure-02/panel-a-freeze-dispositions-20261014.json); all unapproved dispositions remain pending.\n'}
    for p,text in docs.items():p.parent.mkdir(exist_ok=True);updates.append((p,text.encode()))
    inventorypath=ROOT/'records/figure-reviews/figure-02/panel-a-freeze-dispositions-20261014.json'
    inventory=dict(current=artifact(FREEZE),authorization='Freezing does not authorize removal',items=[dict(path=str(ROOT/'records/frozen-analyses/figure-02/panel-a/2026-10-14-pooled-one-second/manifests/freeze.json'),role='previous wider freeze; dependencies also current scientific inputs',recommendation='keep',disposition='pending author choice'),dict(path=str(history),role='prior A draft pair/state, original whole Figure 2 HTML and legend',recommendation='retain as history',disposition='pending author choice'),dict(path=str(STAGE.parent/'figure-02-a-freeze-2026-10-14/assembly-candidate.svg'),role='superseded vertical assembly candidate',recommendation='archive with previous version',disposition='pending author choice')],executed_dispositions=[])
    updates.append((inventorypath,encoded(inventory)))
    artifacts=[(SVG,svg)]
    for p,_ in updates:
        if str(p.resolve()) not in snapshot.hashes:raise ValueError('Missing pre-edit baseline: '+str(p))
    for a in f['source_artifacts']+f['data_artifacts']+f['exports']:snapshot.hashes[a['path']]=a['sha256']
    snapshot.hashes[str(FREEZE)]=e['freeze']['sha256']
    for a in [e['row'],e['candidate_svg'],e['candidate_review']]:snapshot.hashes[a['path']]=a['sha256']
    with state_lock(ASTATE):publish_state(STATE,state,snapshot,artifacts=artifacts,evidence_updates=updates)
    print(REVIEW)
if __name__=='__main__':
    if '--publish' in sys.argv:publish()
    else:prepare()
