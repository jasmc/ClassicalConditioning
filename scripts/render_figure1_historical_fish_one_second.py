"""Apply the approved 1 s recipe to the authenticated legacy F/G/H fish."""
from pathlib import Path
import base64, io, json, sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
from matplotlib.colors import Normalize

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot, file_hash, sha, state_lock, tag
from build_figure_legend_review import svg_editing_page, embed_consistent_svg_fonts


def main():
    archive=Path('/Volumes/JOAQUIM/ClassicalConditioning-Archive/repository-cleanup/2026-10-10/payload')
    folder=archive/'reviews/fgh_c_bout_vs_bins_20261008/outputs'
    sample=folder/'PanelG_sample_data.parquet';manifest=folder/'manifest.json'
    inventory=ROOT/'records/repository-archive/starting-inventory.json'
    index={r['path']:r.get('sha256') for r in json.loads(inventory.read_text())['files']}
    for p in [sample,manifest]:assert file_hash(p)==index[str(p.relative_to(archive))]
    lineage=json.loads(manifest.read_text());assert next(r for r in lineage['source_inputs'] if r['panel']=='G')['fish']=='20230307_12'
    state=ROOT/'docs/paper/figures/figure-01/current-state.json'
    selected=json.loads(state.read_text())['scoped_fgh_freeze']
    frozen=Path(selected['svg']['path']);current=Path(selected['data']['path'])
    assert file_hash(frozen)==selected['svg']['sha256'] and file_hash(current)==selected['data']['sha256']
    snap=Snapshot([sample,manifest,inventory,state,frozen,current,Path(__file__)])
    frames=pd.read_parquet(sample,columns=['trial','FrameID','time_s','eligible','raw_vigor','log_vigor'])
    assert np.array_equal(frames.eligible,frames.log_vigor.notna())
    good=frames.eligible.to_numpy();np.testing.assert_allclose(frames.log_vigor.to_numpy()[good],np.log(frames.raw_vigor.to_numpy()[good]),atol=1e-12,rtol=0)
    edges=np.arange(-20,21);rows=[];tables=[];residuals=[]
    for trial,f in frames.groupby('trial',sort=True):
        f=f[f.eligible];idx=np.searchsorted(edges,f.time_s.to_numpy(),side='right')-1
        values=f.log_vigor.to_numpy();assert np.all((idx>=0)&(idx<40))
        med=f.assign(bin=idx).groupby('bin').log_vigor.median().reindex(range(40)).to_numpy()
        direct=np.array([np.median(values[idx==j]) if np.any(idx==j) else np.nan for j in range(40)])
        np.testing.assert_allclose(med,direct,rtol=0,atol=1e-12,equal_nan=True)
        base=med[5:20];base=base[np.isfinite(base)];reference=float(np.median(base)) if len(base) else np.nan;delta=med-reference
        if len(base):residuals.append(abs(np.nanmedian(delta[5:20])))
        rows.append(delta)
        tables.append(pd.DataFrame(dict(panel='G',fish='20230307_12',acquisition_trial=trial,bin_index=np.arange(40),start_s=edges[:-1],end_s=edges[1:],bin_log_vigor=med,baseline_log_vigor=reference,delta_log_vigor=delta,eligible_frames=np.bincount(idx,minlength=40))))
    matrix=np.asarray(rows);assert matrix.shape==(90,40) and max(residuals)<1e-12
    current_table=pd.read_csv(current);full=pd.concat([current_table[current_table.panel.eq('F')],pd.concat(tables),current_table[current_table.panel.eq('H')]],ignore_index=True)
    root=ET.fromstring(frozen.read_bytes());node=next(n for n in root.iter() if n.get('id')=='fig1__g__heatmap__vigor-cells')
    href='{http://www.w3.org/1999/xlink}href';old=Image.open(io.BytesIO(base64.b64decode(node.get(href).split(',')[1])))
    cmap=matplotlib.colormaps['managua_r'].copy();cmap.set_bad('black')
    rgba=np.asarray(cmap(Normalize(-.25,.25,clip=True)(np.ma.masked_invalid(matrix)))*255,dtype=np.uint8)
    img=Image.fromarray(rgba[::-1]).resize(old.size,resample=Image.Resampling.NEAREST);buffer=io.BytesIO();img.save(buffer,format='PNG');node.set(href,'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode())
    heading=next(n for n in root.iter() if n.get('id')=='fig1__g__heading__fish-id')
    next(n for n in heading.iter() if n.tag==tag('text')).text='20230307_12'
    # The cloned row is a historical-fish review, not the newly frozen selection.
    for parent in root.iter():
        for child in list(parent):
            if child.tag==tag('metadata'):parent.remove(child)
    report={'fish':{'F':'20221115_07','G':'20230307_12','H':'20221115_09'},'legacy_identity_evidence':artifact(manifest),'git_context':{'February_commit':'2f63ef4','March_commit':'bf46bf7','file':'2_ExampleFishPlotting.py','interpretation':'Delay 20221115_07 active; Trace 20230307_12 and Control 20221115_09 listed as commented examples; actual legacy F/G/H set confirmed by hash-authenticated artifact manifest'},'analysis':{'bin_width_s':1,'bin_estimator':'direct original eligible natural-log frame median','baseline_s':[-15,0],'baseline_estimator':'median finite baseline-bin medians','amplitude_scaling':False,'color_limits':[-.25,.25],'palette':'managua_r'},'trace_defined_trials':int(np.any(np.isfinite(matrix),axis=1).sum()),'trace_maximum_baseline_median_residual':float(max(residuals)),'sources':snap.hashes,'active_selection_changed':False,'frozen':False}
    ET.SubElement(root,tag('metadata')).text=json.dumps(report)
    from build_figure_legend_review import figure_font_paths
    defs=ET.SubElement(root,tag('defs'));css=[]
    for weight,path in zip((400,700),figure_font_paths()):
        css.append('@font-face{font-family:"DejaVu Sans";font-weight:'+str(weight)+';src:url(data:font/ttf;base64,'+base64.b64encode(path.read_bytes()).decode()+') format("truetype")}')
    ET.SubElement(defs,tag('style')).text=''.join(css)
    ramp=next(n for n in root.iter() if n.get('id')=='fig1__h__colorbar__colour-ramp');ramp.set('shape-rendering','crispEdges')
    svg=ET.tostring(root)
    legend=('F–H. Legacy example fish: Delay 20221115_07, 3 s Trace 20230307_12 and Control 20221115_09. The exact fish set is authenticated by the archived legacy F/G/H source manifest; February/March Git snapshots retain these IDs in the example configuration (Trace and Control commented). This review applies the newly approved analysis to those fish: direct medians of eligible natural-log original frames in 1 s bins, subtracting each trial’s median of finite baseline-bin medians over [−15,0). No amplitude scaling or numerical clipping; managua_r limits ±0.25, zero midpoint, missing black. Trial numbers 1–90 map to acquisition 5–94. Green CS onset/offset guides, dotted expected training US, phase boundaries and Delay illustrative arrows follow the new frozen layout. Delay and Control are identical to the new freeze; only the Trace fish changes. This historical-fish comparison is not frozen.')
    output=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure-01-historical-fish-one-second-managua-2026-10-13.html');page=svg_editing_page(svg,legend=legend).encode();data=full.to_csv(index=False).encode()
    report['review']={'path':str(output),'sha256':sha(page)};report['data']={'path':str(output.with_suffix('.csv')),'sha256':sha(data)}
    evidence=ROOT/'records/figure-reviews/figure-01/historical-fish-one-second-managua-2026-10-13.json'
    with state_lock(state):
        snap.check()
        for p,raw in [(output,page),(output.with_suffix('.csv'),data),(evidence,(json.dumps(report,indent=2)+'\n').encode())]:
            with p.open('xb') as stream:stream.write(raw)
    Path('/tmp/figure-01-historical-fish-managua.svg').write_bytes(svg)
    print(json.dumps({'review':str(output),'fish':report['fish'],'trace_defined_trials':report['trace_defined_trials']}))


def artifact(p):return {'path':str(p),'sha256':file_hash(p)}
if __name__=='__main__':main()
