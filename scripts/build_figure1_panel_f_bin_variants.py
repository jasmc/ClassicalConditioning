"""Panel F comparison from authenticated frames; leaves active assembly unchanged."""
from pathlib import Path
import argparse, base64, copy, html, io, json, sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import Snapshot,file_hash,sha,tag,publish_state
from build_figure_legend_review import figure_font_paths
from build_figure1_fgh_version_comparison import prefix_svg


def calculate(frames,width):
    """Original eligible log-frame medians; equal-bin baseline, as selected F."""
    edges=np.arange(-20,20+width/2,width); rows=[];tables=[]
    for trial,f in frames.groupby('trial',sort=True):
        good=f[f.eligible];idx=np.searchsorted(edges,good.time_s,side='right')-1
        assert np.all((idx>=0)&(idx<len(edges)-1))
        med=good.assign(bin=idx).groupby('bin').log_vigor.median().reindex(range(len(edges)-1)).to_numpy()
        baseline=med[(edges[:-1]>=-15)&(edges[1:]<=0)];baseline=baseline[np.isfinite(baseline)]
        ref=float(np.median(baseline)) if len(baseline) else np.nan;delta=med-ref;rows.append(delta)
        if len(baseline):assert abs(np.nanmedian(delta[(edges[:-1]>=-15)&(edges[1:]<=0)]))<1e-12
        tables.append(pd.DataFrame(dict(acquisition_trial=trial,start_s=edges[:-1],end_s=edges[1:],median_log=med,baseline_log=ref,delta_log=delta,eligible_frames=np.bincount(idx,minlength=len(med)))))
    return np.array(rows),pd.concat(tables,ignore_index=True)


def bouts(frames,cadence):
    """Bout summaries repeated per eligible frame, so baseline votes track duration."""
    tables=[];check=[]
    for trial,f in frames.groupby('trial',sort=True):
        f=f.sort_values('time_s'); good=f.eligible.to_numpy(); logs=f.log_vigor.to_numpy(); ids=f.bout_id.to_numpy();times=f.time_s.to_numpy()
        med=f[good].groupby('bout_id').log_vigor.median(); summaries=f.bout_id.map(med).to_numpy();summaries[~good]=np.nan
        np.testing.assert_allclose(summaries[good],f.bout_median_log.to_numpy()[good],atol=1e-12,rtol=0)
        base=good&(times>=-15)&(times<0);ref=float(np.median(summaries[base])) if base.any() else np.nan
        # Only consecutive eligible samples within a named bout form a rectangle.
        cuts=np.r_[0,np.flatnonzero((ids[1:]!=ids[:-1])|(good[1:]!=good[:-1])|(np.diff(times)>cadence*1.5))+1,len(f)]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            if not good[lo]:continue
            assert np.all(good[lo:hi]) and np.all(ids[lo:hi]==ids[lo])
            tables.append(dict(acquisition_trial=int(trial),bout_id=int(ids[lo]),start_s=max(-20,float(times[lo]-cadence/2)),end_s=min(20,float(times[hi-1]+cadence/2)),median_log=float(summaries[lo]),baseline_log=ref,delta_log=float(summaries[lo]-ref),eligible_frames=int(hi-lo),first_frame_id=int(f.FrameID.iloc[lo]),last_frame_id=int(f.FrameID.iloc[hi-1])))
        check.append(dict(trial=int(trial),baseline_eligible_frames=int(base.sum()),baseline_log=ref))
    return pd.DataFrame(tables),check


def isolate(raw):
    root=ET.fromstring(raw);root=copy.deepcopy(next(n for n in root.iter() if n.get('id')=='edit-FGH-node-0'))
    root.attrib.update(viewBox='0 0 244 221.102362',width='244',height='221.102362',x='0',y='0',id='panel-f-comparison')
    body=next(n for n in root.iter() if n.get('id')=='edit-FGH-figure_1')
    for n in list(body):
        key=n.get('id','')
        if '__f__' not in key and '__h__colorbar__' not in key:body.remove(n)
        elif '__h__colorbar__' in key:n.set('transform','translate(-277 0)')
        elif '__canvas__' in key:body.remove(n)
    for parent in root.iter():
        for child in list(parent):
            if child.tag==tag('metadata'):parent.remove(child)
    return root


def render(source,matrix=None,table=None):
    r=copy.deepcopy(source);node=next(n for n in r.iter() if n.get('id')=='edit-FGH-fig1__f__heatmap__vigor-cells')
    cmap=matplotlib.colormaps['managua_r'].copy();cmap.set_bad('black')
    if matrix is not None:
        pixels=np.asarray(cmap(np.ma.masked_invalid(np.clip((matrix+.25)/.5,0,1)))*255,dtype=np.uint8)
        im=Image.fromarray(pixels[::-1]).resize((matrix.shape[1]*8,90*8),Image.Resampling.NEAREST);buf=io.BytesIO();im.save(buf,format='PNG')
        node.set('{http://www.w3.org/1999/xlink}href','data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode());node.set('shape-rendering','crispEdges');node.set('preserveAspectRatio','none')
        return r
    parent=next(p for p in r.iter() if node in list(p));pos=list(parent).index(node)
    x=float(node.get('x'));width=float(node.get('width'));height=float(node.get('height'));top=26.519055
    parent.remove(node);g=ET.Element(tag('g'),id='panel-f-bout-cells');parent.insert(pos,g)
    ET.SubElement(g,tag('rect'),x=str(x),y=str(top),width=str(width),height=str(height),fill='black')
    for j,row in enumerate(table.itertuples()):
        rgba=cmap(np.ma.masked_invalid(np.array([(row.delta_log+.25)/.5])))[0]
        color='#'+''.join(f'{int(v*255):02x}' for v in rgba[:3])
        ET.SubElement(g,tag('rect'),id=f'bout-run-{j}',x=str(x+(row.start_s+20)*width/40),y=str(top+(row.acquisition_trial-5)*height/90),width=str((row.end_s-row.start_s)*width/40),height=str(height/90),fill=color,stroke='none')
    return r


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--publish',action='store_true');args=parser.parse_args()
    state_path=ROOT/'docs/paper/figures/figure-01/current-state.json';state=json.loads(state_path.read_text())
    ticket=json.loads(Path('/tmp/panel-f-variants-start.json').read_text())
    for p,h in ticket['starting_hashes'].items():assert file_hash(p)==h,'Concurrent edit: '+p
    archive=Path('/Volumes/JOAQUIM/ClassicalConditioning-Archive/repository-cleanup/2026-10-10/payload');folder=archive/'reviews/fgh_c_bout_vs_bins_20261008/outputs'
    sample=folder/'PanelF_sample_data.parquet';manifest=folder/'manifest.json';inventory=ROOT/'records/repository-archive/starting-inventory.json'
    index={r['path']:r.get('sha256') for r in json.loads(inventory.read_text())['files']}
    for p in [sample,manifest]:assert file_hash(p)==index[str(p.relative_to(archive))]
    lineage=json.loads(manifest.read_text());mapping=next(r for r in lineage['source_inputs'] if r['panel']=='F');assert mapping['fish']=='20221115_07'
    selected=state['scoped_fgh_freeze'];svg=Path(state['draft']['working_svg']['path']);accepted=Path(selected['data']['path'])
    for item in [state['draft']['working_svg'],state['draft']['review'],state['draft']['legend'],selected['data']]:assert file_hash(item['path'])==item['sha256']
    snap=Snapshot([sample,manifest,inventory,state_path,svg,accepted,Path(__file__),ROOT/'scripts/build_figure_legend_review.py',ROOT/'src/classical_conditioning/figure_workspace.py'])
    snap.hashes.update(ticket['starting_hashes']);snap.check()
    frames=pd.read_parquet(sample);assert np.array_equal(frames.eligible,frames.log_vigor.notna())
    np.testing.assert_allclose(frames.log_vigor[frames.eligible],np.log(frames.raw_vigor[frames.eligible]),atol=1e-12,rtol=0)
    matrices={};tables={}
    for width in [1,.5,.25]:matrices[str(width)],tables[str(width)]=calculate(frames,width)
    accepted_table=pd.read_csv(accepted);expected=accepted_table[accepted_table.panel.eq('F')].pivot(index='acquisition_trial',columns='bin_index',values='delta_log_vigor').to_numpy()
    np.testing.assert_allclose(matrices['1'],expected,atol=1e-12,rtol=0,equal_nan=True)
    tables['bouts'],checks=bouts(frames,mapping['interval_ms']/1000)
    original=isolate(svg.read_bytes());labels=['Current 1 s bins','0.5 s bins','0.25 s bins','Bout bins · timepoint-weighted baseline'];keys=['1','0.5','0.25','bouts']
    common='F. Delay fish 20221115_07; trials 1–90 (acquisition 5–94). Natural-log legacy distal angular vigor, original movement eligibility and cadence timing. Baseline [−15,0) s. No amplitude scaling or numerical clipping. Same managua_r colour scale ±0.25; black means missing or ineligible. CS/US guides, phase boundaries and illustrative arrows retained.'
    legends=[common+' Current reference: median eligible original log frames in each 1 s bin; subtract the median of finite baseline-bin medians, each bin one vote.',common+' Median eligible original log frames in each 0.5 s bin; subtract the median of finite baseline-bin medians, each bin one vote. Bin width is the only scientific change.',common+' Median eligible original log frames in each 0.25 s bin; subtract the median of finite baseline-bin medians, each bin one vote. Bin width is the only scientific change.',common+' Each named bout’s median eligible original log frames is displayed over its eligible sample intervals, preserving gaps. For the baseline median, repeat the bout median at every eligible baseline timepoint: one vote per eligible frame, so longer eligible bout periods contribute more. Only the part inside the baseline contributes. Non-bout timepoints remain excluded. Unfrozen scientific draft.']
    page='<!doctype html><meta charset="utf-8"><title>Figure 1 panel F bin comparison</title><style>body{font:15px/1.5 system-ui;margin:24px;color:#222}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:32px;max-width:1100px;margin:auto}svg{width:100%;height:auto}h2{font-size:18px}.legend{border-top:1px solid #ddd;padding-top:12px}@media(max-width:700px){.grid{grid-template-columns:1fr}}</style><main class="grid">'
    for key,label,legend in zip(keys,labels,legends):
        r=original if key=='1' else render(original,matrix=matrices[key]) if key!='bouts' else render(original,table=tables[key])
        fontroot=copy.deepcopy(r);defs=ET.SubElement(fontroot,tag('defs'));css=[]
        for weight,path in zip((400,700),figure_font_paths()):css.append('@font-face{font-family:"DejaVu Sans";font-weight:'+str(weight)+';src:url(data:font/ttf;base64,'+base64.b64encode(path.read_bytes()).decode()+') format("truetype")}')
        ET.SubElement(defs,tag('style')).text=''.join(css)
        raw=prefix_svg(ET.tostring(fontroot),'variant-'+key)
        page+='<section><h2>'+html.escape(label)+'</h2>'+raw+'<p class="legend">'+html.escape(legend)+'</p></section>'
    page+='</main>';page=page.encode()
    output=Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure-01-panel-f-bin-variants-2026-10-14.html');evidence=ROOT/'records/figure-reviews/figure-01/panel-f-bin-variants-2026-10-14.json'
    data=pd.concat([v.assign(variant=k) for k,v in tables.items()],ignore_index=True).to_csv(index=False).encode()
    report=dict(scope=['F'],status='unfrozen-comparison',source_working_svg=state['draft']['working_svg'],source_selection=selected['selected_version'],starting_hashes=snap.hashes,scientific_definition=dict(fixed_bins='direct original eligible log-frame medians; equal-bin baseline median',bout_bins='named bout median repeated over eligible frame intervals; median repeated summaries over eligible baseline frames',baseline_s=[-15,0],timepoint_weight='one per eligible frame, established constant camera cadence',nonbout_frames='excluded as in current F',colour_limits=[-.25,.25],amplitude_scaling=False),verification=dict(one_second_reproduces_accepted_F=True,cached_bout_medians_match_recomputed=True,trials=90,defined_trials={k:int(np.any(np.isfinite(m),axis=1).sum()) for k,m in matrices.items()},bout_runs=len(tables['bouts']),bout_baselines=checks),review=dict(path=str(output),sha256=sha(page)),data=dict(path=str(output.with_suffix('.csv')),sha256=sha(data)),legend_association=dict(scope='F per variant',source='version-bound inline legends in comparison HTML',texts=dict(zip(keys,legends))),active_assembly_changed=False,frozen=False)
    if not args.publish:
        Path('/tmp/panel-f-bin-variants.html').write_bytes(page);Path('/tmp/panel-f-bin-variants-report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'preview':'/tmp/panel-f-bin-variants.html','verification':{k:v for k,v in report['verification'].items() if k!='bout_baselines'}}));return
    report['visual_review']=dict(status='passed',evidence='Browser rendering of exact comparison page inspected: four F-only views with shared colourbar, unchanged guides/axes/arrows, finer fixed bins and sparse bout intervals; no clipping.')
    record=(json.dumps(report,indent=2)+'\n').encode();state['draft'].setdefault('panel_comparisons',{})['F-bin-variants-20261014']=dict(review=report['review'],data=report['data'],evidence=dict(path=str(evidence),sha256=sha(record)),source=state['draft']['working_svg'],status='unfrozen alternatives; active working assembly retained')
    publish_state(state_path,state,snap,artifacts=[(output,page),(output.with_suffix('.csv'),data),(evidence,record)])
    print(json.dumps({'published':str(output),'evidence':str(evidence)}))

if __name__=='__main__':main()
