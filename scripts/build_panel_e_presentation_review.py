"""Isolated Panel E presentation gallery; never modifies assembly or freeze."""
import hashlib, json, html
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw

ROOT=Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure1-assembly')
SRC=ROOT/'cadence-review-v5-20261007'
OUT=ROOT/'panel-e-presentation-review-20261007'
TIME='Time relative to CS onset (s)'
GREY='#777777'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True)
    inventory=[]
    for p in sorted(ROOT.rglob('*.svg')):
        if OUT in p.parents: continue
        if 'PanelE' in p.name or 'vigor' in p.name.lower() or 'same_frame' in p.name:
            side=p.with_suffix('.svg.json')
            inventory.append({'path':str(p),'sha256':sha(p),'sidecar':str(side) if side.exists() else None,
                              'sidecar_content':json.loads(side.read_text()) if side.exists() else None})
    paths=[SRC/'Fig1_PanelsD-E_presumed-cadence_frames_v5.parquet',SRC/'Fig1_PanelsD-E_events_v5.parquet',SRC/'Fig1_PanelF_Delay_presumed-cadence_v5.parquet']
    dm=json.loads((SRC/'Fig1_PanelD_TailAngle_presumed-cadence_v5.svg.json').read_text())
    hm=json.loads((SRC/'Fig1_PanelF_Delay_presumed-cadence_v5.svg.json').read_text())
    assert sha(paths[0])==dm['panel_data_sha256']
    assert sha(paths[2])==hm['panel_data_sha256']
    assembly=json.loads((SRC/'figure1-presumed-acquisition-review-v5.svg.json').read_text())
    # Events have no published standalone digest: record their current digest explicitly.
    f,e,h=[pd.read_parquet(p) for p in paths]
    trials=[9,17,63,66,93]
    for t in trials:
        p=f[f['Trial number']==t]; mask=p.eligible.to_numpy(bool)
        for c in ['Vigor','centred_log','bout_median']:
            np.testing.assert_array_equal(np.isfinite(p[c]),mask)
        baseline=np.median(np.log(p.loc[mask & p[TIME].ge(-15) & p[TIME].lt(0),'Vigor']))
        np.testing.assert_allclose(p.loc[mask,'centred_log'],np.log(p.loc[mask,'Vigor'])-baseline,atol=1e-12)
        rebuilt=p.groupby('bin_index').bout_median.mean().reindex(range(80))
        stored=h[h['Trial number']==t].sort_values('Time bin center (s)')['Signed log vigor']
        np.testing.assert_allclose(rebuilt,stored,atol=1e-12,rtol=0,equal_nan=True)
    plt.rcParams.update({'svg.fonttype':'none','path.simplify':False,'font.size':8})
    rawmax=float(f.Vigor.max())*1.04
    lo=float(f.bout_median.min())-.1; hi=float(f.bout_median.max())+.1
    specs=[('A','Raw + signed bout medians','pair'),('B','Raw only · full amplitude','raw'),('C','Raw full amplitude + temporal detail [0,10] s','zoom'),('D','Raw full amplitude + temporal inset [0,10] s','inset'),('E','Raw + framewise centred log overlay','frame'),('F','Raw + repeated bout median overlay','bout'),('G','Raw + stored 0.5 s bin means overlay','bins')]
    specs.append(('H','Raw full amplitude + labelled y zoom','yzoom'))
    cap=float(np.ceil(f.Vigor.quantile(.995)*10)/10)
    records=[]
    for letter,title,kind in specs:
        cols=2 if kind in ['pair','zoom','inset','yzoom'] else 1
        fig,axs=plt.subplots(5,cols,figsize=(9,5.4),squeeze=False)
        fig.subplots_adjust(left=.12,right=.88 if kind in ['frame','bout','bins'] else .97,bottom=.13,top=.80,hspace=.28,wspace=.28)
        fig.suptitle(f'{letter}  {title}',x=.08,ha='left',fontsize=12,fontweight='bold')
        fig.text(.08,.91,'Delay 20221115_07 · presumed acquisition clock · same eligible bout frames · baseline [−15,0) s',fontsize=8)
        fig.text(.08,.86,('Left preserves all amplitudes; right y zoom clips peaks with counts labelled.' if kind=='yzoom' else 'Raw black (rad/ms); scaled grey. Overlay axes are independent: compare timing, not line heights.' if kind in ['frame','bout','bins'] else 'Raw black (rad/ms); signed log grey. No amplitude clipping.'),fontsize=8)
        for i,t in enumerate(trials):
            p=f[f['Trial number']==t]; a=axs[i,0]
            a.plot(p[TIME],p.Vigor,color='black',lw=.45); a.set_ylim(0,rawmax)
            a.set_ylabel(f'Trial {t}\nrad/ms',fontsize=7)
            targets=[a]
            if kind=='pair':
                b=axs[i,1]; b.plot(p[TIME],p.bout_median,color=GREY,lw=1.2); b.set_ylim(lo,hi); b.set_ylabel('bout log',color=GREY); b.axhline(0,color=GREY,lw=.3);targets.append(b)
            if kind=='zoom':
                b=axs[i,1]; b.plot(p[TIME],p.Vigor,color='black',lw=.55);b.set_ylim(0,rawmax);targets.append(b)
            if kind=='yzoom':
                b=axs[i,1]; b.plot(p[TIME],p.Vigor,color='black',lw=.55);b.set_ylim(0,cap);b.set_ylabel(f'Zoom ≤{cap:g}\n{int(p.Vigor.gt(cap).sum())} clipped',fontsize=7);targets.append(b)
                if i==0:b.set_title('Right: labelled amplitude clipping; left: all peaks',fontsize=8)
            if kind=='inset':
                axs[i,1].set_axis_off()
                b=axs[i,1].inset_axes([.15,.12,.8,.82]);b.plot(p[TIME],p.Vigor,color='black',lw=.5);b.set_xlim(0,10);b.set_ylim(0,rawmax);b.tick_params(labelsize=6);b.set_title('Inset: 0–10 s; full y',fontsize=7)
            if kind in ['frame','bout','bins']:
                b=a.twinx(); b.set_ylabel({'frame':'frame log','bout':'bout log','bins':'bin mean log'}[kind],color=GREY,fontsize=7)
                if kind=='bins':
                    q=h[h['Trial number']==t].sort_values('Time bin center (s)')
                    for x,y in zip(q['Time bin center (s)'],q['Signed log vigor']):
                        if np.isfinite(y): b.plot([x-.25,x+.25],[y,y],color=GREY,lw=1.3)
                    b.set_ylim(lo,hi)
                else:
                    c='centred_log' if kind=='frame' else 'bout_median';b.plot(p[TIME],p[c],color=GREY,lw=.65,ls='--' if kind=='frame' else '-',alpha=.9)
                    b.set_ylim((float(f[c].min())-.1,float(f[c].max())+.1))
                b.tick_params(axis='y',colors=GREY,labelsize=7);b.axhline(0,color=GREY,lw=.3);b.spines['top'].set_visible(False)
            for j,b in enumerate(targets):
                b.set_xlim(0,10) if kind=='zoom' and j==1 else b.set_xlim(-20,20)
                for ev in e[e['Trial number']==t].itertuples(index=False):
                    b.axvline(float(ev[2]),color='#702e78' if ev.Event=='actual US onset' else '#0d8136',lw=.5,ls='--' if ev.Event=='CS offset' else '-')
                b.spines[['top','right']].set_visible(False);b.tick_params(labelsize=7)
                if i==4:b.set_xlabel('Time from measured CS onset (s)')
        stem=OUT/f'PanelE_{letter}_{kind}'
        fig.savefig(stem.with_suffix('.svg'));fig.savefig(stem.with_suffix('.png'),dpi=160);fig.savefig(stem.with_suffix('.pdf'));plt.close(fig)
        rec={'id':letter,'title':title,'kind':kind,'selection_status':'review-only; preprocessing confirmation pending','raw_amplitude_clipping':kind=='yzoom','zoom_cap':cap if kind=='yzoom' else None,'scaled_color':GREY,'sources':[{'path':str(p),'sha256':sha(p)} for p in paths],'svg_sha256':sha(stem.with_suffix('.svg'))}
        stem.with_suffix('.svg.json').write_text(json.dumps(rec,indent=2)); records.append(rec)
    thumbs=[]
    for letter,_,kind in specs:
        im=Image.open(OUT/f'PanelE_{letter}_{kind}.png').convert('RGB');im.thumbnail((900,540));thumbs.append(im)
    gallery=Image.new('RGB',(1800,2160),'white')
    for i,im in enumerate(thumbs): gallery.paste(im,((i%2)*900,(i//2)*540))
    gallery.save(OUT/'gallery.png');ImageOps.grayscale(gallery).save(OUT/'gallery-grayscale.png')
    (OUT/'inventory.json').write_text(json.dumps(inventory,indent=2))
    (OUT/'manifest.json').write_text(json.dumps({'candidates':records,'checks':'exact identical finite masks; baseline centred logs reconstructed; all selected stored bins reproduced within 1e-12; no clipping','event_hash_status':'recorded at review build; no independently published event digest','assembly_sidecar_read':True},indent=2))
    page='<html><meta charset="utf-8"><title>Panel E review</title><style>body{font:16px system-ui;margin:30px}img{width:100%}article{border-bottom:1px solid #ccc;padding:20px}a{color:#333}</style><h1>Panel E presentation review</h1><p>All candidates use identical v5 reconstructed-clock data. Review only. A is recommended for clear amplitude interpretation; E distinguishes framewise scaling from bout medians. Bins in G are separate summaries. Only H right uses labelled clipping; its left companion preserves every peak.</p>'
    for letter,title,kind in specs:page+=f'<article><h2>{html.escape(letter+" · "+title)}</h2><a href="PanelE_{letter}_{kind}.svg"><img src="PanelE_{letter}_{kind}.png"></a></article>'
    page+='<h2>Historical inventory</h2><p>Original arrival-clock designs are preserved. P10–P90 and tail_length_weighted_angular_l1 alternatives are different metrics/scaling and were not substituted.</p><a href="inventory.json">Full original SVG hashes and sidecars</a></html>'
    (OUT/'index.html').write_text(page,encoding='utf-8')
    print(OUT/'gallery.png')
if __name__=='__main__':main()
