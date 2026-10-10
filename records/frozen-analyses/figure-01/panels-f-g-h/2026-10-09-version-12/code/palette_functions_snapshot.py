"""Palette proposals only; no heatmap or scientific definition is changed."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap,to_rgb
from matplotlib.patches import Rectangle
HERE=Path(__file__).resolve().parent
reference=['#81e7ff','#5775b3','#582948','#b26343','#e09f57']
candidate=['#81bddc','#497faa','#383842','#bd7448','#e9b978']
# Interpolate in CIELAB so lightness changes steadily between proposed anchors.
def rgb_lab(rgb):
    rgb=np.asarray(rgb);lin=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    xyz=lin @ np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]]).T
    v=xyz/np.array([.95047,1,1.08883]);d=6/29
    f=np.where(v>d**3,np.cbrt(v),v/(3*d*d)+4/29)
    return np.stack([116*f[...,1]-16,500*(f[...,0]-f[...,1]),200*(f[...,1]-f[...,2])],axis=-1)
def lab_rgb(lab):
    lab=np.asarray(lab);fy=(lab[...,0]+16)/116
    f=np.stack([fy+lab[...,1]/500,fy,fy-lab[...,2]/200],axis=-1);d=6/29
    xyz=np.where(f>d,f**3,3*d*d*(f-4/29))*np.array([.95047,1,1.08883])
    lin=xyz@np.linalg.inv(np.array([[.4124564,.3575761,.1804375],[.2126729,.7151522,.0721750],[.0193339,.1191920,.9503041]])).T
    return np.where(lin<=.0031308,12.92*lin,1.055*np.maximum(lin,0)**(1/2.4)-.055)
positions=np.linspace(0,1,513)
lab=rgb_lab([to_rgb(c) for c in candidate])
interpolated=np.stack([np.interp(positions,np.linspace(0,1,5),lab[:,i]) for i in range(3)],axis=-1)
rgb=lab_rgb(interpolated)
assert rgb.min()>-1e-5 and rgb.max()<1+1e-5
rgb=np.clip(rgb,0,1)
lightness=rgb_lab(rgb)[:,0]
assert np.all(np.diff(lightness[:257])<0) and np.all(np.diff(lightness[256:])>0)
fig=plt.figure(figsize=(11,6.2),facecolor='white')
fig.text(.05,.95,'Palette brainstorm: dark baseline, cool reductions, warm increases',fontsize=14,weight='bold')
rows=[('Current softer managua: five colours',reference,False),
      ('Candidate: blue–charcoal–amber, continuous',candidate,True),
      ('Candidate: five colours',candidate,False),
      ('Candidate: three colours',[candidate[0],candidate[2],candidate[4]],False)]
for n,(name,colours,continuous) in enumerate(rows):
    y=.75-n*.19;fig.text(.05,y+.085,name,fontsize=11)
    ax=fig.add_axes([.05,y,.80,.06]);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    if continuous:ax.imshow(rgb[None,:,:],extent=(0,1,0,1),aspect='auto',interpolation='nearest')
    else:
        for i,c in enumerate(colours):ax.add_patch(Rectangle((i/len(colours),0),1/len(colours),1,facecolor=c))
    for pos,word in [(0,'Reduction'),(.5,'Baseline'),(1,'Increase')]:
        ax.text(pos,-.20,word,ha={0:'left',.5:'center',1:'right'}[pos],va='top',transform=ax.transAxes,fontsize=9)
    nan=fig.add_axes([.90,y,.05,.06]);nan.set_facecolor('black');nan.set_xticks([]);nan.set_yticks([])
    for spine in nan.spines.values():spine.set_visible(False)
    nan.text(.5,-.20,'NaN',ha='center',va='top',transform=nan.transAxes,fontsize=9)
fig.savefig(HERE/'palette_comparison.png',dpi=160,bbox_inches='tight')
plt.close(fig)
(HERE/'palette_proposal.json').write_text(json.dumps({'status':'brainstorm only; not applied to heatmaps',
    'candidate_name':'blue-charcoal-amber (custom)','five_colours':candidate,'three_colours':[candidate[i] for i in [0,2,4]],
    'continuous_interpolation':'CIELAB between five anchors at 0/0.25/0.5/0.75/1',
    'centre_colour':candidate[2],'missing_colour':'#000000','anchor_lightness':lab[:,0].tolist(),
    'sRGB_gamut_and_monotonic_lightness_each_side_checked':True,
    'not_claimed_fully_perceptually_uniform_or_colour_vision_validated':True},indent=2))
print('Created palette comparison; no existing heatmaps changed')
