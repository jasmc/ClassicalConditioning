"""Standard Matplotlib diagnostic and individual views for the isolated report."""
import io
import html
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd


def svg_text(fig,prefix):
    # HTML's SVG parser recognizes canonical xlink:href, not arbitrary XML
    # namespace prefixes. Preserve reusable band/marker <use> elements.
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    stream=io.StringIO();fig.savefig(stream,format="svg")
    root=ET.fromstring(stream.getvalue())
    ids={e.attrib["id"]:prefix+"-"+e.attrib["id"] for e in root.iter() if "id" in e.attrib}
    for e in root.iter():
        for key,value in list(e.attrib.items()):
            if key=="id":e.attrib[key]=ids[value]
            else:
                for old,new in ids.items():
                    value=value.replace(f"url(#{old})",f"url(#{new})")
                    if value=="#"+old:value="#"+new
                e.attrib[key]=value
    return ET.tostring(root,encoding="unicode")


def diagnostic_views(out):
    import matplotlib.pyplot as plt
    from scipy.stats import probplot
    blocks=[]
    for target in ["difference","adjusted_response"]:
        path=out / target / "residuals.parquet"
        if not path.exists():continue
        d=pd.read_parquet(path)
        fig,axes=plt.subplots(2,2,figsize=(9,6),layout="constrained")
        axes[0,0].scatter(d.fitted,d.residual,s=2,alpha=.18,color="#4d5378")
        axes[0,0].axhline(0,color=".5",lw=.7);axes[0,0].set(xlabel="Fitted outcome",ylabel="Residual",title="Residuals versus fitted")
        probplot(d.residual,dist="norm",plot=axes[0,1]);axes[0,1].set_title("Normal quantile plot")
        for condition,color in [("control","#009fd5"),("delay","#d83983")]:
            g=d.loc[d.condition_id==condition].groupby("trial_number",observed=True).residual.std()
            axes[1,0].plot(g.index,g.values,color=color,label=condition,lw=1)
        axes[1,0].set(xlabel="Scheduled CS trial",ylabel="Residual SD",title="Spread across trials");axes[1,0].legend(frameon=False)
        shifted=d[["fish_key","trial_number","residual"]].copy();shifted.trial_number-=1
        adjacent=d.merge(shifted,on=["fish_key","trial_number"],suffixes=("_now","_next"))
        axes[1,1].scatter(adjacent.residual_now,adjacent.residual_next,s=2,alpha=.18)
        axes[1,1].set(xlabel="Residual at t",ylabel="Residual at t+1",title="Actual adjacent trials only")
        for axis in axes.flat:axis.spines[["top","right"]].set_visible(False)
        fig.suptitle(target.replace("_"," ").title()+" — Gaussian adequacy review")
        image=svg_text(fig,target+"-diagnostics");plt.close(fig)
        (out / target / "model-diagnostics.svg").write_text(image)
        blocks.append("<details><summary>"+html.escape(target.replace("_"," "))+" model diagnostics</summary>"+image+"</details>")
    return "".join(blocks)


def individual_gallery(out):
    import matplotlib.pyplot as plt
    path=out / "individual-training-trajectories.parquet"
    if not path.exists():return ""
    d=pd.read_parquet(path);cards=[];options=[]
    if d.empty:return '<p>No accepted individual model trajectories; individual inference is withheld.</p>'
    fish=d[["condition_id","fish_id"]].drop_duplicates().sort_values(["condition_id","fish_id"])
    for index,row in enumerate(fish.itertuples(index=False)):
        fig,axes=plt.subplots(1,2,figsize=(9,3.5),layout="constrained",sharex=True,sharey=True)
        for method,axis,color in zip(["spline","state_space"],axes,["#7543a1","#147d92"]):
            g=d.loc[(d.method==method)&(d.fish_id==row.fish_id)].sort_values("trial_number")
            axis.plot(g.trial_number,g.estimate,color=color,lw=1.2)
            axis.fill_between(g.trial_number,g.conditional_lower,g.conditional_upper,color=color,alpha=.18,lw=0)
            available=g.loc[g.observed]
            axis.scatter(available.trial_number,available.estimate,s=8,color=color)
            axis.axhline(0,color=".5",lw=.7)
            axis.axhline(-np.log(.90),color=".5",lw=.7,ls=":")
            axis.set(title=method.replace("_"," ").title(),xlabel="Scheduled training CS trial",xlim=(14,65))
            axis.spines[["top","right"]].set_visible(False)
        axes[0].set_ylabel("Conditioned-direction expression\n(log units; conditional model)")
        fig.suptitle(row.fish_id+" / "+row.condition_id.title())
        image=svg_text(fig,"individual-"+row.fish_id);plt.close(fig)
        cards.append(f"<div class='individual-card' data-index='{index}' style='display:{'block' if index==0 else 'none'}'>{image}</div>")
        options.append(f"<option value='{index}'>{html.escape(row.condition_id.title()+' / '+row.fish_id)}</option>")
    return ("<h3>Inspect individual training curves</h3><p>Lines and bands are fitted model outputs. "
        "Dots mark trials with an available observed outcome; gaps remain model predictions. "
        "Dotted reference: unapproved 10% candidate threshold. Bands are conditional empirical-Bayes envelopes.</p>"
        "<label for='individual-fish'>Fish </label><select id='individual-fish'>"+"".join(options)+"</select>"+"".join(cards)+
        "<script>document.getElementById('individual-fish').addEventListener('change',function(){"
        "document.querySelectorAll('.individual-card').forEach(e=>e.style.display=e.dataset.index===this.value?'block':'none');});</script>")
