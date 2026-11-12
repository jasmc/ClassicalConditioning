"""Verify each plotted point/curve/band against authenticated full-window arrays."""
from pathlib import Path
import sys,json,re
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.figure_workspace import sha
from classical_conditioning.analysis.review.figure4_profiles import GROUPS
NS='{http://www.w3.org/2000/svg}'
NUM=r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?'

def verify(directory):
    d=Path(directory);root=ET.fromstring((d/'figure-04.svg').read_bytes())
    nodes={n.get('id'):n for n in root.iter()if n.get('id')}
    evidence=json.loads((d/'presentation-evidence.json').read_text());reports={}
    for i,a in enumerate(['delay','3strace']):
        z=np.load(d/a/'model-arrays.npz');obs=pd.read_csv(d/a/'observed-points.csv')
        limits=evidence['axes'][a]['ylim'];lo,hi=limits
        x0=(15+i*91.5)*72/25.4;top=15*72/25.4;w=72*72/25.4;h=43*72/25.4
        def mapped(t,y):return np.column_stack([x0+(t+20)/40*w,top+(hi-y)/(hi-lo)*h])
        t=z['times'];assert np.array_equal(t,np.arange(-20,20,.5)+.25)
        for k,g in enumerate(GROUPS):
            path=next(nodes[a+'-fit-'+str(k)].iter(NS+'path'))
            actual=np.array([float(v)for v in re.findall(NUM,path.get('d'))]).reshape(-1,2)
            np.testing.assert_allclose(actual,mapped(t,z['group_curves'][k]),atol=1e-5,rtol=0)
            uses=list(nodes[a+'-observed-'+str(k)].iter(NS+'use'))
            points=np.array([[float(u.get('x')),float(u.get('y'))]for u in uses])
            p=obs.loc[obs.display_group.eq(g)];assert len(p)==80
            np.testing.assert_allclose(points,mapped(p.time_s.to_numpy(),p['mean'].to_numpy()),atol=1e-5,rtol=0)
            # fill_between stores one closed polygon with a baseline vertex at
            # each end, the lower path forwards and upper path backwards.
            band=nodes[a+'-band-'+str(k)]
            uses=list(band.iter(NS+'use'))
            if uses:
                use=uses[0];path=nodes[use.get('{http://www.w3.org/1999/xlink}href')[1:]]
                offset=np.array([float(use.get('x',0)),float(use.get('y',0))])
            else:path=next(band.iter(NS+'path'));offset=np.zeros(2)
            polygon=np.array([float(v)for v in re.findall(NUM,path.get('d'))]).reshape(-1,2)+offset
            q=np.quantile(z['bootstrap_curves'][:,k],[.025,.975],axis=0)
            expected=np.vstack([mapped(t[:1],q[1,:1]),mapped(t,q[0]),mapped(t[-1:],q[1,-1:]),mapped(t[::-1],q[1,::-1])])
            np.testing.assert_allclose(polygon,expected,atol=1e-5,rtol=0)
            assert np.all(q>=lo)and np.all(q<=hi)
            assert p['mean'].between(lo,hi).all()
        reports[a]={'fitted_vertices_verified':240,'observed_centers_verified':240,
            'band_polygons_verified':3,'full_80_bin_support':True,'extrapolation':False}
    assert evidence['axes']['delay']['ylim']==evidence['axes']['3strace']['ylim']
    report={'passed':True,'svg_sha256':sha((d/'figure-04.svg').read_bytes()),'panels':reports,
        'shared_axes_and_zero':True,'verification_source_sha256':sha(Path(__file__).read_bytes())}
    with (d/'numeric-mapping-verification.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))
if __name__=='__main__':verify(sys.argv[1])
