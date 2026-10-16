"""Verify exact Figure 3 heatmap/color roles and G observed-outcome transfer."""
from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from classical_conditioning.analysis.review.individual_atlas import sha,save_json
from classical_conditioning.figures.layout_frames import assess

NS='{http://www.w3.org/2000/svg}'


def verify(out):
    reference=json.loads((out/'reference-style.json').read_text())
    for record in reference['references']:
        if sha(record['path'])!=record['sha256']:raise ValueError('Changed Figure 2 reference')
    live=json.loads(Path(reference['live_reference_state']).read_text())
    live_svg=Path(live['draft']['working_svg']['path'])
    if sha(live_svg)!=live['draft']['working_svg']['sha256']:raise ValueError('Figure 2 is being published; reconcile reference after completion')
    live_root=ET.fromstring(live_svg.read_bytes())
    for condition,color in reference['condition_colors'].items():
        group=next(n for n in live_root.iter() if n.get('id')==f'fig2__g__native__{condition}-median')
        if 'stroke: '+color not in next(group.iter(NS+'path')).get('style',''):
            raise ValueError('Selected Figure 2 G condition color advanced; reconcile this scoped relationship')
    for record in reference['references']:
        source=record['authenticated_source_path']
        if '/selections/' in source and sha(source)!=record['authenticated_source_sha256']:
            raise ValueError('Selected Figure 2 scientific reference advanced')
    preserved=json.loads((out/'numerical-preservation.json').read_text())
    for record in preserved:
        if sha(out/record['path'])!=record['before']:raise ValueError('Changed original numerical/classifier input')
    frames=pd.read_parquet(out/'trial-data.parquet')
    exported=pd.read_csv(out/'g-logmedian-outcomes.csv')
    keys=['experiment_id','condition_id','fish_id','trial_number']
    if frames.duplicated(keys).any() or exported.duplicated(keys).any():raise ValueError('Duplicate G outcome keys')
    joined=exported.merge(frames,on=keys,validate='one_to_one',suffixes=('_export','_source'))
    if len(joined)!=len(frames) or len(exported)!=len(frames):raise ValueError('Incomplete G outcome inventory')
    np.testing.assert_allclose(joined.response_minus_baseline_logmedian,
        joined.response_logmedian_source-joined.baseline_logmedian_source,rtol=1e-12,atol=1e-12,equal_nan=True)

    cards={k:base64.b64decode(v) for k,v in json.loads((out/'card-svg-data.json').read_text()).items()}
    inventory=pd.read_csv(out/'fish-inventory.csv').set_index('fish_id',verify_integrity=True)
    cmap=matplotlib.colormaps[reference['heatmap_palette']]
    allowed={tuple(rgb) for rgb in cmap(np.arange(cmap.N),bytes=True)[:,:3]}
    allowed.add((0,0,0))
    registry={};checks=[]

    def check_card(root,fish,compact=False):
        condition=inventory.loc[fish,'condition_id'];color=reference['condition_colors'][condition]
        identity=f'fish_{condition}_{fish}'+('_compact' if compact else '')
        title=next(n for n in root.iter(NS+'text') if ''.join(n.itertext())==fish+' · '+condition.capitalize())
        if 'fill: '+color not in title.get('style',''):raise ValueError('Incorrect rendered condition color '+fish)
        main_axis_id=identity+('__axes_3' if compact else '__axes_5')
        main_axis=next(n for n in root.iter() if n.get('id')==main_axis_id)
        available=frames.loc[frames.fish_id.eq(fish),['baseline_logmedian','response_logmedian']].notna().all(axis=1).any()
        if available:
            styles=[n.get('style','') for n in main_axis.iter()]
            if not any('fill: '+color in value for value in styles) or not any('stroke: '+color in value for value in styles):
                raise ValueError('G observed points/median bars do not use the reference condition color '+fish)
        heatmap=next(n for n in root.iter() if n.get('id')==identity+'__axes_1')
        image=next(heatmap.iter(NS+'image'))
        href=image.get('{http://www.w3.org/1999/xlink}href',image.get('href'))
        rgba=np.asarray(Image.open(io.BytesIO(base64.b64decode(href.split(',')[1]))).convert('RGBA'))
        colors={tuple(c) for c in np.unique(rgba[:,:,:3].reshape(-1,3),axis=0)}
        if colors-allowed:raise ValueError('Heatmap contains pixels outside the native reference palette '+fish)
        if not np.all(rgba[:,:,3]==255):raise ValueError('Unexpected transparent heatmap pixels '+fish)
        if not compact:
            colorbar=next(n for n in root.iter() if n.get('id')==identity+'__axes_2')
            labels=[''.join(n.itertext()).replace('−','-') for n in colorbar.iter(NS+'text')]
            if not {'-0.25','0.00','0.25'}.issubset(labels):raise ValueError('Incorrect rendered scale '+fish)
        text=' '.join(''.join(n.itertext()) for n in root.iter(NS+'text'))
        if 'Figure 2 G outcome' not in text:raise ValueError('G outcome not identified '+fish)
        if not compact and 'Exact legacy WIP input' not in text:raise ValueError('Separate classifier input missing '+fish)
        key=fish+('-compact' if compact else '')
        registry[key]={'fish_id':fish,'condition_id':condition,'owner_group':identity+'__figure_1',
            'heatmap':{'element_id':identity+'__axes_1','scientific_role':'heatmap.individual_centered_log_bout_vigor',
                'source':{'path':str(out/'heatmaps.npz'),'sha256':sha(out/'heatmaps.npz'),'array':fish+'__heat'},
                'quantity':'direct 1 s median ln positive bout vigor minus median baseline-bin medians [-15,0)',
                'palette':reference['heatmap_palette'],'limits':reference['heatmap_limits'],'undefined':'black'},
            'main_trial_outcome':{'element_id':main_axis_id,'scientific_role':'observed.individual_response_minus_baseline_logmedian',
                'source':{'path':str(out/'g-logmedian-outcomes.csv'),'sha256':sha(out/'g-logmedian-outcomes.csv')},
                'windows_s':{'baseline':[-15,0],'response':[0,9]},'summary':'five-trial medians; no fitted trajectory or inferential band'},
            'condition_style':{'role':'condition.identity','color':color},
            'other_quantities':'Movement fractions/availability retain Greys on 0..1; legacy classifier input/score scales remain distinct'}
        checks.append({'fish_id':fish,'compact':compact,'heatmap_palette_pixels_verified':True,
            'condition_title_and_main_points_color_verified':True,'full_colorbar_limits_verified':not compact,'native_palette_colors':len(colors)})

    for fish,data in cards.items():check_card(ET.fromstring(data),fish)
    overview=ET.fromstring((out/'overview.svg').read_bytes())
    for selected in json.loads((out/'selected-examples.json').read_text()):
        fish=selected['fish_id'];condition=selected['condition_id']
        owner=next(n for n in overview.iter() if n.get('id')==f'fish_{condition}_{fish}_compact__figure_1')
        check_card(owner,fish,True)
    layout=json.loads((out/'layout-evidence.json').read_text())
    vectors={'overview':(out/'overview.svg').read_bytes(),**cards}
    if set(layout)!=set(vectors):raise ValueError('Incomplete layout evidence')
    for name,data in vectors.items():
        if assess(data,layout[name]['browser'])['status']!='passed':raise ValueError('Stale/failing exact layout evidence')
    save_json(out/'draft-registry.json',{'schema':'figure3-exploratory-registry/1.0','scope':'Figure 3 individual atlas',
        'element_registry':registry,'relationships':[
            {'type':'scientific','property':'shared heatmap frame-log/bin/baseline estimator','reference':'selected Figure 2 A / Figure 1 F-H; individual rather than across-fish pooling'},
            {'type':'scientific','property':'main observed trial outcome','reference':'Figure 2 G definition; individual fish, no transfer of population LMM/bootstrap or learner labels'},
            {'type':'presentation_style','property':'all heatmap palettes/limits and condition colors','reference':reference}]})
    save_json(out/'consistency-evidence.json',{'scope':'Requested Figure 2 heatmap scale, G colors and G observed outcome',
        'status':'passed for listed properties','artifacts':{name:sha(out/name) for name in ['overview.svg','review.html','card-svg-data.json','g-logmedian-outcomes.csv']},
        'reference_style':{'path':str(out/'reference-style.json'),'sha256':sha(out/'reference-style.json')},
        'live_reference_scope_check':{'state_path':reference['live_reference_state'],
            'state_sha256':sha(reference['live_reference_state']),'svg':live['draft']['working_svg'],
            'G_colors_still_match':True,'scientific_selections_unchanged':True,
            'scope':'G color roles and A/G scientific selections; unrelated Figure 2 panel changes are preserved'},
        'checker':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
        'checks':checks,'unchanged_numerical_and_classifier_files':len(preserved),'g_outcome_rows_verified':len(joined),
        'exact_layout_checks':len(layout),'limitations':['Palette pixel membership and labelled limits verified; per-cell raster-coordinate inversion not asserted',
            'Individual observations/medians do not inherit G population-model uncertainty or learner validation']})
    print(json.dumps({'cards_verified':len(checks),'G_trial_rows':len(joined),'layout_checks':len(layout)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    verify(p.parse_args().output_dir.resolve())
