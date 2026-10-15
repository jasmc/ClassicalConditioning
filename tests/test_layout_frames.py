"""Physical layout and bound-rendering evidence; synthetic sources only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from classical_conditioning.figures import layout_frames as f


def spec():
    return dict(scope='figure',calibration=True,inset_mm=1.5,final_size_mm=[89.,30.],
                allocations=[dict(id='A',kind='panel',box_mm=[0,0,44.5,30.],group_ids=['A']),
                             dict(id='B',kind='panel',box_mm=[44.5,0,44.5,30.],group_ids=['B'])],
                background_ids=[],alignments=[])


def svg(s=None):
    return f.embed_spec(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 89 30"><g id="A"><path d="M 3 3 L 30 20"/></g><g id="B"><rect x="48" y="3" width="30" height="20"/></g></svg>',s or spec())


def browser(data):
    s=f.read_spec(data)
    return dict(artifact_sha256=f.sha(data),spec_sha256=f.object_sha(s),collector_sha256=f.sha(f.COLLECTOR.read_bytes()),
                final_size_mm=s['final_size_mm'],scope=s['scope'],user_agent='synthetic unit fixture, not visual review',
                fonts_ready=True,images_ready=True,issues=[],raster_dpi=600,
                allocations={a['id']:dict(painted_bounds_mm=f.inset(a['box_mm'],2),unclipped_bounds_mm=f.inset(a['box_mm'],2)) for a in s['allocations']})


class LayoutFrameTests(unittest.TestCase):
    def test_spacing_requires_equal_measured_gaps_even_when_margins_pass(self):
        s=spec();s['data_axes']=[{'id':p,'element_id':p} for p in ('A','B','C')]
        s['spacing_groups']=[dict(id='related-ABC',target='data_axes',axis='x',ids=['A','B','C'],minimum_gap_mm=2)]
        data=svg(s);b=browser(data)
        # Bounds stay contained; the new requirement catches unequal gaps.
        b['data_axes']={'A':[3,3,10,10],'B':[25,3,10,10],'C':[45,3,10,10]}
        self.assertIn('Spacing failed: related-ABC/data_axes',f.assess(data,b)['issues'])
        b['data_axes']['B'][0]=24
        self.assertEqual(f.assess(data,b)['status'],'passed')

    def test_spacing_cannot_be_hidden_by_clipping_or_unknown_members(self):
        s=spec();s['allocations']=[dict(id=p,kind='panel',box_mm=[i*89/3,0,89/3,30],group_ids=[p]) for i,p in enumerate(('A','B','C'))]
        s['spacing_groups']=[dict(id='row',target='content',axis='x',ids=['A','B','C'],minimum_gap_mm=2)]
        data=svg(s);b=browser(data)
        b['allocations']['B']['unclipped_bounds_mm'][0]+=.4
        self.assertIn('Spacing failed: row/unclipped_bounds_mm',f.assess(data,b)['issues'])
        s['spacing_groups'][0]['ids']=['A','B','missing']
        with self.assertRaises(ValueError):f.validate_spec(s)

    def test_shared_edge_and_interior_spacing(self):
        s=f.validate_spec(spec());a,b=[f.inset(p['box_mm'],s['inset_mm']) for p in s['allocations']]
        self.assertEqual(a[0],1.5);self.assertEqual(b[0]-a[0]-a[2],3)
        self.assertEqual(s['final_size_mm'][0]-b[0]-b[2],1.5)

    def test_unequal_panels_shared_colourbar_and_single_panel(self):
        s=spec();s['allocations'][0]['box_mm'][2]=30;s['allocations'][1]['box_mm']=[30,0,50,30]
        s['allocations'].append(dict(id='key',kind='shared',box_mm=[80,0,9,30],group_ids=['key']))
        f.validate_spec(s)
        s['scope']='panel';s['allocations']=[dict(id='A',kind='panel',box_mm=[0,0,89.,30.],group_ids=['A'])]
        f.validate_spec(s)

    def test_invalid_geometry_spacing_and_pending_policy(self):
        for mutation in (lambda s:s.update(inset_mm=True),lambda s:s.update(final_size_mm=[float('nan'),30]),
                         lambda s:s['allocations'][0].update(box_mm=[1.5,0,43,30]),
                         lambda s:s['allocations'][1].update(box_mm=[43,0,46,30])):
            s=spec();mutation(s)
            with self.assertRaises(ValueError):f.validate_spec(s)
        pending=f.load_policy()[0];pending.update(activation='pending-author-calibration',selected_inset_mm=None)
        ordinary=spec();ordinary['calibration']=False
        with self.assertRaises(ValueError):f.validate_spec(ordinary,pending)
        s=spec();s['allocations'][0]['box_mm'][2]=43
        with self.assertRaisesRegex(ValueError,'grouping'):f.validate_spec(s)
        s['grouping_gaps']=[dict(between=['A','B'],gap_mm=1.5,reason='named group separation')]
        f.validate_spec(s)

    def test_invisible_metadata_does_not_draw_a_frame(self):
        data=svg();root=ET.fromstring(data)
        self.assertEqual(root.get('width'),'89.0mm')
        self.assertEqual(len(root.findall(f.SVG+'rect')),0)
        self.assertEqual(f.read_spec(data),spec())

    def test_browser_measurements_required_and_bound(self):
        data=svg();self.assertEqual(f.assess(data)['status'],'unresolved')
        b=browser(data);self.assertEqual(f.assess(data,b)['status'],'passed')
        b['artifact_sha256']='0'*64
        self.assertIn('Stale/mismatched browser artifact_sha256',f.assess(data,b)['issues'])
        b=browser(data);b['fonts_ready']=False
        self.assertEqual(f.assess(data,b)['status'],'unresolved')

    def test_long_label_stroke_marker_raster_and_clipping_intrusions(self):
        data=svg()
        for target in ('painted_bounds_mm','unclipped_bounds_mm'):
            b=browser(data);b['allocations']['A'][target]=[.1,2,30,20]
            self.assertEqual(f.assess(data,b)['status'],'unresolved')
        b=browser(data);b['issues']=['Unsupported filter']
        self.assertEqual(f.assess(data,b)['status'],'unresolved')

    def test_alignment_and_scale_evidence(self):
        s=spec();s['alignments']=[dict(id='same-top',target='data_axes',property='top',ids=['axA','axB'])]
        data=svg(s);b=browser(data);b['data_axes']={'axA':[3,3,20,10],'axB':[48,3.5,20,10]}
        self.assertIn('Alignment failed: same-top',f.assess(data,b)['issues'])
        b['data_axes']['axB'][1]=3
        self.assertEqual(f.assess(data,b)['status'],'passed')
        b['final_size_mm']=[183,30]
        self.assertEqual(f.assess(data,b)['status'],'unresolved')

    def test_assembly_opt_in_and_legacy_preservation(self):
        layout=dict(canvas=[890,300],panels=[dict(id='A',box=[0,0,445,300]),dict(id='B',box=[445,0,445,300])])
        self.assertEqual(f.assembly_layout(layout),(layout,None))
        layout['layout_frames']=dict(final_width_mm=89.,inset_mm=1.5,calibration=True)
        candidate,s=f.assembly_layout(layout)
        self.assertEqual(candidate['panels'][0]['content_box'],[15,15,415,270])
        for setting in ({'frame':{'stroke':'black'}},{'fit':'none'},{'view_box':[0,0,10,10]}):
            bad=copy.deepcopy(layout);bad['panels'][0].update(setting)
            with self.assertRaises(ValueError):f.assembly_layout(bad)

    def test_freeze_rejects_missing_stale_calibration_and_wrong_scope(self):
        data=svg()
        self.assertTrue(f.freeze_issues(data,None,89,['A','B']))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'evidence.json'
            report=f.assess(data,browser(data));path.write_text(json.dumps(report))
            record=dict(path=str(path),sha256=f.sha(path.read_bytes()))
            self.assertIn('Calibration previews cannot be frozen',f.freeze_issues(data,record,89,['A','B']))
            record['sha256']='0'*64
            self.assertTrue(f.freeze_issues(data,record,89,['A','B']))
            policy=f.load_policy()[0];policy.update(activation='active',selected_inset_mm=1.5,author_selection='synthetic test only')
            policy_path=Path(directory)/'policy.json';policy_path.write_text(json.dumps(policy))
            s=spec();s['calibration']=False
            with patch.object(f,'load_policy',return_value=(policy,f.sha(policy_path.read_bytes()))):
                data=svg(s);report=f.assess(data,browser(data));path.write_text(json.dumps(report));record['sha256']=f.sha(path.read_bytes())
                self.assertEqual(f.freeze_issues(data,record,89,['A','B']),[])
                self.assertTrue(f.freeze_issues(data,record,183,['A','B']))
                self.assertTrue(f.freeze_issues(data,record,89,['A']))
                report['helper_sha256']='stale';path.write_text(json.dumps(report));record['sha256']=f.sha(path.read_bytes())
                self.assertTrue(f.freeze_issues(data,record,89,['A','B']))

    def test_generic_assembly_repeat_and_panel_scale(self):
        from scripts.assemble_svg_figure import render
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'a.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><rect width="10" height="10"/></svg>')
            layout=dict(canvas=[890,300],panels=[dict(id='A',source='a.svg',box=[0,0,445,300]),dict(id='B',source='a.svg',box=[445,0,445,300])],
                        layout_frames=dict(final_width_mm=89.,inset_mm=1.5,calibration=True))
            path=root/'layout.json';path.write_text(json.dumps(layout))
            first=render(path,root/'candidate.svg');second=render(path,root/'candidate.svg')
            self.assertEqual(first['output_sha256'],second['output_sha256'])
            self.assertEqual(f.read_spec((root/'candidate.svg').read_bytes())['final_size_mm'],[89,30])
            render(path,root/'panel.svg',only_panel='A')
            single=f.read_spec((root/'panel.svg').read_bytes())
            self.assertEqual(single['scope'],'panel');self.assertEqual(single['final_size_mm'],[44.5,30])

    def test_matplotlib_export_retains_canvas_despite_tight_theme(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from classical_conditioning.figures.export import FigureMode, FigureProvenance, export_matplotlib_figure
        with tempfile.TemporaryDirectory() as folder:
            fig,ax=plt.subplots(figsize=(89/25.4,30/25.4));ax.plot([0,1],[0,1])
            s=spec();s['scope']='panel';s['allocations']=[dict(id='A',kind='panel',box_mm=[0,0,89.,30.],group_ids=['axes__a__main'])]
            source=Path(__file__)
            provenance=FigureProvenance(figure_id='test',analysis_recipe='test',source_file=str(source),source_symbol='test',source_hash=f.sha(source.read_bytes()),reproduction_snippet='synthetic fixture',input_artifacts=())
            try:
                with matplotlib.rc_context({'savefig.bbox':'tight','savefig.pad_inches':.9}):
                    result=export_matplotlib_figure(fig,Path(folder)/'figure',provenance,mode=FigureMode.PUBLICATION,panel_ids=['A'],allow_dirty_publication=True,formats=('svg',),layout_frames=s)
                    self.assertEqual(matplotlib.rcParams['savefig.bbox'],'tight')
                root=ET.parse(result.outputs[0]).getroot()
                self.assertEqual(root.get('width'),'89.0mm')
                self.assertAlmostEqual(float(root.get('viewBox').split()[2])*25.4/72,89,places=5)
            finally:plt.close(fig)

    def test_new_default_freeze_contract_requires_frame_evidence(self):
        from classical_conditioning.figure_freeze import check_candidate
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'test.svg').write_bytes(svg())
            (root/'candidate.json').write_text(json.dumps(dict(exports=[dict(path='test.svg',sha256=f.sha(svg()))],verification={})))
            result=check_candidate(root/'candidate.json')
            self.assertTrue(any(i['code']=='layout-frames' for i in result['issues']))

if __name__=='__main__':unittest.main()
