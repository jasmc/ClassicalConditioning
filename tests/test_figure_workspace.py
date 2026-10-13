"""Working-copy isolation, measured transforms and stale-writer protection."""
import base64
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from classical_conditioning.figure_workspace import (Snapshot, StaleFigureState, editable_svg,
    measured_presentation, publish_state, tag, verify_mapping_baseline,bind_loaded_state,sha,draft_axis_findings)

class FigureWorkspaceTests(unittest.TestCase):
    def test_authorized_removal_rolls_back_if_state_publication_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=root/'state.json';old=root/'old.svg'
            state.write_text('{}');old.write_bytes(b'preserve until committed')
            with patch('classical_conditioning.figure_workspace.os.replace',side_effect=OSError('publication failed')):
                with self.assertRaises(OSError):
                    publish_state(state,{'default_display':'draft'},Snapshot([state,old]),removals=[old])
            self.assertEqual(old.read_bytes(),b'preserve until committed')
            self.assertEqual(state.read_text(),'{}')

    def test_authorized_removal_commits_with_state_and_rejects_stale_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=root/'state.json';old=root/'old.svg'
            state.write_text('{}');old.write_bytes(b'original');snapshot=Snapshot([state,old])
            old.write_bytes(b'another editor')
            with self.assertRaises(StaleFigureState):publish_state(state,{},snapshot,removals=[old])
            self.assertEqual(old.read_bytes(),b'another editor')
            publish_state(state,{'default_display':'draft'},Snapshot([state,old]),removals=[old])
            self.assertFalse(old.exists());self.assertEqual(json.loads(state.read_text())['default_display'],'draft')

    def test_vectors_are_editable_and_css_resource_ids_remain_isolated(self):
        root=ET.Element(tag('svg'),{'viewBox':'0 0 40 20'})
        for pid,color,x in [('A','red',0),('B','blue',20)]:
            source=(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><style>.mark{{fill:{color}}}</style>'
                    '<defs><clipPath id="clip"><rect width="10" height="10"/></clipPath></defs>'
                    '<path id="line" class="mark" clip-path="url(#clip)" d="M0 0L10 10"/><text x="2" y="5">X</text></svg>').encode()
            g=ET.SubElement(root,tag('g'),{'id':'panel-'+pid})
            ET.SubElement(g,tag('image'),{'x':str(x),'y':'0','width':'20','height':'20','href':'data:image/svg+xml;base64,'+base64.b64encode(source).decode()})
        output,metadata=editable_svg(ET.tostring(root),kind='test',source_records={})
        parsed=ET.fromstring(output)
        self.assertFalse(list(parsed.iter(tag('image'))))
        self.assertFalse(list(parsed.iter(tag('style'))))
        paths={p.get('id'):p for p in parsed.iter(tag('path'))}
        self.assertEqual(paths['edit-A-line'].get('d'),'M0 0L10 10')
        self.assertIn('fill:red',paths['edit-A-line'].get('style'))
        self.assertIn('fill:blue',paths['edit-B-line'].get('style'))
        self.assertEqual(paths['edit-A-line'].get('clip-path'),'url(#edit-A-clip)')
        self.assertEqual(metadata['scientific_registry'],{})
        self.assertEqual(len(list(parsed.iter(tag('text')))),2)

    def test_measurement_accounts_for_nested_viewbox_and_anisotropic_stroke(self):
        data=b'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
          <svg x="20" y="10" width="40" height="40" viewBox="0 0 20 20">
            <g transform="scale(1 0.5)"><path id="line" d="M0 0L10 0" style="stroke:black;stroke-width:1"/>
              <text id="text" x="5" y="10" style="font-size:10px;font-family:DejaVu Sans">0</text></g>
          </svg></svg>'''
        report=measured_presentation(data,width_mm=100*25.4/72)
        self.assertAlmostEqual(report['texts'][0]['font_height_pt'],10)
        self.assertAlmostEqual(report['texts'][0]['font_width_scale_pt'],20)
        self.assertAlmostEqual(report['strokes'][0]['normal_width_pt'],1)
        self.assertEqual(report['strokes'][0]['width_range_pt'],[1.,2.])
        self.assertEqual(report['texts'][0]['anchor_pt'],[30.,20.])
        moved=data.replace(b'scale(1 0.5)',b'scale(1 0.6)')
        with self.assertRaises(StaleFigureState):verify_mapping_baseline(moved,report)

    def test_stale_input_prevents_output_and_state_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=root/'state.json';source=root/'source';source.write_text('old')
            snapshot=Snapshot([state,source]);source.write_text('new')
            with self.assertRaises(StaleFigureState):publish_state(state,{'revision':1},snapshot,[(root/'copy.svg',b'new')])
            self.assertFalse(state.exists());self.assertFalse((root/'copy.svg').exists())

    def test_second_chat_cannot_overwrite_first_chat_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=root/'state.json';state.write_text('{}')
            first=Snapshot([state]);second=Snapshot([state])
            publish_state(state,{'revision':1},first,[(root/'one.svg',b'one')])
            with self.assertRaises(StaleFigureState):publish_state(state,{'revision':2},second,[(root/'two.svg',b'two')])
            self.assertEqual(json.loads(state.read_text())['revision'],1)
            self.assertFalse((root/'two.svg').exists())

    def test_existing_destination_is_preserved_and_new_outputs_rolled_back(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=root/'state.json';existing=root/'existing.svg';existing.write_bytes(b'keep')
            with self.assertRaises(FileExistsError):publish_state(state,{},Snapshot([state]),[(root/'new.svg',b'new'),(existing,b'wrong')])
            self.assertEqual(existing.read_bytes(),b'keep');self.assertFalse((root/'new.svg').exists());self.assertFalse(state.exists())

    def test_state_advancing_between_read_and_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.json';old=b'{}';path.write_bytes(old)
            parsed=json.loads(old);path.write_text('{"new":true}')
            with self.assertRaises(StaleFigureState):bind_loaded_state(Snapshot([path]),parsed,old,path)

    def test_bound_artifact_changed_before_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);state=root/'state.json';artifact=root/'working.svg';artifact.write_bytes(b'old')
            parsed={'frozen_display':{'working_svg':{'path':str(artifact),'sha256':sha(b'old')}},'draft':{}}
            raw=json.dumps(parsed).encode();state.write_bytes(raw);artifact.write_bytes(b'foreign edit')
            with self.assertRaises(StaleFigureState):bind_loaded_state(Snapshot([state,artifact]),parsed,raw,state)

    def test_time_axis_spine_is_not_a_tick_and_actual_width_deviation_is_reported(self):
        svg=b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><g id="time-axis-control"><line x1="0" x2="80" y1="20" y2="20" stroke="black" stroke-width=".5"/><line x1="10" x2="10" y1="20" y2="22" stroke="black" stroke-width=".75"/></g></svg>'
        report=measured_presentation(svg,width_mm=100*25.4/72)
        self.assertEqual(len(report['ticks']),1);self.assertEqual(report['ticks'][0]['length_pt'],2.)
        self.assertEqual(draft_axis_findings(report)[0]['property'],'tick_width_pt')

    def test_transformed_marker_template_is_rejected_instead_of_mismeasured(self):
        svg=b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><defs><path id="mark" transform="scale(2)" d="M0 0L0 2"/></defs><use href="#mark" x="2" y="2"/></svg>'
        with self.assertRaises(ValueError):measured_presentation(svg)

    def test_scientific_colorbar_links_follow_namespaced_target_ids(self):
        inner=b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"><g id="heatmap"><rect width="5" height="5"/></g><g id="colorbar"><rect width="1" height="5"/></g></svg>'
        root=ET.Element(tag('svg'),{'viewBox':'0 0 10 10'});g=ET.SubElement(root,tag('g'),{'id':'panel-F'})
        ET.SubElement(g,tag('image'),{'x':'0','y':'0','width':'10','height':'10','href':'data:image/svg+xml;base64,'+base64.b64encode(inner).decode()})
        _,meta=editable_svg(ET.tostring(root),kind='test',source_records={},registries={'F':{'heatmap':{'scientific_role':'heatmap.vigor'},'colorbar':{'scientific_role':'colorbar.scale','mappable_ids':['heatmap']}}})
        self.assertEqual(meta['scientific_registry']['edit-F-colorbar']['mappable_ids'],['edit-F-heatmap'])
        self.assertEqual(meta['scientific_registry']['edit-F-colorbar']['source_mappable_ids'],['heatmap'])

if __name__=='__main__':unittest.main()
