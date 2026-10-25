"""Text-only review edits must preserve graphical counts and original scale."""
import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('legend_review',ROOT/'scripts/paper/shared/build_legend_review.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
class LegendReviewTests(unittest.TestCase):
    def test_scoped_editing_page_preserves_native_svg_and_only_associated_legend(self):
        svg=b'<svg xmlns="http://www.w3.org/2000/svg"><g id="panel-D"><text>Data label</text></g></svg>'
        page=review.svg_editing_page(svg,scope_label='Figure 1 panel D',legend='# Legend\n\n(D) Tail angle.')
        self.assertIn(svg.decode(),page);self.assertIn('(D) Tail angle.',page)
        self.assertNotIn('<button',page);self.assertNotIn('figure2',page)
        self.assertNotIn('<details',page);self.assertNotIn('<img',page)

    def test_working_review_removes_only_scoped_comparison_preserving_payload(self):
        selected='<div class="version candidate"><img class="figure" src="data:image/svg+xml;base64,EXACT"></div>'
        other='<section id="figure2"><h2>Figure 2</h2><button>Original selected assembly</button></section>'
        page='<header>Candidate shown first. Switch to the authenticated original to compare.</header><section id="figure1"><h2>Figure 1</h2><button>Decluttered draft</button>'+selected+'<div class="version original" hidden><img src="OLD"></div><div class="legend">Legend unchanged</div></section>'+other
        output=review.working_review_page(page,version_label='v10')
        self.assertIn(selected,output);self.assertIn(other,output)
        self.assertIn('<div class="legend">Legend unchanged</div>',output)
        self.assertNotIn('OLD',output);self.assertNotIn('Decluttered draft',output)
        self.assertNotIn('Switch to the authenticated original',output)

    def test_text_cleanup_preserves_duplicate_graphics_and_units(self):
        data=b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><path d="M0 0L5 5"/><path d="M0 0L5 5"/><text>Distal cumulative angle; -15 to 0 s median centred</text><text>Tail angle (\xc2\xb0) \xc2\xb7 Delay fish 20221115_07</text></svg>'
        output,edits=review.transform_panel(1,'D',data)
        root=ET.fromstring(output)
        self.assertEqual(len(list(root.iter(review.tag('path')))),2)
        self.assertEqual([review.label(e) for e in root.iter(review.tag('text'))],['Tail angle (°)'])
        self.assertEqual(len(edits),2)
    def test_unrecognized_source_text_stops_cleanup(self):
        with self.assertRaises(AssertionError):
            review.transform_panel(1,'D',b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text>Different recipe</text></svg>')
    def test_cropping_preserves_meet_scale_and_letterboxing(self):
        old=b'<svg viewBox="0 0 100 100"/>';new=b'<svg viewBox="0 20 100 50"/>'
        extent=review.image_extent(dict(x='0',y='0',width='200',height='100'),old,new)
        self.assertEqual(extent,dict(x=50,y=20,width=100,height=50,scale=1))
    def test_grid_preserves_exact_event_metadata_and_order(self):
        root=ET.Element(review.tag('svg'),{'viewBox':'0 0 1110 355'})
        ET.SubElement(root,review.tag('g'),{'id':'heading'})
        ET.SubElement(root,review.tag('g'),{'id':'shared-time-axis'})
        events=[]
        conditions=[('control',82,'Control','#333333'),('delay',151,'Delay','#0000ff'),
                    ('trace-3s',220,'Trace (3 s)','#ff0000'),('trace-10s',289,'Trace (10 s)','#008000')]
        for name,cy,title,colour in conditions:
            group=ET.SubElement(root,review.tag('g'),{'id':'row-'+name})
            ET.SubElement(group,review.tag('rect'),{'x':'24','fill':colour})
            ET.SubElement(group,review.tag('text')).text=title
            event=ET.SubElement(group,review.tag('circle'),{'id':'event-'+name,'cx':'542.5','cy':str(cy),'r':'7.2','data-time-s':'9'})
            events.append(ET.tostring(event))
        output=review.grid_timing(root)
        cells=[g for g in output if g.get('id','').startswith('cell-')]
        self.assertEqual([g.get('id') for g in cells],['cell-control','cell-delay','cell-trace-3s','cell-trace-10s'])
        self.assertEqual([ET.tostring(next(g.iter(review.tag('circle')))) for g in cells],events)
        self.assertEqual(cells[0].get('transform'),'translate(-180 18)')
        self.assertEqual(cells[1].get('transform'),'translate(530 -51)')
        # Author-selected display names (13 October); source/acquisition labels stay historical.
        display_titles=['Control','Delay','3sTrace','10sTrace']
        for cell,(_,_,_,colour),title in zip(cells,conditions,display_titles):
            label=next(node for node in cell.iter(review.tag('text')) if node.text==title)
            self.assertEqual(label.get('fill'),colour)
            self.assertFalse(any(node.get('x')=='24' for node in cell.iter(review.tag('rect'))))
    def test_protocol_row_moves_preserve_symbol_and_count(self):
        root=ET.Element(review.tag('svg'))
        for name in ['priming','pretraining','testing']:
            phase=ET.SubElement(root,review.tag('g'),{'id':'phase-'+name})
            ET.SubElement(phase,review.tag('rect'),{'height':'209'})
            ET.SubElement(phase,review.tag('circle'),{'cx':'29','cy':'260','r':'6.5'})
            ET.SubElement(phase,review.tag('text'),{'y':'266'}).text='3 separate short US'
        review.align_stimulus_rows(root)
        for phase in root:
            wrapper=phase.find(review.tag('g'))
            # The compact protocol layout predates this test fixture's old +34/250 values.
            self.assertEqual(wrapper.get('transform'),'translate(0 -26)')
            self.assertEqual(wrapper[0].get('cy'),'260')
            self.assertEqual(wrapper[1].text,'3 separate short US')
            self.assertEqual(phase[0].get('height'),'190')
            self.assertEqual(float(wrapper[1].get('y'))-float(wrapper[0].get('cy')),6)
            self.assertEqual(len(list(phase.iter(review.tag('circle')))),1)
            self.assertEqual(len(list(phase.iter(review.tag('text')))),1)
    def test_typography_compensates_nested_display_scale(self):
        import base64
        inner=b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><g id="xtick_1"><text style="font-weight:700;font-size:8px" x="10" y="20">0</text></g><text x="50" y="60">Axis</text></svg>'
        root=ET.Element(review.tag('svg'))
        group=ET.SubElement(root,review.tag('g'),{'id':'panel-X'})
        ET.SubElement(group,review.tag('text')).text='X'
        image=ET.SubElement(group,review.tag('image'),{'x':'0','y':'0','width':'200','height':'200','href':'data:image/svg+xml;base64,'+base64.b64encode(inner).decode()})
        output,report=review.harmonize_typography(review.svg_bytes(root),[{'panel':'X'}],1)
        image=next(ET.fromstring(output).iter(review.tag('image')))
        panel=ET.fromstring(base64.b64decode(image.get('href').split(',')[1]))
        texts=list(panel.iter(review.tag('text')))
        factor=183/25.4*72/1800*2
        self.assertAlmostEqual(float(texts[0].get('font-size'))*factor,7,places=5)
        self.assertAlmostEqual(float(texts[1].get('font-size'))*factor,8,places=5)
        self.assertEqual(texts[0].get('font-weight'),'normal')
        self.assertEqual(report[0]['final_size_pt'],10)
    def test_stroke_normalization_preserves_data_path_and_dash_pattern(self):
        root=ET.fromstring('<svg xmlns="http://www.w3.org/2000/svg"><g id="line2d_9"><path d="M1 2L3 4" style="stroke:black;stroke-width:0.6"/></g><g id="line2d_11"><path d="M8 1L8 10" style="stroke:green;stroke-dasharray:2,3"/></g></svg>')
        report=review.harmonize_strokes(root,1,'D',.5)
        paths=list(root.iter(review.tag('path')))
        self.assertEqual(paths[0].get('d'),'M1 2L3 4')
        self.assertIn('stroke-width:1.200000',paths[0].get('style'))
        self.assertIn('stroke-dasharray:2,3',paths[1].get('style'))
        self.assertEqual(report[1]['final_stroke_pt'],.7)
if __name__=='__main__':unittest.main()
