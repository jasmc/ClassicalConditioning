"""Presentation migration cannot change scientific payload or stretch plots."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scripts.paper.shared.stage_nature import protection, axis_style
from classical_conditioning.figures.layout_frames import embed_spec
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from classical_conditioning.figure_workspace import sha,measured_presentation
from scripts.paper.shared.publish_axis_geometry import verify_metrics
from scripts.paper.shared.register_axes import build_inventory

class NatureMigrationTests(unittest.TestCase):
    def source(self):
        return embed_spec(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 183 60"><g id="owner"><path id="data" d="M1 2L3 4" style="stroke:#112233;fill:none"/><text x="1" y="2">A</text><image href="data:image/png;base64,AAA"/></g></svg>',{'scope':'panel','final_size_mm':[183,60],'inset_mm':1.,'allocations':[{'id':'A','kind':'panel','box_mm':[0,0,183,60],'group_ids':['owner']}],'background_ids':[]})

    def candidate(self):
        return self.source().replace(b'id="owner"',b'id="owner" transform="scale(.72)"').replace(b'>A</text>',b'>a</text>')

    def test_uniform_layout_and_display_alias_preserve_science(self):
        protection(self.source(),self.candidate())
        with self.assertRaises(ValueError):protection(self.source(),self.candidate().replace(b'scale(.72)',b'scale(.72 .7)'))
        with self.assertRaises(ValueError):protection(self.source(),self.candidate().replace(b'M1 2L3 4',b'M1 2L3 5'))
        with self.assertRaises(ValueError):protection(self.source(),self.candidate().replace(b'base64,AAA',b'base64,AAB'))
        with self.assertRaises(ValueError):protection(self.source(),self.candidate().replace(b'#112233',b'#112234'))

    def test_canvas_width_change_preserves_physical_units(self):
        candidate=self.candidate().replace(b'0 0 183 60',b'0 0 136 60')
        with self.assertRaises(ValueError):protection(self.source(),candidate)

    def test_registered_paired_axis_move_cannot_stretch_science(self):
        raw=self.source().replace(b'id="data"',b'id="paired-axis"')
        candidate=raw.replace(b'id="owner"',b'id="owner" transform="scale(.72)"').replace(b'id="paired-axis"',b'id="paired-axis" transform="translate(1 0)"').replace(b'>A</text>',b'>a</text>')
        move={'element_id':'paired-axis','dx_mm':1.,'dy_mm':0.}
        protection(raw,candidate,layout_moves=[move])
        with self.assertRaises(ValueError):protection(raw,candidate)
        with self.assertRaises(ValueError):protection(raw,candidate.replace(b'translate(1 0)',b'scale(1.1)'),layout_moves=[move])

    def test_browser_css_serialization_is_not_a_palette_change(self):
        protection(self.source(),self.candidate().replace(b'stroke:#112233',b'stroke:rgb(17, 34, 51)'))

    def test_E_band_colorbar_keeps_frozen_source_and_current_display_distinct(self):
        inventory=build_inventory(self.source(),1,{},[],{'condition_order_mapping':{'mapping':[{'source_panel':'F','display_panel':'G'}]},'presentation_styles':{'axis_titles':{'roles':[]}}})
        axis=next(a for a in inventory['axes']if a['id']=='E-band-colorbar')
        self.assertEqual(axis['source_heatmap_panel'],'F')
        self.assertEqual(axis['display_heatmap_panel'],'G')
        self.assertIn('displayed G',axis['applies_to'])

    def test_registered_stat_alias_preserves_star_count_and_hypothesis(self):
        raw=self.source().replace(b'<text x="1" y="2">A</text>',b'<g id="edit-G-axes__g-stats__main"><g id="edit-G-fig2__g__stats__D-1"><text>D***</text></g></g>')
        candidate=raw.replace(b'id="owner"',b'id="owner" transform="scale(.72)"').replace(b'>D***</text>',b'>***</text>')
        alias={'xml_index_path':[0,1,0,0],'source_text':'D***','display_text':'***','role':'annotation.statistical_label'}
        protection(raw,candidate,text_aliases=[alias])
        with self.assertRaises(ValueError):protection(raw,candidate)
        with self.assertRaises(ValueError):protection(raw,candidate.replace(b'>***</text>',b'>**</text>'),text_aliases=[alias])

    def test_non_scaling_width_is_physical_under_anisotropic_transform(self):
        raw=b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 183 60"><path d="M0 0L5 5" transform="scale(3 .5)" stroke="black" stroke-width="1.333333333333" vector-effect="non-scaling-stroke"/></svg>'
        stroke=measured_presentation(raw)['strokes'][0]
        self.assertAlmostEqual(stroke['width_range_pt'][0],1)
        self.assertAlmostEqual(stroke['width_range_pt'][1],1)
        with self.assertRaises(ValueError):measured_presentation(raw.replace(b'non-scaling-stroke',b'non-scaling-size'))

    def test_tick_template_inherits_its_instance_width(self):
        raw=b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 183 60"><defs><path id="tick" d="M0 0L0 1" style="stroke:black;stroke-width:inherit"/></defs><use href="#tick" style="stroke:black;stroke-width:.088194444444" transform="scale(2 .5)"/></svg>'
        stroke=measured_presentation(raw)['strokes'][0]
        self.assertAlmostEqual(stroke['normal_width_pt'],.5)

    def test_profile_metrics_require_actual_final_font_sizes(self):
        expected=axis_style(load_presentation_profile())
        report={'artifact_sha256':sha(b'svg'),'axes':[{'id':'x','ticks':[],'tick_labels':[{'size_pt':6,'font_family':'Arial','font_weight':'400'}],'titles':[{'size_pt':7,'font_family':'Arial','font_weight':'400'}],'tick_label_gaps_mm':[3*25.4/72],'title_gap_mm':3*25.4/72}]}
        verify_metrics(report,b'svg',expected)
        report['axes'][0]['tick_labels'][0]['size_pt']=7
        with self.assertRaises(ValueError):verify_metrics(report,b'svg',expected)
        with self.assertRaises(ValueError):verify_metrics(report,b'changed',expected)

if __name__=='__main__':unittest.main()
