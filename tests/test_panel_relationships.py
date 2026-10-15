"""Spacing and role-specific formatting regressions without external artwork."""
import sys
from pathlib import Path
from xml.etree import ElementTree as ET
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from correct_panel_relationships import equal_gaps, style_titles, verify_correction
from classical_conditioning.figures.layout_frames import embed_spec
from classical_conditioning.figure_workspace import measured_presentation


def fixture():
    ids=['edit-B-column-time-label-left','edit-B-column-time-label-right','edit-D-time-label','edit-D-value-label','edit-E-time-label','edit-E-value-label','edit-E-H-fig1__h__colorbar__label-y','edit-FGH-fig1__h__colorbar__label-y',*[f'edit-FGH-fig1__{p}__heatmap__label-x' for p in 'fgh']]
    groups=''.join(f'<g id="{identity}"><text font-size="12" font-weight="bold" x="20" y="20">Title</text></g>' for identity in ids)
    data=f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 183 60"><g id="owner" transform="scale(.5)">{groups}<path id="science" d="M 1 2 L 3 4"/></g></svg>'.encode()
    return embed_spec(data,{'scope':'panel','final_size_mm':[183,60],'inset_mm':1.,'allocations':[{'id':'A','kind':'panel','box_mm':[0,0,183,60],'group_ids':['owner']}],'background_ids':[]})


class PanelRelationshipTests(unittest.TestCase):
    def test_equal_plot_gaps_preserve_endpoints_and_sizes(self):
        boxes=[[14,0,40,58],[74,0,40,58],[124,0,40,58]]
        self.assertEqual(equal_gaps(boxes),[0,-5,0])
        self.assertEqual(boxes[1][0],74)

    def test_font_role_is_measured_after_nested_scale(self):
        raw=fixture();root=ET.fromstring(raw);roles=style_titles(root,1);candidate=ET.tostring(root)
        report=measured_presentation(candidate)
        for item in report['texts']:
            self.assertAlmostEqual(item['font_height_pt'],8.,places=8)
            self.assertEqual(item['weight'],'normal')
        self.assertEqual(verify_correction(raw,candidate,{'axis_titles':roles})['axis_titles'],11)

    def test_style_authorization_does_not_allow_data_or_label_changes(self):
        raw=fixture();root=ET.fromstring(raw);roles=style_titles(root,1)
        candidate=ET.tostring(root)
        with self.assertRaises(ValueError):verify_correction(raw,candidate.replace(b'M 1 2 L 3 4',b'M 1 2 L 3 5'),{'axis_titles':roles})
        with self.assertRaises(ValueError):verify_correction(raw,candidate.replace(b'Title',b'New value',1),{'axis_titles':roles})

if __name__=='__main__':unittest.main()
