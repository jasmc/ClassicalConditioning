"""Registration must preserve semantic distinctions and honest ID coverage."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scripts.paper.shared.register_axes import axis_definitions, build_inventory


class AxisRegistrationTests(unittest.TestCase):
    def test_different_scientific_outcomes_are_not_shared_axes(self):
        first={a['id']:a for a in axis_definitions(1)}
        second={a['id']:a for a in axis_definitions(2)}
        self.assertEqual(first['E-9-y']['units'],'baseline units')
        self.assertIn('not E black traces',first['E-band-colorbar']['applies_to'])
        self.assertEqual(second['H-y']['units'],'dimensionless ratio')
        self.assertEqual(second['G-y']['units'],'natural-log vigor difference')
        self.assertIn('Pending',second['DE-shared-y']['not_members'])
        controls=[a for a in second.values() if a.get('control_identity')]
        self.assertTrue(controls)
        self.assertTrue(all('roster hashes' in a['control_identity'] for a in controls))

    def test_anonymous_and_duplicate_elements_keep_distinct_bound_addresses(self):
        svg=b'''<svg xmlns="http://www.w3.org/2000/svg"><g id="axis"><g id="xtick_1"><text>0</text><path d="M0 0L0 1"/></g><g id="xtick_1"><text>1</text><path d="M1 0L1 1"/></g><defs><path id="resource"/></defs></g></svg>'''
        definition={'id':'test-x','group_ids':['axis'],'title_ids':[]}
        registry={'presentation_styles':{'axis_titles':{'roles':[]}}}
        with patch('register_figure_axes.axis_definitions',return_value=[definition]):
            result=build_inventory(svg,1,{},[],registry)
        axis=result['axes'][0]
        self.assertEqual(len(axis['tick_labels']),2)
        self.assertNotEqual(axis['tick_labels'][0]['xml_index_path'],axis['tick_labels'][1]['xml_index_path'])
        self.assertFalse(any(a['unique_id'] for a in axis['tick_labels']))
        self.assertEqual(len(axis['axis_marks']),2)
        self.assertFalse(result['freeze_requirements']['whole_figure_ready'])
        self.assertEqual(result['source_svg_sha256'],__import__('hashlib').sha256(svg).hexdigest())

    def test_missing_renderer_artist_blocks_registration(self):
        registry={'presentation_styles':{'axis_titles':{'roles':[]}}}
        definition={'id':'test-y','group_ids':['missing-axis'],'title_ids':[]}
        with patch('register_figure_axes.axis_definitions',return_value=[definition]):
            result=build_inventory(b'<svg/>',2,{},[],registry)
        self.assertEqual(result['axes'][0]['status'],'unresolved')
        self.assertEqual(result['unresolved'][0]['elements'],['missing-axis'])


if __name__=='__main__':unittest.main()
