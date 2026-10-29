"""Presentation guards reject changes to results, paths and authentic rasters."""
import unittest
from scripts.paper.figure_02.assembly.three_columns import protection

SOURCE = b'''<svg xmlns="http://www.w3.org/2000/svg" width="156mm" viewBox="0 0 156 150">
<g id="panel-G"><path id="data" d="M 1 2 L 3 4"/><text id="result">***</text>
<image href="data:image/png;base64,original"/></g><metadata id="layout-frames">{}</metadata></svg>'''


class ProtectionTests(unittest.TestCase):
    def test_explicit_panel_translation_preserves_science(self):
        candidate=SOURCE.replace(b'id="panel-G"',b'id="panel-G" transform="translate(2 0)"')
        report=protection(SOURCE,candidate,[{'path':[0],'attributes':{'transform':{}}}])
        self.assertTrue(report['all_data_paths_rasters_text_and_scientific_clips_preserved'])

    def test_statistical_result_text_cannot_change(self):
        with self.assertRaisesRegex(ValueError,'Protected'):
            protection(SOURCE,SOURCE.replace(b'***',b'**'),[])

    def test_data_path_cannot_change(self):
        with self.assertRaisesRegex(ValueError,'Protected'):
            protection(SOURCE,SOURCE.replace(b'M 1 2 L 3 4',b'M 1 2 L 3 5'),[])

    def test_embedded_heatmap_cannot_change(self):
        with self.assertRaisesRegex(ValueError,'Protected'):
            protection(SOURCE,SOURCE.replace(b'base64,original',b'base64,replacement'),[])


if __name__=='__main__':unittest.main()
