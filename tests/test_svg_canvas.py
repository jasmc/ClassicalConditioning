"""Canvas repair must never strip data backgrounds or alter scientific payloads."""
import unittest

from classical_conditioning.figures.svg_canvas import transparent_canvases


def artwork():
    return b'''<svg xmlns="http://www.w3.org/2000/svg" width="156mm" height="30mm" viewBox="0 0 30 10">
<!-- Keep this original serialization and numeric spelling. -->
<svg viewBox="0 0 10 10" x="5.000" width="10" height="10">
<g id="canvas"><path d="M 0 10 L 10 10 L 10 0 L 0 0 z " style="fill: #ffffff" /></g>
<g id="data"><path d="M 1 9 L 9 9 L 9 1 L 1 1 z" style="fill: #ffffff" /></g>
<text x="9">LT</text><path d="M 8 7 L 9 2" style="stroke: red" />
</svg></svg>'''


class CanvasTests(unittest.TestCase):
    def test_only_selected_inline_fill_changes(self):
        raw = artwork()
        result, evidence = transparent_canvases(raw, ('canvas',))
        self.assertEqual(result, raw.replace(b'fill: #ffffff', b'fill: none', 1))
        self.assertTrue(evidence['all_other_bytes_preserved'])
        self.assertIn(b'id="data"><path d="M 1 9 L 9 9 L 9 1 L 1 1 z" style="fill: #ffffff"', result)

    def test_data_background_is_ineligible(self):
        with self.assertRaisesRegex(ValueError, 'full viewport'):
            transparent_canvases(artwork(), ('data',))

    def test_duplicate_missing_and_empty_selection_fail(self):
        for raw, ids in ((artwork(), ()), (artwork(), ('missing',)),
                         (artwork(), ('canvas', 'canvas')),
                         (artwork().replace(b'id="data"', b'id="canvas"'), ('canvas',))):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                transparent_canvases(raw, ids)

    def test_transformed_or_extra_artwork_is_ineligible(self):
        for old, new in ((b'id="canvas"', b'id="canvas" transform="translate(1)"'),
                         (b'id="canvas">', b'id="canvas"><text>keep</text>'),
                         (b'style="fill: #ffffff"', b'style="fill: #ffffff; opacity: .5"'),
                         (b'style="fill: #ffffff"', b'style="fill: red"')):
            with self.subTest(new=new), self.assertRaises(ValueError):
                transparent_canvases(artwork().replace(old, new, 1), ('canvas',))

    def test_namespaced_lexical_xml_and_single_quotes_are_preserved(self):
        raw = artwork().replace(b'<svg ', b'<s:svg ', 1).replace(b'xmlns=', b'xmlns:s=', 1)
        raw = raw.replace(b'<svg ', b'<s:svg ').replace(b'</svg>', b'</s:svg>')
        for name in (b'g', b'path', b'text'):
            raw = raw.replace(b'<' + name, b'<s:' + name).replace(b'</' + name, b'</s:' + name)
        raw = raw.replace(b'"', b"'")
        result, _ = transparent_canvases(raw, ('canvas',))
        self.assertEqual(result, raw.replace(b'fill: #ffffff', b'fill: none', 1))

    def test_declarations_and_non_rectangular_path_fail(self):
        for raw in (b'<!DOCTYPE svg>' + artwork(), artwork().replace(b'M 0 10 L 10 10', b'M 0 10 Q 10 10')):
            with self.assertRaises(ValueError):
                transparent_canvases(raw, ('canvas',))


if __name__ == '__main__':
    unittest.main()
