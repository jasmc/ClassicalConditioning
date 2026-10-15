"""Resource-only font support must reject external CSS and presentation rules."""
import base64,hashlib,unittest
from classical_conditioning.figure_freeze import _embedded_font_hashes
class EmbeddedFontsTest(unittest.TestCase):
    def test_embedded_font_is_hashed(self):
        payload=b'\x00\x01\x00\x00'+b'\x00'*12
        css='@font-face{font-family:"DejaVu Sans";font-weight:400;src:url(data:font/ttf;base64,'+base64.b64encode(payload).decode()+') format("truetype")}'
        self.assertEqual(_embedded_font_hashes(css),{hashlib.sha256(payload).hexdigest()})
        with self.assertRaises(ValueError):_embedded_font_hashes(css+'text{font-size:99px}')
    def test_rejects_external_rules_and_invalid_font(self):
        for css in ['@font-face{font-family:"DejaVu Sans";font-weight:400;src:url(https://example.com/font.ttf) format("truetype")}', 'text{font-size:99px}', '@font-face{font-family:"DejaVu Sans";font-weight:400;src:url(data:font/ttf;base64,YWJj) format("truetype")}']:
            with self.subTest(css=css),self.assertRaises(ValueError):_embedded_font_hashes(css)
