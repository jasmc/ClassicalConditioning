"""Collection isolation and transaction protection, independent of mounted media."""
import importlib.util
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET
import unittest
import tempfile
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import main_paper_figures as paper
import layout_main_paper_figures as layout
from classical_conditioning.figure_workspace import Snapshot,StaleFigureState,publish_state,sha

SVG=b'''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 183 90"><defs><clipPath id="clip"><rect width="10" height="10"/></clipPath><path id="mark" d="M 0 0 L 1 1"/></defs><style>#plot{clip-path:url(#clip)}</style><g id="plot"><use xlink:href="#mark"/><text>Pending</text></g></svg>'''


def test_namespaces_references_and_keeps_science():
    result=paper.embedded_svg(SVG,'fig1--')
    root=ET.fromstring(result)
    assert root.find('.//{http://www.w3.org/2000/svg}path').get('d')=='M 0 0 L 1 1'
    assert 'url(#fig1--clip)' in result and '#fig1--mark' in result
    assert '#fig1--plot' in result
    assert 'Pending' in result
    assert not set(n.get('id') for n in root.iter() if n.get('id')) & {'clip','plot','mark'}


def test_order_captions_and_shadow_isolation():
    figures=[{'id':f'fig{n}','svg_bytes':SVG,'legend_text':f'# Figure {n} legend draft\n\nFigure {n} | Title.\n\n(A) Same wording < & >.'} for n in (1,2)]
    page,metadata=paper.render(figures);text=page.decode()
    assert text.index('id="fig1"')<text.index('Figure 1 | Title.')<text.index('id="fig2"')<text.index('Figure 2 | Title.')
    assert text.count('shadowrootmode="open"')==2
    assert 'legend draft' not in text
    assert '(A) Same wording &lt; &amp; &gt;.' in text
    assert 'width:183mm' in text and 'break-inside:avoid' in text
    assert [f['id'] for f in metadata['figures']]==['fig1','fig2']


def test_rejects_active_or_external_svg():
    for payload in [b'<script/>',b'<foreignObject/>',b'<image href="https://example.com/x.png"/>',b'<path onclick="x()"/>',b'<path style="fill:url(https://example.com/a.svg)"/>']:
        data=b'<svg xmlns="http://www.w3.org/2000/svg">'+payload+b'</svg>'
        with unittest.TestCase().assertRaises(ValueError):paper.embedded_svg(data,'test--')


def test_authentication_preserves_manual_edits():
    tmp_path=Path(tempfile.mkdtemp())
    p=tmp_path/'working.svg';p.write_bytes(SVG);ref=paper.reference(p)
    p.write_bytes(SVG+b'<!-- manual -->')
    with unittest.TestCase().assertRaises(StaleFigureState):paper.authenticate(ref,Snapshot([]))
    assert p.read_bytes().endswith(b'<!-- manual -->')


def test_publication_rolls_back_without_switching_pointer():
    tmp_path=Path(tempfile.mkdtemp())
    state=tmp_path/'state.json';state.write_text('{"old":true}')
    html=tmp_path/'paper.html';html.write_bytes(b'old html')
    snapshot=Snapshot([state,html]);original=paper.publish_state.__globals__['os'].replace
    def failure(source,dest):
        if Path(dest)==state:raise OSError('simulated state switch failure')
        return original(source,dest)
    with patch.object(paper.publish_state.__globals__['os'],'replace',failure):
        with unittest.TestCase().assertRaises(OSError):publish_state(state,{'new':True},snapshot,evidence_updates=[(html,b'new html')])
    assert html.read_bytes()==b'old html' and json.loads(state.read_text())=={'old':True}
    assert not Path(str(state)+'.lock').exists()


def test_layout_rejects_scientific_geometry_change():
    assert layout.preserved_content(SVG,SVG)['all_drawable_attributes_and_text_preserved']
    with unittest.TestCase().assertRaises(ValueError):layout.preserved_content(SVG,SVG.replace(b'L 1 1',b'L 2 2'))


def test_fonts_are_registered_with_per_figure_names():
    svg=SVG.replace(b'<defs>',b'<style>@font-face{font-family:"ExampleFace";src:url(data:font/ttf;base64,AA)}</style><defs>')
    page,_=paper.render([{'id':'fig1','svg_bytes':svg,'legend_text':'Figure 1 | Test'}])
    head=page.decode().split('</head>')[0]
    assert 'font-family:"fig1--ExampleFace"' in head


def test_duplicate_unreferenced_legacy_handles_are_isolated():
    svg=SVG.replace(b'<g id="plot">',b'<g id="legacy-tick"/><g id="legacy-tick"/><g id="plot">')
    result=paper.embedded_svg(svg,'fig2--')
    assert 'fig2--legacy-tick--instance-1' in result and 'fig2--legacy-tick--instance-2' in result
    with unittest.TestCase().assertRaises(ValueError):
        paper.embedded_svg(svg.replace(b'<use xlink:href="#mark"/>',b'<use xlink:href="#legacy-tick"/>'),'fig2--')


def test_native_intermediate_width_is_not_scaled_to_double_column():
    svg=SVG.replace(b'viewBox="0 0 183 90"',b'width="136mm" viewBox="0 0 136 90"')
    page,meta=paper.render([{'id':'fig1','svg_bytes':svg,'legend_text':'Figure 1 | Test'}])
    assert meta['figures'][0]['publication_size_mm']==[136,90]
    assert b'data-width-mm="136.0"' in page and b'width:136mm' in page
    assert b'width:183mm' not in page


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(fn) for name,fn in globals().items() if name.startswith("test_"))
