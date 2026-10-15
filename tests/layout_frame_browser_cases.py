"""Deterministic renderer regression cases, used by a separate browser review."""
import base64
import io
from PIL import Image
from classical_conditioning.figures.layout_frames import embed_spec


def cases():
    image=Image.new('RGBA',(2,2),(0,120,200,255));buffer=io.BytesIO();image.save(buffer,format='PNG')
    uri='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
    examples={
        'PASS nested SVG, text, stroke, marker and raster': f'''<defs><marker id="arrow" markerWidth="3" markerHeight="3" refX="2" refY="1.5" orient="auto"><path d="M0 0 L3 1.5 L0 3Z"/></marker></defs><g id="A"><svg x="3" y="3" width="20" height="12" viewBox="0 0 20 12"><g transform="translate(1 1)"><text x="1" y="3" font-size="2">Time (s)</text><path d="M2 6 L12 6" stroke="black" fill="none" stroke-width=".3" marker-end="url(#arrow)"/><image x="13" y="2" width="3" height="3" href="{uri}"/></g></svg></g>''',
        'FAIL long label': '<g id="A"><text x="2" y="10" font-size="5">Label extends beyond this panel</text></g>',
        'FAIL thick stroke': '<g id="A"><path d="M2 3 L20 3" stroke="black" stroke-width="5"/></g>',
        'FAIL marker in margin': '<defs><marker id="m" markerWidth="20" markerHeight="8" refX="0" refY="4"><path d="M0 0 L20 4 L0 8Z"/></marker></defs><g id="A"><path d="M3 10 L27 10" stroke="black" marker-end="url(#m)"/></g>',
        'FAIL raster in margin': f'<g id="A"><image x="0" y="4" width="5" height="5" href="{uri}"/></g>',
        'FAIL clipped overflow': '<defs><clipPath id="c"><rect x="3" y="3" width="24" height="14"/></clipPath></defs><g id="A" clip-path="url(#c)"><rect x="0" y="4" width="22" height="8"/></g>',
        'FAIL CSS clipped overflow': '<style>.crop{clip-path:url(#c)}</style><defs><clipPath id="c"><rect x="3" y="3" width="24" height="14"/></clipPath></defs><g id="A" class="crop"><rect x="0" y="4" width="22" height="8"/></g>',
        'FAIL nested viewport crop': '<g id="A"><svg x="3" y="3" width="24" height="14" viewBox="0 0 24 14" overflow="hidden"><rect x="-4" y="2" width="20" height="8"/></svg></g>',
        'FAIL unowned visible mark': '<g id="A"><rect x="4" y="4" width="20" height="8"/></g><circle cx="5" cy="5" r="1"/>',
    }
    spec=dict(scope='panel',calibration=True,inset_mm=1.5,final_size_mm=[30.,20.],
              allocations=[dict(id='A',kind='panel',box_mm=[0,0,30.,20.],group_ids=['A'])],background_ids=[],alignments=[])
    return [dict(name=name,svg_base64=base64.b64encode(embed_spec(('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 20">'+body+'</svg>').encode(),spec)).decode()) for name,body in examples.items()]
