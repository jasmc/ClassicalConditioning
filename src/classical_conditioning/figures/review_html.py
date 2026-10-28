"""Minimal scoped HTML review and historical font compatibility helpers.

The DejaVu font resolver reproduces existing artwork. New paper styling resolves
roles through the selected Nature profile, not this compatibility helper.
"""
from pathlib import Path
import html
import re
import xml.etree.ElementTree as ET
from classical_conditioning.figure_workspace import tag

def legend_html(text):
    return ''.join('<p>'+html.escape(p).replace('\n','<br>')+'</p>' for p in text.strip().split('\n\n') if not p.startswith('#'))

def figure_font_paths():
    from matplotlib.font_manager import FontProperties,findfont
    return [Path(findfont(FontProperties(family='DejaVu Sans',weight=weight),fallback_to_default=False)) for weight in ('normal','bold')]

def svg_editing_page(svg, *, scope_label='Figure 1', legend=None):
    """Display exactly the scoped native SVG and its associated legend."""
    if ET.fromstring(svg).tag!=tag('svg'):raise ValueError('Expected a scoped SVG')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page+='<title>'+html.escape(scope_label)+'</title><style>body{margin:0;background:white;color:#20272c;font:16px/1.55 system-ui}main{max-width:1100px;margin:auto;padding:24px}main>svg{display:block;width:100%;height:auto}.legend{border-top:1px solid #ddd;margin-top:24px;padding-top:16px;max-width:85ch}@media print{main{padding:0}main>svg{width:183mm;max-width:100%}}</style>'
    page+='<main aria-label="'+html.escape(scope_label,quote=True)+'">'+svg.decode('utf-8')
    if legend is not None:page+='<div class="legend" aria-label="Associated legend">'+legend_html(legend)+'</div>'
    return page+'</main></html>'

def working_review_page(page, figure=1, version_label='Current working version'):
    """Remove a scoped historical comparison without changing its working SVG."""
    pattern=rf'(<section id="figure{figure}">)(.*?)(</section>)'
    match=re.search(pattern,page,re.S)
    if match is None:raise ValueError('Working figure section missing')
    body=match[2]
    candidate=re.search(r'<div class="version candidate">.*?</div>',body,re.S)
    original=re.search(r'<div class="version original" hidden>.*?</div>',body,re.S)
    if candidate is None or original is None:raise ValueError('Expected one working and one original figure')
    before=candidate[0]
    body=body.replace(original[0],'',1)
    body=re.sub(r'<button\b[^>]*>.*?</button>','',body,flags=re.S)
    body=body.replace(f'<h2>Figure {figure}</h2>',f'<h2>Figure {figure} · {html.escape(version_label)}</h2>',1)
    if before not in body:raise ValueError('Working figure changed during display cleanup')
    page=page[:match.start()]+match[1]+body+match[3]+page[match.end():]
    return page.replace('Candidate shown first. Switch to the authenticated original to compare.',
                        f'Figure {figure} shows only the selected working version.',1)
