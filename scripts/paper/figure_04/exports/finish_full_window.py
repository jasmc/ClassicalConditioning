"""Attach invisible ownership frames and export the requested raster PDF.

Run with the bundled PDF Python. The Matplotlib PNG is embedded losslessly at
exactly 600 pixels/inch; its rounded pixel dimensions determine PDF page size.
"""
from pathlib import Path
import sys
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
# ReportLab and pypdf are pure-Python consumers here; retain the project's
# NumPy/Pillow/runtime first and use the bundled PDF libraries as a fallback.
sys.path.append('/Users/joaquim/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
import json
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader
from classical_conditioning.figures.layout_frames import embed_spec,load_policy
from classical_conditioning.figure_workspace import Snapshot,sha

def finish(directory):
    directory=Path(directory)
    files=[directory/n for n in ['figure-04.svg','figure-04.html','legend.md','figure-04-600dpi.png','presentation-evidence.json']]
    guard=Snapshot([*files,Path(__file__)])
    original=directory/'sources'/'initial-render'
    inputs=[original/p.name for p in files] if original.exists() else files
    guard.include(inputs)
    raw=inputs[0].read_bytes();root=ET.fromstring(raw);ns='{http://www.w3.org/2000/svg}'
    fig=next(n for n in root if n.get('id')=='figure_1')
    letters=[]
    for label in ['a','b']:
        group=next(n for n in fig if any(t.text==label for t in n.iter(ns+'text')))
        letters.append(group.get('id'))
    # Put keys below the plots so timing lines and observed data cannot run
    # through key text. Keep every data coordinate, font and stroke unchanged.
    for group in root.iter(ns+'g'):
        if group.get('id','').startswith('legend_'):
            group.set('transform',f'translate(0 {20*72/25.4})')
    vb=[float(v)for v in root.get('viewBox').split()];vb[3]=83*72/25.4
    root.set('viewBox',' '.join(map(str,vb)))
    policy,_=load_policy()
    spec={'scope':'figure','final_size_mm':[183.,83.],'inset_mm':policy['selected_inset_mm'],
        'allocations':[{'id':a,'kind':'panel','box_mm':[i*91.5,0,91.5,83.],
            'group_ids':[a+'-panel-a',letters[i]]}for i,a in enumerate(['delay','3strace'])],
        'background_ids':['patch_1'],
        'scientific_clips':[{'id':n.get('id'),'evidence':'Data rectangle clipping of the full [-20,20) profile; common y limits cover observed means and complete bands.'}for n in root.iter(ns+'clipPath')]}
    framed=embed_spec(ET.tostring(root),spec)
    page=inputs[1].read_text();old=page[page.index('<svg'):page.index('</svg>')+6]
    svg=framed.decode();svg=svg[svg.index('<svg'):]
    page=page.replace(old,svg).replace('dotted red:','dotted violet:')
    legend=inputs[2].read_text().replace('dotted red:','dotted violet:')
    evidence=json.loads(inputs[4].read_text())
    evidence['layout_frames']=spec
    for a in evidence['axes']:evidence['axes'][a]['zero_figure_y']*=73/83
    evidence['frame_coverage']='Owned physical allocations with shared outside boundary and active 1 mm insets; exact render checked separately.'
    evidence['finalizer_source']= {'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__).read_bytes())}
    guard.check()
    if not original.exists():
        original.mkdir(parents=True,exist_ok=False)
        for p in files:(original/p.name).write_bytes(p.read_bytes())
    files[0].write_bytes(framed);files[1].write_text(page);files[2].write_text(legend)
    files[4].write_text(json.dumps(evidence,indent=2)+'\n')
    png=directory/'figure-04-600dpi.png'
    node=Path('/Users/joaquim/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node')
    sharp=Path('/Users/joaquim/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp')
    # Use explicit pixel dimensions: this bundled SVG loader applies density
    # twice to mm dimensions. Pixel-sized input avoids that loader behaviour.
    js="const fs=require('fs'),sharp=require(process.argv[1]);let svg=fs.readFileSync(process.argv[2],'utf8').replace('width=\"183.0mm\"','width=\"4323px\"').replace('height=\"83.0mm\"','height=\"1961px\"');sharp(Buffer.from(svg),{density:72}).flatten({background:'#fff'}).withMetadata({density:600}).png().toFile(process.argv[3]).catch(e=>{console.error(e);process.exit(1)});"
    subprocess.run([str(node),'-e',js,str(sharp),str(files[0]),str(png)],check=True)
    image=Image.open(png)
    size=(image.width/600*72,image.height/600*72)
    pdf=directory/'figure-04.pdf'
    if pdf.exists():raise FileExistsError(pdf)
    c=canvas.Canvas(str(pdf),pagesize=size,pageCompression=1)
    c.setTitle('Figure 4 — full-window Delay and 3sTrace profiles')
    c.drawImage(ImageReader(str(png)),0,0,width=size[0],height=size[1]);c.showPage();c.save()
    reader=PdfReader(pdf);assert len(reader.pages)==1
    objects=reader.pages[0]['/Resources']['/XObject'].get_object()
    images=[v.get_object()for v in objects.values()if v.get_object().get('/Subtype')=='/Image']
    assert len(images)==1
    assert images[0]['/Width']==image.width and images[0]['/Height']==image.height
    assert images[0].get_data()==image.convert('RGB').tobytes()
    evidence['raster_pdf']={'png_pixels':[image.width,image.height],'page_points':list(size),
        'effective_dpi':[600.,600.],'lossless_rgb_pixel_equivalence':True,'embedded_images':1,
        'page_size_mm':[v/72*25.4 for v in size]}
    files[4].write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence['raster_pdf']))

if __name__=='__main__': finish(sys.argv[1])
