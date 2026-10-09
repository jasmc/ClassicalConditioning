"""Show recovered Figure 2 drafts alongside the unchanged historical reviews."""
from __future__ import annotations
import html
import json
from pathlib import Path
import shutil
from urllib.parse import quote

from assemble_svg_figure import digest

BASE = Path('J:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly')
OLD = BASE.parent


def main():
    recoveries = sorted((BASE/'sources').glob('*/recovery.json'))
    if not recoveries:
        raise FileNotFoundError('No completed panel recovery yet')
    recovery_path = recoveries[-1]
    source = recovery_path.parent
    manifest = json.loads(recovery_path.read_text())
    gallery = source/'gallery'
    gallery.mkdir(exist_ok=True)
    entries=[]

    def add(path: Path, title: str, note: str, *, svg: Path|None=None, copy: bool=False):
        original=path
        if copy:
            path=gallery/(str(len(entries))+'-'+path.name)
            shutil.copy2(original,path)
            sidecar=original.with_suffix('.figure.json')
            if sidecar.is_file():shutil.copy2(sidecar,path.with_suffix('.figure.json'))
        entries.append({'title':title,'note':note,'preview':str(path),'original':str(original),
                        'sha256':digest(path),'svg':str(svg) if svg else None})

    for letter, item in sorted(manifest['selected'].items()):
        svg=Path(item['svg']);ident=item['provenance']
        note='Main provisional draft; legacy distal metric, baseline [-15,0) s; no inference marks. '
        note+=('Equal-fish signed bout-log heatmap; upstream audit remains open.' if letter in 'AB' else
               'Blocks 10-14 / 65-69 / 90-94.' if letter in 'DE' else 'Condition median and fish IQR.')
        add(svg.with_suffix('.png'),f'Current {letter}',note,svg=svg)
    for svg in sorted(source.glob('Fig2_PanelE_historical5-9*.svg')):
        add(svg.with_suffix('.png'),svg.stem.replace('Fig2_Panel',''),
            'Recovered display alternative: same 59-fish ratio inputs and [-15,0) baseline; historical Pre-Train 5-9, excluded from main E. No test annotations.',svg=svg)
    for svg in sorted(source.glob('Fig2_PanelH_*bootstrap.svg')):
        add(svg.with_suffix('.png'),svg.stem.replace('Fig2_Panel',''),
            'Recovered alternative: 59-fish cohort, [-15,0) baseline, condition median and 95% bootstrap CI; 100 resamples, seed 10. CI and fish IQR have different meanings.',svg=svg)
    history=OLD/'trace-plot-versions/20261006'
    for name, description in [
        ('figure-2E_v1-legacy-boxplot.png','E v1: legacy grouped boxes'),
        ('figure-2E_v2-descriptive-iqr.png','E v2: points and median/IQR'),
        ('figure-2E_v3-paired-legacy-tests.png','E v3: paired fish with exploratory test annotations'),
        ('figure-2H_v1-legacy-bootstrap.png','H v1: bootstrap bands'),
        ('figure-2H_v2-descriptive-iqr.png','H v2: fish-IQR bands'),
        ('figure-2H_v3-separate-conditions.png','H v3: split condition curves')]:
        add(history/name,'Original '+description,
            'Unchanged 2026-10-06 review from “Data processing”; 59 fish, legacy metric, response [0,13) s. E uses Pre-Train 5-9. Inference annotations in original v3 E are historical and not selected for the main figure.',copy=True)
    add(OLD/'baseline-window-review/figure2-pre15/figure-2A_delay-control_tail_length_weighted_angular_l1.png',
        'Original A: baseline comparison',
        'From “Compare panel layouts in Figures 1–2”: [-15,0) signed heatmap, angular L1 metric. This sensitivity version does not substitute for the paper legacy distal metric.',copy=True)
    add(OLD/'trace-legacy-preview/20260925-window13/figures/figure2-3strace-window13-legacy-53fish/signed-heatmap/figure-2B-3strace_signed_legacy_distal_angular_speed.png',
        'Original B: 53-fish signed heatmap',
        'Historical legacy-metric version: 34 trace and 19 controls, [-20,0) baseline. Kept unchanged; current B is rebuilt for all 59 fish with [-15,0).',copy=True)
    add(BASE/'reference-20261008/uploaded-Asset7.png','Uploaded complete A-I reference',
        'Layout reference supplied by the user. Historical cohort, ratio and scaling definitions are unauthenticated. Embedded stars/LME rows are not transferred. Current 10sTrace panels remain pending.',svg=BASE/'reference-20261008/uploaded-Asset7.svg')
    cards=[]
    for e in entries:
        preview=quote(Path(e['preview']).relative_to(BASE).as_posix(),safe='/')
        link=quote(Path(e['svg']).relative_to(BASE).as_posix(),safe='/') if e['svg'] else preview
        cards.append(f'<article><h2>{html.escape(e["title"])}</h2><p>{html.escape(e["note"])}</p><a href="{link}"><img src="{preview}" alt="{html.escape(e["title"])}"></a></article>')
    page='''<!doctype html><meta charset="utf-8"><title>Figure 2 recovered versions</title>
<style>body{font-family:"DejaVu Sans",sans-serif;margin:28px;color:#222;background:#f4f5f7}main{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px}article{padding:20px;background:white;border:1px solid #ddd;border-radius:8px}img{width:100%;height:auto}p{line-height:1.5}h2{font-size:20px}@media(max-width:850px){main{grid-template-columns:1fr}}</style>
<h1>Figure 2: recovered drafts and historical versions</h1>
<p>Current sources remain provisional. The baseline is [-15,0) s. No significance marks are in the current main figure. Historical differences are explicit below.</p>
<p>Sources: “Data processing” and “Compare panel layouts in Figures 1–2”. Click each preview to open its editable SVG where available. The uploaded reference is preserved unchanged.</p><main>'''+''.join(cards)+'</main>'
    (BASE/'available-versions.html').write_text(page,encoding='utf-8')
    (source/'version-catalog.json').write_text(json.dumps({'recovery':str(recovery_path),'entries':entries},indent=2)+'\n',encoding='utf-8')
    print(BASE/'available-versions.html')


if __name__=='__main__':main()
