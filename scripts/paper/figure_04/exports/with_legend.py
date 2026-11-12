"""Export selected Figure 4 artwork with its caption; never recompute science.

Prepare into a fresh ticketed SSD version, inspect the saved PDF, then publish
with --visual-review. Earlier artwork, PDF and presentation locks stay intact.
"""
from pathlib import Path
import argparse
import copy
import html
import json
import subprocess
import sys
import tempfile

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'pyproject.toml').is_file())
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
sys.path.append('/Users/joaquim/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/lib/python3.12/site-packages')
from PIL import Image
from pypdf import PdfReader
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph
from classical_conditioning.figure_workspace import Snapshot, file_hash, publish_state, sha

TITLE = 'Figure 4 | Pooled catch-trial conditional-vigor profiles.'
PARAGRAPHS = [
    '(a) Delay; (b) 3sTrace. Catch trials 25, 39, 53, 59 and 65 are pooled within each fish by taking the median at each 0.5 s bin. Dots show equal-fish group means of these fish-level medians. The signed conditional bout ln-vigor estimator uses each trial\'s [-20,0) s baseline and requires at least 90% tracking coverage. Bins without bouts have missing conditional intensity, rather than zero activity.',
    'Solid colored curves are group-specific cubic spline Gaussian mixed-model fits (six spline degrees of freedom per group), fitted to actual data throughout [-20,+20) s. Shading shows pointwise 95% whole-fish bootstrap percentile ranges (399 requested refits); these are not simultaneous confidence bands. Cyan: unsplit controls. Delay: magenta learner-flagged, orange nonlearner-flagged; 3sTrace: orange learner-flagged, gray nonlearner-flagged. Group sizes appear in each line key.',
    'Both panels use identical axes and aligned zero. Solid green: CS onset; dashed green: CS offset at 10 s; dotted violet: expected paired-training US at 9 s (Delay) or 13 s (3sTrace). Catch trials omit the paired US.',
    'Saved learner flags are retrospective, unvalidated and overlap the evaluation trials; unclassified experimental fish are excluded. These exploratory fits do not validate learner status or conditioned-response peak/onset timing. Informative bout availability, temporal residual dependence and unresolved model/uncertainty calibration remain limitations. Conditional fitted intensity in bins where a fish has no bout does not reconstruct its movement.'
]


def record(path):
    return {'path': str(path), 'sha256': file_hash(path)}


def save_new(path, data):
    with path.open('xb') as stream:
        stream.write(data)


def dump(data):
    return (json.dumps(data, indent=2, allow_nan=False) + '\n').encode()


def guard_from_ticket(directory):
    ticket = json.loads((directory / 'start-ticket.json').read_text())
    guard = Snapshot([])
    guard.bind_hashes(ticket['starting_hashes'])
    guard.include([directory / 'start-ticket.json', Path(__file__)])
    guard.check()
    return ticket, guard


def prepare(directory):
    ticket, guard = guard_from_ticket(directory)
    fonts = [Path('/System/Library/Fonts/Supplemental') / name for name in
             ['Times New Roman.ttf', 'Times New Roman Bold.ttf']]
    guard.include(fonts)
    for name, path in zip(['CaptionTimes', 'CaptionTimesBold'], fonts):
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily('CaptionTimes', normal='CaptionTimes', bold='CaptionTimesBold')
    source = ticket['inputs']
    plot = Image.open(source['png']['path']).convert('RGB')
    width_pt = plot.width / 600 * 72
    style = ParagraphStyle('caption', fontName='CaptionTimes', fontSize=9,
                           leading=12.15, alignment=TA_LEFT)
    texts = ['<b>' + html.escape(TITLE) + '</b>', *map(html.escape, PARAGRAPHS)]
    paragraphs = [Paragraph(text, style) for text in texts]
    heights = [p.wrap(width_pt, 10000)[1] for p in paragraphs]
    spacing = 1.8 * mm
    caption_height = sum(heights) + spacing * (len(paragraphs) - 1) + 2
    with tempfile.TemporaryDirectory(prefix='figure4-caption-') as tmp:
        tmp = Path(tmp)
        cap_pdf = tmp / 'caption.pdf'
        c = canvas.Canvas(str(cap_pdf), pagesize=(width_pt, caption_height))
        y = caption_height - 1
        for p, height in zip(paragraphs, heights):
            y -= height
            p.drawOn(c, 0, y)
            y -= spacing
        c.save()
        subprocess.run(['pdftoppm', '-r', '600', '-png', '-singlefile',
                        str(cap_pdf), str(tmp / 'caption')], check=True)
        cap = Image.open(tmp / 'caption.png').convert('RGB')
        assert abs(cap.width - plot.width) <= 1
        gap = round(3 / 25.4 * 600)
        combined = Image.new('RGB', (plot.width, plot.height + gap + cap.height), 'white')
        combined.paste(plot, (0, 0))
        combined.paste(cap, (0, plot.height + gap))
        assert combined.crop((0, 0, plot.width, plot.height)).tobytes() == plot.tobytes()
        save_new(directory / 'figure-04.svg', Path(source['working_svg']['path']).read_bytes())
        save_new(directory / 'figure-04-artwork-600dpi.png', Path(source['png']['path']).read_bytes())
        png = directory / 'figure-04-600dpi.png'
        with png.open('xb') as stream:
            combined.save(stream, format='PNG', dpi=(600, 600))
        caption = TITLE + '\n\n' + '\n\n'.join(PARAGRAPHS) + '\n'
        save_new(directory / 'legend.md', caption.encode())
        svg = (directory / 'figure-04.svg').read_text()
        svg = svg[svg.index('<svg'):]
        body = '<p><strong>' + html.escape(TITLE) + '</strong></p>'
        body += ''.join('<p>' + html.escape(p) + '</p>' for p in PARAGRAPHS)
        page = '<!doctype html><meta charset="utf-8"><title>Figure 4</title><style>body{margin:10mm auto;width:183mm;max-width:96vw}svg{width:100%;height:auto;display:block}figcaption{font:9pt/1.35 "Times New Roman",Times,serif;margin-top:3mm}p{margin:0 0 1.8mm}@page{size:A4;margin:10mm 13.5mm}</style>' + svg + '<figcaption>' + body + '</figcaption>'
        save_new(directory / 'figure-04.html', page.encode())
        pdf = directory / 'figure-04.pdf'
        assert not pdf.exists()
        c = canvas.Canvas(str(pdf), pagesize=A4)
        c.setTitle(TITLE)
        height_pt = combined.height / 600 * 72
        assert height_pt + 20 * mm < A4[1]
        c.drawImage(ImageReader(combined), (A4[0] - width_pt) / 2,
                    A4[1] - 10 * mm - height_pt, width=width_pt, height=height_pt)
        c.showPage()
        c.save()
    reader = PdfReader(str(pdf))
    assert len(reader.pages) == 1
    images = list(reader.pages[0].images)
    assert len(images) == 1
    assert images[0].image.convert('RGB').tobytes() == combined.tobytes()
    assert not reader.pages[0].extract_text().strip()
    guard.check()
    outputs = {p.name: record(p) for p in directory.iterdir() if p.is_file()}
    evidence = {'date': ticket['date'], 'scope': 'Legend and raster-PDF export only',
                'starting_hashes': guard.hashes, 'outputs': outputs,
                'checks': {'svg_bytes_unchanged': True, 'artwork_pixels_unchanged': True,
                           'pdf_page_count': 1, 'page_size': 'A4', 'image_dpi': 600,
                           'pdf_image_pixels_equal_combined_png': True,
                           'caption': '9 pt Times, 1.35 leading, bold title, 3 mm gap',
                           'scientific_recomputation': False},
                'style_reference_reconciliation': ticket['reconciliation']}
    save_new(directory / 'export-evidence.json', dump(evidence))
    print(json.dumps({'pdf': str(pdf), 'png_pixels': combined.size, 'caption_words': len(caption.split())}))


def publish(directory, note):
    ticket, guard = guard_from_ticket(directory)
    evidence = json.loads((directory / 'export-evidence.json').read_text())
    guard.bind_hashes(evidence['starting_hashes'])
    guard.bind_hashes({item['path']: item['sha256'] for item in evidence['outputs'].values()})
    guard.include([directory / 'export-evidence.json'])
    guard.check()
    state_path = Path(ticket['source_state'])
    state = json.loads(state_path.read_text())
    draft = state['draft']
    prior_verification = copy.deepcopy(draft['verification'])
    for key, name in [('working_svg', 'figure-04.svg'), ('review', 'figure-04.html'), ('legend', 'legend.md')]:
        draft[key] = record(directory / name)
    draft['exports'] = {key: record(directory / name) for key, name in [('png', 'figure-04-600dpi.png'), ('pdf', 'figure-04.pdf')]}
    draft['exports']['artwork_png'] = record(directory / 'figure-04-artwork-600dpi.png')
    review = {'date': ticket['date'], 'review': note, 'export_evidence': record(directory / 'export-evidence.json'), 'prior_artwork_verification': prior_verification}
    review_path = directory / 'visual-review.json'
    review_bytes = dump(review)
    draft['verification'] = {'path': str(review_path), 'sha256': sha(review_bytes)}
    lock = {'date': ticket['date'], 'kind': 'provisional-presentation-lock',
            'authorization': ticket['authorization'], 'whole_figure_frozen': False,
            'files': evidence['outputs'], 'export_evidence': record(directory / 'export-evidence.json'),
            'visual_review_sha256': sha(review_bytes), 'previous_lock': state['presentation_lock']}
    lock_path = directory / 'version-lock.json'
    lock_bytes = dump(lock)
    state['presentation_lock'] = {'path': str(lock_path), 'sha256': sha(lock_bytes),
        'status': 'locked', 'kind': 'provisional-presentation-lock', 'date': ticket['date'],
        'scope': state['scope'] + '; pooled catch-trial caption included',
        'retain_other_versions': True, 'future_edits_require_new_version': True,
        'publication_freeze_completed': False, 'previous_lock': lock['previous_lock']}
    publish_state(state_path, state, guard, artifacts=[(review_path, review_bytes), (lock_path, lock_bytes)])
    print('Published Figure 4 caption export; earlier versions retained.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--visual-review', help='Publish only after actual saved PDF inspection')
    args = parser.parse_args()
    if args.visual_review:
        publish(args.directory, args.visual_review)
    else:
        prepare(args.directory)
