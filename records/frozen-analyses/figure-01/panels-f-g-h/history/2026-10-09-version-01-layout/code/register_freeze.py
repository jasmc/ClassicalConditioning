"""Register the scoped F/G/H selection without changing the A-D freeze."""
from pathlib import Path
import json, hashlib
from pypdf import PdfReader, PdfWriter
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / 'polished-layout'
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

writer = PdfWriter()
for panel in 'FGH':
    reader = PdfReader(OUT / f'Fig1_Panel{panel}_C_BoutSamples_polished.pdf')
    assert len(reader.pages) == 1
    writer.add_page(reader.pages[0])
bundle = OUT / 'FGH_C_BoutSamples_polished.pdf'
with bundle.open('wb') as stream:
    writer.write(stream)
assert len(PdfReader(bundle).pages) == 3

images = [Image.open(OUT / 'pdf_checks' / f'Panel{p}.png').convert('RGB') for p in 'FGH']
preview = Image.new('RGB', (sum(im.width for im in images), max(im.height for im in images)), 'white')
x = 0
for im in images:
    preview.paste(im, (x, 0)); x += im.width
preview.save(OUT / 'pdf_checks' / 'FGH_pdf_readback.png')

manifest = HERE / 'frozen-version1' / 'freeze.json'
validation = OUT / 'layout_validation.json'
frozen = json.loads(manifest.read_text())
for item in frozen['snapshot_files'] + frozen['original_sample_tables']:
    assert sha(Path(item['path'])) == item['sha256']
checks = json.loads(validation.read_text())
for panel in checks['panels']:
    for item in panel['outputs']:
        assert sha(Path(item['path'])) == item['sha256']
record = {
    'figure': 'Figure 1', 'panels': ['F', 'G', 'H'],
    'selection': 'Version 1: C_BoutSamples',
    'status': 'Processing and numeric display frozen; layout refined as requested',
    'user_approval_date_local': '2026-10-08', 'record_date_local': '2026-10-09',
    'freeze_manifest': str(manifest), 'freeze_manifest_sha256': sha(manifest),
    'layout_validation': str(validation), 'layout_validation_sha256': sha(validation),
    'baseline_voting_unit': 'eligible sample', 'halfsecond_binning': False,
    'scale': 'C = clip((x - P50) / ((P90 - P10) / 2), -1, +1)',
    'palette': 'managua_r', 'limits': [-1, 1],
    'bundle': str(bundle), 'bundle_sha256': sha(bundle),
    'scope': 'Only selected single-fish F/G/H; preserves existing A-D freeze and does not assemble Figure 1',
}
config = REPO / 'configs/paper-figures/figure1-fgh-version1-freeze-20261009.json'
config.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
main = REPO / 'configs/paper-figures/figure1-freeze.json'
previous = json.loads(main.read_text(encoding='utf-8'))
original_panels = previous['panels']
previous['single_fish_heatmap_processing_freeze'] = {
    'panels': ['F', 'G', 'H'], 'config': str(config), 'status': record['status'],
}
previous['pending_panels'].pop('F-H', None)
assert previous['panels'] == original_panels
main.write_text(json.dumps(previous, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'config': str(config), 'pdf': str(bundle)}, indent=2))
