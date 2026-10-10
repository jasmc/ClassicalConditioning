"""Build one self-contained review with a verified, content-deduplicated archive.

Deletion is deliberately a separate, explicit PowerShell operation after checks.
"""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(0, str(_ArchivePath(__file__).resolve().parents[1] / "src"))
from classical_conditioning.external_artifacts import resolve_artifact, external_output
from pathlib import Path
import base64, hashlib, html, io, json, zipfile, re
ROOT=Path(__file__).resolve().parents[1]
REVIEW=Path('F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review')
FREEZE=REVIEW.parent/'frozen/20261009-G-historical-logmedian'
SELECTED='20261009T153803735311Z-delay-logmedian'
NAMES=['20261009T153013526348Z-delay-boutonly-lme','20261009T153143538697Z-delay-display','20261009T153313320243Z-delay-display','20261009T153314805058Z-delay-display','20261009T153317754141Z-delay-phase-lmm','20261009T153341564032Z-delay-logmedian',SELECTED,'20261009T173417074522Z-delay-lines-only','20261009T174114195765Z-delay-meanlog-check','all-versions-comparison']
OUTPUT = external_output('reviews/figure2_delay_all_versions_20261009.html')
def digest(b):return hashlib.sha256(b).hexdigest()
def sha(p):return digest(p.read_bytes())
def image(p):
    mime='image/svg+xml' if p.suffix=='.svg' else 'image/png'
    return f'<img alt="{html.escape(p.name)}" src="data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}">'
def main():
    assert (FREEZE/'Fig2_G_Historical_LogMedian.freeze.json').exists()
    dirs=[REVIEW/n for n in NAMES]
    assert all(p.resolve().parent==REVIEW.resolve() for p in dirs)
    files=[p for d in dirs for p in sorted(d.rglob('*')) if p.is_file()]
    assert not any(p.is_symlink() for p in files+dirs)
    manifest=[];contents={}
    for p in files+[REVIEW/'f-recovery-current.json']:
        b=p.read_bytes();h=digest(b);ref=None
        if p.parent.name==SELECTED and p.suffix not in {'.png','.svg'}:
            target=FREEZE/'data'/p.name
            assert sha(target)==h;ref=str(target)
        redundant_preview=(p.suffix=='.png' and (p.with_suffix('.svg').exists() or 'pdf-qa' in p.parts)) or p.suffix=='.pdf'
        entry=dict(original_path=str(p),relative_path=p.relative_to(REVIEW).as_posix(),sha256=h,bytes=len(b),canonical_freeze_path=ref,archive_member=None if ref or redundant_preview else 'objects/'+h,discarded_derivative=redundant_preview,discard_reason='Redundant raster/PDF preview; vector source or plot inputs and documentation retained in consolidated review' if redundant_preview else None)
        manifest.append(entry)
        if ref is None and not redundant_preview: contents.setdefault(h,b)
    docs=sorted((ROOT/'docs/analysis/figures').glob('FIGURE2_DELAY_*.md'))
    for p in docs:
        contents.setdefault(sha(p),p.read_bytes())
    meta=dict(date='2026-10-09',review_root=str(REVIEW),selected_freeze=str(FREEZE/'Fig2_G_Historical_LogMedian.freeze.json'),files=manifest,documentation=[dict(path=str(p),archive_member='objects/'+sha(p),sha256=sha(p)) for p in docs],directories=[str(p) for p in dirs],raw_bytes=sum(x['bytes'] for x in manifest),policy='Original contents retained by SHA-256, deduplicated; selected scientific payload points to immutable hash-verified freeze data; J inaccessible and untouched; original Digested Data untouched')
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        z.writestr('manifest.json',json.dumps(meta,indent=2))
        for h,b in contents.items(): z.writestr('objects/'+h,b)
    archive=stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        assert z.testzip() is None
        for entry in manifest:
            if entry['discarded_derivative']:continue
            b=Path(entry['canonical_freeze_path']).read_bytes() if entry['canonical_freeze_path'] else z.read(entry['archive_member'])
            assert digest(b)==entry['sha256'] and len(b)==entry['bytes']
    rows=[
        ('All-frame arithmetic means (superseded)','mean(response)/mean(baseline), included non-bout frames','Not eligible under current bout-only rule; old J artifacts unavailable, retained there. Local audit explains original stats.'),
        ('Bout-only arithmetic means','mean(response bout vigor)/mean(baseline bout vigor)','Median across fish; strong bursts influence each window mean; ratio reference1. Own LMM results preserved.'),
        ('Log arithmetic-mean ratio','ln(mean(response bout vigor)/mean(baseline bout vigor))','Median across fish; logging occurs after arithmetic means. Baseline-adjusted log-response global and phase LMM alternatives.'),
        ('Literal within-window medians','median(response bout vigor)/median(baseline bout vigor)','Includes finite zeros. Median across fish; descriptive only, no own LMM.'),
        ('Historical LogMedian (selected)','median(ln positive response bout vigor) − median(ln positive baseline bout vigor)','Median across fish; one log, no second log(x+1), no smoothing. Selected fresh phase-aware LMM statistics.'),
        ('Exponentiated Historical LogMedian','exp(per-fish Historical LogMedian difference)','Median across fish; descriptive ratio-scale view. Not identical to literal median ratios with zeros or even-count interpolation.'),
        ('Mean of logged frames','mean(ln positive response bout vigor) − mean(ln positive baseline bout vigor)','Median across fish; geometric-mean ratio on log scale. Same positive-frame inputs as selected LogMedian. Descriptive only; no LMM marks.'),
        ('Fitted contrasts: global versus phase-aware','Baseline-adjusted fitted control − Delay change relative to mean Pre contrast','Shown separately from observed medians; each outcome has its own model. A global spline borrowing across phases is a sensitivity analysis.'),
        ('Confidence interval versus IQR','Same observed median lines, different uncertainty summaries','95% bootstrap CI quantifies uncertainty in median; IQR spans middle50% of observed fish. Selected freeze uses CI.'),
    ]
    table='<table><tr><th>Version</th><th>Per-fish calculation</th><th>Display / status</th></tr>'+''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</table>'
    head='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Delay — all versions and selected freeze</title><style>body{font:16px/1.5 system-ui;margin:32px auto;max-width:1100px;padding:0 24px;color:#17202a}h1,h2{line-height:1.2}img{max-width:100%;display:block}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:10px;text-align:left;vertical-align:top}figure{margin:28px 0}figcaption{font-size:14px;color:#445}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.5 monospace}button{padding:10px;font-size:16px}.selected{border:2px solid #789;padding:16px}</style>
<h1>Delay: Historical LogMedian frozen; all alternatives documented</h1><p>Author selection: 9 October 2026. Scientific outcomes and fitted statistics are preserved; this is a panel G freeze at 183 × 98 mm. The whole figure assembly has not been refrozen.</p>'''
    explanation='''<h2>What is plotted</h2><p><strong>Blue: control (28 fish). Magenta: Delay (29 fish).</strong> Each trial line is the median across eligible fish of that fish’s response-window median log vigor minus baseline-window median log vigor. Vigor is the frozen legacy distal angular-speed metric. Only valid adjacent moving/bout frames with finite positive vigor enter these logged windows. Baseline is −15 to 0 seconds; response is 0 to 9 seconds after CS onset. Non-bout frames and empty windows remain missing.</p><p>The shaded regions are pointwise percentile 95% confidence intervals. <strong>5,000 resamples, seed10:</strong> repeatedly draw fish with replacement within each condition, taking each selected fish’s entire trial trajectory together, including its missing trials. This keeps repeated trials from the same fish together. The fixed seed makes rerunning the resampling reproducible. The middle95% of the resampled condition medians defines each trial’s interval.</p><h2>Statistics included in the freeze</h2><p>D: six condition × block differences from Pre, Holm8. M: five local block condition mean differences, BH9. R: two local condition slope differences, both raw and BH9. Black trial stars: phase-aware LMM control-minus-Delay changes from the average Pre5–14 contrast, BH90, trials19–71. These tests concern fitted adjusted contrasts, not tests of the displayed medians. One black star means adjusted p&lt;0.05; black-star count is not a p-value tier. Block marks use * / ** / *** for p&lt;0.05 /0.01 /0.001.</p><p>Models adjust for the matching log-baseline summary. The block and phase models use fish random intercepts and trial slopes; local block models use fish random intercepts. The phase fit separates Pre, training and test, with separate phase intercepts and training/test cubic spline features. <strong>Test3 local M/R is unavailable:</strong> its singular fit is retained, not replaced. Both BH9 families keep all nine scheduled tests. Heavy residual tails remain; significance does not establish learning onset. Thirteen of fourteen attempted LogMedian fits pass numerical gates; all57 phase leave-one-fish-out checks pass.</p><h2>How to interpret the alternatives</h2><p>The selected median of logged positive bout frames measures typical intensity conditional on being in a bout. A mean within the window answers a different question: high-intensity bursts contribute to average intensity. Mean(log vigor) gives a geometric-average measure; log(mean vigor) still carries the influence of bursts already included in the arithmetic mean. Both can be useful sensitivity comparisons, but neither inherits the selected LogMedian statistics.</p><p>“Historical” identifies the within-window log-median summary. It does not restore every February/March preprocessing step: no rolling median, no downsampling, no second log(x+1). Current corrected bout masking and positivity filtering apply. The selected definition agrees with the B outcome in the scoped D/E freezes on shared fish/trials. It does not improve residual tails here; selection is a scientific outcome preference, not evidence of better model calibration.</p>'''
    figures=[];seen=set()
    frozen_svg=FREEZE/'Fig2_G_Historical_LogMedian.svg'
    figures.append('<div class="selected"><h2>Frozen G</h2>'+image(frozen_svg)+'</div>')
    for p in files:
        if p.suffix=='.svg' or (p.suffix=='.png' and not p.with_suffix('.svg').exists() and 'pdf-qa' not in p.parts):
            h=sha(p)
            if h in seen:continue
            seen.add(h);figures.append('<figure>'+image(p)+'<figcaption>'+html.escape(p.relative_to(REVIEW).as_posix())+' — exploratory/source preview; only the selected panel above is current.</figcaption></figure>')
    b64=base64.b64encode(archive).decode()
    download='''<h2>Provenance and recoverable archive</h2><p>The embedded ZIP preserves earlier small analysis tables, fitted model outputs, source snapshots and vector or unique raster previews. Contents are deduplicated by SHA-256. Redundant PNG renders, PDF QA images and the superseded comparison PDF are discarded after their figures and documentation have been consolidated here; their original hashes remain in the manifest. The manifest maps original filenames to archived objects, discarded derivatives or immutable selected freeze data. Original experiment inputs and historical freezes remain untouched. J was inaccessible and was not cleaned.</p><button onclick="downloadArchive()">Download archived alternatives and manifest</button><script type="application/octet-stream" id="archive">'''+b64+'''</script><script>function downloadArchive(){const s=document.getElementById('archive').textContent;const b=Uint8Array.from(atob(s),c=>c.charCodeAt(0));const u=URL.createObjectURL(new Blob([b],{type:'application/zip'}));const a=document.createElement('a');a.href=u;a.download='delay-alternatives-20261009.zip';a.click();setTimeout(()=>URL.revokeObjectURL(u),10000)}</script>'''
    doc_text=''.join('<details><summary>'+html.escape(p.name)+' (historical audit snapshot)</summary><pre>'+html.escape(p.read_text(encoding='utf-8-sig'))+'</pre></details>' for p in docs)
    output=head+explanation+table+''.join(figures)+download+doc_text+'</html>'
    OUTPUT.write_text(output,encoding='utf-8')
    readback=OUTPUT.read_text(encoding='utf-8')
    embedded=re.search(r'id="archive">([^<]+)</script>',readback).group(1)
    assert digest(base64.b64decode(embedded))==digest(archive)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(embedded))) as z:
        for h in contents: assert digest(z.read('objects/'+h))==h
    audit=dict(status='verified-awaiting-cleanup',html_path=str(OUTPUT),html_sha256=sha(OUTPUT),html_bytes=OUTPUT.stat().st_size,embedded_archive_sha256=digest(archive),embedded_archive_bytes=len(archive),raw_bytes=meta['raw_bytes'],files=manifest,directories=meta['directories'],preserve=[str(FREEZE),str(REVIEW.parent/'sources')],skipped='J inaccessible; original data and other panels not in cleanup scope')
    external_output('reviews/delay_versions_cleanup_20261009.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k not in {'files','directories','preserve'}},indent=2))
if __name__=='__main__':main()
