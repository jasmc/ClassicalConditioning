"""Assemble a single scientific comparison PDF using the bundled PDF runtime."""
from pathlib import Path
import hashlib
import json
from datetime import datetime,timezone
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A3,landscape
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader

OUT=Path('F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs/figure2-assembly/row3-trial-ratio-review/all-versions-comparison')
W,H=landscape(A3)
M=40
BODY=ParagraphStyle('body',fontName='Helvetica',fontSize=12,leading=17,textColor=HexColor('#253447'))
SMALL=ParagraphStyle('small',parent=BODY,fontSize=10,leading=14)
TITLE=ParagraphStyle('title',parent=BODY,fontName='Helvetica-Bold',fontSize=25,leading=29)

def paragraph(c,text,y,width=W-2*M,style=BODY,x=M):
    text=text.replace('\u2013','-').replace('\u2014','-').replace('\u2011','-')
    p=Paragraph(text,style); _,height=p.wrap(width,y-50)
    p.drawOn(c,x,y-height)
    return y-height-14

def heading(c,title,page):
    c.setFillColor(HexColor('#15283b')); c.setFont('Helvetica-Bold',22)
    c.drawString(M,H-44,title)
    c.setFont('Helvetica',10); c.setFillColor(HexColor('#64748b'))
    c.drawString(M,24,'Delay | legacy metric | bout frames only | exploratory comparison | 9 October 2026')
    c.drawRightString(W-M,24,str(page))

def table(c,rows,y,widths):
    cells=[[Paragraph(str(cell),SMALL) for cell in row] for row in rows]
    t=Table(cells,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#e6edf4')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.7,HexColor('#91a4b8')),
        ('LINEBELOW',(0,1),(-1,-1),.35,HexColor('#d9e1e9')),
        ('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9)]))
    _,h=t.wrap(W-2*M,H); t.drawOn(c,M,y-h)
    return y-h-18

def trial_ranges(values):
    ranges=[]
    for value in values:
        value=int(value)
        if ranges and value==ranges[-1][1]+1: ranges[-1][1]=value
        else: ranges.append([value,value])
    return ', '.join(str(a) if a==b else f'{a}-{b}' for a,b in ranges) or 'none'

def main():
    report=json.loads((OUT/'comparison-report.json').read_text())
    pages=json.loads((OUT/'pdf-pages.json').read_text())
    pdf=OUT/'Delay_all_versions_comparison.pdf'
    c=canvas.Canvas(str(pdf),pagesize=(W,H))
    c.setTitle('Delay: all bout-intensity versions on matched inputs')
    c.setAuthor('ClassicalConditioning analysis review')
    page=1; heading(c,'What is plotted, and what changes between versions',page)
    y=H-90
    y=paragraph(c,'<b>Blue = control (28 fish). Magenta = Delay (29 fish).</b> Each observed line is the median across fish at each CS trial. A fish contributes one window summary per trial. Thin connecting lines are observed summaries, not fitted curves.',y)
    y=paragraph(c,'The metric remains <b>legacy distal angular speed</b>, in rad/ms before normalization. Only valid, adjacent <b>bout frames</b> are considered. Non-bout periods are ignored. Baseline is [-15, 0) seconds; response is [0, 9) seconds relative to CS onset. Trials 5-14 are Pre, 15-64 training, and 65-94 test. Missing windows remain NaN.',y)
    y=table(c,[['Version','Per-fish calculation','Population line / reference','Purpose'],
      ['Arithmetic-mean ratio','mean(response bout frames) / mean(baseline bout frames)','Median of fish ratios / 1','Average intensity while bouting; strong bursts contribute'],
      ['Log arithmetic-mean ratio','ln[mean(response) / mean(baseline)]','Median of fish log ratios / 0','Same window means, different display scale'],
      ['Literal window-median ratio','median(response bout frames) / median(baseline bout frames)','Median of fish ratios / 1','Typical within-bout intensity; descriptive matched comparison'],
      ['Historical LogMedian summary','median(ln positive response frames) - median(ln positive baseline frames)','Median of fish differences / 0','One log before within-window medians; new LMM fits'],
      ['Observed + fitted contrasts','Observed lines above; adjusted control-minus-Delay change from average Pre below','Separate population model contrast / 0','Evaluate condition differences accounting for repeated fish trials']],y,[205,385,260,W-2*M-850])
    y=paragraph(c,'<b>Why mean and median differ:</b> for bout vigor [1, 1, 10], mean = 4 and median = 1. Logging the mean gives ln(4), so the large value still affects it. The median of logged values is 0. A median answers a typical-intensity question; a mean includes the contribution of strong bursts to average intensity.',y)
    y=paragraph(c,'The new LogMedian uses the historical <i>summary</i> on the same current corrected input files and initial bout mask. Means/raw medians retain zero vigor, but LogMedian excludes 18,610 baseline and 10,859 response nonpositive frames before logging. It does not reproduce the old rolling-median/downsampling pipeline and omits the second log(x+1). The technical cohort and models remain exploratory; no new panel freeze is inferred.',y,style=SMALL)
    c.showPage(); page+=1
    heading(c,'How to read bands, letters and stars',page); y=H-90
    y=paragraph(c,'<b>5,000 resamples, seed 10, sampling whole fish trajectories within each condition and preserving missingness.</b> Draw 28 control fish and 29 Delay fish with replacement. A selected fish brings every trial and missing trial with it. Calculate the median at each trial, ignoring NaNs. Repeat 5,000 times; the 2.5th and 97.5th percentiles give a pointwise 95% confidence interval. Seed 10 makes the random draws repeatable.',y)
    y=paragraph(c,'<b>Bootstrap is a method; CI and IQR describe different things.</b> The CI estimates uncertainty about a condition median. The IQR covers the middle 50% of observed fish. Use the bootstrap CI for uncertainty of the plotted median, and an IQR or individual-fish view to show heterogeneity. These are pointwise intervals, not simultaneous bands over 90 trials. Band overlap is not the mixed-model test.',y)
    y=table(c,[['Mark','Exact question / correction','Interpretation'],
      ['D','Condition x block interaction relative to Pre; eight terms, Holm adjustment','Does the adjusted condition difference change between that block and Pre?'],
      ['M','Local block condition coefficient at centered trial; nine tests, BH FDR','Adjusted condition difference at the block center; not a direct test of the observed median ratio'],
      ['R (raw / FDR)','Local condition x centered-trial coefficient; nine slope tests, raw and separate BH FDR','Does the within-block slope differ between conditions?'],
      ['Black stars','Two-sided fitted control-minus-Delay difference at trial, minus its average in Pre5-14; BH over 90 trials','One star marks adjusted p < .05. Constant star size. Global and phase models can give different answers.']],y,[170,520,W-2*M-690])
    y=paragraph(c,'Letter suffixes encode * p &lt; .05, ** p &lt; .01, *** p &lt; .001, using that row’s raw or adjusted p-value. Families are corrected separately; this is not a single paper-wide correction.',y)
    y=paragraph(c,'<b>Phase-aware fit:</b> Pre has a linear shape; training and test each have their own cubic spline and intercept. A global spline can borrow curvature across phase boundaries. Pre stars test departures from the average Pre condition contrast, so they cannot establish learning before training. A phase-aware comparison is a sensitivity check, not a proof of learning onset.',y)
    y=paragraph(c,'The Plans correctly identify fish as the experimental units and require a simultaneous band and persistent, meaningful contrast for onset. A first significant trial is insufficient. Legacy separate per-trial random-intercept models have one observation per fish in each fit, so their random-intercept and residual variances are not separately identifiable. This comparison instead uses longitudinal repeated measurements.',y,style=SMALL)
    c.showPage(); page+=1
    for item in pages:
        heading(c,item['title'],page)
        y=H-83
        p=Paragraph(item['caption'].replace('\u2013','-').replace('\u2014','-'),BODY); _,ch=p.wrap(W-2*M,110)
        p.drawOn(c,M,49)
        image=ImageReader(item['image']); iw,ih=image.getSize()
        available_h=y-(49+ch+20)
        scale=min((W-2*M)/iw,available_h/ih)
        c.drawImage(image,(W-iw*scale)/2,49+ch+20+(available_h-ih*scale)/2,width=iw*scale,height=ih*scale)
        c.showPage(); page+=1
    heading(c,'Model confirmation, limitations and provenance',page); y=H-90
    m=report['mean']; lm=report['logmedian']; ph=report['phase']
    y=table(c,[['Outcome / trial model','D Holm8','M BH9','R raw / BH9','Black-star trials'],
      ['Arithmetic means / global spline',m['D_significant'],m['M_significant'],f"{m['R_raw_significant']} / {m['R_FDR_significant']}",'5-71 (67), including all Pre'],
      ['Arithmetic means / phase model',m['D_significant'],m['M_significant'],f"{m['R_raw_significant']} / {m['R_FDR_significant']}",trial_ranges(ph['phase']['significant_trials'])],
      ['Historical LogMedian / phase model',lm['D'],lm['M'],f"{lm['R_raw']} / {lm['R_fdr']}",trial_ranges(lm['trial_significant']['phase'])]],y,[300,110,110,150,W-2*M-670])
    y=paragraph(c,'<b>Confirmed scope:</b> all 15 mean main/local/sensitivity fits and four global/phase comparison fits pass their numerical gates. LogMedian has 14 attempted fits: 13 pass, but the Test3 local random-intercept fit is singular and fails. Its M/R results are unavailable (n/a), with the nine-block correction families retained. Gates cover convergence, design rank, fixed covariance and Hessian curvature. A valid optimizer result does not by itself validate the assumptions.',y)
    failures={'mean':m['influence_failed_refits'],'mean phase':ph['influence']['failed'],'LogMedian phase':lm['influence_failed']}
    y=paragraph(c,'Recorded influence failures: '+str(failures)+'. LogMedian residual excess kurtosis by model: '+', '.join(f"{r['model']}: {r['residual_excess_kurtosis']:.3f}" for r in lm['adequacy'])+'.',y,style=SMALL)
    y=paragraph(c,'<b>Criticism:</b> dense stars are a model result, not an onset detector. LogMedian does not improve the residual tails here: phase-model excess kurtosis is 8.027 versus 7.363 for means. Covariance assumptions, possible day/rig effects and informative missing bout windows remain concerns. These results measure bout intensity, not movement probability or total activity. Comparing outcomes on the same data is exploratory; do not select by counting stars.',y)
    y=paragraph(c,'For an average-intensity claim, retain arithmetic means and a log-response LMM with log-baseline adjustment and fish effects. For a typical-intensity claim, the literal medians / one-log historical summary are sensible candidates. Select the scientific target first. Keep observed medians visible and fitted contrasts separate. Current Gaussian Wald inference remains exploratory; onset/extinction require the additional calibrated analysis specified in the Plans.',y)
    y=paragraph(c,'<b>Storage and history:</b> J was inaccessible. These previews were regenerated on F after matching all 402 recorded source-data hashes to the local sidecar, retaining the 57-fish cohort hash and 4,811 eligible rows. No raw/frame data copies were created. February and March LogMedian snapshots were compared earlier: logging occurred before window medians, and the later second log(x+1) is omitted here. Old all-frame panels are superseded by the bout-only correction and are not accepted candidates.',y,style=SMALL)
    y=paragraph(c,'Source plans: Plans/02_ANALYSIS_AND_STATISTICS.md and Plans/03_LEARNING_ONSET_IMPLEMENTATION.md. Historical commits: February 2f63ef48361d6e80d1b8e1393894d7e5a3dedf55; March bf46bf7b02baaf6d8138d9772f255881f86c3c78. Mixed-model documentation: https://www.statsmodels.org/stable/mixed_linear.html (checked 9 October 2026). Full authenticated tables, formulas, draws, source hashes and diagnostics are in the adjacent version directories.',y,style=SMALL)
    if y<40: raise RuntimeError('Final page text exceeds safe page area')
    c.showPage(); c.save()
    reader=PdfReader(str(pdf)); assert len(reader.pages)==len(pages)+3
    texts=[p.extract_text() for p in reader.pages]
    assert all(t.strip() for t in texts)
    manifest={'path':str(pdf),'sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
        'pages':len(reader.pages),'created_at':datetime.now(timezone.utc).isoformat(),
        'page_sources':pages,'text_extraction_checked':True}
    (OUT/'pdf-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
