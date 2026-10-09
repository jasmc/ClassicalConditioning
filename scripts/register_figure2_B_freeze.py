"""Register completed D/E freezes and provide a local review landing page."""
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from classical_conditioning.artifacts import sha256_file
import pandas as pd

OUT=ROOT/"reviews/figure2_B_freeze_20261009"
def record(path): return dict(path=str(path.resolve()),sha256=sha256_file(path))

def main():
    panels={}
    for p in ("D","E"):
        frozen=json.loads((OUT/f"{p}.freeze.json").read_text(encoding="utf-8"))
        for item in frozen["exports"]:
            assert sha256_file(Path(item["path"]))==item["sha256"]
        panels[p]=dict(freeze=record(OUT/f"{p}.freeze.json"),svg=record(OUT/f"Fig2_{p}_B.svg"),pdf=record(OUT/f"Fig2_{p}_B.pdf"),png=record(OUT/f"Fig2_{p}_B.png"))
    selection=dict(figure_id="Fig2",panels=["D","E"],selected_version="B",status="frozen via explicit gate",authorization="Joaquim: correct statistical tests, then freeze version B, 2026-10-09",sampling_policy="valid-bout-frames-only-v1",processing="median log positive valid bout-frame response minus baseline; median eligible trials per fish/block",baseline_s=[-15,0],response_s={"D":[0,9],"E":[0,13]},statistics=record(OUT/"statistics.csv"),family="36 two-sided tests across D/E; exact sign and Brunner-Munzel t; Holm36",current_panels=panels,specification_version="1.1.1",intended_panel_size_mm=[54.9,53.2],scope="D/E only; F unavailable; whole assembly older than these scoped panel freezes",handoff=str(ROOT/"Plans/HANDOFF_BOUT_LOG_VIGOR_PANELS_2026-10-09.md"))
    config=ROOT/"configs/paper-figures/figure2-DE-B-freeze-20261009.json"
    config.write_text(json.dumps(selection,indent=2)+"\n",encoding="utf-8")
    current=dict(policy="valid-bout-frames-only-v1",author_approval_date="2026-10-09",active_review="figure2_B_freeze_20261009/comparison.html",selected_version="B",status="frozen",scoped_selection=record(config),superseded_reviews=["figure2_ratio_vs_log_20261009","figure2_bout_only_ratio_vs_log_20261009"],note="Earlier reviews/statistics preserved as history; A remains a sensitivity outcome, not the selected panel",F="placeholder; no authenticated inputs")
    (ROOT/"reviews/CURRENT_FIGURE2_VIGOR_REVIEW.json").write_text(json.dumps(current,indent=2)+"\n",encoding="utf-8")
    tests=pd.read_csv(OUT/"statistics.csv")
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Frozen B · Figure 2 D/E</title><style>body{max-width:1100px;margin:28px auto;padding:20px;font:16px/1.5 system-ui;background:#f3f5f7;color:#20242a}section,article{background:white;padding:20px;border-radius:10px;margin:18px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:22px}img{width:100%}a{color:#235fa4}table{font-size:12px;border-collapse:collapse}td,th{padding:5px;border-bottom:1px solid #ddd}.scroll{overflow:auto}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style><h1>Frozen B · Figure 2 D and E</h1><p>B = median natural-log bout vigor in response minus baseline, then median eligible trials within each fish/block. No-bout/invalid frames are NaN and excluded. Reference 0 = unchanged.</p><section><h2>Corrected statistics</h2><p>Exact sign tests for paired changes and comparisons with zero; Brunner–Munzel t tests for independent conditioned/control comparisons. One Holm correction covers all 36 tests across D/E. Fish are the units.</p><p>Delay ET is below baseline (p=0.0000552); its PT→ET change differs from controls (p=0.000120), and its ET→LT increase differs from controls (p=0.0000131). Trace has no significant result after Holm36. These analyses remain exploratory and assume independent fish; no day/tank adjustment.</p><p>Main whiskers are fish IQR. † denotes a test against zero; lines denote within/between-block group comparisons. Direct changes and pointwise bootstrap effects are in the files below.</p></section><div class="grid">'''
    for p,label in (("D","Delay"),("E","3 s Trace")):
        page+=f'<article><h2>{p} · {label}</h2><img src="Fig2_{p}_B.png"><p><a href="Fig2_{p}_B.svg">SVG</a> · <a href="Fig2_{p}_B.pdf">PDF</a> · <a href="{p}.freeze.json">Freeze record</a></p></article>'
    page+='''</div><section><h2>Processing handoff</h2><p><a href="handoff.md">Shared scaffold for Figure 1 E–H and whole Figure 2</a> · <a href="README.md">Frozen methods and limitations</a> · <a href="statistics.csv">36 tests</a> · <a href="bootstrap-effects.csv">Bootstrap effects</a></p><p>Both freezes passed specification 1.1.1 at 54.9 × 53.2 mm per panel. Whole figures require their own assembly checks. F/10 s Trace remains unavailable. Earlier comparisons are historical and have superseded statistics.</p></section><section><h2>All corrected tests</h2><div class="scroll">'''+tests[["panel","family","condition","left","right","test","n_pairs","n_conditioned","n_control","p_raw","p_holm36","stars"]].to_html(index=False,float_format=lambda x:f"{x:.5g}",na_rep="—")+'''</div></section></html>'''
    (OUT/"comparison.html").write_text(page,encoding="utf-8")
    (OUT/"handoff.md").write_text((ROOT/"Plans/HANDOFF_BOUT_LOG_VIGOR_PANELS_2026-10-09.md").read_text(encoding="utf-8"),encoding="utf-8")
    print("Registered frozen B D/E selection; historical assembly and earlier reviews preserved.")

if __name__=="__main__":main()
