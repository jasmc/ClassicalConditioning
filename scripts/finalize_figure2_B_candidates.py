"""Record the performed B visual review and preserve small code snapshots.

Run only after inspecting the exported D/E panels at the recorded 54.9-mm
width. This prepares evidence; the explicit freeze command remains mandatory.
"""
from pathlib import Path
import json
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from classical_conditioning.artifacts import sha256_file

OUT=ROOT/"reviews/figure2_B_freeze_20261009"

def record(path): return dict(path=str(path.resolve()),sha256=sha256_file(path))

def main():
    if any(OUT.glob("*.freeze.json")): raise ValueError("Existing freezes are immutable")
    snapshot=OUT/"code-snapshot";snapshot.mkdir(exist_ok=True)
    paths=["scripts/prepare_figure2_B_freeze.py","scripts/review_figure2_block_log_median.py","scripts/review_figure2_bout_only.py","src/classical_conditioning/analysis/bout_block_statistics.py","src/classical_conditioning/analysis/bout_vigor.py","src/classical_conditioning/preprocessing/candidate_metric_kernel.py","src/classical_conditioning/config/experiments.py","src/classical_conditioning/figures/export.py","src/classical_conditioning/figure_freeze.py","configs/paper-figures/figure-elements.json","docs/analysis/figures/FIGURE_ELEMENT_SPECIFICATION.md"]
    snapshots=[]
    for name in paths:
        src=ROOT/name;dest=snapshot/Path(name).name
        shutil.copyfile(src,dest);snapshots.append(record(dest))
    for name in ("inputs.json","manifest.json"):
        src=ROOT/"reviews/figure2_bout_only_ratio_vs_log_20261009"/name
        dest=OUT/("authenticated-source-"+name)
        shutil.copyfile(src,dest);snapshots.append(record(dest))
    review={"status":"passed","evidence":"Codex inspected final D and E PNGs after correcting an encoding defect; labels, zero-reference dagger, group titles, PT/ET/LT, sample counts, paired fish, IQR, significance lines and negative/positive log scales legible and unclipped. Actual panel size 54.9 x 53.2 mm, with 8-pt labels/headings and 7-pt ticks/annotations; renderer text-bound checks pass. All eligible values within limits, no data clipping. Missing fish/block values remain gaps. SVG freeze check measures semantic styles and protected geometry.","exports":[record(OUT/f"Fig2_{p}_B.png") for p in ("D","E")],"renderer_evidence":[record(OUT/f"{p}_renderer_review.json") for p in ("D","E")]}
    (OUT/"visual-review.json").write_text(json.dumps(review,indent=2)+"\n",encoding="utf-8")
    for panel in ("D","E"):
        path=OUT/f"{panel}_candidate.json";candidate=json.loads(path.read_text(encoding="utf-8"))
        candidate["verification"]["visual_review"]={"status":"passed","evidence":review["evidence"],"record":record(OUT/"visual-review.json")}
        candidate["source_artifacts"].extend(snapshots+[record(OUT/"visual-review.json"),record(OUT/f"{panel}_renderer_review.json")])
        path.write_text(json.dumps(candidate,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Bound actual visual-review evidence and code/source snapshots to both candidates.")

if __name__=="__main__":main()
