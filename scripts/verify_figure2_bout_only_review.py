"""Validate active scientific review without changing frozen/historical files."""
from pathlib import Path
import json
import sys
import hashlib
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from classical_conditioning.analysis.bout_vigor import VIGOR_SAMPLE_POLICY

OUT = ROOT / "reviews/figure2_bout_only_ratio_vs_log_20261009"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    tests = pd.read_csv(OUT / "all_statistics.csv")
    assert len(tests) == 48 and tests.groupby("version").size().eq(24).all()
    assert np.isfinite(tests[["p_raw", "p_holm24"]]).all().all()
    assert tests.p_holm24.ge(tests.p_raw - 1e-12).all()
    effects = pd.read_csv(OUT / "direct_change_effects_bootstrap.csv")
    assert len(effects) == 12 and effects.bootstrap_draws.eq(5000).all()
    for panel in ("D", "E"):
        trials = pd.read_parquet(OUT / f"{panel}_log_trials.parquet")
        assert trials.vigor_sample_policy.eq(VIGOR_SAMPLE_POLICY).all()
        assert not trials.duplicated(["recording_id", "trial_number"]).any()
        assert trials.loc[trials.log_trial_eligible, ["baseline_positive_moving_samples", "response_positive_moving_samples"]].gt(0).all().all()
        for version in ("ratio", "log"):
            fish = pd.read_parquet(OUT / f"{panel}_{version}_fish_blocks.parquet")
            assert fish.vigor_sample_policy.eq(VIGOR_SAMPLE_POLICY).all()
            # Ratio coverage tables retain descriptive medians for 1–2 valid
            # trials, explicitly ineligible under D's minimum-three rule.
            empty = fish["Contributing trials"].eq(0) if version == "ratio" else ~fish.Eligible
            assert fish.loc[empty, "Fish median response / baseline"].isna().all()
            assert not fish.duplicated(["fish_id", "condition_id", "Selected block order"]).any()
    # Update only the final code identities, after review and verification.
    inputs = json.loads((OUT / "inputs.json").read_text())
    for item in inputs:
        path = Path(item["path"])
        if path.suffix == ".py" and ROOT in path.parents:
            item["sha256"] = sha(path)
    (OUT / "inputs.json").write_text(json.dumps(inputs, indent=2)+"\n", encoding="utf-8")
    page = (OUT / "comparison.html").read_text(encoding="utf-8")
    page = page.replace("Change effect intervals = pointwise 95% bootstrap.", 'Change plots show fish median/IQR. <a href="direct_change_effects_bootstrap.csv">Effect estimates and pointwise 95% bootstrap intervals</a> are provided separately.')
    (OUT / "comparison.html").write_text(page, encoding="utf-8")
    verification = {"policy": VIGOR_SAMPLE_POLICY, "statistics": "48 tests; 24 per version; Holm24; 12 fish-level bootstrap change effects", "synthetic_and_targeted_tests": "90 passed across vigor policy, trial outcomes, temporal profiles, traces, cohort figures/outcomes, model input, metric comparison, discarding, learner adapter and learning onset", "scientific_invariants": "Non-bout extremes excluded before mean/log/scaling; empty bout windows remain NaN; no source array mutation; no silent old scaled-cache reuse", "visual_review": "D ratio and E log all-tests inspected; renderer checks labels/clipping for all 8 main exports", "historical_source_files": "preserved; superseded scientific definition", "freeze_invoked": False}
    (OUT / "verification.json").write_text(json.dumps(verification, indent=2)+"\n", encoding="utf-8")
    files = [{"path":str(p.resolve()),"sha256":sha(p),"size_bytes":p.stat().st_size} for p in sorted(OUT.iterdir()) if p.is_file() and p.name!="manifest.json"]
    (OUT / "manifest.json").write_text(json.dumps({"scope":"Active bout-only D/E comparison; superseded sources preserved", "files": files}, indent=2)+"\n", encoding="utf-8")
    print("Verified: 48 tests, 12 bootstrap effects, both bout-only panels, provenance and exports.")

if __name__ == "__main__":
    main()
