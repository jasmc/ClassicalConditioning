"""Recheck B rank tests and separately audit its zero-baseline reference."""
import sys as _archive_sys
from pathlib import Path as _ArchivePath
_archive_sys.path.insert(0, str(_ArchivePath(__file__).resolve().parents[1] / "src"))
from classical_conditioning.external_artifacts import resolve_artifact, external_output
from pathlib import Path
import sys
import hashlib
import json
import re
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, binomtest, skew
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import review_figure2_block_log_median as review

OUT = external_output("reviews/figure2_bout_only_ratio_vs_log_20261009")

def main():
    data = {p: pd.read_parquet(OUT / f"{p}_log_fish_blocks.parquet") for p in ("D", "E")}
    recomputed, _, diagnostics = review.block_statistics(data, "log")
    saved = pd.read_csv(OUT / "log_tests.csv")
    keys = ["panel", "family", "condition", "left", "right"]
    matched = recomputed.merge(saved, on=keys, validate="one_to_one", suffixes=("_check", "_saved"))
    assert len(matched) == 24
    np.testing.assert_allclose(matched.p_raw_check, matched.p_raw_saved, atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(matched.p_holm24_check, matched.p_holm24_saved, atol=1e-14, rtol=1e-12)
    rows, counts = [], []
    for panel, fish in data.items():
        for (condition, block), group in fish.loc[fish.Eligible].groupby(["condition_id", "Selected block order"]):
            values = group[review.VALUE].to_numpy()
            nonzero = values[values != 0]
            ties = len(np.unique(np.abs(nonzero))) != len(nonzero)
            method = "exact" if len(nonzero) == len(values) and not ties else "asymptotic"
            p = wilcoxon(values, alternative="two-sided", zero_method="wilcox", method=method).pvalue if len(nonzero) else 1.
            sign = binomtest(int((nonzero < 0).sum()), len(nonzero), .5).pvalue if len(nonzero) else 1.
            rows.append(dict(panel=panel, condition=condition, block=review.LABELS[block], n_fish=len(values), n_below_zero=int((values<0).sum()), median_B=np.median(values), median_multiplicative_factor=np.exp(np.median(values)), wilcoxon_p_raw=p, sign_p_raw=sign, skewness=skew(values,bias=False), wilcoxon_method=method))
        trials = pd.read_parquet(OUT / f"{panel}_log_trials.parquet")
        for condition, group in trials.groupby("condition_id"):
            counts.append(dict(panel=panel, condition=condition, selected_trials=len(group), eligible=int(group.log_trial_eligible.sum()), excluded=int((~group.log_trial_eligible).sum()), no_positive_baseline=int(group.baseline_positive_moving_samples.eq(0).sum()), no_positive_response=int(group.response_positive_moving_samples.eq(0).sum())))
    baseline = pd.DataFrame(rows)
    assert len(baseline) == 12
    baseline["wilcoxon_p_holm12"] = multipletests(baseline.wilcoxon_p_raw, method="holm")[1]
    baseline["sign_p_holm12"] = multipletests(baseline.sign_p_raw, method="holm")[1]
    combined = multipletests(np.r_[saved.p_raw, baseline.wilcoxon_p_raw], method="holm")[1]
    baseline["wilcoxon_p_holm36_sensitivity"] = combined[24:]
    saved["p_holm36_with_zero_tests"] = combined[:24]
    baseline.to_csv(OUT / "B_zero_reference_audit.csv", index=False)
    saved.to_csv(OUT / "B_combined36_sensitivity.csv", index=False)
    pd.DataFrame(counts).to_csv(OUT / "B_trial_exclusion_audit.csv", index=False)
    print(baseline.to_string(index=False))
    print(pd.DataFrame(counts).to_string(index=False))
    print("Reproduced all 24 B tests. Zero-reference audit is exploratory; existing plot stars unchanged.")
    inputs = [{"path":str((OUT / f"{p}_log_fish_blocks.parquet").resolve()), "sha256":review.sha(str(OUT / f"{p}_log_fish_blocks.parquet"))} for p in ("D","E")]
    (OUT / "B_statistics_audit.json").write_text(json.dumps({"code":{"path":str(Path(__file__).resolve()),"sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},"inputs":inputs,"existing_24_tests_reproduced":True,"zero_reference_tests":"Exploratory two-sided tests; Holm12 separately for Wilcoxon and sign tests; Holm36 sensitivity combines 24 existing + 12 zero-reference Wilcoxon tests; no plot stars changed"},indent=2)+"\n",encoding="utf-8")
    page_path = OUT / "comparison.html"
    page = page_path.read_text(encoding="utf-8")
    page = re.sub(r'<!-- B-AUDIT-BEGIN -->.*?<!-- B-AUDIT-END -->', '', page, flags=re.S)
    section = '<!-- B-AUDIT-BEGIN --><section><h2>B: reductions relative to the zero baseline</h2><p>Existing stars compare blocks or conditions, not zero. A separate exploratory audit finds Delay ET below zero: 27/29 fish; median B=-0.0752 (about 7.2% reduction), Holm12 Wilcoxon p=0.00000755; symmetry-free sign test p=0.0000195. Trace ET does not establish a reduction below zero after correction. Original stars are unchanged.</p><p><a href="B_STATISTICAL_REVIEW.md">Statistical review and processing bullets</a> · <a href="B_zero_reference_audit.csv">Zero-reference results</a> · <a href="B_combined36_sensitivity.csv">Combined Holm36 sensitivity</a> · <a href="B_trial_exclusion_audit.csv">Trial exclusions</a></p></section><!-- B-AUDIT-END -->'
    page_path.write_text(page.replace('</html>', section+'</html>'), encoding="utf-8")

if __name__ == "__main__":
    main()
