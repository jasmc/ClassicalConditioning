"""Fish-level B statistics without a symmetry or equal-shape requirement.

The primary family is 36 two-sided tests across authenticated D/E panels:
12 exact sign tests versus zero, 12 paired-change sign tests, 6 independent
block comparisons and 6 independent paired-change comparisons. Independent
comparisons use Brunner–Munzel with t calibration (probabilistic superiority).
"""
import numpy as np
import pandas as pd
from scipy.stats import binomtest, brunnermunzel
from statsmodels.stats.multitest import multipletests

VALUE = "Fish median response / baseline"  # compatibility name in B tables
PAIRS = [(0, 1), (1, 2), (0, 2)]


def exact_sign(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Sign test requires finite eligible fish values")
    nonzero = values[values != 0]
    npositive = int((nonzero > 0).sum())
    p = binomtest(npositive, len(nonzero), .5, alternative="two-sided").pvalue if len(nonzero) else 1.
    return dict(statistic=npositive, p_raw=p, n_nonzero=len(nonzero), zero_differences=len(values)-len(nonzero),
                n_negative=int((nonzero < 0).sum()), median=float(np.median(values)))


def independent_test(t, c):
    if min(len(t), len(c)) < 10:
        raise ValueError("The frozen t-calibrated Brunner–Munzel recipe requires at least 10 fish per independent group")
    result = brunnermunzel(t, c, alternative="two-sided", distribution="t")
    if not np.isfinite(result.pvalue):
        raise ValueError("Degenerate Brunner–Munzel comparison: resolve before freezing")
    probability = float(((t[:, None] > c[None, :]).mean() + .5*(t[:, None] == c[None, :]).mean()))
    return dict(statistic=float(result.statistic), p_raw=float(result.pvalue),
                probability_conditioned_greater=probability, rank_biserial=2*probability-1)


def corrected_b_statistics(panels):
    if set(panels) != {"D", "E"}:
        raise ValueError("This family is defined for D/E only; unavailable F is not a tested panel")
    rows, changes = [], []
    for panel, data in panels.items():
        if data.duplicated(["condition_id", "fish_id", "Selected block order"]).any():
            raise ValueError("Duplicate fish/block values would pseudoreplicate the analysis")
        conditioned = "delay" if panel == "D" else "trace"
        w = data.loc[data.Eligible].pivot(index=["condition_id", "fish_id"], columns="Selected block order", values=VALUE).reindex(columns=[0,1,2])
        for block in range(3):
            groups = {}
            for condition in (conditioned, "control"):
                values = w.xs(condition)[block].dropna().to_numpy()
                groups[condition] = values
                rows.append(dict(panel=panel, family="zero", condition=condition, left=block, right=block,
                    test="exact two-sided sign test versus zero", n_pairs=len(values), **exact_sign(values)))
            t, c = groups[conditioned], groups["control"]
            rows.append(dict(panel=panel, family="between", condition=conditioned+" vs control", left=block, right=block,
                test="two-sided Brunner-Munzel; t calibration", n_conditioned=len(t), n_control=len(c), **independent_test(t,c)))
        for lo, hi in PAIRS:
            deltas = {}
            for condition in (conditioned, "control"):
                paired = w.xs(condition)[[lo,hi]].dropna()
                delta = (paired[hi]-paired[lo]).to_numpy()
                deltas[condition] = delta
                rows.append(dict(panel=panel, family="within", condition=condition, left=lo, right=hi,
                    test="exact two-sided sign test on paired fish changes", n_pairs=len(delta), **exact_sign(delta)))
                for fish, change in zip(paired.index, delta):
                    changes.append(dict(panel=panel, condition=condition, fish_id=fish, left=lo, right=hi, change=float(change)))
            t, c = deltas[conditioned], deltas["control"]
            rows.append(dict(panel=panel, family="change", condition=conditioned+" vs control", left=lo, right=hi,
                test="two-sided Brunner-Munzel on paired fish changes; t calibration", n_conditioned=len(t), n_control=len(c),
                median_change_conditioned=float(np.median(t)), median_change_control=float(np.median(c)), **independent_test(t,c)))
    tests = pd.DataFrame(rows)
    assert len(tests) == 36 and not tests.duplicated(["panel","family","condition","left","right"]).any()
    tests["p_holm36"] = multipletests(tests.p_raw, method="holm")[1]
    tests["stars"] = tests.p_holm36.map(lambda p: "****" if p<.0001 else "***" if p<.001 else "**" if p<.01 else "*" if p<.05 else "ns")
    tests["statistical_unit"] = "fish; paired fish changes where applicable"
    return tests, pd.DataFrame(changes)
