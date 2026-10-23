"""Guarded exploratory 3sTrace CR profile pilot; never selects a learner method.

Run extract, model, render, then publish after exact browser/visual verification.
Permanent data, SVG and HTML are stored only on the authenticated external SSD.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from classical_conditioning.analysis.figure4 import verify_expected_us
from classical_conditioning.analysis.metric_comparison import _verify_temporal_profiles
from classical_conditioning.analysis.movement_state import resolve_candidate_metric_source
from classical_conditioning.analysis.review.figure4_assay_profiles import (
    GROUPS, CONTRASTS, model_data, fit_bootstrap, differences, peaks, individual_predictions,
)
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.config import Alignment, get_experiment_spec
from classical_conditioning.figure_workspace import Snapshot, StaleFigureState, publish_state, state_lock

TICKET = ROOT / "records/figure-reviews/figure-04/3strace-model-draft-start-20261022.json"
STATE = ROOT / "docs/paper/figures/figure-04/3strace-model-draft-state.json"
ATLAS = Path(r"J:\ClassicalConditioning Outputs\current\reviews\figures-03-04-3strace-20261020")
PROJECT = Path(r"F:\Digested Data\all3sTrace-full-v1")
OUT = Path(r"J:\ClassicalConditioning Outputs\current\reviews\figure4-3strace-model-draft-20261022")
METRIC = "legacy_distal_angular_speed"
SVG = "{http://www.w3.org/2000/svg}"
COLORS = ("#00aeef", "#f15a29", "#777777")
PAIR_NAMES = ("Control − L", "Control − N", "N − L")


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def save(p, value):
    with Path(p).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False, default=str)
        f.write("\n")


def ref(p):
    return {"path": str(Path(p).resolve()), "sha256": sha256_file(Path(p))}


def verify(r):
    for attempt in range(3):
        try:
            actual = sha256_file(Path(r['path']), block_size=256 * 1024)
            break
        except OSError as error:
            print(f"Source read failed ({attempt+1}/3): {r['path']}: {error}", flush=True)
            if attempt == 2:
                raise OSError('Unable to authenticate source: ' + r['path']) from error
            time.sleep((.5, 2.)[attempt])
    if actual != r["sha256"]:
        raise ValueError("Changed bound source: " + r["path"])


class PublicationSnapshot(Snapshot):
    """Hash originals under lock; retain their authenticated file states there.

    The publication helper repeats its checks without writing these immutable
    inputs. Like the repository's verified-artifact states, later checks in
    this same transaction reject a changed identity/size/time fingerprint.
    Working artifacts, states, code and other small inputs rehash every time.
    No cached authentication is carried across publication attempts.
    """
    def __init__(self, immutable_paths=()):
        super().__init__([])
        self.immutable_paths = {str(Path(p).resolve()) for p in immutable_paths}
        self.authenticated = {}

    @staticmethod
    def fingerprint(path):
        s = Path(path).stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def check(self, ignore=()):
        ignored = {str(Path(p).resolve()) for p in ignore}
        for index, (path, expected) in enumerate(self.hashes.items(), 1):
            if path in ignored:
                continue
            exists = Path(path).is_file()
            if exists and expected is not None:
                before = self.fingerprint(path)
                if path in self.authenticated:
                    if self.authenticated[path] != (expected, before):
                        raise StaleFigureState('Original changed within publication lock: ' + path)
                else:
                    verify({'path': path, 'sha256': expected})
                    if self.fingerprint(path) != before:
                        raise StaleFigureState('Source changed while hashing: ' + path)
                    if path in self.immutable_paths:
                        self.authenticated[path] = (expected, before)
            elif exists or expected is not None:
                raise StaleFigureState('Figure input changed during work: ' + path)
            if index % 100 == 0 or index == len(self.hashes):
                print(f'Locked publication hash checks {index}/{len(self.hashes)}', flush=True)


def guards():
    ticket = load(TICKET)
    for r in ticket["protected"]:
        p = Path(r["path"])
        actual = sha256_file(p) if p.is_file() else None
        if actual != r["sha256"]:
            raise ValueError("Changed starting authority: " + str(p))
    for r in ticket["inputs"]:
        verify(r)
    if not OUT.parent.is_dir() or not PROJECT.is_dir():
        raise FileNotFoundError("JOAQUIM and its authenticated 3sTrace project are required")
    return ticket


def membership():
    from classical_conditioning.analysis.figure4 import load_classification_manifest
    prior = load(ATLAS / 'figure4-classified-preliminary/analysis.json')
    labels, meta = load_classification_manifest(Path(prior['classifier_table']['path']), METRIC, ('all3sTrace',))
    if meta['classifier_execution_id'] != 'legacy-wip-3strace-distal-window13-59fish':
        raise ValueError('Unexpected classifier identity')
    verify(prior['classifier_table']); verify(prior['classifier_manifest'])
    verify({'path': meta['source_comparison'], 'sha256': meta['source_comparison_sha256']})
    comparison = load(meta['source_comparison'])
    for key, hash_key in [('cohort_path', 'cohort_sha256'), ('outcomes_path', 'outcomes_sha256')]:
        verify({'path': comparison[key], 'sha256': comparison[hash_key]})
    if comparison['metric_id'] != METRIC or comparison['cohort_hash'] != meta['cohort_hashes']['all3sTrace']:
        raise ValueError('Classifier comparison metric/cohort identity mismatch')
    roster = pd.read_csv(ATLAS / 'acquisition-scores.csv')[['fish_id', 'condition_id']]
    if roster.duplicated('fish_id').any() or roster.condition_id.value_counts().to_dict() != {'control': 90, 'trace': 89}:
        raise ValueError('Current authenticated 3sTrace roster changed')
    result = roster.merge(labels[['fish_id', 'condition_id', 'classifier_label', 'classification_eligible', 'ineligible_reason']],
                          on=['fish_id', 'condition_id'], how='left', validate='one_to_one', indicator=True)
    missing = result['_merge'].eq('left_only')
    result['label_source_status'] = np.where(missing, 'No saved classification', 'Saved legacy-WIP result')
    result.loc[missing, 'classifier_label'] = 'Unclassified'
    result.loc[missing, 'classification_eligible'] = False
    result.loc[missing, 'ineligible_reason'] = 'No saved legacy-WIP result; never assigned to nonlearner'
    result['display_group'] = 'Unclassified 3sTrace'
    result.loc[result.condition_id.eq('control'), 'display_group'] = GROUPS[0]
    for label, group in [('Learner', GROUPS[1]), ('Non-learner', GROUPS[2])]:
        result.loc[result.condition_id.eq('trace') & result.classifier_label.eq(label), 'display_group'] = group
    if result.display_group.value_counts().to_dict() != {GROUPS[0]: 90, 'Unclassified 3sTrace': 50, GROUPS[2]: 23, GROUPS[1]: 16}:
        raise ValueError('Saved WIP/current-roster overlap changed')
    return result.drop(columns='_merge')


def extract():
    guards()
    OUT.mkdir(exist_ok=True)
    cache = OUT / 'fish'; cache.mkdir(exist_ok=True)
    members = membership()
    if (OUT / 'membership.csv').exists():
        pd.testing.assert_frame_equal(pd.read_csv(OUT / 'membership.csv'), members, check_dtype=False)
    else:
        members.to_csv(OUT / 'membership.csv', index=False)
    spec = get_experiment_spec('all3sTrace')
    catches = tuple(spec.catch_trial_numbers(Alignment.CS))
    expected_us = float(spec.conditions[1].us_latency_s[0])
    sources = {r['fish_id']: r for r in load(ATLAS / 'source-authentication.json')}
    trial = pd.read_parquet(ATLAS / 'trial-bins.parquet')
    if set(trial.fish_id) != set(members.fish_id) or len(trial) != len(members) * 90 * 80:
        raise ValueError('Saved supported-reader profile roster/cardinality differs')
    parts = []
    for number, fish in enumerate(members.itertuples(index=False), 1):
        rid = fish.fish_id
        report = cache / (rid + '.json'); dest = cache / (rid + '.parquet')
        if report.exists():
            record = load(report)
            verify(record['output'])
            # Same checkpoint reuse boundary as Delay: retain the previously
            # authenticated source map; all originals rehash under publication
            # lock. Also compare resumed catch values against the bound master.
            source_hashes = {r['path']: r['sha256'] for r in sources[rid]['sources']}
            for r in record['inputs']:
                if r['path'] in source_hashes and r['sha256'] != source_hashes[r['path']]:
                    raise ValueError('Checkpoint/source authority mismatch: ' + rid)
            cached = pd.read_parquet(dest)
            expected = trial.loc[trial.fish_id.eq(rid) & trial.trial_number.isin(catches),
                ['fish_id', 'condition_id', 'trial_number', 'time_s', 'coverage', 'signed_log_vigor', 'movement_probability']].rename(
                    columns={'signed_log_vigor': 'signed', 'movement_probability': 'movement'})
            expected['display_group'] = fish.display_group
            pd.testing.assert_frame_equal(cached.reset_index(drop=True), expected.reset_index(drop=True))
            parts.append(cached)
            print(f'Reused authenticated fish {number}/{len(members)}: {rid}', flush=True)
            continue
        source = sources[rid]
        for r in [source['output'], *source['sources']]: verify(r)
        protocol_path = PROJECT / 'Processed data' / rid / 'stimulus_events.parquet'
        protocol = pd.read_parquet(protocol_path)
        cycles = protocol.loc[protocol.Type.eq('Cycle')].sort_values('Beg', kind='stable').reset_index(drop=True)
        durations = (cycles.End - cycles.Beg).to_numpy() / 1000
        if np.max(np.abs(durations - spec.cs_duration_s)) > .1:
            raise ValueError('Recorded CS duration differs from configuration: ' + rid)
        timing = verify_expected_us(protocol, 'all3sTrace') if fish.condition_id == 'trace' else None
        # Authenticate the temporal completion/config/upstream pairing as well.
        resolved = resolve_candidate_metric_source(metric_recipe='tail-candidate-corrected')
        temporal = _verify_temporal_profiles(PROJECT, rid, resolved)
        metadata = load(temporal.summary_path)
        by_name = {Path(r['path']).name: r for r in source['sources']}
        for key, name in [('candidate_frames', resolved.metrics_name), ('movement_state', resolved.movement_artifact_name), ('protocol', 'stimulus_events.parquet')]:
            if metadata['inputs'][key]['sha256'] != by_name[name]['sha256']:
                raise ValueError('Temporal upstream mismatch: ' + rid)
        original = pd.read_parquet(ATLAS / 'fish' / (rid + '.parquet'))
        saved = trial.loc[trial.fish_id.eq(rid)].reset_index(drop=True)
        pd.testing.assert_frame_equal(saved[original.columns], original.reset_index(drop=True), check_dtype=False)
        bins = saved.loc[saved.trial_number.isin(catches), ['fish_id', 'condition_id', 'trial_number', 'time_s', 'coverage', 'signed_log_vigor', 'movement_probability']].copy()
        bins = bins.rename(columns={'signed_log_vigor': 'signed', 'movement_probability': 'movement'})
        bins['display_group'] = fish.display_group
        if bins.duplicated(['trial_number', 'time_s']).any() or len(bins) != len(catches)*80:
            raise ValueError('Catch identity/grid mismatch: ' + rid)
        if bins.loc[bins.coverage < .9, ['signed', 'movement']].notna().any().any():
            raise ValueError('Insufficient coverage has finite outcome')
        # Independent numerical reproduction for the first fish, using the exact
        # supported reader and original corrected data rather than saved signed columns.
        if number == 1:
            from analyze_figures34_3strace import read_processed
            check, _, _, _ = read_processed(rid)
            check = check.loc[check.trial_number.isin(catches)].sort_values(['trial_number', 'time_s'])
            actual = bins.sort_values(['trial_number', 'time_s'])
            np.testing.assert_allclose(actual.signed, check.signed_log_vigor, atol=1e-12, equal_nan=True)
            np.testing.assert_allclose(actual.movement, check.movement_probability, atol=1e-12, equal_nan=True)
        bins.to_parquet(dest, index=False)
        inputs = [*source['sources'], source['output'], ref(ATLAS / 'fish' / (rid + '.json')),
                  ref(temporal.summary_path), ref(temporal.marker_path)]
        save(report, {'output': ref(dest), 'inputs': inputs, 'expected_us': timing,
                     'cs_duration_range_s': [float(durations.min()), float(durations.max())],
                     'catches': catches, 'equivalence_to_supported_reader': number == 1})
        parts.append(bins)
        print(f'Authenticated original sources and catches {number}/{len(members)}: {rid}', flush=True)
    trials = pd.concat(parts, ignore_index=True)
    fish = trials.groupby(['fish_id', 'condition_id', 'display_group', 'time_s'], sort=False).agg(
        signed=('signed', 'median'), signed_trials=('signed', 'count'),
        movement=('movement', 'median'), movement_trials=('movement', 'count')).reset_index()
    trials.to_parquet(OUT / 'trial-bins.parquet', index=False)
    fish.to_parquet(OUT / 'fish-bins.parquet', index=False)
    group_support = fish.groupby(['display_group', 'time_s']).agg(signed_fish=('signed', 'count'),
        movement_fish=('movement', 'count'), signed_trials=('signed_trials', 'sum'), movement_trials=('movement_trials', 'sum')).reset_index()
    group_support.to_csv(OUT / 'group-support.csv', index=False)
    guards()
    save(OUT / 'data-complete.json', {'artifacts': [ref(OUT / f) for f in ('membership.csv', 'trial-bins.parquet', 'fish-bins.parquet', 'group-support.csv')],
        'fish_reports': [ref(cache / (rid + '.json')) for rid in members.fish_id], 'roster': len(members),
        'modeled_groups': {g: int((members.display_group == g).sum()) for g in GROUPS},
        'unclassified_trace': 50, 'all_unclassified_labels': int(members.classifier_label.eq('Unclassified').sum()),
        'catches': catches, 'expected_us_s': expected_us, 'cs_duration_s': spec.cs_duration_s,
        'source_atlas': ref(ATLAS / 'artifact-manifest.json'),
        'source_project': str(PROJECT), 'source_recipe': 'original unversioned corrected metrics/movement/temporal, authenticated against saved 3sTrace source records'})


def model(repetitions):
    guards()
    model_source = OUT / ('model-build-source-' + sha256_file(Path(__file__))[:12] + '.py')
    if model_source.exists():
        verify({'path': str(model_source), 'sha256': sha256_file(Path(__file__))})
    else:
        with model_source.open('xb') as stream:
            stream.write(Path(__file__).read_bytes())
    for r in load(OUT / "data-complete.json")["artifacts"]:
        verify(r)
    # Bind the independently authenticated classifier dependencies as well as
    # its saved table. Missing/new labels are never estimated by this pilot.
    membership()
    prior = load(ATLAS / 'figure4-classified-preliminary/analysis.json')
    label_meta = load(prior['classifier_manifest']['path'])
    comparison = load(label_meta['source_comparison'])
    classification_inputs = [prior['classifier_table'], prior['classifier_manifest'],
        {'path': label_meta['source_comparison'], 'sha256': label_meta['source_comparison_sha256']}]
    classification_inputs += [{'path': comparison[k], 'sha256': comparison[h]} for k, h in
        [('cohort_path', 'cohort_sha256'), ('outcomes_path', 'outcomes_sha256')]]
    assessment = label_meta['selection_assessments']['all3sTrace']
    classification_inputs.append({'path': assessment['path'], 'sha256': assessment['sha256']})
    assessment_summary = load(assessment['path'])
    assessment_files = {'technical': 'technical-assessment.parquet', 'exploratory': 'exploratory-assessment.parquet',
        'rules': 'legacy-rule-results.parquet', 'flow': 'discarding-flow.parquet', 'details': 'rule-details.parquet'}
    classification_inputs += [{'path': str(Path(assessment['path']).parent / name),
        'sha256': assessment_summary['artifacts'][key]} for key, name in assessment_files.items()]
    for record in classification_inputs: verify(record)
    old = next(r for r in prior['inputs'] if r['path'].endswith('figure4\\analysis.json'))
    verify(old)
    current_sources = {r['path']: r['sha256'] for row in load(ATLAS / 'source-authentication.json') for r in row['sources']}
    for record in load(old['path'])['inputs']:
        if record['path'] in current_sources and current_sources[record['path']] != record['sha256']:
            raise ValueError('Saved classifier/current profile source mismatch: ' + record['path'])
    fish = pd.read_parquet(OUT / "fish-bins.parquet")
    data = model_data(fish)
    times = np.sort(data.time_s.unique())
    # Illustrations are selected deterministically by median observed-bin
    # support within each experimental group, independent of peak size/timing.
    illustrative = {}
    for group in GROUPS[1:]:
        supported = data.loc[data.display_group.eq(group)].groupby('fish_id').size().sort_values(kind='stable')
        illustrative[group] = str(supported.index[len(supported) // 2])
    result, grid, curves, draws, diagnostics = fit_bootstrap(data, times, repetitions=repetitions,
        progress=lambda n, ok: print(f"Whole-fish LMM refits {n}/{repetitions}: {ok} accepted", flush=True))
    pre_us = (times >= 0) & (times < 13)
    contrast = differences(curves)
    cd = np.asarray([differences(d) for d in draws])
    sd = cd[:, :, pre_us].std(axis=0, ddof=1)
    standardized = np.abs(cd[:, :, pre_us] - contrast[None, :, pre_us]) / np.maximum(sd, 1e-10)
    critical = float(np.quantile(standardized.max(axis=(1, 2)), .95))
    amplitude, latency = peaks(contrast[:, pre_us], times[pre_us])
    ampd, latd = peaks(cd[:, :, pre_us], times[pre_us])
    predictions = individual_predictions(result, data, times, diagnostics["random"],
        next(x["residual_variance"] for x in reversed(diagnostics["full_fit_attempts"]) if x["diagnostic_status"] == "ok"))
    prediction_rows = [{"fish_id": rid, "time_s": float(t), "fitted": float(v)}
                       for rid, values in predictions.items() for t, v in zip(times, values)]
    pd.DataFrame(prediction_rows).to_parquet(OUT / "individual-predictions.parquet", index=False)
    np.savez_compressed(OUT / "model-arrays.npz", times=times, group_curves=curves, bootstrap_curves=draws,
        contrasts=contrast, bootstrap_contrasts=cd, contrast_sd=sd, contrast_critical=critical,
        peak_amplitude=amplitude, peak_latency=latency, bootstrap_peak_amplitude=ampd, bootstrap_peak_latency=latd)
    peak_rows = []
    support = pd.read_csv(OUT / 'group-support.csv')
    source_reports = [load(p) for p in (OUT / 'fish').glob('*.json')]
    measured_us = [r['expected_us'][0] for r in source_reports if r['expected_us'] is not None]
    for i, name in enumerate(PAIR_NAMES):
        aq, tq = np.quantile(ampd[:, i], [.025, .975]), np.quantile(latd[:, i], [.025, .975])
        peak_rows.append({"contrast": name, "maximum": float(amplitude[i]), "grid_peak_s": float(latency[i]),
            "amplitude_range": aq.tolist(), "timing_range_s": tq.tolist(),
            "bootstrap_fraction_nonpositive_maximum": float(np.mean(ampd[:, i] <= 0)),
            "bootstrap_fraction_boundary_maximum": float(np.mean((latd[:, i] == times[pre_us][0]) | (latd[:, i] == times[pre_us][-1]))),
            "original_maximum_at_boundary": bool(latency[i] in (times[pre_us][0], times[pre_us][-1])),
            "exact_grid_ties": int(np.sum(contrast[i, pre_us] == amplitude[i])),
            "bootstrap_peak_grid_distribution": {str(t): int(np.sum(latd[:, i] == t)) for t in times[pre_us]},
            "fish_support_at_maximum": {GROUPS[k]: int(support.loc[support.display_group.eq(GROUPS[k]) & support.time_s.eq(latency[i]), 'signed_fish'].iloc[0]) for k in CONTRASTS[i]},
            "minimum_pre_us_fish_support": {GROUPS[k]: int(support.loc[support.display_group.eq(GROUPS[k]) & support.time_s.between(0, 13, inclusive='left'), 'signed_fish'].min()) for k in CONTRASTS[i]},
            "status": "Descriptive fitted extremum; timing not validated or biologically localized"})
    summary = {"scope": "3sTrace-only exploratory Figure 4 model pilot", "metric": METRIC,
        "profile": "0.5 s mean of bout-median log-vigor values, centered by trial pre [-20,0) median; valid coverage >=0.9",
        "aggregation": "Median over five configured catches within fish; Gaussian spline LMM of fish medians",
        "membership": "Saved legacy-wip exploratory flags, fixed for this retrospective pilot; all controls unsplit",
        "classifier_execution_id": "legacy-wip-3strace-distal-window13-59fish",
        "classification_inputs": classification_inputs + [old],
        "illustrative_fish": illustrative, "illustration_selection": "Median observed-bin support within each experimental stratum; no peak/size selection",
        "observed_model_bins": len(data), "modeled_fish": int(data.fish_id.nunique()),
        "all_unclassified_labels_in_inventory": 122,
        "source_project": str(PROJECT),
        "timing_evidence": {'trace_fish_checked': len(measured_us),
            'recorded_paired_us_median_range_s': [min(measured_us), max(measured_us)],
            'paired_trial_counts': sorted({r['expected_us'][1] for r in source_reports if r['expected_us'] is not None}),
            'recorded_cs_duration_range_s': [min(r['cs_duration_range_s'][0] for r in source_reports), max(r['cs_duration_range_s'][1] for r in source_reports)]},
        "differences_from_delay": "CS remains 0–10 s; expected US 13 rather than 9 s; fit −5 to 16 rather than −5 to 12 s; maxima searched at 26 rather than 18 bin centers; same df=6 basis, seed and covariance/refit stability rule; assay-specific saved WIP roster/flags and original corrected recipes",
        "counts": load(OUT / "data-complete.json")["modeled_groups"], "all_roster_fish": 179,
        "unclassified_trace_not_in_modeled_groups": 50, "cs_s": [0, 10], "expected_us_s": 13,
        "fit_window_s": [-5, 16], "peak_search_bin_centers_s": times[pre_us].tolist(),
        "model": "Group-specific degree-3 B-splines, df=6; Gaussian fish intercept/time slope if supported; ML",
        "diagnostics": diagnostics, "residual_excess_kurtosis": float(kurtosis(result.resid, fisher=True)),
        "nominal_simultaneous_family": "Three pairwise contrasts across 26 half-second pre-US bin centers",
        "nominal_simultaneous_critical": critical, "peaks": peak_rows,
        "movement_panel": "Observed equal-fish mean of within-fish catch medians; whole-fish percentile bootstrap, no Gaussian movement LMM",
        "limitations": diagnostics["limitations"] + ["Source groups use acquisition plus recovery, overlapping evaluation trials",
            "No p-values or validated CR peak/onset claim; all bands and extrema are exploratory",
            "Controls have unpaired US events; source cohort and protocol flags retained",
            "Missing conditional intensity is not zero activity; no-bout prediction is conditional, not reconstructed movement"],
        "implementation": [ref(model_source), ref(ROOT / "src/classical_conditioning/analysis/review/figure4_assay_profiles.py"),
                           ref(ROOT / "src/classical_conditioning/analysis/review/figure4_profiles.py"),
                           ref(ROOT / "src/classical_conditioning/config/experiments.py"),
                           ref(ROOT / "src/classical_conditioning/analysis/inference/panel_g_fast_lmm.py")],
        "artifacts": [ref(OUT / n) for n in ("model-arrays.npz", "individual-predictions.parquet")]}
    save(OUT / "analysis-summary.json", summary)
    guards()
    print(json.dumps({"accepted": diagnostics["bootstrap_accepted"], "peaks": peak_rows}), flush=True)


def render():
    # The existing shared review-page module has archived Figure 1 imports.
    # Resolve its recorded archive on this host; no Figure 1 build is invoked.
    os.environ.setdefault("CLASSICAL_CONDITIONING_ARCHIVE_ROOT",
        r"J:\ClassicalConditioning-Archive\repository-cleanup\2026-10-10")
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from classical_conditioning.figures.theme import apply_theme, style_axes
    from classical_conditioning.figures.presentation_profiles import load_presentation_profile
    from classical_conditioning.figures.layout_frames import embed_spec, load_policy
    from build_figure_legend_review import svg_editing_page
    guards()
    summary = load(OUT / "analysis-summary.json")
    for r in summary["implementation"] + summary["artifacts"]:
        verify(r)
    arrays = np.load(OUT / "model-arrays.npz")
    fish = pd.read_parquet(OUT / "fish-bins.parquet")
    individual = pd.read_parquet(OUT / "individual-predictions.parquet")
    times = arrays["times"]; curves = arrays["group_curves"]; draws = arrays["bootstrap_curves"]
    mask = (times >= 0) & (times < 13)
    profile = load_presentation_profile("nature-main")
    width, height, key_height = 183., 135., 9.
    cellw, cellh = width / 3, (height - key_height) / 2
    roles = profile.contract["styles"]
    registry = {"scientific_status": "Exploratory; no learner or timing validation", "sources":
        [ref(OUT / p) for p in ("analysis-summary.json", "model-arrays.npz", "fish-bins.parquet", "membership.csv")],
        "presentation_profile": profile.inputs, "elements": {}}
    with mpl.rc_context():
        theme = apply_theme(profile=profile)
        mpl.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "figure4-3strace-model-draft", "figure.autolayout": False,
            "figure.constrained_layout.use": False, "savefig.bbox": None, "savefig.pad_inches": 0})
        fig = plt.figure(figsize=(width / 25.4, height / 25.4))
        allocations = [{"id": "key", "kind": "shared", "box_mm": [0, 0, width, key_height], "group_ids": ["condition-key"]}]
        axes = []
        titles = ("Pooled catch response", "Suppression contrasts", "Fitted maximum", "Movement occurrence", "Illustrative fish fits", "Timing of fitted maximum")
        for i, title in enumerate(titles):
            row, col = divmod(i, 3); x, y = col * cellw, key_height + row * cellh
            ax = fig.add_axes([(x + 15) / width, 1 - (y + cellh - 12) / height, (cellw - 19) / width, (cellh - 24) / height])
            letter = chr(97 + i); ax.set_gid("panel-" + letter + "-axes")
            ax.set_title(title, pad=8, fontsize=theme.axes_titlesize)
            fig.text((x + 2) / width, 1 - (y + 5) / height, letter, fontsize=roles["panel_letter"]["font_size_pt"], fontweight="bold", gid="panel-" + letter + "-letter")
            allocations.append({"id": letter, "kind": "panel", "box_mm": [x, y, cellw, cellh],
                "group_ids": ["panel-" + letter + "-axes", "panel-" + letter + "-letter"]})
            axes.append(ax)
        def time_axis(ax, ylabel, pre_only=False):
            ax.axhline(0, color="#999999", lw=.35, zorder=0)
            ax.axvline(0, color=theme.cs_color, lw=theme.cs_linewidth, zorder=0)
            ax.axvline(10, color=theme.cs_color, ls='--', lw=theme.cs_linewidth, zorder=0)
            ax.axvline(13, color=theme.us_color, ls=":", lw=theme.us_linewidth, zorder=0)
            ax.set_xlim(0 if pre_only else -5, 13 if pre_only else 16)
            ax.set_xticks([0, 5, 10, 13] if pre_only else [-5, 0, 5, 10, 15])
            style_axes(ax, theme=theme, xlabel="Time from CS (s)", ylabel=ylabel)
        lines = []
        lo, hi = np.quantile(draws, [.025, .975], axis=0)
        for i, group in enumerate(GROUPS):
            line, = axes[0].plot(times, curves[i], color=COLORS[i], label=group,
                lw=theme.lines_linewidth, ls="--" if i == 0 else "-", gid="response-" + str(i))
            axes[0].fill_between(times, lo[i], hi[i], color=COLORS[i], alpha=.12, linewidth=0, gid="response-band-" + str(i))
            observed = fish.loc[fish.display_group.eq(group)].groupby("time_s").signed.mean().reindex(times)
            axes[0].plot(times, observed, ".", color=COLORS[i], ms=1.8, alpha=.55, gid="observed-" + str(i))
            lines.append(line)
            registry["elements"]["response-" + str(i)] = {"role": "curve", "quantity": summary["profile"], "group": group, "estimator": summary["model"]}
        time_axis(axes[0], "Signed bout ln vigor")
        pair_colors = (COLORS[1], COLORS[2], "#555555")
        c = arrays["contrasts"][:, mask]; sd = arrays["contrast_sd"]; critical = float(arrays["contrast_critical"])
        for i in range(3):
            axes[1].plot(times[mask], c[i], color=pair_colors[i], lw=theme.lines_linewidth, ls=("-", "--", ":")[i], gid="contrast-" + str(i))
            axes[1].fill_between(times[mask], c[i] - critical * sd[i], c[i] + critical * sd[i], color=pair_colors[i], alpha=.10, linewidth=0)
            registry["elements"]["contrast-" + str(i)] = {"role": "condition_contrast", "groups": [GROUPS[k] for k in CONTRASTS[i]], "family": summary["nominal_simultaneous_family"]}
        time_axis(axes[1], "Difference (ln vigor)", pre_only=True)
        for j, (ax, field, range_key, xlabel) in enumerate(((axes[2], "maximum", "amplitude_range", "Maximum difference (ln vigor)"),
                                                        (axes[5], "grid_peak_s", "timing_range_s", "Time from CS (s)"))):
            for i, p in enumerate(summary["peaks"]):
                low, high = p[range_key]
                ax.plot([low, high], [2 - i] * 2, color=pair_colors[i], lw=theme.lines_linewidth)
                ax.plot(p[field], 2 - i, "o", ms=3, color=pair_colors[i])
            ax.set_yticks([2, 1, 0], PAIR_NAMES); ax.set_ylim(-.65, 2.65)
            if j:
                ax.set_xlim(0, 13); ax.set_xticks([0, 5, 10, 13])
            else:
                ax.axvline(0, color="#999999", lw=.35)
            style_axes(ax, theme=theme, xlabel=xlabel)
        rng = np.random.default_rng(20261022)
        for i, group in enumerate(GROUPS):
            matrix = fish.loc[fish.display_group.eq(group)].pivot(index="fish_id", columns="time_s", values="movement").reindex(columns=times).to_numpy()
            movement_draws = np.asarray([np.nanmean(matrix[rng.choice(len(matrix), len(matrix), replace=True)], axis=0) for _ in range(999)])
            mean = np.nanmean(matrix, axis=0); a, b = np.nanquantile(movement_draws, [.025, .975], axis=0)
            axes[3].plot(times, mean, color=COLORS[i], lw=theme.lines_linewidth, ls="--" if i == 0 else "-")
            axes[3].fill_between(times, a, b, color=COLORS[i], alpha=.12, linewidth=0)
        time_axis(axes[3], "Moving fraction")
        axes[3].set_ylim(bottom=0)
        for rid, color, marker in ((summary["illustrative_fish"][GROUPS[1]], COLORS[1], "o"), (summary["illustrative_fish"][GROUPS[2]], COLORS[2], "s")):
            obs = fish.loc[fish.fish_id.eq(rid)].set_index("time_s").reindex(times)
            pred = individual.loc[individual.fish_id.eq(rid)].set_index("time_s").reindex(times)
            axes[4].plot(times, pred.fitted, color=color, lw=theme.lines_linewidth)
            axes[4].plot(times, obs.signed, marker, color=color, ms=2, markeredgewidth=.35, markerfacecolor="white", ls="none")
        time_axis(axes[4], "Signed bout ln vigor")
        key = fig.legend(lines, GROUPS, loc="upper right", bbox_to_anchor=(1 - 2 / width, 1 - 1 / height),
                         ncol=3, frameon=False, fontsize=theme.legend_fontsize, handlelength=2)
        key.set_gid("condition-key")
        stream = io.BytesIO(); fig.savefig(stream, format="svg", metadata={"Date": None}); plt.close(fig)
        root = ET.fromstring(stream.getvalue())
        root.set("width", f"{width:g}mm")
        root.set("height", f"{height:g}mm")
        clip_ids = [n.get("id") for n in root.iter(SVG + "clipPath")]
        spec = {"scope": "figure", "final_size_mm": [width, height], "inset_mm": load_policy()[0]["selected_inset_mm"],
            "allocations": allocations, "background_ids": ["patch_1"],
            "scientific_clips": [{"id": k, "evidence": "Explicit data-axis clipping rectangle of the named model-profile panel"} for k in clip_ids]}
        svg = embed_spec(ET.tostring(root), spec)
    legend = ("Figure 4 3sTrace | Exploratory CR profile model draft. "
        "Saved legacy-WIP groups: 16 learner-flagged (L), 23 nonlearner-flagged (N), all 90 matched controls; "
        "50 unclassified trace fish remain in the 179-fish inventory. Controls are pooled regardless of label; "
        "122 fish overall lack a usable saved classification. Groups are retrospective, not validated learner classes. "
        "Catches 25, 39, 53, 59 and 65 are pooled by median within fish. Conditional signed bout-ln-vigor uses "
        "0.5-s bins, each trial's [−20,0) s baseline and coverage ≥0.9; no-bout intensity stays unavailable. "
        "a, Group spline-LMM means (lines), observed across-fish means (dots), pointwise whole-fish bootstrap ranges (shading). "
        "b, Direct contrasts: Control−L (orange), Control−N (gray dashed), N−L (dark gray dotted); nominal joint 95% bands cover three contrasts at 26 pre-US bin centers. "
        "c,f, Fitted-contrast maxima and times on 0.25–12.75 s; bars are bootstrap 2.5–97.5% ranges. "
        "These are unvalidated extrema summaries, not localized CR-peak confidence intervals. "
        "d, Observed mean moving fraction with pointwise fish-bootstrap ranges. "
        f"e, Conditional fits and observed catch medians for illustrative fish {summary['illustrative_fish'][GROUPS[1]]} (circles) and "
        f"{summary['illustrative_fish'][GROUPS[2]]} (squares), selected by median bin support; fitted gaps do not reconstruct movement. "
        "Green marks CS onset (solid) and end at 10 s (dashed); violet marks expected US at 13 s (omitted in 3sTrace catches). "
        f"The fixed-basis Gaussian ML model uses {'fish intercepts' if summary['diagnostics']['random'] == '1' else 'fish intercepts and time slopes'}; {summary['diagnostics']['bootstrap_accepted']} "
        "whole-fish refits passed. Labels overlap evaluation data; no independent learner comparison or timing claim is made. "
        "Controls have unpaired reinforcers. LME/GAMM comparison, informative availability and calibration remain open.")
    registry["remaining_coverage"] = "Named scientific curves/contrasts; no full freeze-role inventory or production certification"
    files = {"figure4-3strace.svg": svg, "legend.md": (legend + "\n").encode(),
        "figure4-3strace.html": svg_editing_page(svg, scope_label="Figure 4 3sTrace exploratory model draft", legend=legend).encode(),
        "semantic-registry.json": (json.dumps(registry, indent=2) + "\n").encode()}
    with state_lock(OUT / "render"):
        if (OUT / "render-manifest.json").exists():
            prior = load(OUT / "render-manifest.json")
            for r in prior["artifacts"]:
                verify(r)
            history = OUT / "history" / prior["artifacts"][0]["sha256"][:12]
            history.mkdir(parents=True, exist_ok=True)
            for r in prior["artifacts"]:
                p = Path(r["path"]); target = history / p.name
                if not target.exists():
                    target.write_bytes(p.read_bytes())
        for name, content in files.items():
            (OUT / name).write_bytes(content)
        meta = {"artifacts": [ref(OUT / p) for p in files], "implementation": ref(Path(__file__)),
                "profile": profile.inputs, "legend_words": len(legend.split())}
        (OUT / "render-manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    guards()
    print(json.dumps({"svg": str(OUT / "figure4-3strace.svg"), "legend_words": len(legend.split())}))


def publish(note):
    from classical_conditioning.figures.layout_frames import assess
    guards()
    rendered = load(OUT / "render-manifest.json")
    for r in rendered["artifacts"]:
        verify(r)
    svg = (OUT / "figure4-3strace.svg").read_bytes()
    browser = load(OUT / "layout-browser.json")
    assessment = assess(svg, browser)
    if assessment["status"] != "passed":
        raise ValueError("Exact rendered layout unresolved: " + str(assessment["issues"]))
    # Reauthenticate all consumed original numerical sources before publication.
    bound = list(rendered["artifacts"])
    reports = sorted((OUT / "fish").glob("*.json"))
    if len(reports) != load(OUT / 'data-complete.json')['roster']:
        raise ValueError('Original-source authentication is incomplete')
    for index, p in enumerate(reports, 1):
        report = load(p)
        verify(report["output"])
        bound += [ref(p), report["output"], *report["inputs"]]
    bound += load(OUT / "data-complete.json")["artifacts"]
    bound += load(OUT / "analysis-summary.json")["implementation"]
    bound += load(OUT / "analysis-summary.json")["classification_inputs"]
    bound += load(OUT / "analysis-summary.json")["artifacts"]
    bound += [ref(OUT / n) for n in ("analysis-summary.json", "data-complete.json", "render-manifest.json", "layout-browser.json")]
    bound += [rendered["implementation"]]
    # Original checks completed at extraction; publish_state rehashes the
    # complete original-source map under lock. Authenticate other outputs now.
    original_paths = {r['path'] for p in reports for r in load(p)['inputs']}
    for r in bound:
        if r['path'] not in original_paths:
            verify(r)
    evidence = OUT / "layout-evidence.json"
    review = OUT / "visual-review.json"
    for destination, record in [(evidence, assessment), (review,
            {"artifact": ref(OUT / "figure4-3strace.svg"), "note": note, "scope": "3sTrace only", "performed": True})]:
        if destination.exists():
            if load(destination) != record:
                raise ValueError('Existing review evidence differs; preserve and reconcile: ' + str(destination))
        else:
            save(destination, record)
    ticket = load(TICKET)
    expected = ticket["protected"] + ticket["inputs"] + bound + [ref(evidence), ref(review)]
    # Use the captured expected bytes, without a redundant fresh snapshot of
    # all large originals. publish_state checks this complete map under lock.
    # Large immutable Parquet inputs get a fresh hash in this locked attempt;
    # subsequent redundant helper checks verify their captured file states.
    snapshot = PublicationSnapshot([p for p in original_paths if Path(p).suffix == '.parquet'])
    for r in expected:
        snapshot.hashes[str(Path(r["path"]).resolve())] = r["sha256"]
    state = {"schema_version": 1, "figure_id": "fig4", "scope": "3sTrace exploratory model draft", "whole_figure_frozen": False,
        "default_display": "draft", "default_display_format": "html-svg-and-legend",
        "scientific_selection": {"status": "Unfrozen exploratory pilot; saved legacy groups, no method approval", "records": []},
        "draft": {"working_svg": ref(OUT / "figure4-3strace.svg"), "review": ref(OUT / "figure4-3strace.html"),
                  "legend": ref(OUT / "legend.md"), "semantic_registry": ref(OUT / "semantic-registry.json"),
                  "analysis": ref(OUT / "analysis-summary.json"), "layout_frames": ref(evidence), "visual_review": ref(review)},
        "start_ticket": ref(TICKET), "limitations": load(OUT / "analysis-summary.json")["limitations"]}
    publish_state(STATE, state, snapshot)
    print(json.dumps({"state": str(STATE), "review": state["draft"]["review"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "model", "render", "publish"))
    parser.add_argument("--bootstrap", type=int, default=399)
    parser.add_argument("--visual-review-note")
    args = parser.parse_args()
    if args.stage == "extract": extract()
    elif args.stage == "model": model(args.bootstrap)
    elif args.stage == "render": render()
    elif not args.visual_review_note: parser.error("Actual visual-review note required")
    else: publish(args.visual_review_note)
