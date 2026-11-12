"""Session fixtures independently encode the author-confirmed acquisition order."""
import json
from pathlib import Path
import unittest
import warnings
from unittest.mock import patch

import pandas as pd

from classical_conditioning.config import Alignment, get_experiment_spec
from classical_conditioning.config.experiments import resolve_condition
from classical_conditioning.config.trial_map import require_current_trial_map, trial_map_sha256
from classical_conditioning.exceptions import ArtifactIntegrityError, SchemaValidationError
from classical_conditioning.ingestion.protocol_verification import evaluate_protocol, verify_recording_protocol


def session(latency=13000, *, control=False):
    base = 1_700_000_000_000
    events = []
    for trial in range(1, 95):
        onset = base + (trial - 1) * 100000
        if control and 15 <= trial <= 64:
            onset += 40000 + (trial % 7) * 1000
        events.append(("Cycle", onset, onset + 10000))
    short = ([1000 + i * 20000 for i in range(15)]
             + [450000, 750000, 1050000])
    for onset in short:
        events.append(("Reinforcer", base + onset, base + onset + 50))
    for trial in range(15, 65):
        if trial not in {25, 39, 53, 59}:
            onset = base + (trial - 1) * 100000 + latency
            events.append(("Reinforcer", onset, onset + 100))
    for i in range(14):
        onset = base + 6450000 + i * 200000
        events.append(("Reinforcer", onset, onset + 50))
    events.append(("Reinforcer", base + 9400000, base + 9400500))
    return pd.DataFrame(events, columns=["Type", "Beg", "End"]).sort_values("Beg").reset_index(drop=True)


def audit(frame=None, token="3sTrace", experiment="all3sTrace", **kwargs):
    return evaluate_protocol(session() if frame is None else frame,
                             f"20260101_01_{token}_black-1_test_6dpf",
                             experiment_name=experiment, **kwargs)


def codes(report):
    return {f["code"] for f in report["findings"]}


class ProtocolVerificationTests(unittest.TestCase):
    def test_exact_totals_are_always_confirmed_for_every_assay_and_control(self):
        from classical_conditioning.config.export import _json_value
        from classical_conditioning.config.trial_map import get_experiment_trial_map
        for exp, token, latency in (("allDelay", "delay", 9000),
                                   ("all3sTrace", "3sTrace", 13000),
                                   ("all10sTrace", "10sTrace", 20000)):
            config = _json_value(get_experiment_spec(exp))
            self.assertEqual(config["expected_cs_count"], 94)
            self.assertEqual(config["expected_us_count"], 79)
            self.assertNotIn("minimum_cs_trials", config)
            self.assertNotIn("minimum_us_trials", get_experiment_trial_map(exp))
            for control in (False, True):
                name = "control" if control else token
                frame = session(latency, control=control)
                for kind, expected in (("CS", 94), ("US", 79)):
                    event_type = "Cycle" if kind == "CS" else "Reinforcer"
                    row = frame.loc[frame.Type.eq(event_type)].iloc[[-1]]
                    complete = audit(frame, name, exp)
                    self.assertEqual(complete["count_checks"][kind],
                                     {"expected": expected, "observed": expected, "matches": True})
                    for changed, observed in ((frame.drop(row.index), expected - 1),
                            (pd.concat([frame, row], ignore_index=True), expected + 1)):
                        report = audit(changed, name, exp)
                        self.assertEqual(report["count_checks"][kind],
                                         {"expected": expected, "observed": observed, "matches": False})
                        self.assertIn(kind.lower() + "_count", codes(report))
                        self.assertEqual(report["status"], "warn")

    def test_all_assays_and_controls(self):
        for exp, token, latency, name in (("allDelay", "delay", 9000, "Delay"),
                ("all3sTrace", "3sTrace", 13000, "3sTrace"),
                ("all10sTrace", "10sTrace", 20000, "10sTrace")):
            with self.subTest(exp=exp):
                result = audit(session(latency), token, exp)
                self.assertEqual(result["status"], "pass", result["findings"])
                self.assertEqual(result["observed_schedule"], name)
                self.assertEqual(result["counts"], {"CS": 94, "US": 79})
                control = audit(session(latency, control=True), "control", exp)
                self.assertEqual(control["status"], "pass", control["findings"])
                self.assertEqual(control["observed_schedule"], "variable_or_unpaired")

    def test_every_filename_and_timing_conflict(self):
        self.assertIn("paired_timing_mismatch", codes(audit(session(9000))))
        self.assertIn("filename_experiment_mismatch", codes(audit(token="delay")))
        self.assertIn("control_has_fixed_pairing", codes(audit(token="control")))
        self.assertIn("filename_timing_mismatch", codes(audit(session(9000), "trace", None)))
        result = audit(experiment=None, token="trace")
        self.assertEqual(result["status"], "not_checkable")
        self.assertIsNone(result["resolved_experiment"])

    def test_tolerance_boundaries_and_single_bad_pair(self):
        self.assertEqual(audit(session(13200))["status"], "pass")
        self.assertEqual(audit(session(12800))["status"], "pass")
        self.assertEqual(audit(session(20200), "10sTrace", "all10sTrace")["status"], "pass")
        frame = session()
        idx = frame.index[frame.Type.eq("Reinforcer")][18 + 10]
        frame.loc[idx, ["Beg", "End"]] += 201
        self.assertIn("paired_timing_mismatch", codes(audit(frame)))

    def test_catches_extra_missing_and_pulse_durations(self):
        frame = session()
        base = frame.Beg.min()
        catch_onset = frame.loc[frame.Type.eq("Cycle"), "Beg"].iloc[24]
        extra = pd.DataFrame([("Reinforcer", catch_onset + 13000, catch_onset + 13100)], columns=frame.columns)
        result = audit(pd.concat([frame, extra], ignore_index=True))
        self.assertIn("us_at_catch", codes(result))
        self.assertIn("us_count", codes(result))
        idx = frame.index[frame.Type.eq("Reinforcer")][18]
        self.assertIn("training_us_sequence", codes(audit(frame.drop(idx))))
        frame.loc[idx, "End"] += 400
        self.assertIn("us_pulse_duration", codes(audit(frame)))
        # Maintenance pulses near pre/test CSs are valid; no blanket no-US rule.
        self.assertNotIn("long_us_at_nontraining_cs", codes(audit()))

    def test_control_coincidences_are_descriptive(self):
        frame = session(control=True)
        idx = frame.index[frame.Type.eq("Cycle")][14]
        us_onset = frame.loc[frame.Type.eq("Reinforcer"), "Beg"].iloc[18]
        frame.loc[idx, "Beg"] = us_onset - 13000
        frame.loc[idx, "End"] = us_onset - 3000
        result = audit(frame, "control")
        self.assertEqual(result["status"], "pass", result["findings"])
        self.assertEqual(result["pairing_counts"]["3sTrace"], 1)

    def test_variable_trace_is_not_accepted_as_fixed_trace_or_unpaired_control(self):
        frame = session()
        ids = frame.index[frame.Type.eq("Reinforcer")][18:64]
        for i, idx in enumerate(ids):
            frame.loc[idx, ["Beg", "End"]] += round(i * 2500 / 45) - 2500
        self.assertIn("paired_timing_mismatch", codes(audit(frame)))
        self.assertIn("control_all_training_cs_have_unique_nearby_us", codes(audit(frame, "control")))

    def test_integrity_missingness_order_and_coverage(self):
        frame = session()
        self.assertEqual(audit(frame.iloc[::-1])["status"], "pass")
        result = audit(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
        self.assertIn("duplicate_onset", codes(result))
        self.assertIn("cs_count", codes(audit(frame.iloc[:40])))
        frame.loc[0, "End"] = frame.loc[0, "Beg"]
        self.assertIn("nonpositive_duration", codes(audit(frame)))
        frame.loc[0, "Type"] = "Unknown"
        self.assertIn("unknown_event_type", codes(audit(frame)))
        bounds = (int(frame.Beg.min()) + 1, int(frame.End.max()) - 1)
        self.assertIn("outside_acquisition", codes(audit(acquisition_bounds_ms=bounds)))
        frame["Beg"] = frame["Beg"].astype(float)
        frame.loc[0, "Beg"] = 1.5
        self.assertIn("invalid_timestamps", codes(audit(frame)))

    def test_training_numbering_and_aliases(self):
        expected = [list(range(a,b)) for a,b in ((19,29),(29,38),(38,47),(47,56),(56,65))]
        for exp in ("allDelay", "all3sTrace", "all10sTrace"):
            spec = get_experiment_spec(exp)
            self.assertEqual([list(v) for _,v in spec.trial_blocks(Alignment.US)], expected)
        for token in ("trace", "3sTrace", "traceNoMK801"):
            self.assertEqual(resolve_condition("all3sTrace", token), "trace")
        result = audit()
        self.assertEqual([p["us_ids"][0] for p in result["training_pairs"] if not p["catch"]], list(range(19,65)))
        us = [e for e in result["events"] if e["type"] == "US"]
        self.assertEqual(us[0]["event_number"], 1)
        self.assertEqual(len(us), 79)

    def test_stale_trial_mapping_cannot_pass(self):
        with self.assertRaises(ArtifactIntegrityError):
            require_current_trial_map({}, "allDelay")
        require_current_trial_map({"trial_map_sha256": trial_map_sha256("allDelay")}, "allDelay")

    def test_outcomes_retain_first_us_and_apply_corrected_training_labels(self):
        from tests.test_trial_outcomes import TrialOutcomeTests
        from classical_conditioning.analysis.trial_outcomes import aggregate_trial_outcomes
        fixture = TrialOutcomeTests()
        fixture.setUp()
        outcomes, coverage = aggregate_trial_outcomes(
            fixture.frames, fixture.movement, session(), identity=fixture.identity,
            experiment_name="allDelay", config=fixture.config)
        us = outcomes.loc[outcomes.alignment.eq("US")]
        self.assertEqual(us.trial_number.nunique(), 79)
        self.assertTrue(us.loc[us.trial_number.eq(18), "phase"].isna().all())
        self.assertTrue(us.loc[us.trial_number.eq(19), "block_10_name"].eq("Train 1").all())
        self.assertTrue(us.loc[us.trial_number.eq(64), "block_10_name"].eq("Train 5").all())
        self.assertTrue(us.loc[us.trial_number.eq(65), "phase"].isna().all())

    def test_session_schematic_values_match_runtime(self):
        from classical_conditioning.config.protocols import SESSION
        original = json.loads((Path(__file__).parents[1] / "configs/paper-figures/figure1-session-protocol.json").read_text())
        for old, phase in zip(original["phases"], SESSION.phases, strict=True):
            self.assertEqual(old["cs_count"], phase.cs_count)
            self.assertEqual(old.get("short_us_count", old.get("paired_long_us_count")), phase.us_count)
        self.assertEqual(original["final_check_us_duration_ms"], SESSION.final_check_duration_ms)


class ProtocolIntegrationTests(unittest.TestCase):
    def test_alias_filters_preserve_historical_raw_names(self):
        import tempfile
        from classical_conditioning.run_config import load_pipeline_run_config
        from classical_conditioning.pipeline import resolve_pipeline_recording_ids
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            raw.mkdir()
            config = root / "run.json"
            config.write_text(json.dumps({"raw_dir": str(raw), "save_dir": str(root / "save"),
                "experiment": "all3sTrace", "analysis_id": "aliases", "keep_conditions": ["3sTrace"]}))
            settings = load_pipeline_run_config(config)
            self.assertEqual(settings.keep_conditions, ("trace",))
            inventory = {"records": [{"recording_id": "old", "condition_id": "trace"},
                                      {"recording_id": "new", "condition_id": "3strace"},
                                      {"recording_id": "control", "condition_id": "control"}]}
            self.assertEqual(resolve_pipeline_recording_ids(settings, inventory), ("old", "new"))

    def test_population_loader_rejects_resealed_old_mapping(self):
        import tempfile
        from classical_conditioning.artifacts import sha256_file
        from classical_conditioning.analysis.inference.model_input import load_authenticated_trial_outcomes
        from tests.test_cohort_outcomes import write_authenticated_trial_outcomes, cohort_outcomes_fixture
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_authenticated_trial_outcomes(root, "recording-a", cohort_outcomes_fixture())
            path = root / "Quality checks/recording-a/candidate-trial-outcomes-corrected_summary.json"
            summary = json.loads(path.read_text())
            summary.pop("trial_map_sha256")
            path.write_text(json.dumps(summary))
            marker_path = root / "Metadata/recording-a_candidate-trial-outcomes-corrected_complete.json"
            marker = json.loads(marker_path.read_text())
            marker["summary_sha256"] = sha256_file(path)
            marker_path.write_text(json.dumps(marker))
            with self.assertRaisesRegex(ArtifactIntegrityError, "trial mapping"):
                load_authenticated_trial_outcomes(root, ["recording-a"])

    def test_fresh_evidence_for_resume_and_changed_context(self):
        from tests.test_intake import IntakeTests
        from classical_conditioning.intake import intake_recording
        fixture = IntakeTests()
        fixture.setUp()
        self.addCleanup(fixture.tearDown)
        intake_recording(fixture.raw, fixture.project)
        with warnings.catch_warnings(record=True) as emitted:
            first = verify_recording_protocol(fixture.project, "20260101_01", experiment_name="allDelay")
            second = verify_recording_protocol(fixture.project, "20260101_01", experiment_name="all3sTrace")
        self.assertTrue(emitted)
        self.assertNotIn("filename_experiment_mismatch", codes(first))
        self.assertIn("filename_experiment_mismatch", codes(second))
        self.assertTrue((fixture.project / "Quality checks/20260101_01/protocol_verification.html").is_file())
        protocol = fixture.project / "Processed data/20260101_01/stimulus_events.parquet"
        protocol.write_bytes(b"tampered")
        with self.assertRaises(ValueError):
            verify_recording_protocol(fixture.project, "20260101_01")

    def test_fractional_raw_timestamps_are_rejected(self):
        import tempfile
        from classical_conditioning.ingestion.readers import read_protocol
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stim.txt"
            path.write_text("Type Beg End\nReinforcer 1.5 3\n")
            with self.assertRaises(SchemaValidationError):
                read_protocol(path)
