from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib as mpl
import numpy as np

from classical_conditioning.analysis.protocol_timing import build_timing_evidence, read_stimulus_log, schedule_comparisons, summarize_lags
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.config.protocol_timing import default_timing_config, timing_config_from_dict
from classical_conditioning.exceptions import ArtifactIntegrityError, ConfigurationError
from classical_conditioning.ingestion.stimulus_timing import _camera_evidence, run_protocol_timing
from tests.test_protocol_verification import session


def rows(frame):
    return [{"source_row": i+1, "source_line": i+2, "raw_type": str(r.Type),
             "raw_beg": str(int(r.Beg)), "raw_end": str(int(r.End)),
             "raw_text": f"{r.Type} {int(r.Beg)} {int(r.End)}", "column_count_valid": True}
            for i, r in enumerate(frame.itertuples())]


def evidence(frame=None, *, condition="delay", experiment="allDelay", config=None, rid="fish", **kwargs):
    return build_timing_evidence(rows(session(9000) if frame is None else frame),
        config or default_timing_config(experiment), recording_id=rid, condition=condition, **kwargs)


def future_settings():
    return {"experiment_id": "future-assay", "event_types": {"CS": "Light", "US": "Pulse"},
        "phases": [{"name": "novel-training", "cs_count": 3, "us_count": 2,
                    "us_role": "training", "us_duration_ms": 100, "block_size": 2},
                   {"name": "health", "cs_count": 0, "us_count": 1, "us_role": "viability", "us_duration_ms": 500}],
        "conditions": {"novel": {"role": "conditioned", "catch_cs": [2],
            "pairings": [{"cs": 1, "us": 1, "latency_s": 2}, {"cs": 3, "us": 2, "latency_s": 4}]},
                       "control": {"role": "control"}},
        "cs_duration_ms": 1000, "target_latencies_s": [2, 4],
        "local_window_s": [-5, 10], "lag_views": [{"horizon_s": 10, "bin_s": 1}],
        "recordings": {"20260101_01": {"condition": "novel", "day": "20260101",
            "batch_id": "generator-A", "batch_provenance": "fixture generator version",
            "cs_schedule_id": "cs-A", "us_schedule_id": "us-A", "schedule_provenance": "fixture generator",
            "phase_bounds_s": {"novel-training": [0, 30], "health": [30, 40]},
            "phase_bounds_provenance": {name: {"kind":"recorded", "source":"fixture phase log"} for name in ("novel-training", "health")}}}}


class StimulusTimingEvidenceTests(unittest.TestCase):
    def test_all_assays_use_current_pairs_and_preserve_every_pulse_role(self):
        for experiment, condition, latency in [("allDelay", "delay", 9000), ("all3sTrace", "trace", 13000), ("all10sTrace", "trace", 20000)]:
            r = evidence(session(latency), experiment=experiment, condition=condition)
            self.assertEqual(len(r["events"]), 173)
            self.assertEqual(r["counts"]["US"]["expected"], 79)
            self.assertEqual(len(r["expected_pairs"]), 46)
            self.assertEqual({p["latency_ms"] for p in r["expected_pairs"]}, {latency})
            us = sorted([e for e in r["events"] if e["type"] == "US"], key=lambda e:e["event_number"])
            self.assertEqual(us[17]["phase"], "pretraining")
            self.assertEqual(us[18]["phase"], "training")
            self.assertEqual(us[63]["role"], "training")
            self.assertEqual(us[64]["role"], "maintenance")
            self.assertEqual(us[78]["role"], "viability")
            self.assertEqual([e["expected_slot"] for e in r["events"] if e["is_catch"]], [25,39,53,59,65])
            self.assertEqual(us[28]["block"], "training 2")
            control = evidence(session(latency, control=True), experiment=experiment, condition="control")
            self.assertEqual(control["expected_pairs"], [])
            self.assertFalse(any(e["is_catch"] for e in control["events"]))
            training = [lag for lag in control["lags"] if lag["phase"] == "training"]
            self.assertTrue(training)
            self.assertTrue(all(lag["anchor_count"] == 50 for lag in training))

    def test_missing_extra_or_duplicate_events_do_not_shift_expected_slots(self):
        frame = session(9000)
        cs = frame.index[frame.Type.eq("Cycle")]
        for changed in (frame.drop(cs[20]), __import__('pandas').concat([frame, frame.loc[[cs[20]]]], ignore_index=True)):
            r = evidence(changed)
            self.assertEqual(r["mapping_status"]["CS"], "unresolved")
            self.assertTrue(all(e["expected_slot"] is None for e in r["events"] if e["type"] == "CS"))
            self.assertTrue(all(p["status"] == "uncheckable" for p in r["expected_pairs"]))
            self.assertTrue(any(l["phase"] == "unresolved_recording" for l in r["lags"]))

    def test_source_order_is_preserved_without_treating_interleaving_as_corruption(self):
        frame = session(9000).iloc[::-1]
        r = evidence(frame)
        self.assertEqual(r["events"][0]["raw_beg"], str(int(frame.Beg.iloc[0])))
        self.assertEqual(r["events"][0]["source_row"], 1)
        self.assertEqual({p["latency_ms"] for p in r["expected_pairs"]}, {9000})

    def test_damaged_raw_values_remain_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"stim.txt"
            path.write_text("Type Beg End  \nCycle 1.66911E+12 1669110001000\nReinforcer broken 20\nBad 30 -2\n")
            raw = read_stimulus_log(path)
        config = timing_config_from_dict({"experiment_id": "generic", "phases": [], "conditions": {}})
        r = build_timing_evidence(raw, config, recording_id="test")
        self.assertEqual(len(r["events"]), 3)
        self.assertEqual(r["events"][0]["raw_beg"], "1.66911E+12")
        self.assertEqual(r["unplottable_source_rows"], [2])
        self.assertIn("nonpositive_duration", {f["code"] for f in r["findings"]})
        self.assertEqual(r["counts"]["US"]["observed"], 1)
        self.assertEqual(r["counts"]["US"]["usable_onsets"], 0)

    def test_all_pairs_backward_lags_and_exact_targets_are_not_nearest_neighbor_statistics(self):
        import pandas as pd
        cfg = timing_config_from_dict({"experiment_id":"simple", "phases":[{"name":"train","cs_count":2,"us_count":3,"us_role":"training"}],
            "conditions":{"control":{"role":"control"}}, "target_latencies_s":[2], "timing_tolerance_ms":0,
            "lag_views":[{"horizon_s":10,"bin_s":1}]})
        frame = pd.DataFrame([("Cycle",10000,14000),("Cycle",13000,14000),
                              ("Reinforcer",8000,8050),("Reinforcer",12000,12050),("Reinforcer",15000,15050)], columns=["Type","Beg","End"])
        r = evidence(frame, config=cfg, condition="control", acquisition_bounds_ms=(0,25000), camera_gaps_ms=[(11000,12000)])
        lag = r["lags"][0]
        self.assertEqual(sum(lag["counts"]), 6)
        self.assertEqual(lag["target_coincidences"]["2"], 2)
        self.assertEqual(len(lag["us_onsets_inside_cs"]), 1)
        self.assertEqual(sum(lag["shifted_counts"]), 6)
        np.testing.assert_allclose(np.diff(lag["shifted_edges_s"]),1)
        self.assertAlmostEqual(sum(lag["observable_exposure_s"]), 38)
        self.assertFalse(lag["phase_time_verified"])
        self.assertEqual(r["expected_pairs"], [])

    def test_pair_latency_outside_legacy_search_is_still_measured(self):
        frame = session(9000)
        idx = frame.index[frame.Type.eq("Reinforcer")][18]
        frame.loc[idx,["Beg","End"]] += 50000
        r = evidence(frame)
        self.assertEqual(r["expected_pairs"][0]["latency_ms"], 59000)
        self.assertIn("pair_latency_deviation", {f["code"] for f in r["findings"]})
        self.assertEqual(r["local_window_coverage"]["observed_pairs_outside_cs_window"],1)
        for latency in (8800,9200):
            self.assertNotIn("pair_latency_deviation",{f["code"] for f in evidence(session(latency))["findings"]})

    def test_schedule_translation_is_removed_but_drift_and_incomparability_remain(self):
        first = evidence(rid="first")
        shifted = session(9000)
        shifted[["Beg","End"]] += 123456
        second = evidence(shifted, rid="second")
        matrix = next(c for c in schedule_comparisons([first,second]) if c["phase"]=="training" and c["type"]=="CS")
        self.assertEqual(matrix["maximum_residual_ms"][0][1], 0)
        shifted[["Beg","End"]] += np.arange(len(shifted))[:,None]*10
        drift = evidence(shifted, rid="drift")
        matrix = next(c for c in schedule_comparisons([first,drift]) if c["phase"]=="training" and c["type"]=="CS")
        self.assertGreater(matrix["maximum_residual_ms"][0][1], 0)

    def test_individual_opposite_patterns_and_reuse_survive_aggregation(self):
        import pandas as pd
        config = timing_config_from_dict({"experiment_id":"reuse", "phases":[{"name":"train","cs_count":1,"us_count":1,"us_role":"training"}],
            "conditions":{"control":{"role":"control"}}, "lag_views":[{"horizon_s":10,"bin_s":1}],
            "recordings":{rid:{"cs_schedule_id":schedule,"us_schedule_id":schedule,"schedule_provenance":"known generator"} for rid,schedule in [("a","one"),("b","one"),("c","two")]}})
        reports = []
        for rid,lag in [("a",5000),("b",5000),("c",-5000)]:
            frame = pd.DataFrame([("Cycle",10000,11000),("Reinforcer",10000+lag,10100+lag)],columns=["Type","Beg","End"])
            reports.append(evidence(frame,config=config,condition="control",rid=rid))
        group = summarize_lags(reports)[0]
        self.assertEqual(group["known_intended_schedule_count"],2)
        self.assertNotEqual(group["curves"][0],group["curves"][2])
        self.assertAlmostEqual(group["equal_fish_counts_per_anchor"][15],2/3)
        self.assertEqual(group["equal_intended_schedule_counts_per_anchor"][15],.5)
        self.assertEqual(group["equal_intended_schedule_counts_per_anchor"][5],.5)

    def test_configured_bounds_do_not_claim_recorded_phase_coverage(self):
        settings = future_settings()
        settings["recordings"]["20260101_01"].pop("phase_bounds_provenance")
        config = timing_config_from_dict(settings)
        r = evidence(config=config,rid="20260101_01",condition="novel",acquisition_bounds_ms=(0,10000000))
        self.assertTrue(all(not lag["phase_time_verified"] for lag in r["lags"]))

    def test_missing_raw_event_does_not_get_a_nearest_processed_match(self):
        frame = session(9000)
        frame = frame.drop(frame.index[frame.Type.eq("Cycle")][20])
        r = evidence(frame,processed_markers=[{"type":"CS","event_number":21,"beg_ms":1234}])
        self.assertEqual(r["alignment"]["matches"][0]["status"],"unmatched_processed")

    def test_future_phases_variable_pair_latencies_and_explicit_annotations(self):
        cfg = timing_config_from_dict(future_settings())
        raw = [{"source_row": i+1,"source_line": i+2,"raw_type":t,"raw_beg":str(a),"raw_end":str(b),"raw_text":"", "column_count_valid":True} for i,(t,a,b) in enumerate([
            ("Light",0,1000),("Pulse",2000,2100),("Light",10000,11000),("Light",20000,21000),("Pulse",24000,24100),("Pulse",35000,35500)])]
        r = build_timing_evidence(raw,cfg,recording_id="20260101_01",acquisition_bounds_ms=(0,40000))
        self.assertEqual([p["latency_ms"] for p in r["expected_pairs"]],[2000,4000])
        self.assertEqual(r["batch_id"],"generator-A")
        self.assertTrue(all(l["phase_time_verified"] for l in r["lags"]))
        markers = [{"type":"CS","source_row":1,"beg_ms":2,"end_ms":998}]
        r = build_timing_evidence(raw,cfg,recording_id="20260101_01",processed_markers=markers)
        self.assertEqual(r["alignment"]["matches"][0]["beg_residual_ms"],2)
        self.assertEqual(r["alignment"]["matches"][0]["end_residual_ms"],-2)

    def test_config_rejects_ambiguous_or_unsupported_definitions(self):
        good = future_settings()
        for bad in ({**good,"extra":1}, {**good,"local_window_s":[1,2]}, {**good,"lag_views":[{"horizon_s":10,"bin_s":3}]},
                    {**good,"recordings":{"fish":{"batch_id":"unproven"}}}):
            with self.assertRaises(ConfigurationError):
                timing_config_from_dict(bad)

    def test_recorded_boundaries_reveal_disagreement_and_phase_specific_cs_durations(self):
        import pandas as pd
        settings = {"experiment_id":"phases", "phases":[{"name":"first","cs_count":1,"us_count":1,"us_role":"training","cs_duration_ms":1000},
            {"name":"second","cs_count":1,"us_count":0,"cs_duration_ms":2000}],
            "conditions":{"delay":{"role":"conditioned","pairings":[{"cs":1,"us":1,"latency_s":2}]}},
            "recordings":{"fish":{"phase_bounds_s":{"first":[0,5],"second":[5,20]},
                "phase_bounds_provenance":{p:{"kind":"recorded","source":"phase log"} for p in ("first","second")}}}}
        frame = pd.DataFrame([("Cycle",1000,2000),("Reinforcer",6000,6100),("Cycle",10000,12000)],columns=["Type","Beg","End"])
        r = evidence(frame,config=timing_config_from_dict(settings),acquisition_bounds_ms=(0,20000))
        self.assertIn("recorded_configured_phase_disagreement",{f["code"] for f in r["findings"]})
        self.assertEqual(r["expected_pairs"][0]["expected_cs_duration_ms"],1000)
        cs = [e for e in r["events"] if e["type"]=="CS"]
        self.assertEqual([e["expected_duration_ms"] for e in cs],[1000,2000])


class StimulusTimingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.raw, self.project = self.root/"raw", self.root/"output"
        self.raw.mkdir()
        self.log = self.raw/"20260101_01_novel_black-1_test_6dpf_stim control.txt"
        self.log.write_text("Type Beg End\nLight 0 1000\nPulse 2000 2100\nLight 10000 11000\nLight 20000 21000\nPulse 24000 24100\nPulse 35000 35500\n")
        self.config = self.root/"timing.json"
        self.config.write_text(json.dumps(future_settings()))

    def test_raw_only_run_renders_all_views_and_preserves_inputs_and_rc(self):
        before, rc = self.log.read_bytes(), dict(mpl.rcParams)
        result = run_protocol_timing(self.project, raw_dir=self.raw, config_path=self.config)
        self.assertEqual(result["status"],"complete",result["errors"])
        self.assertEqual(before,self.log.read_bytes())
        self.assertEqual(rc,dict(mpl.rcParams))
        root = self.project/"Quality checks/20260101_01/stimulus-timing/protocol-timing"
        for name in ["timeline.png","trials.png","timing.png","alignment.png"]:
            self.assertGreater((root/name).stat().st_size,5000)
        self.assertTrue(any(Path(p).name.startswith("lags-") for p in result["outputs"]))
        self.assertTrue(any(Path(p).name.startswith("schedules-") for p in result["outputs"]))
        for p,digest in result["outputs"].items():
            self.assertEqual(sha256_file(Path(p)),digest)
        second = run_protocol_timing(self.project, raw_dir=self.raw, config_path=self.config)
        self.assertEqual(second["status"],"complete")
        page = Path(result["collection"])
        import re
        from urllib.parse import unquote
        for link in re.findall(r'(?:href|src)="([^"]+)"',page.read_text()):
            self.assertTrue((page.parent/unquote(link)).exists(),link)

    def test_manual_changes_and_concurrent_input_changes_are_preserved(self):
        result = run_protocol_timing(self.project,raw_dir=self.raw,config_path=self.config)
        image = next(Path(p) for p in result["outputs"] if p.endswith("timeline.png"))
        image.write_bytes(b"author edit")
        with self.assertRaisesRegex(ArtifactIntegrityError,"manually changed"):
            run_protocol_timing(self.project,raw_dir=self.raw,config_path=self.config)
        self.assertEqual(image.read_bytes(),b"author edit")

    def test_changed_source_at_publication_rolls_back_all_outputs(self):
        from classical_conditioning.artifacts import publish_transaction
        def conflict(units,staging,**kwargs):
            self.log.write_text(self.log.read_text()+"Light 40000 41000\n")
            return publish_transaction(units,staging,**kwargs)
        with patch("classical_conditioning.ingestion.stimulus_timing.publish_transaction",side_effect=conflict):
            with self.assertRaisesRegex(ArtifactIntegrityError,"publication verification"):
                run_protocol_timing(self.project,raw_dir=self.raw,config_path=self.config)
        self.assertFalse((self.project/"Metadata/protocol-timing_stimulus_timing.json").exists())
        self.assertFalse(list(self.project.rglob("*.png")))

    def test_authenticated_intake_and_corrupt_optional_markers_are_separate(self):
        from tests.test_intake import IntakeTests
        from classical_conditioning.intake import intake_recording
        fixture = IntakeTests();fixture.setUp();self.addCleanup(fixture.tearDown)
        intake_recording(fixture.raw,fixture.project)
        result = run_protocol_timing(fixture.project,["20260101_01"],experiment_name="allDelay")
        self.assertEqual(result["status"],"complete",result["errors"])
        path = fixture.project/"Quality checks/20260101_01/stimulus-timing/protocol-timing/evidence.json"
        report = json.loads(path.read_text())
        self.assertIsNotNone(report["source"]["intake_protocol_sha256"])
        self.assertEqual(report["acquisition_bounds_ms"],[1000,1003])

    def test_ambiguous_recording_does_not_hide_other_logs_and_bad_camera_is_optional(self):
        another = self.raw/"another"
        another.mkdir()
        (another/self.log.name).write_bytes(self.log.read_bytes())
        second = self.raw/"20260101_02_control_black-1_test_6dpf_stim control.txt"
        second.write_bytes(self.log.read_bytes())
        (self.raw/"20260101_02_control_black-1_test_6dpf_cam.txt").write_text("invalid camera\n")
        result = run_protocol_timing(self.project,raw_dir=self.raw,config_path=self.config)
        self.assertEqual(result["status"],"partial")
        self.assertEqual(result["completed"],["20260101_02"])
        self.assertIn("Ambiguous",result["errors"][0]["error"])
        report = json.loads((self.project/"Quality checks/20260101_02/stimulus-timing/protocol-timing/evidence.json").read_text())
        self.assertIsNone(report["acquisition_bounds_ms"])
        self.assertTrue(report["camera_findings"])

    def test_unowned_output_collision_is_not_overwritten(self):
        from classical_conditioning.figures.protocol_timing import render_collection
        run_protocol_timing(self.project,raw_dir=self.raw,config_path=self.config)
        unowned = self.project/"Figures/Stimulus timing/protocol-timing/new-plot.png"
        unowned.write_bytes(b"author work")
        def extra(*args,**kwargs):
            paths = render_collection(*args,**kwargs)
            path = args[2]/"new-plot.png"
            path.write_bytes(b"generated")
            return [*paths,path]
        with patch("classical_conditioning.figures.protocol_timing.render_collection",side_effect=extra):
            with self.assertRaisesRegex(ArtifactIntegrityError,"unowned"):
                run_protocol_timing(self.project,raw_dir=self.raw,config_path=self.config)
        self.assertEqual(unowned.read_bytes(),b"author work")

    def test_optional_markers_validate_raw_binding_and_keep_onset_offset_residuals(self):
        markers = self.root/"markers.json"
        markers.write_text(json.dumps([{"type":"CS","source_row":1,"beg_ms":2,"end_ms":998}]))
        settings = future_settings()
        settings["recordings"]["20260101_01"]["processed_markers"] = {"path":str(markers),"sha256":sha256_file(markers),"raw_protocol_sha256":sha256_file(self.log)}
        self.config.write_text(json.dumps(settings))
        result = run_protocol_timing(self.project,raw_dir=self.raw,config_path=self.config)
        report = json.loads((self.project/"Quality checks/20260101_01/stimulus-timing/protocol-timing/evidence.json").read_text())
        self.assertEqual(report["alignment"]["matches"][0]["end_residual_ms"],-2)
        self.assertEqual(result["status"],"complete")
        settings["recordings"]["20260101_01"]["processed_markers"]["raw_protocol_sha256"] = "wrong"
        self.config.write_text(json.dumps(settings))
        result = run_protocol_timing(self.project,raw_dir=self.raw,config_path=self.config)
        report = json.loads((self.project/"Quality checks/20260101_01/stimulus-timing/protocol-timing/evidence.json").read_text())
        self.assertEqual(report["alignment"]["status"],"integrity_failure")
        self.assertEqual(report["status"],"review")

    def test_missing_expected_recording_is_retained_and_not_success(self):
        result = run_protocol_timing(self.project, raw_dir=self.raw, config_path=self.config, expected_recording_ids=["missing"])
        self.assertEqual(result["status"],"partial")
        self.assertEqual(result["errors"][0]["recording_id"],"missing")
        self.assertIn("missing",result["recording_ids"])

    def test_camera_gaps_and_clock_resets_do_not_invent_continuous_coverage(self):
        import pandas as pd
        chunks = [pd.DataFrame({"FrameID":[1,2],"AbsoluteTime":[0,1]}),pd.DataFrame({"FrameID":[5,6],"AbsoluteTime":[4,5]})]
        bounds,gaps,issues = _camera_evidence(chunks)
        self.assertEqual(bounds,(0,5));self.assertEqual(gaps,[(1,4)]);self.assertEqual(issues,[])
        chunks[1]["AbsoluteTime"] = [-2,-1]
        self.assertIsNone(_camera_evidence(chunks)[0])

    def test_cli_supports_raw_future_protocol_and_batch_selection(self):
        from classical_conditioning.cli import build_parser
        args = build_parser().parse_args(["verify-protocol","--project-dir",str(self.project),"--raw-dir",str(self.raw),"--timing-config",str(self.config),"--recording-id","a","--recording-id","b"])
        self.assertEqual(args.recording_id,["a","b"])
        self.assertEqual(args.timing_config,self.config)


if __name__ == "__main__":
    unittest.main()
