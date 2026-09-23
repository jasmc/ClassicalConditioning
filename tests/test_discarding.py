from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from classical_conditioning.analysis.candidate_runner import run_candidate_development_pipeline
from classical_conditioning.analysis.discarding import (
    LEGACY_SOURCES,
    RULE_ORDER,
    assess_discarding,
    evaluate_learner_inputs,
    evaluate_legacy_bouts,
)
from classical_conditioning.cli import build_parser
from classical_conditioning.exceptions import ConfigurationError


def _protocol_and_movement(*, include_cs: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    protocol_rows = []
    movement_rows = []
    if include_cs:
        for trial in range(1, 95):
            start = trial * 100_000
            protocol_rows.append({"Type": "Cycle", "Beg": start, "End": start + 10_000})
            for offset in (-5_000, 1_000):
                # Only three of the five Early Test trials move in the CR
                # window. Trial 65 must count as one of those three.
                moving = trial not in range(65, 70) or trial in (65, 66, 67) or offset < 0
                movement_rows.append({"AbsoluteTime": start + offset, "moving": moving, "valid": True})
    for trial in range(1, 79):
        start = 10_000_000 + trial * 100_000
        protocol_rows.append({"Type": "Reinforcer", "Beg": start, "End": start + 1_000})
        movement_rows.append({"AbsoluteTime": start + 1_000, "moving": True, "valid": True})
    return pd.DataFrame(movement_rows), pd.DataFrame(protocol_rows)


def _learner_outcomes() -> pd.DataFrame:
    return pd.DataFrame({
        "alignment": ["CS"] * 90,
        "metric_id": ["legacy_distal_angular_speed"] * 90,
        "trial_number": list(range(5, 95)),
        "block_10_name": ["present"] * 90,
        "baseline_total_activity": [1.0] * 90,
        "response_total_activity": [0.5] * 90,
    })


class DiscardingTests(unittest.TestCase):
    def test_one_cli_command_exposes_two_stage_assessment(self) -> None:
        args = build_parser().parse_args([
            "assess-discarding", "--raw-dir", "raw", "--project-dir", "derived",
            "--analysis-id", "audit", "--experiment", "allDelay",
            "--metric", "legacy_distal_angular_speed",
            "--disable-check", "last_us",
        ])
        self.assertEqual(args.command, "assess-discarding")
        self.assertEqual(args.disable_check, ["last_us"])

    def test_all_published_rules_have_legacy_sources(self) -> None:
        self.assertEqual(set(RULE_ORDER), set(LEGACY_SOURCES))
        self.assertTrue(all("Archive/historical-scripts/" in source for source in LEGACY_SOURCES.values()))

    def test_legacy_bouts_include_early_test_trial_65(self) -> None:
        movement, protocol = _protocol_and_movement()
        result, details = evaluate_legacy_bouts(movement, protocol, experiment="allDelay")
        self.assertTrue(all(status == "pass" for status, _ in result.values()))
        early = [row for row in details if row.get("rule_id") == "cr_bouts" and row.get("block") == "Early Test"]
        self.assertEqual(early[0]["bout_trial_count"], 3)
        movement.loc[movement["AbsoluteTime"].eq(65 * 100_000 + 1_000), "moving"] = False
        result, _ = evaluate_legacy_bouts(movement, protocol, experiment="allDelay")
        self.assertEqual(result["cr_bouts"][0], "fail")

    def test_trace_windows_and_missing_us_checks(self) -> None:
        movement, protocol = _protocol_and_movement()
        movement.loc[
            movement["AbsoluteTime"].eq(65 * 100_000 + 1_000), "AbsoluteTime"
        ] = 65 * 100_000 + 12_000
        delay, _ = evaluate_legacy_bouts(movement, protocol, experiment="allDelay")
        trace, _ = evaluate_legacy_bouts(movement, protocol, experiment="all3sTrace")
        self.assertEqual(delay["cr_bouts"][0], "fail")
        self.assertEqual(trace["cr_bouts"][0], "pass")
        protocol.loc[protocol["Type"].eq("Reinforcer"), "End"] = None
        result, _ = evaluate_legacy_bouts(movement, protocol, experiment="all3sTrace")
        self.assertEqual(result["last_us"][0], "fail")

    def test_legacy_empty_cs_bypass_is_visible(self) -> None:
        movement, protocol = _protocol_and_movement(include_cs=False)
        result, _ = evaluate_legacy_bouts(movement, protocol, experiment="allDelay")
        self.assertEqual(result["baseline_bouts"], ("pass", "legacy_empty_cs_bypass"))
        self.assertEqual(result["cr_bouts"], ("pass", "legacy_empty_cs_bypass"))

    def test_merged_learner_inputs_require_all_blocks_and_positive_values(self) -> None:
        frame = _learner_outcomes()
        status, _, _ = evaluate_learner_inputs(frame, metric_id="legacy_distal_angular_speed")
        self.assertEqual(status, "pass")
        reduced = frame.loc[~frame["trial_number"].isin([65, 66, 67])]
        status, reason, _ = evaluate_learner_inputs(reduced, metric_id="legacy_distal_angular_speed")
        self.assertEqual(status, "fail")
        self.assertIn("Early Test_fewer_than_three", reason)
        frame.loc[frame["trial_number"].isin([65, 66, 67]), "response_total_activity"] = 0.0
        status, _, _ = evaluate_learner_inputs(frame, metric_id="legacy_distal_angular_speed")
        self.assertEqual(status, "fail")
        frame = _learner_outcomes()
        frame.loc[frame["trial_number"].isin([65, 66, 67]), "block_10_name"] = None
        status, reason, _ = evaluate_learner_inputs(frame, metric_id="legacy_distal_angular_speed")
        self.assertEqual(status, "fail")
        self.assertIn("Early Test_fewer_than_three", reason)

    def test_incomplete_recording_is_retained_and_hash_changes_with_toggle(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            save = root / "save"
            raw.mkdir()
            save.mkdir()
            inventory = {
                "records_sha256": "inventory-hash",
                "records": [{
                    "recording_id": "day_1", "recording_name": "day_1_control",
                    "condition_id": "control", "status": "INCOMPLETE",
                    "tracking_schema": None,
                }],
            }
            first = assess_discarding(raw, save, analysis_id="audit", experiment="allDelay",
                                       metric_id="legacy_distal_angular_speed", inventory=inventory)
            second = assess_discarding(raw, save, analysis_id="audit", experiment="allDelay",
                                        metric_id="legacy_distal_angular_speed", inventory=inventory)
            self.assertEqual(first.assessment_hash, second.assessment_hash)
            technical = pd.read_parquet(first.technical_path)
            self.assertEqual(technical.iloc[0]["inventory_status"], "INCOMPLETE")
            self.assertFalse(technical.iloc[0]["technical_ready"])
            self.assertTrue(pd.isna(technical.iloc[0]["primary_candidate"]))
            summary = json.loads(first.summary_path.read_text(encoding="utf-8"))
            self.assertFalse(summary["primary_cohort_changed"])
            flow = pd.read_parquet(first.flow_path)
            self.assertEqual(flow.iloc[0]["rule_id"], "technical")
            self.assertEqual(int(flow.iloc[0]["selected_fish_count"]), 1)
            changed = assess_discarding(
                raw, save, analysis_id="audit", experiment="allDelay",
                metric_id="legacy_distal_angular_speed", inventory=inventory,
                disabled_rules=["last_us"],
            )
            self.assertNotEqual(first.assessment_hash, changed.assessment_hash)
            self.assertEqual(first.summary_path, changed.summary_path)
            self.assertEqual(first.technical_path, changed.technical_path)
            self.assertTrue(raw.exists())
            changed.technical_path.write_bytes(b"tampered assessment")
            with self.assertRaises(ConfigurationError):
                assess_discarding(raw, save, analysis_id="audit", experiment="allDelay",
                                   metric_id="legacy_distal_angular_speed", inventory=inventory,
                                   disabled_rules=["last_us"])

    def test_requested_missing_recording_stays_in_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            save = root / "save"
            raw.mkdir()
            save.mkdir()
            result = assess_discarding(
                raw, save, analysis_id="missing", experiment="allDelay",
                metric_id="legacy_distal_angular_speed",
                recording_ids=["day_1"],
                inventory={"records_sha256": "empty", "records": []},
            )
            technical = pd.read_parquet(result.technical_path)
            self.assertEqual(technical.iloc[0]["inventory_status"], "NOT_DISCOVERED")
            self.assertIn("inventory_not_discovered", technical.iloc[0]["technical_reason"])
            self.assertFalse(technical.iloc[0]["technical_ready"])

    def test_trace_source_condition_maps_to_canonical_condition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            save = root / "save"
            raw.mkdir()
            save.mkdir()
            result = assess_discarding(
                raw, save, analysis_id="trace", experiment="all10sTrace",
                metric_id="legacy_distal_angular_speed",
                inventory={"records_sha256": "trace-inventory", "records": [{
                    "recording_id": "day_1", "recording_name": "day_1_10sFixedTrace_fish_1_session",
                    "condition_id": "10sfixedtrace", "status": "INCOMPLETE",
                    "tracking_schema": None,
                }]},
            )
            technical = pd.read_parquet(result.technical_path)
            self.assertEqual(technical.iloc[0]["condition_id"], "trace")
            self.assertEqual(technical.iloc[0]["source_condition"], "10sfixedtrace")
            self.assertNotIn("invalid_condition", technical.iloc[0]["technical_reason"])

    def test_complete_raw_record_without_processed_fish_fails_legacy_readability(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            save = root / "save"
            raw.mkdir()
            save.mkdir()
            result = assess_discarding(
                raw, save, analysis_id="unreadable", experiment="allDelay",
                metric_id="legacy_distal_angular_speed",
                inventory={"records_sha256": "complete", "records": [{
                    "recording_id": "day_1", "recording_name": "day_1_control_fish_1_session",
                    "condition_id": "control", "status": "COMPLETE",
                    "tracking_schema": {"ok": True},
                }]},
            )
            rules = pd.read_parquet(result.rules_path)
            readability = rules.loc[rules["rule_id"].eq("readable_fish")].iloc[0]
            self.assertEqual(readability["status"], "fail")
            self.assertEqual(readability["reason"], "processed_fish_unreadable_or_unavailable")

    @patch("classical_conditioning.analysis.candidate_runner._run_recording_candidate_stages")
    @patch("classical_conditioning.analysis.candidate_runner._verify_comparison", return_value="ok")
    @patch("classical_conditioning.analysis.candidate_runner.build_candidate_metric_comparison")
    def test_runner_calls_assessment_before_comparison(self, comparison, _verify, stages) -> None:
        order = []
        stages.side_effect = lambda *_args, **_kwargs: order.append("fish")
        comparison.side_effect = lambda *_args, **_kwargs: order.append("comparison")
        with tempfile.TemporaryDirectory() as temporary:
            run_candidate_development_pipeline(
                Path(temporary), ["fish-a"], analysis_id="audit-order",
                before_comparison=lambda _successful, _steps: order.append("assessment"),
                overwrite=True,
            )
        self.assertEqual(order, ["fish", "assessment", "comparison"])


if __name__ == "__main__":
    unittest.main()
