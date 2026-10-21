"""Historical source aliases preserve raw identity and canonical outcomes."""
import unittest
import tempfile
from pathlib import Path
from classical_conditioning.analysis.trial_outcomes import parse_recording_identity
from classical_conditioning.exceptions import SchemaValidationError
from classical_conditioning.intake import RecordingSources, intake_recording
from tests import test_intake as intake_tests
from scripts.process_trace10_additions import protocol

RECORDING = intake_tests.RECORDING


class Trace10AliasTests(unittest.TestCase):
    def test_fixed_trace_aliases_have_one_canonical_identity(self):
        for label in ("10sTrace", "10sFixedTrace", "trace"):
            result = parse_recording_identity(
                f"20250714_05_{label}_orange-1_test_5dpf", experiment_name="all10sTrace")
            self.assertEqual(result["condition_id"], "trace")
            self.assertEqual(result["fish_id"], "20250714_05")

    def test_treatment_and_other_assay_labels_are_not_implicitly_accepted(self):
        for experiment, label in (("all10sTrace", "traceMK801"), ("all3sTrace", "10sTrace")):
            with self.assertRaises(SchemaValidationError):
                parse_recording_identity(f"20250714_05_{label}_orange-1_test_5dpf",
                                         experiment_name=experiment)


class ExplicitIntakeTests(unittest.TestCase):
    setUp = intake_tests.IntakeTests.setUp
    tearDown = intake_tests.IntakeTests.tearDown
    def test_missing_delimiter_sources_keep_original_raw_paths(self):
        paths = {}
        for kind, suffix in (("camera", "cam.txt"), ("tracking", "mp tail tracking.txt"),
                             ("protocol", "stim control.txt")):
            before = self.raw / f"{RECORDING}_{suffix}"
            path = self.raw / f"{RECORDING}{suffix}"
            before.rename(path)
            paths[kind] = path
        sources = RecordingSources(RECORDING, "20260101_01", **paths)
        intake_recording(self.raw, self.project, recording_id="20260101_01", sources=sources)
        import json
        manifest = json.loads((self.project / "Metadata/20260101_01_source_manifest.json").read_text())
        for kind, path in paths.items():
            self.assertEqual(manifest["sources"][kind]["path"], str(path.resolve()))
            self.assertTrue(path.is_file())

    def test_explicit_sources_reject_different_identity(self):
        sources = RecordingSources(RECORDING, "20260101_02",
            self.raw / f"{RECORDING}_cam.txt", self.raw / f"{RECORDING}_mp tail tracking.txt",
            self.raw / f"{RECORDING}_stim control.txt")
        with self.assertRaises(ValueError):
            intake_recording(self.raw, self.project, sources=sources)


class Trace10ScheduleTests(unittest.TestCase):
    def test_interleaved_control_log_keeps_events_and_uses_onset_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"stim.txt"
            p.write_text("Type Beg End\nCycle 30000 40000\nReinforcer 29000 29050\n")
            result=protocol(p)
            self.assertFalse(result["ordered"])
            self.assertEqual(result["errors"],[])
            self.assertEqual([e["line"] for e in result["events"]],[2,3])

    def test_short_trace_counts_final_training_cs_without_a_later_us(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"stim.txt"
            rows=["Type Beg End"]
            for t in range(1,54):
                beg=t*120000
                rows.append(f"Cycle {beg} {beg+10000}")
                if 15<=t<=53 and t not in {25,39,53}:
                    rows.append(f"Reinforcer {beg+20000} {beg+20050}")
            p.write_text("\n".join(rows))
            result=protocol(p)
            self.assertEqual(len(result["training"]),39)
            self.assertEqual(result["near20_training_count"],36)
            self.assertEqual(result["unpaired_training_cs"],[25,39,53])
            self.assertFalse(result["standard_event_counts"])


if __name__ == "__main__":
    unittest.main()
