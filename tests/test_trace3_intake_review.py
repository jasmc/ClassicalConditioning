"""Regression checks for timing and parser failures found in the intake review."""
import runpy
import unittest
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

REVIEW = runpy.run_path(str(Path(__file__).resolve().parents[1]/"scripts"/"process_trace3_additions.py"))
PARSE = REVIEW["parse_protocol"]
CATCHES = REVIEW["CATCHES"]


def protocol(latencies, *, short_us=False, reverse=False):
    base = 1_700_000_000_000
    events = [(base+i*100_000, f"Cycle {base+i*100_000} {base+i*100_000+10_000}") for i in range(94)]
    for i in range(18):
        start = base-2_000_000+i*50_000
        events.append((start,f"Reinforcer {start} {start+50}"))
    for index, trial in enumerate(t for t in range(15,65) if t not in CATCHES):
        start = base+(trial-1)*100_000+latencies[index]
        events.append((start,f"Reinforcer {start} {start+(2 if short_us and index==0 else 500)}"))
    for i in range(15):
        start = base+10_000_000+i*50_000
        events.append((start,f"Reinforcer {start} {start+50}"))
    lines=[line for _,line in sorted(events,reverse=reverse)]
    return ("Type Beg End\n"+"\n".join(lines)+"\n").encode()


class Trace3IntakeReviewTests(unittest.TestCase):
    def test_fixed13_checks_every_pair_catches_and_us_numbers(self):
        result=PARSE(protocol([13000]*46))
        self.assertEqual(result["schedule"],"fixed-3s-trace")
        self.assertTrue(result["catch_schedule_matches"])
        self.assertTrue(result["paired_us_numbers_unique_and_consecutive"])
        self.assertEqual(result["paired_us_numbers"],list(range(19,65)))
        self.assertEqual(result["errors"],[])

    def test_single_nonfixed_pair_cannot_pass(self):
        timings=[13000]*46
        timings[17]=12000
        self.assertNotEqual(PARSE(protocol(timings))["schedule"],"fixed-3s-trace")

    def test_increasing_schedule_is_separate(self):
        timings=[10500+round(i*2500/45) for i in range(46)]
        self.assertEqual(PARSE(protocol(timings))["schedule"],"increasing-trace")

    def test_fixed20_above20_seconds_is_not_discarded(self):
        self.assertEqual(PARSE(protocol([20100]*46))["schedule"],"fixed-10s-trace")

    def test_exact_tolerance_boundary_and_unsorted_rows(self):
        self.assertEqual(PARSE(protocol([13200]*46,reverse=True))["schedule"],"fixed-3s-trace")

    def test_bad_rows_are_reported_and_short_us_is_descriptive(self):
        result=PARSE(protocol([13000]*46,short_us=True)+b"Unknown 1 2\nCycle 1.5 3\n")
        self.assertEqual(len(result["errors"]),2)
        self.assertEqual(result["short_us_screen_only"][0]["duration_ms"],2)
        self.assertEqual(result["schedule"],"fixed-3s-trace")


class TrialMarkerIntegrityReviewTests(unittest.TestCase):
    def check_fixture(self, *, corrupt_marker=False, corrupt_output=False):
        """Exercise the real marker schema missed by the preliminary audit."""
        from classical_conditioning.artifacts import artifact_staging
        parent=Path(__file__).resolve().parents[1]/".cache"
        with artifact_staging(parent,prefix="trace3-marker-test-") as root:
            rid="20230227_03"
            data=root/"Processed data"/rid
            qc=root/"Quality checks"/rid
            meta=root/"Metadata"
            for p in (data,qc,meta): p.mkdir(parents=True)
            artifacts={}
            for key,name in (("outcomes","candidate-trial-outcomes-corrected.parquet"),("coverage","candidate-trial-outcomes-corrected_coverage.parquet")):
                p=data/name
                p.write_bytes(key.encode())
                artifacts[key]={"path":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
            summary={"recipe":"candidate-trial-outcomes-corrected","recording_id":rid,
                     "experiment":"all3sTrace","config":{"response_window_s":[0.0,13.0]},
                     "inputs":{"protocol":"a"*64},"artifacts":artifacts}
            summary_path=qc/"candidate-trial-outcomes-corrected_summary.json"
            summary_path.write_text(json.dumps(summary))
            marker={"recipe":summary["recipe"],"recording_id":rid,"status":"complete",
                    "summary_sha256":hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                    "artifact_sha256":{k:v["sha256"] for k,v in artifacts.items()}}
            if corrupt_marker: marker["artifact_sha256"]["outcomes"]="0"*64
            if corrupt_output: (data/"candidate-trial-outcomes-corrected.parquet").write_bytes(b"changed")
            (meta/f"{rid}_candidate-trial-outcomes-corrected_complete.json").write_text(json.dumps(marker))
            (meta/f"{rid}_source_manifest.json").write_text(json.dumps({"recording_id":rid,"artifacts":{"protocol":{"sha256":"a"*64}},"sources":{}}))
            check=REVIEW["existing_status"]
            with patch.dict(check.__globals__,{"PROJECT":root}):
                return check(rid,False)

    def test_actual_marker_hash_map_is_checked(self):
        self.assertEqual(self.check_fixture()["state"],"trial-outputs-verified-upstream-pending")
        self.assertEqual(self.check_fixture(corrupt_marker=True)["state"],"invalid-or-incomplete")

    def test_changed_output_bytes_are_detected(self):
        self.assertEqual(self.check_fixture(corrupt_output=True)["state"],"invalid-or-incomplete")


class UntreatedSourceAliasTests(unittest.TestCase):
    def test_explicit_untreated_aliases_preserve_identity(self):
        from classical_conditioning.analysis.trial_outcomes import parse_recording_identity
        for label,condition in (("traceNoMK801","trace"),("controlNoMK801","control")):
            result=parse_recording_identity(f"20230419_01_{label}_orange-1_test_6dpf",experiment_name="all3sTrace")
            self.assertEqual(result["condition_id"],condition)
            self.assertEqual(result["recording_id"],"20230419_01")

    def test_drug_and_other_protocol_aliases_are_rejected(self):
        from classical_conditioning.analysis.trial_outcomes import parse_recording_identity
        from classical_conditioning.exceptions import SchemaValidationError
        for label in ("traceMK801","controlMK801","tracePuromycin","10sTrace","increasingTrace"):
            with self.assertRaises(SchemaValidationError):
                parse_recording_identity(f"20230419_01_{label}_orange-1_test_6dpf",experiment_name="all3sTrace")


if __name__=="__main__":
    unittest.main()
