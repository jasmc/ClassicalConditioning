"""A launched worker or a partial result list must never imply completion."""
import runpy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

scripts=Path(__file__).resolve().parents[1]/"scripts"
with patch.object(sys,"path",[str(scripts),*sys.path]):
    CHECK=runpy.run_path(str(scripts/"finalize_trace3_additions.py"))["check_run"]


def result(rid):
    return {"recording_id":rid,"status":"completed","lineage_verified":True,
            "stages":["intake","corrected-preprocess","candidate-metrics","movement-state","temporal-profiles","trial-outcomes"]}


class CompletionEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.plan={"selected":[{"recording_id":"A"},{"recording_id":"B"}],"held":[]}

    def test_complete_results_require_verification(self):
        completed={"results":[result("A"),result("B")],"pending":[],"existing_files_changed":[]}
        self.assertEqual(CHECK(self.plan,completed,lambda *_:None)["status"],"completed-and-verified")
        def bad(*_): raise ValueError("artifact changed")
        self.assertNotEqual(CHECK(self.plan,completed,bad)["status"],"completed-and-verified")

    def test_partial_or_duplicate_results_cannot_claim_completion(self):
        for rows in ([result("A")],[result("A"),result("A")]):
            c={"results":rows,"pending":["B"],"existing_files_changed":[]}
            self.assertNotEqual(CHECK(self.plan,c,lambda *_:None)["status"],"completed-and-verified")

    def test_preservation_failure_is_not_success(self):
        c={"results":[result("A"),result("B")],"pending":[],"existing_files_changed":["old-file"]}
        self.assertNotEqual(CHECK(self.plan,c,lambda *_:None)["status"],"completed-and-verified")

    def test_missing_stage_is_not_success(self):
        a=result("A");a["stages"].pop()
        c={"results":[a,result("B")],"pending":[],"existing_files_changed":[]}
        self.assertNotEqual(CHECK(self.plan,c,lambda *_:None)["status"],"completed-and-verified")


if __name__=="__main__": unittest.main()
