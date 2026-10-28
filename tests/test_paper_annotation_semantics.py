"""Condition spelling must not alter time units or scientific values."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from scripts.paper.shared.stage_annotations import wording

class AnnotationSemanticsTests(unittest.TestCase):
    def test_trace_condition_spelling_only(self):
        self.assertEqual(wording('3 s Trace / 10 s Trace'),'3sTrace / 10sTrace')
        self.assertEqual(wording('baseline −15 s; 3 s after CS; 10 s CS'),'baseline −15 s; 3 s after CS; 10 s CS')
        self.assertEqual(wording('3sTrace (n=40); Control (n=19)'),'3sTrace (n=40); Control (n=19)')

if __name__=='__main__':unittest.main()
