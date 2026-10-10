"""Independent AST-loaded detailed legacy oracle; never run legacy top-level code."""
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
_original_path = sys.path.copy()
try:
    sys.path.insert(0, str(ROOT / 'legacy/helpers'))
    import analysis_utils as current
    import data_io
    from general_configuration import config
finally:
    sys.path[:] = _original_path

def oracle(name):
    tree = ast.parse((ROOT / 'legacy/modules/my_functions.py').read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in (name, 'rolling_window')]
    namespace = {'np': np, 'pd': pd, 'gen_var': SimpleNamespace(
        cols=[f'Angle of point {i} (deg)' for i in range(16)],
        time_trial_frame=config.time_trial_frame_label,
        tail_angle=config.tail_angle_label, expected_framerate=700)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<detailed-legacy-oracle>', 'exec'), namespace)
    return namespace[name]

class LegacyPreprocessingParityTests(unittest.TestCase):
    def test_spatial_and_temporal_filter_match_original(self):
        rng = np.random.default_rng(190)
        d = pd.DataFrame(rng.normal(size=(1000,16)).cumsum(axis=1), columns=[f'Angle of point {i} (deg)' for i in range(16)])
        d.insert(0, config.time_trial_frame_label, np.arange(len(d)))
        expected = oracle('filter_data')(d.copy(),3,10)
        actual = current.filter_data(d.copy(),3,10)
        pd.testing.assert_frame_equal(actual,expected)
    def test_peak_threshold_and_boundaries_match_original(self):
        n=160
        d=pd.DataFrame({'Vigor for bout detection (deg/ms)':np.r_[np.zeros(10),np.full(60,5.),np.zeros(20),np.full(60,5.),np.zeros(10)], config.tail_angle_label:np.r_[np.arange(80)*.1,np.arange(80)*5.]})
        expected=oracle('find_beg_and_end_of_bouts')(d.copy(),4,40,10,1)
        actual=current.find_beg_and_end_of_bouts(d.copy(),4,40,10,1)
        pd.testing.assert_frame_equal(actual,expected)
        self.assertFalse(actual.Bout.iloc[10:70].any())
        self.assertTrue(actual.Bout.iloc[90:150].all())

    def test_partial_trial_keeps_actual_relative_time(self):
        d=pd.DataFrame({config.time_trial_frame_label:np.arange(5),
            'CS beg':[0,0,1,0,0]})
        actual=current.identify_trials(d,-10,10)
        np.testing.assert_array_equal(actual[config.time_trial_frame_label],[-2,-1,0,1,2])

    def test_numeric_final_tracking_row_is_preserved(self):
        import tempfile
        import data_io
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tracking.txt'
            p.write_text(' '.join(config.cols_to_use_orig)+'\n'+
                '\n'.join(' '.join([str(i)]+['0.1']*16) for i in range(3))+'\n')
            d=data_io.read_tail_tracking_data(p)
            self.assertIsNotNone(d)
            self.assertEqual(d.FrameID.tolist(),[0,1,2])

    def test_nonnumeric_tracking_summary_is_removed(self):
        import tempfile
        import data_io
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tracking.txt'
            p.write_text(' '.join(config.cols_to_use_orig)+'\n'+
                ' '.join(['1']+['0.1']*16)+'\n'+
                ' '.join(['summary']+['0']*16)+'\n')
            d=data_io.read_tail_tracking_data(p)
            self.assertIsNotNone(d)
            self.assertEqual(d.FrameID.tolist(),[1])

if __name__=='__main__': unittest.main()
