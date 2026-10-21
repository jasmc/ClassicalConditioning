"""Recovery must reuse complete stages and reject partial publications."""
import importlib.util
from pathlib import Path
import sys
import unittest

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('trace10_recovery_under_test',SCRIPTS/'resume_trace10_additions.py')
recovery=importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class RecoveryDispositionTests(unittest.TestCase):
    def test_only_missing_tail_is_built(self):
        flags={s:[i<4]*3 for i,s in enumerate(recovery.STAGES)}
        actions=recovery.dispositions(flags)
        self.assertEqual([s for s,a in actions.items() if a=='build-missing'],
                         ['temporal-profiles','trial-outcomes'])

    def test_complete_fish_is_verified_without_rebuilding(self):
        actions=recovery.dispositions({s:[True]*3 for s in recovery.STAGES})
        self.assertEqual(set(actions.values()),{'verify-existing'})

    def test_partial_stage_is_never_selected_for_replacement(self):
        flags={s:[False]*3 for s in recovery.STAGES}
        flags['intake']=[True,False,True]
        with self.assertRaisesRegex(ValueError,'Partially published'):
            recovery.dispositions(flags)

    def test_missing_upstream_and_existing_downstream_is_rejected(self):
        flags={s:[False]*3 for s in recovery.STAGES}
        flags['candidate-metrics']=[True]*3
        with self.assertRaisesRegex(ValueError,'missing upstream'):
            recovery.dispositions(flags)


if __name__=='__main__':unittest.main()
