"""Contract identity, explicit historical recovery and guide source capture."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from classical_conditioning.external_artifacts import resolve_artifact
from classical_conditioning.figures.layout_frames import load_policy
from classical_conditioning.figures.presentation_profiles import ROOT, load_presentation_profile


class ConsolidationTests(unittest.TestCase):
    def test_exact_prior_configuration_hashes_recover_local_compatibility_bytes(self):
        legacy = ROOT/'records/figure-history/shared/contracts/2026-10-31'
        for name in ('figure-elements.json', 'nature-main.json', 'layout-frames.json'):
            with self.subTest(name=name):
                relative = 'configs/paper-figures/'+name
                expected = hashlib.sha256((legacy/relative).read_bytes()).hexdigest()
                resolved = resolve_artifact(relative, expected_sha256=expected)
                self.assertEqual(resolved, legacy/relative)
                self.assertEqual(hashlib.sha256(resolved.read_bytes()).hexdigest(), expected)

    def test_current_profile_and_policy_share_one_raw_contract_identity(self):
        profile = load_presentation_profile()
        policy, identity = load_policy()
        self.assertEqual(profile.layout, policy)
        self.assertEqual(list(profile.inputs.values()), [identity])
        legacy = ROOT/'records/figure-history/shared/contracts/2026-10-31/configs/paper-figures/layout-frames.json'
        old_policy, old_identity = load_policy(legacy)
        self.assertEqual(old_policy, policy)
        self.assertNotEqual(old_identity, identity)

    def test_invalid_embedded_policy_and_unknown_schema_never_fall_back(self):
        contract = json.loads((ROOT/'configs/paper-figures/figure-elements.json').read_bytes())
        for mutation in (lambda c: c.update(schema_version=99),
                         lambda c: c['layout_frames']['policy'].update(selected_inset_mm=float('nan')),
                         lambda c: c['layout_frames']['policy'].update(units='px')):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'contract.json'
                data = copy.deepcopy(contract)
                mutation(data)
                path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    load_presentation_profile(contract_path=path)

    def test_study_capture_reads_single_guide_and_keeps_historical_source_key(self):
        from scripts.paper.figure_02.panel_g.build_study_guide import Guide
        guide = Guide.__new__(Guide)
        guide.sources = {}
        calls = []
        class StopCapture(Exception):
            pass
        def source(key, title, path, *args, **kwargs):
            calls.append((key, Path(path)))
            guide.sources[key] = {'title': title, 'path': str(path), 'copy': 'sources/'+key+'.md'}
            if key == 'selection':
                return b'{}'
            if key == 'figure-state':
                return b'{"scientific_selection":{"records":[]}}'
            if key == 'workflow':
                return Path(path).read_bytes()
            raise StopCapture()
        guide.source = source
        with self.assertRaises(StopCapture):
            guide.capture()
        self.assertEqual(guide.sources['nature']['path'], guide.sources['workflow']['path'])
        self.assertEqual(guide.sources['nature']['copy'], guide.sources['workflow']['copy'])
        self.assertEqual(sum(path.name == 'iteration-workflow.md' for _, path in calls), 1)
        self.assertFalse(any(path.name == 'nature-style.md' for _, path in calls))
