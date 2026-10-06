"""Guard the author-selected metric across routine and paper entry points."""
import json
import tempfile
import unittest
from pathlib import Path

from classical_conditioning.cli import build_parser
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.figures.paper_panels import build_render_plan
from classical_conditioning.metric_policy import PAPER_METRIC_ID
from classical_conditioning.run_config import load_pipeline_run_config


class MetricPolicyTests(unittest.TestCase):
    def test_all_cli_single_metric_selectors_default_to_paper_metric(self):
        parser = build_parser()
        subparsers = next(a for a in parser._actions if hasattr(a, "choices") and isinstance(a.choices, dict))
        count = 0
        for name, command in subparsers.choices.items():
            for action in command._actions:
                if "--metric" in action.option_strings:
                    with self.subTest(command=name):
                        self.assertEqual(action.default, PAPER_METRIC_ID)
                        self.assertFalse(action.required)
                    count += 1
        self.assertGreater(count, 10)

    def test_paper_plan_routes_every_metric_argument_and_rejects_other_metrics(self):
        steps = build_render_plan(Path("project"), Path("review"))
        for step in steps:
            if "--metric" in step.argv:
                self.assertEqual(step.argv[step.argv.index("--metric") + 1], PAPER_METRIC_ID)
        with self.assertRaisesRegex(ValueError, "frozen paper metric"):
            build_render_plan(Path("project"), Path("review"), metric_id="tail_length_weighted_angular_l1")
        with self.assertRaisesRegex(ValueError, "saved Figure 2G LME"):
            build_render_plan(Path("project"), Path("review"), inference_review=True)

    def test_routine_defaults_and_rejection_of_stale_metric_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raw").mkdir()
            path = root / "run.json"
            config = {"raw_dir": str(root / "raw"), "save_dir": str(root / "save"),
                      "experiment": "allDelay", "analysis_id": "metric-policy"}
            path.write_text(json.dumps(config), encoding="utf-8")
            loaded = load_pipeline_run_config(path)
            self.assertEqual(loaded.metric, PAPER_METRIC_ID)
            self.assertEqual(loaded.assessment_metric, PAPER_METRIC_ID)
            for field in ("metric", "assessment_metric"):
                path.write_text(json.dumps({**config, field: "tail_length_weighted_angular_l1"}), encoding="utf-8")
                with self.subTest(field=field), self.assertRaisesRegex(ConfigurationError, "frozen paper metric"):
                    load_pipeline_run_config(path)


if __name__ == "__main__":
    unittest.main()
