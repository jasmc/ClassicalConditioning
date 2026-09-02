from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from classical_conditioning.cli import build_parser
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.run_config import load_pipeline_run_config


class PipelineRunConfigTests(unittest.TestCase):
    def test_loads_valid_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"
            save = Path(temporary) / "save"
            raw.mkdir()
            save.mkdir()
            config_path = Path(temporary) / "run.json"
            config_path.write_text(
                json.dumps(
                    {
                        "raw_dir": str(raw),
                        "save_dir": str(save),
                        "experiment": "allDelay",
                        "analysis_id": "delay-v1",
                        "keep_conditions": ["control", "delay"],
                        "routes": ["legacy", "candidate"],
                    }
                ),
                encoding="utf-8",
            )
            config = load_pipeline_run_config(config_path)
            self.assertEqual(config.experiment, "allDelay")
            self.assertEqual(config.resolved_legacy_analysis_id(), "delay-v1-legacy")
            self.assertEqual(
                config.resolved_candidate_analysis_id(),
                "delay-v1-candidate",
            )

    def test_rejects_missing_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "raw"
            save = Path(temporary) / "save"
            raw.mkdir()
            save.mkdir()
            config_path = Path(temporary) / "run.json"
            config_path.write_text(
                json.dumps(
                    {
                        "raw_dir": str(raw),
                        "save_dir": str(save),
                        "analysis_id": "delay-v1",
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ConfigurationError):
                load_pipeline_run_config(config_path)

    def test_cli_exposes_run_pipeline(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            ["run-pipeline", "--config", "configs/example-run.json"]
        )
        self.assertEqual(args.command, "run-pipeline")
        self.assertEqual(args.config, Path("configs/example-run.json"))


if __name__ == "__main__":
    unittest.main()
