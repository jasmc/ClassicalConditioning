from __future__ import annotations

import io
import unittest
from unittest import mock

from classical_conditioning.environment import ensure_supported_runtime
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.progress import PipelineProgress


class RuntimeGuardTests(unittest.TestCase):
    def test_accepts_supported_python(self) -> None:
        ensure_supported_runtime()

    def test_rejects_python_314(self) -> None:
        with mock.patch(
            "classical_conditioning.environment.sys.version_info",
            (3, 14, 0),
        ):
            with self.assertRaises(ConfigurationError) as context:
                ensure_supported_runtime()
        self.assertIn("3.12 or 3.13", str(context.exception))
        self.assertIn("pyarrow", str(context.exception).lower())


class ProgressTests(unittest.TestCase):
    def test_stage_timer_and_items_emit_messages(self) -> None:
        stream = io.StringIO()
        progress = PipelineProgress(enabled=True, stream=stream)
        with progress.stage_timer("Intake", detail="2 recordings"):
            progress.item_done(1, 2, "fish_a", status="completed")
            for item in progress.iter_items(["fish_b"], description="intake"):
                self.assertEqual(item, "fish_b")
        text = stream.getvalue()
        self.assertIn("==> Intake", text)
        self.assertIn("fish_a: completed", text)
        self.assertIn("finished in", text)

    def test_step_timer_reports_running_and_done(self) -> None:
        stream = io.StringIO()
        progress = PipelineProgress(enabled=True, stream=stream)
        with progress.step_timer("movement-state"):
            pass
        text = stream.getvalue()
        self.assertIn("> movement-state: running", text)
        self.assertIn("> movement-state: done in", text)

    def test_step_timer_reports_failure(self) -> None:
        stream = io.StringIO()
        progress = PipelineProgress(enabled=True, stream=stream)
        with self.assertRaises(RuntimeError):
            with progress.step_timer("temporal-profiles"):
                raise RuntimeError("boom")
        text = stream.getvalue()
        self.assertIn("> temporal-profiles: failed after", text)


if __name__ == "__main__":
    unittest.main()
