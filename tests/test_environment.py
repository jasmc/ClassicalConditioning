from __future__ import annotations

import importlib
import importlib.metadata
import json
import tempfile
import unittest
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import classical_conditioning
from classical_conditioning.environment import (
    LOCKED_DISTRIBUTIONS,
    build_environment_report,
    write_environment_report,
)
from classical_conditioning.exceptions import (
    AmbiguousArtifactError,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ClassicalConditioningError,
    ConfigurationError,
    ModelDiagnosticError,
    SchemaValidationError,
    ScientificValidationError,
)


class PackageEnvironmentTests(unittest.TestCase):
    def test_package_version_matches_installed_metadata(self) -> None:
        self.assertEqual(
            classical_conditioning.__version__,
            importlib.metadata.version("classical-conditioning"),
        )

    def test_complete_scientific_environment_imports(self) -> None:
        import_names = {
            "matplotlib": "matplotlib",
            "numba": "numba",
            "numpy": "numpy",
            "pandas": "pandas",
            "plotly": "plotly",
            "pyarrow": "pyarrow",
            "scikit-learn": "sklearn",
            "scipy": "scipy",
            "seaborn": "seaborn",
            "statannotations": "statannotations",
            "statsmodels": "statsmodels",
            "tqdm": "tqdm",
        }
        for distribution in LOCKED_DISTRIBUTIONS:
            with self.subTest(distribution=distribution):
                self.assertIsNotNone(
                    importlib.import_module(import_names[distribution])
                )

    def test_agg_backend_renders_without_display(self) -> None:
        figure = Figure(figsize=(1, 1))
        canvas = FigureCanvasAgg(figure)
        figure.subplots().plot([0, 1], [0, 1])
        canvas.draw()
        self.assertGreater(len(canvas.buffer_rgba()), 0)

    def test_environment_report_records_reproduction_settings(self) -> None:
        report = build_environment_report()

        self.assertEqual(report["package"]["version"], classical_conditioning.__version__)
        self.assertEqual(
            report["figures"]["matplotlib_version"],
            matplotlib.__version__,
        )
        self.assertTrue(report["numerical_libraries"]["blas"]["name"])
        self.assertTrue(report["figures"]["resolved_sans_serif"])
        self.assertEqual(
            set(report["distributions"]),
            set(LOCKED_DISTRIBUTIONS),
        )

    def test_environment_report_is_written_as_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "environment.json"

            result = write_environment_report(output)

            self.assertEqual(result, output.resolve())
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8"))["package"]["name"],
                "classical-conditioning",
            )

    def test_test_directory_is_located_relative_to_this_file(self) -> None:
        tests_directory = Path(__file__).resolve().parent
        self.assertEqual(tests_directory.name, "tests")
        self.assertTrue((tests_directory / "test_environment.py").is_file())

    def test_package_exception_categories_share_one_boundary(self) -> None:
        categories = (
            AmbiguousArtifactError,
            ArtifactIntegrityError,
            ArtifactNotFoundError,
            ConfigurationError,
            ModelDiagnosticError,
            SchemaValidationError,
            ScientificValidationError,
        )
        for category in categories:
            with self.subTest(category=category.__name__):
                self.assertTrue(issubclass(category, ClassicalConditioningError))
        self.assertTrue(issubclass(ArtifactNotFoundError, FileNotFoundError))


if __name__ == "__main__":
    unittest.main()
