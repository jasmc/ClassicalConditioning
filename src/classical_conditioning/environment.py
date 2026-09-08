"""Reproducible runtime-environment reporting."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.ft2font
import numpy as np

from classical_conditioning.artifacts import write_json_atomic
from classical_conditioning.exceptions import ConfigurationError
from classical_conditioning.figures.theme import resolve_sans_serif_fonts

# PyArrow does not yet ship usable Windows wheels for CPython 3.14.
_SUPPORTED_PYTHON = ((3, 12), (3, 13))

LOCKED_DISTRIBUTIONS = (
    "matplotlib",
    "numba",
    "numpy",
    "pandas",
    "plotly",
    "pyarrow",
    "scikit-learn",
    "scipy",
    "seaborn",
    "statannotations",
    "statsmodels",
    "tqdm",
)
THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _distribution_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in LOCKED_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _numerical_libraries() -> dict[str, dict[str, Any]]:
    build_dependencies = np.show_config(mode="dicts").get(
        "Build Dependencies",
        {},
    )
    return {
        library: {
            field: details.get(field)
            for field in ("name", "version", "openblas configuration")
            if details.get(field) is not None
        }
        for library, details in build_dependencies.items()
        if library in {"blas", "lapack"}
    }


def ensure_supported_runtime() -> None:
    """Fail fast on Python builds that cannot import the locked PyArrow wheel."""
    version = sys.version_info[:2]
    if version not in _SUPPORTED_PYTHON:
        raise ConfigurationError(
            "classical-conditioning requires CPython 3.12 or 3.13. "
            f"This interpreter is {platform.python_version()}. "
            "Python 3.14 is not supported yet because pyarrow has no compatible "
            "wheel (pyarrow.lib import fails). Create/use the project 3.12 venv, "
            "for example: uv sync --python 3.12"
        )
    try:
        import pyarrow  # noqa: F401
    except Exception as error:  # pragma: no cover - depends on local install
        raise ConfigurationError(
            "PyArrow failed to import. Install the project environment with "
            "Python 3.12 or 3.13 (uv sync --python 3.12). "
            f"Original error: {error}"
        ) from error


def build_environment_report() -> dict[str, Any]:
    """Return versions and numerical/figure settings needed for reproduction."""
    return {
        "package": {
            "name": "classical-conditioning",
            "version": importlib.metadata.version("classical-conditioning"),
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "distributions": _distribution_versions(),
        "numerical_libraries": _numerical_libraries(),
        "threads": {
            "logical_cpu_count": os.cpu_count(),
            "environment": {
                variable: os.environ.get(variable)
                for variable in THREAD_ENVIRONMENT_VARIABLES
            },
            "policy": (
                "Unset variables use the numerical library defaults recorded "
                "under numerical_libraries."
            ),
        },
        "figures": {
            "matplotlib_version": matplotlib.__version__,
            "backend": str(matplotlib.get_backend()),
            "freetype_version": matplotlib.ft2font.__freetype_version__,
            "font_family": list(matplotlib.rcParams["font.family"]),
            "font_sans_serif": list(matplotlib.rcParams["font.sans-serif"]),
            "resolved_sans_serif": list(resolve_sans_serif_fonts()),
        },
    }


def write_environment_report(output_path: Path) -> Path:
    """Write an environment report atomically."""
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, build_environment_report())
    return output_path


def format_environment_report() -> str:
    """Return a deterministic human-readable JSON representation."""
    return json.dumps(build_environment_report(), indent=2, sort_keys=True)
