from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path


class RepositoryOrganizationTests(unittest.TestCase):
    LEGACY_MODULES = {
        "my_experiment_specific_variables",
        "my_functions",
        "my_general_variables",
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]

    def test_archived_modules_are_not_imported_by_active_python(self) -> None:
        active_files = [
            *self.root.glob("*.py"),
            *(self.root / "src").rglob("*.py"),
            *(self.root / "tests").rglob("*.py"),
        ]
        violations: list[str] = []
        for path in active_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported = {alias.name.split(".", 1)[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported = {node.module.split(".", 1)[0]}
                else:
                    continue
                forbidden = imported.intersection(self.LEGACY_MODULES)
                if forbidden:
                    violations.append(
                        f"{path.relative_to(self.root)}: {sorted(forbidden)}"
                    )
        self.assertEqual(violations, [])

    def test_archived_modules_are_outside_the_default_import_path(self) -> None:
        module_directory = (self.root / "legacy" / "modules").resolve()
        import_roots = {
            Path(entry or Path.cwd()).resolve()
            for entry in sys.path
        }
        self.assertNotIn(module_directory, import_roots)
        for module_name in self.LEGACY_MODULES:
            self.assertFalse((self.root / f"{module_name}.py").exists())

    def test_legacy_modules_are_preserved_in_archive(self) -> None:
        module_directory = self.root / "legacy" / "modules"
        self.assertEqual(
            {
                path.stem
                for path in module_directory.glob("my_*.py")
            },
            self.LEGACY_MODULES,
        )

    def test_reviewed_stale_root_artifacts_are_absent(self) -> None:
        self.assertFalse((self.root / "jasmc.code-profile").exists())
        self.assertFalse((self.root / "tmp_axis_title_spine_anchor.png").exists())


if __name__ == "__main__":
    unittest.main()
