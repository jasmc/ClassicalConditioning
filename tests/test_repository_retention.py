"""Concrete guards for source moves and retained numerical import contracts."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]

class MaintainedDependencies(unittest.TestCase):
    def test_declared_local_command_dependencies_exist(self):
        missing = []
        for path in (ROOT / 'scripts').rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                if (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
                    and isinstance(node.left, ast.Name) and node.left.id in ('REPO', 'ROOT')
                    and isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)
                    and node.right.value.startswith('scripts/') and node.right.value.endswith('.py')
                    and not (ROOT / node.right.value).is_file()):
                    missing.append(f'{path.relative_to(ROOT)}:{node.lineno}: {node.right.value}')
        self.assertEqual(missing, [])

    def test_analysis_figure4_has_no_rendering_dependencies(self):
        path = ROOT / 'src/classical_conditioning/analysis/figure4.py'
        imported = [n.module for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any(m and m.startswith('classical_conditioning.figures') for m in imported))

    def test_old_numerical_imports_remain_compatible(self):
        from classical_conditioning.analysis.cohort_selection import FrozenCohortSelection, load_primary_cohort
        from classical_conditioning.figures.cohort_response import FrozenCohortSelection as old_type, _load_primary_cohort
        from classical_conditioning.analysis.signed_bout_heatmap import calculate_fish_heatmaps
        from classical_conditioning.figures.signed_bout_heatmap import calculate_fish_heatmaps as old_calculate
        self.assertIs(FrozenCohortSelection, old_type)
        self.assertIs(load_primary_cohort, _load_primary_cohort)
        self.assertIs(calculate_fish_heatmaps, old_calculate)
