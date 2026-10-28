"""Local implementation closure for guarded commands; never imports a command."""
from functools import lru_cache
from pathlib import Path
import ast
import re
from classical_conditioning.external_artifacts import REPOSITORY


@lru_cache(maxsize=128)
def command_sources(command):
    """Return code/resources affecting a command so guards bind extracted helpers.

    Only source files inside the current checkout are followed. Existing scientific
    data/state bindings remain the caller's explicit responsibility.
    """
    root = REPOSITORY.resolve()
    pending, found = [Path(command).resolve(), Path(__file__).resolve()], set()
    def module_path(name):
        if name.startswith('classical_conditioning'):
            base = root / 'src' / name.replace('.', '/')
        elif name.startswith('scripts.'):
            base = root / name.replace('.', '/')
        else:
            return
        for parent in base.parents:
            if not parent.is_relative_to(root):
                break
            init = parent / '__init__.py'
            if init.is_file():
                pending.append(init)
        for candidate in [base.with_suffix('.py'), base / '__init__.py']:
            if candidate.is_file():
                pending.append(candidate)
                return
    while pending:
        path = pending.pop().resolve()
        if path in found or not path.is_relative_to(root) or not path.is_file():
            continue
        found.add(path)
        text = path.read_text()
        literals = []
        if path.suffix == '.py':
            tree = ast.parse(text)
            relative = path.relative_to(root / 'src') if path.is_relative_to(root / 'src') else path.relative_to(root)
            package = '.'.join(relative.with_suffix('').parts[:-1])
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_path(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    prefix = package.split('.')[:len(package.split('.')) - node.level + 1] if node.level else []
                    module = '.'.join([*prefix, *([node.module] if node.module else [])])
                    module_path(module)
                    for alias in node.names:
                        module_path(module + '.' + alias.name)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    literals.append(node.value)
        else:
            literals = re.findall(r"['\"]([^'\"\n]+)['\"]", text)
        for literal in literals:
            if literal.endswith(('.py', '.js', '.cjs')):
                if literal.startswith(('src/classical_conditioning/', 'scripts/')):
                    pending.append(root / literal)
                elif literal.startswith('./'):
                    pending.append(path.parent / literal)
    return tuple(sorted(found))
