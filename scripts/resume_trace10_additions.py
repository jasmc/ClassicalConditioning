"""Compatibility entry point; maintained implementation is scoped in scripts."""
from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / 'pyproject.toml').is_file() and (p / 'src/classical_conditioning').is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / 'src'), str(_COMMAND_REPOSITORY / 'scripts')]

from scripts.workflows.trace import resume_fixed10_additions as _implementation
if __name__ == '__main__':
    if hasattr(_command_sys.stdout,'reconfigure'):
        _command_sys.stdout.reconfigure(encoding='utf-8')
    _implementation.main()
else:
    _command_sys.modules[__name__] = _implementation
