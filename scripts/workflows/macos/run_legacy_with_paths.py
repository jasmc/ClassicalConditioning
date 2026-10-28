"""Run an existing script on macOS with opt-in Windows path relocation.

Stored manifests and frozen artifacts are never rewritten. Only pathlib input
paths are relocated in this process; ordinary hash and scientific checks remain.
"""

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]

import argparse
import json
import os
import posixpath
import pathlib
import runpy
import sys

REPO = _COMMAND_REPOSITORY
from classical_conditioning.external_artifacts import relocate_directory_reference


def relocate(value, volume, repo, directory_moves=()):
    text = os.fspath(value)
    normalized = text.replace("\\", "/")
    if '..' in normalized.split('/'):
        raise ValueError('Legacy input path cannot contain parent traversal')
    # Avoid constructing Path objects here: this function runs inside Path.init.
    repo_text = os.fspath(repo).replace("\\", "/")
    volume_text = os.fspath(volume).replace("\\", "/").rstrip("/")
    mappings = [
        ("C:/Users/joaquim/Documents/ClassicalConditioning", repo_text),
        ("F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs", volume_text + "/ClassicalConditioning-Mac-20261009/pc-outputs"),
        ("F:/Digested Data", volume_text + "/Digested Data"),
        ("J:", volume_text),
    ]
    for old, new in mappings:
        if normalized.lower() == old.lower() or normalized.lower().startswith(old.lower() + "/"):
            normalized = new + normalized[len(old):]
            break
    if normalized==volume_text or normalized.startswith(volume_text+'/'):
        canonical='/Volumes/JOAQUIM'+normalized[len(volume_text):]
        migrated=relocate_directory_reference(canonical,dict(directory_moves))
        if migrated=='/Volumes/JOAQUIM' or migrated.startswith('/Volumes/JOAQUIM/'):
            return volume_text+migrated[len('/Volumes/JOAQUIM'):]
        raise ValueError('Legacy storage relocation escapes JOAQUIM')
    return normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=pathlib.Path, default=pathlib.Path("/Volumes/JOAQUIM"))
    parser.add_argument("--check-paths", action="store_true")
    parser.add_argument("script", nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    volume = args.volume.resolve()
    index=REPO/'records/repository-archive/relocations.json'
    directory_moves=json.loads(index.read_bytes()).get('external_directory_moves',{}) if index.is_file() else {}
    if args.check_paths:
        for value in [r"J:\Raw Data\allDelay", r"F:\Digested Data\all3sTrace-full-v1",
                      r"C:\Users\joaquim\Documents\ClassicalConditioning\reviews"]:
            print(value, "=>", relocate(value, volume, REPO, directory_moves))
        return
    if sys.platform != "darwin":
        parser.error("Execution mode is for macOS; --check-paths works on any OS")
    if not args.script:
        parser.error("Provide a Python script or :cli for the project CLI")
    if not volume.is_dir() or not volume.is_mount():
        raise FileNotFoundError('JOAQUIM is unavailable; no path substitution was performed')
    # Python 3.12/3.13 parse path arguments in __init__, after choosing PosixPath.
    original_init = pathlib.Path.__init__
    def mapped_init(self, *parts):
        original_init(self, *(relocate(p, volume, REPO, directory_moves) for p in parts))
    pathlib.Path.__init__ = mapped_init
    sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
    os.chdir(REPO)
    sys.argv = [args.script, *args.arguments]
    if args.script == ":cli":
        from classical_conditioning.cli import main as cli_main
        cli_main()
    else:
        target = pathlib.Path(args.script)
        if not target.is_absolute():
            target = REPO / target
        from classical_conditioning.external_artifacts import resolve_artifact
        target = resolve_artifact(target, repo=REPO)
        if not target.resolve().is_relative_to(REPO.resolve()):
            raise ValueError("Retired recipe: recover its exact source and recorded environment; use the current scoped command for new work")
        runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
