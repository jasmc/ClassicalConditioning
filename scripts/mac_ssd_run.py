"""Run an existing script on macOS with opt-in Windows path relocation.

Stored manifests and frozen artifacts are never rewritten. Only pathlib input
paths are relocated in this process; ordinary hash and scientific checks remain.
"""
import argparse
import os
import posixpath
import pathlib
import runpy
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]


def relocate(value, volume, repo):
    text = os.fspath(value)
    normalized = text.replace("\\", "/")
    # Avoid constructing Path objects here: this function runs inside Path.init.
    repo_text = os.fspath(repo).replace("\\", "/")
    volume_text = os.fspath(volume).replace("\\", "/").rstrip("/")
    mappings = [
        ("C:/Users/joaquim/Documents/ClassicalConditioning", repo_text),
        ("F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs", posixpath.dirname(repo_text) + "/pc-outputs"),
        ("F:/Digested Data", volume_text + "/Digested Data"),
        ("J:", volume_text),
    ]
    for old, new in mappings:
        if normalized.lower() == old.lower() or normalized.lower().startswith(old.lower() + "/"):
            return new + normalized[len(old):]
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", type=pathlib.Path, default=REPO.parent.parent)
    parser.add_argument("--check-paths", action="store_true")
    parser.add_argument("script", nargs="?")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    volume = args.volume.resolve()
    if args.check_paths:
        for value in [r"J:\Raw Data\allDelay", r"F:\Digested Data\all3sTrace-full-v1",
                      r"C:\Users\joaquim\Documents\ClassicalConditioning\reviews"]:
            print(value, "=>", relocate(value, volume, REPO))
        return
    if sys.platform != "darwin":
        parser.error("Execution mode is for macOS; --check-paths works on any OS")
    if not args.script:
        parser.error("Provide a Python script or :cli for the project CLI")
    # Python 3.12/3.13 parse path arguments in __init__, after choosing PosixPath.
    original_init = pathlib.Path.__init__
    def mapped_init(self, *parts):
        original_init(self, *(relocate(p, volume, REPO) for p in parts))
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
        runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
