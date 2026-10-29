"""Validate explicit candidate/evidence locations without creating anything."""
from pathlib import Path, PureWindowsPath
from classical_conditioning.external_artifacts import REPOSITORY


def candidate_directory(value, *, existing=False, repo=REPOSITORY):
    path=Path(value).expanduser()
    if PureWindowsPath(str(value)).drive and not path.is_absolute():
        raise ValueError('Use an absolute path for this host, not a foreign drive path')
    path=path.resolve()
    if path.is_relative_to(Path(repo).resolve()):
        raise ValueError('Candidate/evidence directories must be outside the repository')
    if any(p.casefold() in {'raw data', 'results (paper)'} for p in path.parts):
        raise ValueError('Candidate/evidence output cannot enter protected raw/paper inputs')
    if path.parts[:2]==('/', 'Volumes') and len(path.parts)>2:
        volume=Path(*path.parts[:3])
        if not volume.is_mount():
            raise FileNotFoundError('External volume is unavailable: '+str(volume))
    if existing:
        if not path.is_dir():raise FileNotFoundError('Existing candidate/evidence directory required: '+str(path))
    elif not path.parent.is_dir():
        raise FileNotFoundError('Candidate parent must already exist: '+str(path.parent))
    return path


def evidence_file(value):
    path=Path(value).expanduser().resolve()
    candidate_directory(path.parent,existing=True)
    if not path.is_file():raise FileNotFoundError('Existing browser evidence required: '+str(path))
    return path
