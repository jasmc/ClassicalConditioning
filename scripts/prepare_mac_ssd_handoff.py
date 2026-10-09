"""Prepare the requested SSD handoff without altering existing source artifacts."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
SSD = Path('J:/')
HANDOFF = SSD / 'ClassicalConditioning-Mac-20261009'
REPO = HANDOFF / 'ClassicalConditioning'
TRACE = Path('F:/Digested Data/all3sTrace-full-v1')
EXCLUDE_DIRS = {'.venv', '.venv-trace', '.cache', '__pycache__', 'fsmonitor--daemon.ipc'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def files(root):
    import os
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS
                   and (not d.startswith('.') or d in {'.git', '.vscode'})]
        for name in names:
            if name.endswith(('.pyc', '.log')) or name == 'fsmonitor--daemon.ipc':
                continue
            yield Path(base) / name


def main():
    jobs = []
    for path in files(ROOT):
        if path.name.startswith('.tmp_'):
            continue
        # Author steering: transfer code/figures only, never fish data tables.
        if path.suffix.lower() in {'.parquet', '.pkl', '.pickle', '.npy', '.npz', '.h5', '.hdf5'}:
            continue
        if path.suffix.lower() == '.csv' and path.parts[-2] != 'Plans':
            continue
        jobs.append((path, REPO / path.relative_to(ROOT), 'repository'))
    for path in files(Path('F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs')):
        if path.suffix.lower() not in {'.svg', '.pdf', '.png', '.html', '.json', '.md', '.ttf', '.otf', '.py'}:
            continue
        jobs.append((path, HANDOFF / 'pc-outputs' / path.relative_to(Path('F:/ClassicalConditioning Outputs/ORGER-JOAQUIM/outputs')), 'pc-outputs'))
    # Exported Trace figures are useful for editing without copying the project.
    figure_root = TRACE / 'Figures'
    for path in files(figure_root):
        if path.suffix.lower() in {'.svg', '.pdf', '.png', '.html', '.json'}:
            jobs.append((path, HANDOFF / 'trace-figure-exports' / path.relative_to(figure_root), 'trace-figure-export'))
    required = sum(src.stat().st_size for src, dest, _ in jobs if not dest.exists())
    free = shutil.disk_usage(SSD).free
    print(json.dumps({'files':len(jobs), 'copy_GiB':required/1024**3, 'free_GiB':free/1024**3}), flush=True)
    if required + 8*1024**3 > free:
        raise RuntimeError('Insufficient SSD room with 8 GiB reserve; nothing copied')
    HANDOFF.mkdir(exist_ok=True)
    manifest_path = HANDOFF / 'transfer-manifest.jsonl'
    started = time.time()
    total = 0
    with manifest_path.open('w', encoding='utf-8') as manifest:
        for index, (src, dest, group) in enumerate(jobs, 1):
            dest.parent.mkdir(parents=True, exist_ok=True)
            before = src.stat()
            source_hash = sha(src)
            if dest.exists():
                if sha(dest) != source_hash:
                    raise RuntimeError(f'Existing different file preserved; cannot overwrite: {dest}')
            else:
                shutil.copy2(src, dest)
                if sha(dest) != source_hash:
                    raise RuntimeError(f'Copy hash mismatch: {dest}')
            after = src.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f'Source changed during transfer: {src}')
            entry={'source':str(src),'destination':str(dest.relative_to(SSD)), 'bytes':before.st_size,'sha256':source_hash,'group':group}
            manifest.write(json.dumps(entry)+'\n'); manifest.flush()
            total += before.st_size
            if index % 100 == 0 or before.st_size > 1024**3:
                print(f'{index}/{len(jobs)} verified; {total/1024**3:.2f} GiB; {src.name}', flush=True)
    # Keep history plus exact local modifications; disable PC-only fsmonitor.
    subprocess.run(['git','config','--file',str(REPO/'.git/config'),'core.fsmonitor','false'],check=True)
    subprocess.run(['git','config','--file',str(REPO/'.git/config'),'core.untrackedCache','false'],check=True)
    summary={'status':'complete','scope':'code and figure artifacts only; author explicitly prohibited fish-data transfer','files':len(jobs),'verified_bytes':total,'elapsed_s':time.time()-started,'free_bytes':shutil.disk_usage(SSD).free,'verification':'SHA-256 source and destination for every transfer; existing raw/Delay trees not fully rehashed','git_config_exception':'Destination .git/config disables fsmonitor and untrackedCache after transfer; source config preserved','missing':'Processed 3sTrace project and review data tables remain on PC; 10sTrace raw/corrected data absent on SSD'}
    (HANDOFF/'transfer-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
