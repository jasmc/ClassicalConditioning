"""Rebuild only stale 3sTrace candidate trial outcomes for the 0–13 s window."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Lock
from datetime import datetime, timezone


PROJECT = Path(r"F:\Digested Data\all3sTrace-full-v1")
EXPECTED_WINDOW = [0.0, 13.0]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _current(project: Path, recording_id: str) -> bool:
    summary = project / "Quality checks" / recording_id / "candidate-trial-outcomes-corrected_summary.json"
    marker = project / "Metadata" / f"{recording_id}_candidate-trial-outcomes-corrected_complete.json"
    if not summary.is_file() or not marker.is_file():
        return False
    try:
        details = json.loads(summary.read_text(encoding="utf-8"))
        completion = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (details.get("experiment") == "all3sTrace"
            and details.get("config", {}).get("response_window_s") == EXPECTED_WINDOW
            and completion.get("status") == "complete")


def _rebuild(project: Path, recording_id: str, log_dir: Path) -> tuple[str, str]:
    command = [
        sys.executable, "-m", "classical_conditioning", "candidate-trial-outcomes",
        "--project-dir", str(project), "--recording-id", recording_id,
        "--experiment", "all3sTrace", "--recipe", "candidate-trial-outcomes-corrected",
        "--overwrite",
    ]
    with (log_dir / f"{recording_id}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0 or not _current(project, recording_id):
        return recording_id, f"failed: exit={result.returncode}"
    return recording_id, "rebuilt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=PROJECT)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    project = args.project_dir.resolve()
    if project != PROJECT.resolve() or args.workers not in (1, 2):
        raise ValueError("This rebuild is restricted to the named F: 3sTrace project and 1–2 workers")
    pipeline = json.loads((project / "Metadata" / "all3sTrace-full_pipeline_run.json").read_text(encoding="utf-8"))
    ids = tuple(str(value) for value in pipeline["recording_ids"])
    if (pipeline.get("status") != "complete" or len(ids) != 59
            or len(set(ids)) != len(ids) or set(ids) != set(pipeline["active_recording_ids"])):
        raise ValueError("The prior 59-fish candidate pipeline is not complete")
    output = Path(__file__).resolve().parents[1] / "outputs" / "trace-transfer-review"
    log_dir = output / "window13-fish-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = output / "window13-rebuild-status.json"
    state = {
        "status": "running", "project": str(project), "response_window_s": EXPECTED_WINDOW,
        "fish_total": len(ids), "workers": args.workers,
        "fish": {name: "current" if _current(project, name) else "pending" for name in ids},
    }
    lock = Lock()

    def save() -> None:
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        state["completed"] = sum(value in ("current", "rebuilt") for value in state["fish"].values())
        state["failed"] = sum(value.startswith("failed") for value in state["fish"].values())
        _write_json(status_path, state)

    save()
    pending = [name for name in ids if state["fish"][name] == "pending"]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_rebuild, project, name, log_dir): name for name in pending}
        for future in as_completed(futures):
            try:
                name, result = future.result()
            except Exception as error:
                name, result = futures[future], f"failed: {type(error).__name__}: {error}"
            with lock:
                state["fish"][name] = result
                save()
                print(f"{name}: {result} ({state['completed']}/{len(ids)})", flush=True)
    state["status"] = "complete" if state["completed"] == len(ids) else "failed"
    save()
    if state["status"] != "complete":
        raise RuntimeError(f"{state['failed']} fish failed; inspect {log_dir}")
    print(f"All {len(ids)} 3sTrace fish use the 0–13 s candidate response window.")


if __name__ == "__main__":
    main()
