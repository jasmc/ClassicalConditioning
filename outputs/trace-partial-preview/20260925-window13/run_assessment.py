"""Assess the immutable partial 3sTrace snapshot using its saved hashed intake."""

import json
from pathlib import Path

from classical_conditioning.analysis.discarding import assess_discarding


PROJECT = Path(r"F:\Digested Data\all3sTrace-full-v1")
RAW = Path(r"J:\Raw Data\all3sTtrace")
SNAPSHOT = Path(__file__).with_name("snapshot.json")


def main() -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    inventory = json.loads(
        (PROJECT / "Metadata" / "recording_inventory.json").read_text(encoding="utf-8")
    )
    if not inventory.get("source_hashes_included"):
        raise ValueError("Saved intake inventory lacks source hashes")
    ids = snapshot["recording_ids"]
    if len(ids) != 36 or len(set(ids)) != 36:
        raise ValueError("Preview snapshot must contain 36 distinct fish")
    for recording_id in ids:
        summary = json.loads(
            (PROJECT / "Quality checks" / recording_id
             / "candidate-trial-outcomes-corrected_summary.json").read_text(encoding="utf-8")
        )
        if summary.get("config", {}).get("response_window_s") != [0.0, 13.0]:
            raise ValueError(f"{recording_id} lacks the corrected 0–13 s window")
    result = assess_discarding(
        RAW, PROJECT,
        analysis_id="all3sTrace-window13-partial-36fish",
        experiment="all3sTrace",
        metric_id="tail_length_weighted_angular_l1",
        metric_recipe="tail-candidate-corrected",
        recording_ids=ids,
        inventory=inventory,
    )
    print(result.summary_path)


if __name__ == "__main__":
    main()
