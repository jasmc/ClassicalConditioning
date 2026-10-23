#!/usr/bin/env python3
"""Append the dated chat supplement, preserving the original scientific snapshot.

No fits, artwork edits or selection changes. Back up the complete old bundle
before publication; keep notebook keys and old chapter markup unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[1]
SLUG = "14-chat-completion-clarifications-model-teaching-and-next-steps"
TITLE = "14. Chat completion: clarifications, model teaching and next steps"
SOURCES = {
    "completion-plan03": "Plans/03_FIGURE_2_LEARNING_ONSET.md",
    "completion-plan06": "Plans/06_ANALYSIS_AND_STATISTICS.md",
    "completion-population-code": "src/classical_conditioning/analysis/inference/panel_g_review.py",
    "completion-fast-backend": "src/classical_conditioning/analysis/inference/panel_g_fast_lmm.py",
    "completion-model-reference": "docs/running-analysis/learning-models.md",
    "completion-figure3-methods": "docs/running-analysis/figure3-main-paper-draft.md",
    "completion-figure3-state": "docs/paper/figures/figure-03/current-state.json",
    "completion-individual-methods": "docs/paper/methods/individual-learning.md",
    "completion-guide-manuscript": "docs/paper/figures/figure-02/panel-g/analysis-guide.md",
}


def bundle_markdown(markdown):
    """Keep saved-source links usable in both repository and HTML contexts."""
    return re.sub(
        r"\]\(<(/Volumes/JOAQUIM/ClassicalConditioning Outputs/current/reviews/figure-02-g-analysis-guide-20261017-v01/sources/[^>]+)>\)",
        lambda match: "](sources/" + Path(match[1]).name + ")",
        markdown,
    )


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guide-dir", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    parser.add_argument("--node", required=True)
    parser.add_argument("--node-modules", required=True)
    parser.add_argument("--pasted-summary", type=Path, required=True)
    args = parser.parse_args()
    root, backup = args.guide_dir.resolve(), args.backup_dir.resolve()
    if not root.is_relative_to(Path("/Volumes/JOAQUIM")) or backup.exists():
        raise ValueError("Mounted JOAQUIM and a new backup directory are required")
    old = (root / "review.html").read_text()
    if f"id='{SLUG}'" in old:
        raise ValueError("Supplement already published; inspect before revising")
    manifest = json.loads((root / "evidence-manifest.json").read_text())
    for artifact in manifest["artifacts"]:
        if sha((root / artifact["path"]).read_bytes()) != artifact["sha256"]:
            raise ValueError("Existing bundle differs from manifest: " + artifact["path"])
    # Back up every existing byte, including historical sources and figures.
    shutil.copytree(root, backup)
    protected = {str(p.relative_to(root)): sha(p.read_bytes()) for p in root.rglob("*")
                 if p.is_file() and p.suffix in {".svg", ".png", ".parquet"}}
    manuscript = (REPO / SOURCES["completion-guide-manuscript"]).read_text()
    content = manuscript.split("## " + TITLE + "\n", 1)[1]
    env = dict(os.environ, NODE_PATH=args.node_modules)
    rendered = subprocess.check_output(
        [args.node, "-e", "const fs=require('fs');const {marked}=require('marked');process.stdout.write(marked.parse(fs.readFileSync(0,'utf8')));"],
        input=bundle_markdown(content).encode(), env=env).decode()
    rendered = re.sub(r"<table>.*?</table>", lambda m: "<div class='table-wrap' tabindex='0'>" + m[0] + "</div>", rendered, flags=re.S)
    # Reuse the compatible notebook markup; old section IDs/semantics stay intact.
    note = re.search(r"<details class='notebook no-print'.*?</details>", old, re.S)[0]
    first = manifest["guide"]["sections"][0]
    note = note.replace(first["id"], SLUG).replace(html.escape(first["title"]), html.escape(TITLE))
    section = f"<section class='chapter' id='{SLUG}'><h2>{html.escape(TITLE)}</h2>{rendered}{note}</section>"
    new = old.replace("<section class='chapter' id='source-inventory'>", section + "<section class='chapter' id='source-inventory'>", 1)
    new = new.replace("<a href='#source-inventory'>Source inventory</a>", f"<a href='#{SLUG}'>{html.escape(TITLE)}</a><a href='#source-inventory'>Source inventory</a>", 1)
    banner = f"<div class='callout'><b>Chat completeness supplement · 23 October 2026.</b> Later clarifications, model teaching, GAMM proposal and Figure 3 context are preserved in <a href='#{SLUG}'>chapter 14</a>. Original analyses/artwork remain unchanged. Export personal notebook notes before deleting the chat.</div>"
    new = new.replace("<section class='chapter' id='01-how-to-use-this-guide'>", banner + "<section class='chapter' id='01-how-to-use-this-guide'>", 1)
    config_match = re.search(r"(<script type='application/json' id='guide-config'>)(.*?)(</script>)", new, re.S)
    config = json.loads(config_match[2])
    original_key = (config["guide_id"], config["source_fingerprint"])
    config["sections"].append({"id": SLUG, "title": TITLE})
    config["documentation_supplement"] = "2026-10-23; bound separately in evidence-manifest.json; core analysis snapshot and notebook key preserved"
    new = new[:config_match.start(2)] + json.dumps(config) + new[config_match.end(2):]
    additions = {}
    source_paths = {key: REPO / rel for key, rel in SOURCES.items()}
    source_paths["completion-other-chat-summary"] = args.pasted_summary
    for key, path in source_paths.items():
        raw = path.read_bytes()
        suffix = ".txt" if key == "completion-other-chat-summary" else path.suffix
        relative = f"sources/{key}{suffix}"
        (root / relative).write_bytes(raw)
        additions[key] = {"title": path.name, "path": str(path), "sha256": sha(raw), "copy": relative,
                          "verification": "Dated documentation/code snapshot, not proof of an executed fit", "note": "Chat completeness supplement, 2026-10-23"}
        card = f"<div class='source-card' id='source-{key}'><h4>23 October supplement: {html.escape(path.name)}</h4><a href='{relative}'>Open saved source</a><p>{html.escape(additions[key]['note'])}</p><code>{html.escape(str(path))}</code><br><code>SHA-256 {sha(raw)}</code></div>"
        new = new.replace("</section><footer>", card + "</section><footer>", 1)
    assert original_key == (config["guide_id"], config["source_fingerprint"])
    # Verify old chapter markup is byte-identical, including embedded artwork.
    original_chapters = re.findall(r"<section class='chapter' id='(?!source-inventory)([^']+)'>.*?</section>", old, re.S)
    for chapter_id in original_chapters:
        pattern = rf"<section class='chapter' id='{re.escape(chapter_id)}'>.*?</section>"
        assert re.search(pattern, old, re.S)[0] == re.search(pattern, new, re.S)[0]
    for relative, expected in protected.items():
        assert sha((root / relative).read_bytes()) == expected
    topics = json.loads((root / "topic-inventory.json").read_text())
    topics["chapters"].append({"section": SLUG, "title": TITLE, "has_notebook": True})
    for topic in ["Author conditional-intensity clarification and frame weighting", "Configured versus fitted parameters and actual model structures", "Baseline and multiplicity worked examples", "GAMM proposal, boundary argument and saved exploration plan", "Figure 3 acquisition-only score versus legacy recovery and extinction", "Clean-chat handover, unresolved questions and deletion-safe source copies"]:
        topics["coverage"].append({"topic": topic, "section": SLUG})
    topics["source_ids"].extend(additions)
    write_json(root / "topic-inventory.json", topics)
    manifest["sources"].update(additions)
    manifest["guide"] = config
    manifest["guide"]["source_fingerprint_role"] = "Original analysis-evidence and notebook identity; later documentation revisions are hash-bound separately, with compatible chapter IDs preserved."
    manifest.setdefault("maintenance", []).append({"date": "2026-10-23", "scope": "Chat completeness, explanatory supplement only", "backup": str(backup), "previous_html_sha256": sha(old.encode()), "new_html_sha256": sha(new.encode()), "new_source_ids": list(additions), "original_chapters_byte_identical": len(original_chapters), "notebook_key_unchanged": True, "artwork_payloads_unchanged": len(protected), "no_models_refitted": True})
    # Publish the document atomically after the support files exist.
    with tempfile.NamedTemporaryFile(dir=root, delete=False) as temp:
        temp.write(new.encode())
        temp_name = temp.name
    os.replace(temp_name, root / "review.html")
    manifest["artifacts"] = [{"path": str(p.relative_to(root)), "sha256": sha(p.read_bytes())} for p in sorted(root.rglob("*")) if p.is_file() and p.name != "evidence-manifest.json"]
    write_json(root / "evidence-manifest.json", manifest)
    print(json.dumps(manifest["maintenance"][-1], indent=2))


if __name__ == "__main__":
    main()
