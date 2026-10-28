"""Check review navigation and publish a portable reading library on external storage.

This catalogue selects reading links, never scientific or presentation state.
Existing reviews, numerical files and notebook storage are never rewritten.
"""
from __future__ import annotations

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]


import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import quote

ROOT = _COMMAND_REPOSITORY
CATALOGUE = ROOT / "docs/paper/figures/review-library.json"
OUTPUTS = Path("/Volumes/JOAQUIM/ClassicalConditioning Outputs")
ARCHIVE = Path("/Volumes/JOAQUIM/ClassicalConditioning-Archive")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def packed(data):
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode()


def contained(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative:
        raise ValueError("Catalogue paths must be relative without traversal: " + relative)
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Catalogue path escapes its root: " + relative)
    return result


def check(catalogue, repo=ROOT, outputs=OUTPUTS, archive=ARCHIVE, *, external=False, check_browser_sources=True):
    roots = {"repo": repo, "outputs": outputs, "archive": archive}
    issues, checked, ids = [], [], set()
    groups = set()
    for group in catalogue["groups"]:
        if group["id"] in groups:
            issues.append("Duplicate group: " + group["id"])
        groups.add(group["id"])
        for entry in group["entries"]:
            if entry["id"] in ids:
                issues.append("Duplicate entry: " + entry["id"])
            ids.add(entry["id"])
            storage = entry["storage"]
            try:
                path = contained(roots[storage], entry["path"])
                if storage != "repo" and not re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256", "")):
                    raise ValueError("External review needs its exact observed hash")
                if storage == "repo" or external:
                    if not path.is_file():
                        raise ValueError("Missing reading target: " + str(path))
                    raw = path.read_bytes()
                    if entry.get('kind') == 'generated_navigation':
                        publication = outputs / 'review-library/catalogue-publication.json'
                        saved = json.loads(publication.read_bytes()) if publication.is_file() else {}
                        expected = next((r['sha256'] for r in saved.get('generated_pages', [])
                                         if r['path'] == entry['path']), entry.get('sha256'))
                        if digest(raw) != expected:
                            raise ValueError('Generated reading target has unrecorded edits: ' + str(path))
                    elif entry.get("sha256") and digest(raw) != entry["sha256"]:
                        raise ValueError("Reading target changed: " + str(path))
                    checked.append({"id": entry["id"], "path": str(path), "sha256": digest(raw)})
            except (KeyError, ValueError, OSError) as error:
                issues.append(entry["id"] + ": " + str(error))
    if external and catalogue.get('version_browser') == 1:
        from scripts.reviews.ordered_library import verify_package
        issues.extend(verify_package(outputs, sources=check_browser_sources))
        model_path = outputs / 'review-library/versions.json'
        if check_browser_sources and model_path.is_file():
            from scripts.reviews.versions import source_files, GENERATED
            model = json.loads(model_path.read_bytes())
            recorded = {(r['storage'], r['path']) for r in model['inventory']
                        if r.get('storage') and r.get('path') and Path(r['path']).name not in GENERATED}
            observed = {(storage, str(path)) for storage, path in source_files(outputs.resolve(), archive.resolve())
                        if path.name not in GENERATED}
            for _, path in sorted(observed - recorded):
                issues.append('Uninventoried browser source; rebuild required: ' + path)
            for _, path in sorted(recorded - observed):
                issues.append('Inventoried browser source missing: ' + path)
    return {"issues": issues, "groups": len(groups), "entries": len(ids),
            "checked": checked, "external": external}


CSS = """body{font:17px/1.6 system-ui,sans-serif;color:#20272b;background:#f7f5ef;
margin:0}main{max-width:1050px;margin:auto;padding:32px 24px 80px}h1{line-height:1.15}
nav{display:flex;gap:16px;flex-wrap:wrap}a{color:#125b76;text-underline-offset:3px}
section{margin-top:40px}article{background:white;border:1px solid #d7dcd9;
border-radius:8px;padding:18px 22px;margin:14px 0}h3{margin:0 0 8px;font-size:1.1em}
.status{font-size:.8em;font-weight:650;color:#4d5960}p{margin:8px 0}
code,pre{font:14px/1.6 ui-monospace,monospace}pre{white-space:pre-wrap;overflow-wrap:anywhere;
background:white;padding:24px;border:1px solid #d7dcd9}small{overflow-wrap:anywhere}
input{font:inherit;padding:10px;width:min(90%,550px)}[hidden]{display:none!important}
@media print{input{display:none}body{background:white}article{break-inside:avoid}}
"""


def page(title, content):
    return ("<!doctype html><html lang='en'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style><main>{content}</main></html>").encode()


def atomic(path, raw):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temp.write(raw)
        name = temp.name
    os.replace(name, path)


def build(catalogue, output_root, archive_root=ARCHIVE, repo=ROOT):
    if catalogue.get('version_browser') == 1:
        from scripts.reviews.ordered_library import build as ordered_build
        return ordered_build(catalogue, output_root, archive_root, repo)
    output_root = output_root.resolve()
    if not output_root.is_dir() or output_root.is_relative_to(repo.resolve()):
        raise ValueError("Use an existing output root outside the repository")
    if str(output_root).startswith("/Volumes/") and not os.path.ismount("/Volumes/JOAQUIM"):
        raise ValueError("JOAQUIM is not mounted; no mount will be fabricated")
    report = check(catalogue, repo, output_root, archive_root, external=True)
    if report["issues"]:
        raise ValueError("\n".join(report["issues"]))
    index = output_root / "START-HERE.html"
    markdown_index = output_root / "REVIEW-LIBRARY.md"
    control = output_root / "review-library/catalogue-publication.json"
    before = index.read_bytes() if index.exists() else None
    markdown_before = markdown_index.read_bytes() if markdown_index.exists() else None
    if before is not None:
        if not control.is_file() or digest(before) != json.loads(control.read_text())["html_sha256"]:
            raise ValueError("START-HERE.html has unrecorded edits; preserve and reconcile them")
    if markdown_before is not None:
        if not control.is_file() or digest(markdown_before) != json.loads(control.read_text()).get("markdown_sha256"):
            raise ValueError("REVIEW-LIBRARY.md has unrecorded edits; preserve and reconcile them")
    identities = {row["id"]: row["sha256"] for row in report["checked"]}
    inventory = []
    for storage, root in (("outputs", output_root), ("archive", archive_root)):
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if catalogue.get('storage_layout') == 'figure-panel-v1':
                if rel.parts[0] not in {'figures', 'shared'} or 'reading-originals' in rel.parts:
                    continue  # Recovery/transport originals are not everyday reading guides.
            if (path.is_file() and path.suffix.lower() in {".html", ".md"}
                    and "review-library" not in rel.parts and not path.name.startswith("._")
                    and path.name not in {"START-HERE.html", "REVIEW-LIBRARY.md"}):
                inventory.append({"storage": storage, "path": rel.as_posix()})
    fingerprint = digest(packed({"catalogue": catalogue, "sources": identities,
                                "inventory": inventory}))[:16]
    relative = "review-library/" + fingerprint
    destination = output_root / relative
    files = {"catalogue.json": packed(catalogue)}
    roots = {"repo": repo, "outputs": output_root, "archive": archive_root}
    cards = []
    markdown = ["# Review library", "", catalogue["purpose"], "",
                "Open [START-HERE.html](../../START-HERE.html) for the main reading page.", ""]
    stable_markdown = ["# Review library", "", catalogue["purpose"], "",
        "[Searchable HTML index](START-HERE.html). Reading navigation does not select science or artwork.", ""]
    for group in catalogue["groups"]:
        entries = []
        markdown.extend(["## " + group["title"], "", group["context"], ""])
        stable_markdown.extend(["## " + group["title"], "", group["context"], ""])
        for entry in group["entries"]:
            source = contained(roots[entry["storage"]], entry["path"])
            raw = source.read_bytes()
            if digest(raw) != identities[entry["id"]]:
                raise ValueError("Source changed during build: " + str(source))
            if entry["storage"] == "repo" or source.suffix == ".md":
                snapshot = "reading/" + ("" if entry["storage"] == "repo" else entry["storage"] + "/") + entry["path"]
                files[snapshot] = raw
                href = relative + "/" + snapshot
                if source.suffix == ".md":
                    viewer = snapshot + ".html"
                    back = quote(os.path.relpath(index, (destination / viewer).parent), safe="/.")
                    files[viewer] = page(entry["title"], f"<p><a href='{back}'>Back to review library</a></p>"
                        f"<h1>{html.escape(entry['title'])}</h1><p>Verbatim Markdown reading snapshot · {html.escape(catalogue['reviewed'])}</p>"
                        f"<p><a href='{quote(source.name)}'>Read-only Markdown snapshot</a> · <a href='{quote(os.path.relpath(source, (destination / viewer).parent), safe='/ .').replace(' ', '%20')}'>Canonical source</a></p><pre>{html.escape(raw.decode())}</pre>")
                    href = relative + "/" + viewer
            else:
                href = os.path.relpath(source, output_root).replace(os.sep, "/")
            link = quote(href, safe="/.")
            stable_markdown.append(f"- [{entry['title']}]({link}) — {entry['status']}. {entry['description']}")
            markdown.append(f"- [{entry['title']}]({quote(os.path.relpath(output_root / href, destination), safe='/ .').replace(' ', '%20')}) — {entry['status']}. {entry['description']}")
            extra = (f" · <a href='{quote(relative + '/reading/' + entry['path'], safe='/ .').replace(' ', '%20')}'>Markdown/source</a>"
                     if entry["storage"] == "repo" else "")
            entries.append(f"<article data-search><span class='status'>{html.escape(entry['status'])}</span>"
                f"<h3><a href='{link}'>{html.escape(entry['title'])}</a></h3><p>{html.escape(entry['description'])}</p>"
                f"<small>{html.escape(entry['path'])}{extra}</small></article>")
        cards.append(f"<section id='{html.escape(group['id'])}'><h2>{html.escape(group['title'])}</h2><p>{html.escape(group['context'])}</p>{''.join(entries)}</section>")
    files["README.md"] = ("\n".join(markdown) + "\n").encode()
    all_links = []
    for item in inventory:
        source = contained(roots[item["storage"]], item["path"])
        link = quote(os.path.relpath(source, destination), safe="/.")
        role = "Archive" if item["storage"] == "archive" else "Saved output"
        if any(part in Path(item['path']).parts for part in ('sources', 'code-snapshot')):
            role += ' · source snapshot'
        if any(token in item["path"].lower() for token in ("/history/", "previous-selected", "preview", "-qa", "render-history")):
            role += " · history/QA"
        all_links.append(f"<article data-search><span class='status'>{role}</span>"
                         f"<a href='{link}'>{html.escape(item['path'])}</a></article>")
    files["ALL-REVIEWS.html"] = page("All saved reviews", f"<h1>All saved HTML and Markdown reviews</h1><p><a href='../../START-HERE.html'>Back to curated review library</a></p>"
        "<p>This inventory includes retained figure versions, source snapshots and QA. Transport and repository-recovery records remain separately in Archive. Its order does not select a current result.</p>"
        "<label>Filter paths <input id='search' type='search'></label>" + "".join(all_links)
        + "<script>document.querySelector('#search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('[data-search]').forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(q))})</script>")
    manifest = {"catalogue_sha256": digest(packed(catalogue)), "source_identities": report["checked"],
                "files": [{"path": key, "sha256": digest(raw)} for key, raw in sorted(files.items())]}
    files["manifest.json"] = packed(manifest)
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        if any(not (destination / key).is_file() or (destination / key).read_bytes() != raw for key, raw in files.items()):
            raise ValueError("Existing reading snapshot differs; preserve its edits")
    else:
        with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
            stage = Path(temp) / "snapshot"
            stage.mkdir()
            for key, raw in files.items():
                path = stage / key
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            os.replace(stage, destination)
    navigation = " · ".join(f"<a href='#{html.escape(g['id'])}'>{html.escape(g['title'])}</a>" for g in catalogue["groups"])
    content = (f"<h1>Classical Conditioning<br>Review library</h1><p>{html.escape(catalogue['purpose'])}</p>"
        f"<p>Reviewed {html.escape(catalogue['reviewed'])}. These are reading links. Accepted science and selected artwork remain in repository selection/state records.</p>"
        f"<nav>{navigation}</nav><p><label>Find a topic or approach <input id='search' type='search' placeholder='e.g. GAMM, 10sTrace, model'></label></p>"
        f"<p><a href='REVIEW-LIBRARY.md'>Markdown index</a> · <a href='{relative}/manifest.json'>Source identities</a> · <a href='{relative}/ALL-REVIEWS.html'>All saved HTML and Markdown reviews</a></p>"
        + "".join(cards) + "<script>document.querySelector('#search').addEventListener('input',e=>{const q=e.target.value.toLowerCase();document.querySelectorAll('[data-search]').forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(q))})</script>")
    result = page("Classical Conditioning — Review library", content)
    markdown_result = ("\n".join(stable_markdown) + "\n").encode()
    if (index.read_bytes() if index.exists() else None) != before:
        raise ValueError("Navigation edited during build; completed snapshot preserved, pointer unchanged")
    if (markdown_index.read_bytes() if markdown_index.exists() else None) != markdown_before:
        raise ValueError("Markdown navigation edited during build; pointer unchanged")
    atomic(index, result)
    atomic(markdown_index, markdown_result)
    publication = {"html_sha256": digest(result), "snapshot": relative,
                   "markdown_sha256": digest(markdown_result),
                   "manifest_sha256": digest(files["manifest.json"]), "entries": report["entries"]}
    atomic(control, packed(publication))
    return publication


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "build"])
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--output-root", type=Path, default=OUTPUTS)
    parser.add_argument("--archive-root", type=Path, default=ARCHIVE)
    parser.add_argument("--external", action="store_true")
    args = parser.parse_args()
    catalogue = json.loads(args.catalogue.read_text())
    if args.command == "build":
        result = build(catalogue, args.output_root, args.archive_root)
    else:
        result = check(catalogue, ROOT, args.output_root, args.archive_root, external=args.external)
    print(json.dumps(result, indent=2))
    if result.get("issues"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
