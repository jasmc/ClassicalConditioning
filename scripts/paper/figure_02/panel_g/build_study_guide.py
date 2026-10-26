#!/usr/bin/env python3
"""Build an evidence-bound teaching guide; never fit models or publish figure state.

Permanent output must be a NEW directory on mounted JOAQUIM. Markdown prose is
resolved through the authenticated SSD source registry; numerical tables are
read from saved artifacts.
Requires Node's `marked` package (the desktop dependency runtime supplies it).
"""
from __future__ import annotations

from pathlib import Path as _CommandPath
import sys as _command_sys
_COMMAND_REPOSITORY = next(p for p in _CommandPath(__file__).resolve().parents
                           if (p / "pyproject.toml").is_file() and (p / "src/classical_conditioning").is_dir())
_command_sys.path[:0] = [str(_COMMAND_REPOSITORY), str(_COMMAND_REPOSITORY / "src")]


import argparse
import base64
import copy
import csv
import hashlib
import html
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
import xml.etree.ElementTree as ET
import zipfile

ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")

REPO = _COMMAND_REPOSITORY
sys.path.insert(0, str(REPO / "src"))
from classical_conditioning.external_artifacts import resolve_artifact

CONTENT = REPO / "docs/paper/figures/figure-02/panel-g/analysis-guide.md"
NOTEBOOK = REPO / "src/classical_conditioning/reviews/assets/figure2_g_notebook.js"
OUTPUTS = Path("/Volumes/JOAQUIM/ClassicalConditioning Outputs")
RUN = OUTPUTS / 'figures/figure-02/panel-g/versions/figure2-G-independent-review/20261014-analysis-02'
ARCHIVE = Path('/Volumes/JOAQUIM/ClassicalConditioning-Archive/repository-history/retired-source/configuration-cleanup/2026-10-11/hash-recovery/payload/objects/6ebf338595492b7930145eac9de74232992be59824c4b4a92c1444e2f307d9a3/figure2_delay_all_versions_20261009.html')
HISTORICAL = Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/shared/studies/legacy-learner-comparison/allDelay-full-v1-distal/comparison.json')
FROZEN = Path('/Volumes/JOAQUIM/ClassicalConditioning Outputs/figures/figure-02/panel-g/versions/20261009-G-historical-logmedian')
FROZEN_LOCAL = REPO / "records/frozen-analyses/figure-02/panel-g/2026-10-09-historical-log-median"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def rows(data: bytes | str):
    return list(csv.DictReader(io.StringIO(data.decode() if isinstance(data, bytes) else data)))


def number(value, places=4):
    if value in (None, "", "inf", "nan"):
        return "—"
    f = float(value)
    return f"{f:.{places}f}".replace("-", "−")


def interval(row, prefix="simultaneous"):
    return f"{number(row['estimate'])} [{number(row[prefix+'_lower'])}, {number(row[prefix+'_upper'])}]"


def table(headers, data, caption=""):
    # HTML is escaped for all values sourced from artifacts.
    head = "".join(f"<th scope='col'>{html.escape(str(x))}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(x))}</td>" for x in r) + "</tr>" for r in data)
    return f"<div class='table-wrap' tabindex='0'><table><caption>{html.escape(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


class ArchivedImages(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "img" and a.get("src", "").startswith("data:image/"):
            self.images.append(a)


CSS = r"""
:root{--ink:#202f35;--muted:#53666d;--paper:#fff;--back:#f3f5f3;--line:#d7e0dd;--accent:#176451;--pale:#edf6f1;--warn:#79511d;--warn-bg:#fff7e9;--blue:#1762ae;--magenta:#ad2678}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;background:var(--back);color:var(--ink);font:18px/1.7 Georgia,'Times New Roman',serif}a{color:var(--accent);text-underline-offset:3px}a:hover{color:#0c3f33}button,input,select,textarea{font:inherit}button,select{cursor:pointer}button{border:1px solid var(--line);border-radius:6px;background:white;padding:7px 12px;font:14px/1.4 system-ui,sans-serif;color:var(--ink)}button:hover{background:var(--pale)}button:focus-visible,a:focus-visible,input:focus-visible,textarea:focus-visible,summary:focus-visible{outline:3px solid #448dca;outline-offset:3px}
.shell{display:grid;grid-template-columns:248px minmax(0,1040px);max-width:1360px;margin:auto;gap:36px;padding:32px 24px 80px}.rail{position:sticky;top:24px;align-self:start;max-height:calc(100vh - 48px);overflow:auto;font:13px/1.5 system-ui,sans-serif}.brand{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--accent);font-weight:700}.rail h2{font:600 19px/1.3 system-ui;margin:12px 0}.rail nav a{display:block;padding:7px 9px;border-left:2px solid transparent;text-decoration:none;color:var(--muted)}.rail nav a.active{border-color:var(--accent);background:var(--pale);color:var(--accent);font-weight:600}.rail input{width:100%;font:14px system-ui;padding:10px;border:1px solid var(--line);border-radius:6px;background:#fff}.rail .tools{display:flex;flex-wrap:wrap;gap:7px;margin:15px 0}.rail .small{font-size:12px}.rail .chapter-menu{border:0;padding:0;margin:10px 0}.rail .chapter-menu>summary{display:none}.rail .note-help{font:12px/1.6 system-ui;border:0;padding:0;margin:8px 0}.rail .note-help>summary{font-size:12px;color:var(--muted)}.read-progress{height:4px;background:#dce6e1;margin:14px 0}.read-progress span{height:100%;display:block;background:var(--accent);width:0}.search-results{font-size:12px;min-height:20px}
main{min-width:0}.hero{background:#173f37;color:#fff;border-radius:12px;padding:38px 42px;margin-bottom:28px}.hero .eyebrow{font:12px/1.5 system-ui;letter-spacing:.12em;text-transform:uppercase;color:#b7d8ca}.hero h1{font:500 clamp(30px,3vw,43px)/1.15 Georgia;margin:18px 0}.hero p{color:#e0eee8;font-size:18px;max-width:720px}.hero .meta{font:13px/1.6 system-ui;color:#cee0d8}.hero a{color:#d8f1e4}section.chapter{background:var(--paper);border:1px solid var(--line);border-radius:10px;padding:32px 40px;margin:0 0 26px;scroll-margin-top:20px;overflow-wrap:anywhere}.chapter>h2{font:500 29px/1.25 Georgia;margin:0 0 25px;color:#153f34}.chapter h3{font:600 20px/1.4 system-ui;margin:32px 0 12px}.chapter h4{font:600 16px/1.4 system-ui;margin:24px 0 10px}.chapter p{margin:0 0 17px}.chapter li{margin:7px 0}.chapter ul,.chapter ol{padding-left:24px}.chapter strong{font-weight:700}.chapter code{font:14px/1.5 ui-monospace,SFMono-Regular,monospace;background:#eff3f1;padding:2px 4px;border-radius:3px}.formula{background:#f0f5f3;border-left:3px solid #82ac98;padding:16px 20px;margin:18px 0;font:16px/1.8 ui-monospace,SFMono-Regular,monospace;white-space:normal}.callout,blockquote{border-left:4px solid var(--accent);background:var(--pale);padding:18px 22px;margin:20px 0}.callout p:last-child,blockquote p:last-child{margin-bottom:0}.caution{background:var(--warn-bg);border-color:#c28b3d}.label{font:600 11px/1.4 system-ui;text-transform:uppercase;letter-spacing:.07em;color:var(--accent)}.small,.source-line{font:13px/1.6 system-ui;color:var(--muted)}.source-line{margin-top:20px!important}.source-line a{margin-right:8px}.table-wrap{overflow-x:auto;margin:22px 0;max-width:100%;border:1px solid var(--line);border-radius:6px}table{border-collapse:collapse;width:100%;font:13px/1.5 system-ui;min-width:480px}th,td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #e2e8e5}thead{background:#eef3f0}th{font-weight:650;color:#234b3e}tbody tr:nth-child(even){background:#fafcfb}tbody tr:last-child td{border-bottom:0}caption{caption-side:bottom;text-align:left;font-size:12px;padding:9px 12px;color:var(--muted);background:#f7f9f8}.num{font-variant-numeric:tabular-nums}
figure{margin:26px 0;padding:15px;border:1px solid var(--line);border-radius:6px;background:white}figure img{display:block;width:100%;height:auto;max-height:none}figure figcaption{font:13px/1.65 system-ui;color:var(--muted);padding:14px 3px 0}.artwork{overflow:auto;max-width:100%;background:white}.artwork svg{display:block;max-width:100%;height:auto}figure.compact{max-width:680px;margin-left:auto;margin-right:auto}details{border:1px solid var(--line);border-radius:6px;padding:12px 15px;margin:18px 0}details>summary{font:600 15px/1.5 system-ui;color:#345b4b;cursor:pointer}details[open]>summary{margin-bottom:18px}details figure{border:0;padding:0}.stage-map{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;font:14px/1.6 system-ui;margin:20px 0}.stage-map div{padding:14px;background:#f1f5f2;border-radius:6px}.stage-map b{display:block;color:var(--accent)}.notebook{margin-top:30px;padding:20px;background:#f7f8f5;border:1px dashed #aabeb3;border-radius:8px;font:14px/1.6 system-ui}.notebook summary{font-size:15px}.notebook label{display:block;margin:12px 0 5px;font-weight:600}.notebook textarea{display:block;resize:vertical;width:100%;min-height:85px;font:14px/1.6 system-ui;border:1px solid #bdcec4;border-radius:5px;padding:10px;background:white}.notebook select{font:14px system-ui;padding:7px;border:1px solid #bdcec4;border-radius:4px;max-width:100%}.notebook .check-row{display:flex;align-items:center;gap:8px;font-weight:400}.notebook .check-row input{width:17px;height:17px}.save-status{font:12px/1.6 system-ui;color:var(--muted)}.status-live{display:block;color:var(--accent);margin-top:12px}.quiz{background:#f7f8fa}.legend-copy{white-space:pre-wrap;font:14px/1.7 system-ui;background:#f7f9f8;padding:18px;border-radius:6px}.source-card{font:13px/1.6 system-ui;border-top:1px solid var(--line);padding:14px 0}.source-card h4{margin:0 0 5px;font-size:15px}.source-card code{font-size:11px;word-break:break-all}.source-card p{margin:5px 0}#import-message{white-space:normal;overflow-wrap:anywhere}.no-print{display:block}dialog{max-width:600px;width:90%;border:1px solid var(--line);border-radius:10px;padding:24px}dialog::backdrop{background:#173f3755}dialog pre{max-height:35vh;overflow:auto;white-space:pre-wrap;font:13px/1.5 system-ui}.search-hit{outline:3px solid #d5ad43;outline-offset:3px}.skip-link{position:absolute;left:12px;top:-70px;background:white;padding:8px;z-index:3}.skip-link:focus{top:12px}.source-badge{display:inline-block;font:11px system-ui;background:#eef4f0;border-radius:3px;padding:2px 4px;margin-left:4px}footer{font:13px/1.7 system-ui;color:var(--muted);padding:8px 0 30px}
@media(max-width:1050px){.shell{grid-template-columns:200px minmax(0,1fr);gap:20px;padding:20px 16px}.hero{padding:28px}.chapter{padding:28px!important}.stage-map{grid-template-columns:1fr}body{font-size:17px}}
@media(max-width:760px){.shell{display:block;padding:12px}.rail{position:static;max-height:none;padding:14px 8px;margin-bottom:14px}.rail .chapter-menu>summary{display:block;padding:8px 0}.rail nav{display:grid;grid-template-columns:1fr 1fr}.rail nav a{font-size:12px;padding:6px}.hero{padding:25px}.chapter{padding:22px!important}.chapter>h2{font-size:25px}.table-wrap{font-size:12px}.rail .read-progress{display:none}table{min-width:440px}.hero h1{font-size:34px}}
@page{size:A4;margin:16mm 14mm}@media print{html{scroll-behavior:auto}body{background:white;font:10pt/1.5 Georgia;color:black}.shell{display:block;padding:0;margin:0;max-width:none}.rail,.no-print,dialog{display:none!important}.hero{background:white;color:black;padding:0;border:0;margin:0 0 8mm}.hero p,.hero .meta,.hero .eyebrow{color:#333}.hero h1{font-size:26pt}.chapter{border:0!important;border-radius:0!important;padding:0!important;margin:0!important;break-before:page}.chapter h2{font-size:20pt}.chapter h3{font-size:13pt;break-after:avoid}.chapter h4{font-size:11pt;break-after:avoid}.chapter p{orphans:3;widows:3}.table-wrap{overflow:visible;border:0}table{min-width:0;font:8pt/1.4 Arial;width:100%;table-layout:auto}tr{break-inside:avoid}thead{display:table-header-group}th,td{padding:5px;overflow-wrap:anywhere}figure{break-inside:avoid;padding:4mm;margin:5mm 0}figure img{max-height:220mm;object-fit:contain}figcaption{font-size:8pt!important}.notebook{break-inside:avoid;background:white}.notebook textarea{border:0;min-height:20mm;font-size:9pt}.notebook select{appearance:none;border:0}.source-card{break-inside:avoid}details{border:0;padding:0}details>summary{font-size:10pt}a{color:black}footer{font-size:8pt}.source-line{font-size:8pt}.formula{font-size:9pt}.artwork svg{max-width:100%}}
"""


class Guide:
    def __init__(self, out: Path, manuscript: Path = CONTENT, manuscript_sha256=None):
        self.out = out
        if manuscript != CONTENT and not manuscript_sha256:
            raise ValueError("An explicit manuscript requires --manuscript-sha256")
        self.manuscript = resolve_artifact(manuscript, expected_sha256=manuscript_sha256, repo=REPO)
        self.manuscript_sha256 = digest(self.manuscript.read_bytes())
        self.sources = {}
        self.tokens = {}
        self.protected = {}
        self.checks = {}
        self.figure_hashes = []

    def source(self, key, title, path: Path, expected=None, note="", copy_file=True):
        path = resolve_artifact(path, expected_sha256=expected, repo=REPO)
        raw = path.read_bytes()
        actual = digest(raw)
        if expected and actual != expected:
            raise ValueError(f"Hash mismatch for {title}: {path}")
        self.protected[str(path)] = actual
        suffix = path.suffix or ".txt"
        relative = f"sources/{key}{suffix}"
        if copy_file:
            (self.out / relative).parent.mkdir(parents=True, exist_ok=True)
            (self.out / relative).write_bytes(raw)
        self.sources[key] = {"title": title, "path": str(path), "sha256": actual,
                             "verification": "matched recorded SHA-256" if expected else "snapshotted current source; identity recorded",
                             "copy": relative if copy_file else None, "note": note}
        return raw

    def archive_source(self, key, title, member, raw, note=""):
        relative = f"sources/{key}{Path(title).suffix or '.bin'}"
        (self.out / relative).parent.mkdir(parents=True, exist_ok=True)
        (self.out / relative).write_bytes(raw)
        is_git=member.startswith('git:')
        self.sources[key] = {"title": title, "path": member if is_git else f"{ARCHIVE} :: ZIP/{member}",
                             "sha256": digest(raw), "copy": relative,
                             "verification": "Git object matched recorded source-catalog SHA-256" if is_git else "member authenticated inside hash-verified historical review", "note": note}
        return raw

    def saved_figure(self, key, title, path, expected, caption, legend=None):
        raw = self.source(key, title, path, expected)
        relative = f"assets/{key}{path.suffix}"
        (self.out / relative).parent.mkdir(parents=True, exist_ok=True)
        (self.out / relative).write_bytes(raw)
        self.figure_hashes.append({"id": key, "source_sha256": digest(raw), "asset": relative, "asset_sha256": digest(raw)})
        caption += f" <a href='#source-{key}'>Source and identity</a> · <a href='{relative}' target='_blank' rel='noopener'>Open artwork</a>."
        if legend:
            text = self.source(key + "-legend", title + " — associated legend", Path(legend["path"]), legend.get("sha256")).decode()
            caption += "<details><summary>Associated legend, verbatim</summary><div class='legend-copy'>" + html.escape(text) + "</div></details>"
        self.tokens["figure:" + key] = f"<figure id='figure-{key}'><a href='{relative}' target='_blank' rel='noopener'><img loading='lazy' src='{relative}' alt='{html.escape(title)}'></a><figcaption>{caption}</figcaption></figure>"

    def capture(self):
        selection_path = REPO / "configs/paper-figures/selections/figure2-G-logmedian-freeze-20261009.json"
        selection = json.loads(self.source("selection", "Current scoped scientific selection of G", selection_path))
        state = json.loads(self.source("figure-state", "Current Figure 2 presentation state", REPO / "docs/paper/figures/figure-02/current-state.json"))
        self.source("workflow", "Authoritative figure workflow", REPO / "docs/paper/figures/shared/iteration-workflow.md")
        self.source("nature", "Maintained Nature presentation guidance", REPO / "docs/paper/figures/shared/nature-style.md")
        for rec in state["scientific_selection"]["records"]:
            if Path(rec["path"]).name == selection_path.name:
                assert digest(selection_path.read_bytes()) == rec["sha256"]
        # Preserve the later explanatory supplement on future dated builds.
        # These snapshots do not change the selected scientific calculations.
        from classical_conditioning.reviews.figure2_g_sources import SOURCES as completion_sources
        for key, relative in completion_sources.items():
            self.source(key, "23 October discussion: " + Path(relative).name, REPO / relative)
        self.source("completion-other-chat-summary", "Pasted other-chat GAMM proposal (not an executed analysis)",
                    OUTPUTS / 'figures/figure-02/panel-g/studies/analysis-guide/sources/completion-other-chat-summary.txt')
        self.source("frozen-definition", "Frozen G scientific definitions and annotations", FROZEN_LOCAL / "manifests/scientific-selection.json",selection["scientific_selection"]["sha256"])
        self.source("freeze-record", "October 9 scoped G freeze record", FROZEN_LOCAL / "manifests/freeze.json",selection["freeze_record"]["sha256"])
        inv = json.loads(self.source("frozen-inventory", "Selected freeze source inventory", FROZEN_LOCAL / "sources/selected-source-inventory.json"))
        self.source("frozen-sidecar", "Selected G source/data element mappings", FROZEN / "Fig2_G_Historical_LogMedian.figure.json", "ea90c79d4dc9c3f3fe0c5c3786d16dbc39780c74891bcb128c0e3e9402fb65b2")
        self.saved_figure("frozen-g", "G: October 9 selected scientific freeze", FROZEN / "Fig2_G_Historical_LogMedian.svg", selection["svg"]["sha256"], "Original selected panel, 183 × 98 mm. This authentic artwork is preserved; it is not a newly fitted result.")
        for key, title, path in [
            ("plan03", "Plan 03: learning onset", "Plans/03_FIGURE_2_LEARNING_ONSET.md"),
            ("plan06", "Plan 06: analysis and statistics", "Plans/06_ANALYSIS_AND_STATISTICS.md"),
            ("comparison", "Current G versus independent results — October 15", "records/scientific-evidence/analyses/figure-02-g-independent/current-G-comparison-2026-10-15.json"),
            ("comparison-prose", "Interpretation of the matched G comparison", "records/scientific-evidence/analyses/figure-02-g-independent/current-G-comparison-2026-10-15.md"),
            ("review01", "First independent review and identified defects", "records/scientific-evidence/analyses/figure-02-g-independent/review-2026-10-14.md"),
            ("review02", "Completed implementation and review — October 15", "records/scientific-evidence/analyses/figure-02-g-independent/implementation-and-review-2026-10-15.md"),
            ("legacy-matrix", "Legacy learner behavior matrix", "records/scientific-evidence/legacy/04_LEARNER_VARIANT_BEHAVIOR_MATRIX.md"),
            ("legacy-review", "Legacy learner comparison documentation", "records/scientific-evidence/legacy/05_LEGACY_LEARNER_COMPARISON.md"),
            ("legacy-audit", "Legacy output authentication and caveats", "records/scientific-evidence/audits/figure3-individual-display-legacy-review-2026-10-15.json"),
        ]:
            self.source(key, title, REPO / path)
        comparison = read_json(REPO / "records/scientific-evidence/analyses/figure-02-g-independent/current-G-comparison-2026-10-15.json")
        legacy_html = self.source("old-review", "Authenticated October 9 all-versions review", ARCHIVE, selection["all_versions_review"]["sha256"])
        parser = ArchivedImages(); parser.feed(legacy_html.decode())
        # Retain original embedded bytes. These are archived output illustrations,
        # not newly generated/selected panels and not a transfer of their inference.
        for index, title in [(1,"Bout-only arithmetic means and global LMM"),(2,"Arithmetic-mean residual diagnostics"),(5,"Arithmetic means: observed and global/phase fitted contrasts"),(6,"Phase-model residual diagnostics"),(7,"LogMedian: observed and fitted contrasts"),(8,"LogMedian statistical rows"),(9,"Four outcome definitions, matched lines"),(10,"Mean of logged frames versus LogMedian"),(11,"Pointwise confidence interval versus IQR"),(14,"Matched window summaries"),(15,"Arithmetic-mean phase statistics")]:
            item = parser.images[index]
            raw = base64.b64decode(item["src"].split(",",1)[1])
            ext = ".svg" if "svg+xml" in item["src"][:40] else ".png"
            rel = f"assets/archive-{index}{ext}"
            (self.out / rel).write_bytes(raw)
            self.figure_hashes.append({"id": f"archive-{index}", "asset": rel, "asset_sha256": digest(raw), "source": "old-review", "embedded_image_index": index, "source_sha256": digest(raw)})
            self.tokens[f"archive:{index}"] = f"<details class='archive-figure'><summary>{html.escape(title)} — saved output</summary><figure><a href='{rel}' target='_blank' rel='noopener'><img loading='lazy' src='{rel}' alt='{html.escape(title)}'></a><figcaption>Unmodified embedded illustration from the authenticated October 9 review. <a href='#source-old-review'>Source</a> · <a href='{rel}' target='_blank' rel='noopener'>Open artwork</a>. Its analysis/status is described in the surrounding text.</figcaption></figure></details>"
        m = re.search(r'<script[^>]*id=["\x27]archive["\x27][^>]*>(.*?)</script>', legacy_html.decode(), re.S)
        z = zipfile.ZipFile(io.BytesIO(base64.b64decode(m[1])))
        # Recover by selected content hashes, not by an ambiguous relocated name.
        zip_hashes = {digest(z.read(n)): n for n in z.namelist()}
        selected = {}
        for name in ["phase-contrasts.csv","D-interaction-tests.csv","joint-interaction-test.csv","model-input.parquet","bootstrap-summary.parquet","candidate-specification.json","phase-diagnostics.json"]:
            rec = next(x for x in inv if x["path"].replace("\\","/").endswith("/"+name))
            member = zip_hashes.get(rec["sha256"])
            if not member:
                raise ValueError(f"Selected hash not in authenticated archive: {name}")
            raw = self.archive_source("selected-" + name.split(".")[0], name, member, z.read(member), "Exact SHA-256 match to selected freeze inventory.")
            assert digest(raw) == rec["sha256"]
            selected[name] = raw
        mr_expected = next(x["sha256"] for x in inv if x["path"].replace("\\","/").endswith("/MR-local-block-tests.csv"))
        self.checks["selected_MR_csv_recovered"] = mr_expected in zip_hashes
        self.checks["selected_model_input_recovered_from_archive"] = True
        d = rows(selected["D-interaction-tests.csv"])
        self.tokens["table:current-d"] = table(["Block", "Delay − control change vs Pre", "Raw p", "Holm p (8)"], [(x["block_10_name"], number(x["estimate"]), f"{float(x['p_raw']):.4g}",f"{float(x['p_holm']):.4g}") for x in d],"Exact selected D table. This sign is opposite to the positive suppression convention used for the trial and new phase contrasts.")
        marks = comparison["old_MR_displayed_annotations_from_saved_sidecar"]
        names = ["Pre","Train 1","Train 2","Train 3","Train 4","Train 5","Test 1","Test 2","Test 3"]
        self.tokens["table:current-mr"] = table(["Block", "Saved annotation", "p", "Status"],[(names[x["row"]], x["column"], "unavailable" if x["p"] is None else f"{x['p']:.7g}", x["status"]) for x in marks], "Only annotation entries actually recorded in the authenticated selected sidecar; this is not the complete original M/R table.")
        astate = json.loads(self.source("analysis-state", "Independent analysis selected stage", RUN / "analysis-state.json"))
        report = RUN / astate["latest"]["directory"]
        am = json.loads(self.source("analysis-manifest", "Independent report artifact manifest", report / "artifact-manifest.json", astate["latest"]["manifest_sha256"]))
        hashes = {x["path"]:x["sha256"] for x in am["artifacts"]}
        saved = {}
        def saved_file(relative, key=None):
            key = key or "analysis-" + relative.replace("/","-").rsplit(".",1)[0]
            raw = self.source(key, "Independent analysis: " + relative, report / relative, hashes[relative])
            saved[relative] = raw
            return raw
        files = ["legend.md","review.html","input-verification.json","input-import.json","legacy-source-catalog.json","legacy-assessment-status.json","legacy-cohort-phase-results.csv","legacy-cohort-timing-results.csv","legacy-sensitivity-membership.csv","legacy-discard-rule-details.csv","legacy-cumulative-flow.csv","technical-phase-results.csv","technical-timing-results.csv","technical-recording-audit.csv","technical-policy-before-fitting.json","raw-protocol-trace.json","technical-coverage-sensitivity.json","unsmoothed-fish-change-phases.csv","unsmoothed-fish-change-result.json","unsmoothed-phase-contributors.csv","fish-phase-robustness.csv","calibration-pilot.csv","calibration-status.json","unsmoothed-calibration-summary.csv","unsmoothed-calibration-status.json","individual-model-diagnostics.json","individual-heldout-summary.csv","individual-calibration-pilot.csv","individual-covariance-sensitivity-status.json","individual-missingness-review.json","movement-occurrence-companion.csv","day-condition-balance.csv","actual-US-window-overlaps.csv","report-sources.json","review-supplements-binding.json"]
        for name in files:
            saved_file(name)
        for target in ["difference","adjusted_response"]:
            for name in ["result.json","planned-phase-contrasts.csv","population-trajectory.csv","fit-diagnostics.json","model-adequacy.json","categorical-trial-diagnostics.json","categorical-trial-sensitivity.csv","leave-one-fish-out.csv","day-cluster-diagnostics.json","day-cluster-phase-sensitivity.csv","random-effects-sensitivity.json"]:
                saved_file(target+"/"+name)
        self.saved_figure("population", "Independent population candidates: direct change and adjusted response", report / "population-candidates.svg", hashes["population-candidates.svg"], "Saved independent report illustration. Its observed summaries and fitted means must be distinguished from G’s displayed medians.", {"path":str(report / "legend.md"),"sha256":hashes["legend.md"]})
        for key, name, caption in [("unsmoothed","unsmoothed-fish-change.svg","Available fish changes anchored to each fish’s available Pre mean. The estimator differs from the mixed-model target."),("movement","movement-occurrence-companion.svg","Descriptive movement-occurrence/time-moving companion; not a validated replacement population analysis.")]:
            self.saved_figure(key, name, report/name, hashes[name], caption)
        for target in ["difference","adjusted_response"]:
            self.saved_figure(target+"-diagnostics",target+" model diagnostics",report/target/"model-diagnostics.svg",hashes[target+"/model-diagnostics.svg"], "Saved residual/influence diagnostics; successful optimization does not establish model adequacy.")
        self.data_tables(saved, comparison)
        old_trials=rows(selected['phase-contrasts.csv'])
        new_trials=rows(saved['adjusted_response/population-trajectory.csv'])
        assert [int(x['trial_number']) for x in old_trials]==list(range(5,95))
        assert [int(x['trial_number']) for x in new_trials]==list(range(5,95))
        old_rejections=[int(x['trial_number']) for x in old_trials if float(x['change_p_fdr'])<.05]
        new_positive=[int(x['trial_number']) for x in new_trials if float(x['simultaneous_lower'])>0 and x['coverage_adequate']=='True']
        assert old_rejections==comparison['old_BH90_significant_trials']
        assert new_positive==comparison['new_adjusted_trial_positive_joint_interval']
        self.checks['adjusted_curve_max_absolute_difference']=max(abs(float(a['change_estimate'])-float(b['estimate'])) for a,b in zip(old_trials,new_trials))
        self.checks['adjusted_SE_max_absolute_difference']=max(abs(float(a['change_se'])-float(b['se'])) for a,b in zip(old_trials,new_trials))
        assert self.checks['adjusted_curve_max_absolute_difference'] < 2e-8
        assert self.checks['adjusted_SE_max_absolute_difference'] < 6e-8
        ex = json.loads(self.source("exploratory-state", "Separate exploratory G presentation state", REPO / "docs/paper/figures/figure-02/panel-g/exploratory/current-state.json"))
        variants = [("bracket-g", "Exploratory G with bracket and asterisks", ex["draft"]),("plain-g","Exploratory G without asterisks",ex["previous_versions"][0]["draft"])]
        for key,title,s in variants:
            self.saved_figure(key,title,Path(s["working_svg"]["path"]),s["working_svg"]["sha256"],"Separate unfrozen presentation of the saved adjusted-response results; calibration remains unresolved.",s["legend"])
        self.source("asterisk-decisions", "Exact meaning and qualifying trials of the new asterisk bracket", Path(ex["draft"]["asterisk_decisions"]["path"]), ex["draft"]["asterisk_decisions"]["sha256"])
        dense = OUTPUTS / 'figures/figure-02/panel-g/versions/figure-02-g-exploratory-20261016-asterisks'
        dense_sources = {x['path']:x['sha256'] for x in read_json(dense/"render-outputs.json")}
        # This is an unpublished historical preview. Do not give it selected authority.
        self.source("dense-output-record", "Dense-star preview output record", dense/"render-outputs.json")
        self.saved_figure("dense-g", "Unpublished dense-star exploratory preview", dense/"figure2-G-exploratory.svg", dense_sources[str(dense/"figure2-G-exploratory.svg")], "Superseded unpublished preview with individual trial stars; retained only to explain the presentation iteration.", {"path":str(dense/"legend.md"),"sha256":dense_sources[str(dense/"legend.md")]})
        h = json.loads(self.source("september-learners", "September 24 reapplication of historical learner rules", HISTORICAL,"b9ebad4dd6a06d635d98a2e268f861689e24266ad4ff0a387dd110e58cbed356"))
        assert h["historical_equivalence"] is False
        self.tokens["table:september"] = table(["Rule", "Delay flagged / eligible", "Control flagged / eligible", "Unclassified Delay / control"],[(v["variant"].replace("legacy-",""),f"{v['learner_delay']} / {v['eligible_delay']}",f"{v['control_flagged']} / {v['eligible_control']}","0 / 0") for v in h["variants"]],"September 24, corrected activity units translated to legacy inputs. Historical equivalence explicitly false; not the original February/March experiment output.")
        ats = json.loads(self.source("atlas-state", "Later bout-only learner comparison state (context)",REPO / "docs/paper/figures/figure-03/exploratory-state.json"))
        atmanifest = json.loads(self.source("atlas-manifest", "Later learner output manifest",Path(ats["manifest"]["path"]),ats["manifest"]["sha256"]))
        entries = atmanifest.get("artifacts", atmanifest.get("files",[]))
        if isinstance(entries,dict): entries=[{"path":k,"sha256":v} for k,v in entries.items()]
        atroot = Path(ats["manifest"]["path"]).parent
        at_hashes = {x["path"]:x["sha256"] for x in entries}
        at = json.loads(self.source("bout-learners", "Later bout-only learner summary",atroot/"classification-summary.json",at_hashes["classification-summary.json"]))
        atrows = rows(self.source("bout-classification", "Later learner eligibility and decisions (optional detailed source)",atroot/"classification-manifest.csv",at_hashes["classification-manifest.csv"]))
        self.source("atlas-review", "Existing detailed individual-fish atlas (optional external link)",Path(ats["review"]["path"]),ats["review"]["sha256"],copy_file=False)
        by = {(x["variant"],x["condition"]):x for x in at}
        counts=[]
        for v in h["variants"]:
            key=v["variant"];de=by[key,"delay"];co=by[key,"control"]
            assert de["eligible"]+de["unclassified"]==29 and co["eligible"]+co["unclassified"]==28
            assert len([x for x in atrows if x["variant"]==key]) == 57
            for condition,expected in [('delay',de),('control',co)]:
                rr=[x for x in atrows if x['variant']==key and x['condition_id']==condition]
                assert sum(x['decision']=='Unclassified' for x in rr)==expected['unclassified']
                assert sum(x['decision'] in ('Learner','Flagged') for x in rr)==expected['flagged']
                assert sum(x['decision']!='Unclassified' for x in rr)==expected['eligible']
            counts.append((key.replace("legacy-",""),f"{de['flagged']} / {de['eligible']}",f"{co['flagged']} / {co['eligible']}",f"{de['unclassified']} / {co['unclassified']}"))
        self.tokens["table:bout-learners"] = table(["Rule", "Delay flagged / eligible", "Control flagged / eligible", "Unclassified Delay / control"], counts,"October 15 bout-only reapplication from the later individual-fish workstream, supplied as context. Flags remain exploratory; this is not a new classifier run.")
        self.checks["learner_denominators_reconciled"] = True
        self.source("current-assembly", "Current Figure 2 assembly (G is isolated in this guide)",Path(state["draft"]["working_svg"]["path"]),state["draft"]["working_svg"]["sha256"])
        self.source("current-legend", "Current Figure 2 associated legend",Path(state["draft"]["legend"]["path"]),state["draft"]["legend"]["sha256"])
        self.source("current-registry", "Current Figure 2 semantic registry",Path(state["draft"]["semantic_registry"]["path"]),state["draft"]["semantic_registry"]["sha256"])
        root = ET.fromstring(Path(state["draft"]["working_svg"]["path"]).read_bytes())
        ns="{http://www.w3.org/2000/svg}"
        layout = json.loads(root.find(f"{ns}metadata[@id='layout-frames']").text)
        box = next(x["box_mm"] for x in layout["allocations"] if x["id"]=="G")
        vb = list(map(float,root.get("viewBox").split()))
        scale=vb[2]/float(root.get("width").replace("mm",""))
        # HTML-only view into the exact assembly. Keep the root definitions and
        # inherited styles and retain G's transforms; no new working SVG exists.
        group=copy.deepcopy(root.find(f".//*[@id='panel-G']"))
        isolated=ET.Element(root.tag,dict(root.attrib))
        isolated.set("viewBox"," ".join(str(x*scale) for x in box));isolated.set("width",f"{box[2]}mm");isolated.set("height",f"{box[3]}mm")
        isolated.set("id","guide-current-g");isolated.set("aria-label","Current Figure 2 G presentation only")
        for node in list(root):
            if node.tag in (ns+"defs",ns+"style"): isolated.append(copy.deepcopy(node))
        isolated.append(group)
        self.tokens["figure:current-g"] = "<figure class='compact' id='figure-current-g'><div class='artwork' data-native-width-mm='"+str(box[2])+"' data-native-height-mm='"+str(box[3])+"'>"+ET.tostring(isolated,encoding="unicode")+"</div><figcaption>G isolated from the authenticated current Figure 2 presentation, enlarged on screen for reading. Its native allocated width is "+number(box[2],2)+" mm. The October 9 scientific selection is unchanged. <a href='#source-current-assembly'>Assembly identity</a> · <a href='#source-current-legend'>Matching current legend</a>.</figcaption></figure>"
        current_text=Path(state["draft"]["legend"]["path"]).read_text()
        legend_match=re.search(r'^\(g\).*?(?=^\(h\)|\Z)',current_text,re.M|re.S)
        if not legend_match:raise ValueError('Cannot resolve current panel g legend paragraph')
        self.tokens["legend:current"]="<details><summary>Current panel g legend, verbatim excerpt</summary><div class='legend-copy'>"+html.escape(legend_match[0].strip())+"</div></details>"
        self.checks["old_BH_trials"] = comparison["old_BH90_significant_trials"]
        self.checks["new_joint_positive_trials"] = comparison["new_adjusted_trial_positive_joint_interval"]
        assert self.checks["old_BH_trials"]==list(range(19,72))
        assert self.checks["new_joint_positive_trials"]==list(range(22,62))
        self.source("guide-content", "Guide explanatory source",self.manuscript,expected=self.manuscript_sha256)
        self.source("guide-builder", "Guide builder",Path(__file__))
        self.source("guide-notebook", "Review notebook browser code",NOTEBOOK)
        # Record the original technical supplement binding as well as the copy
        # authenticated by the selected report's manifest.
        self.source("supplement-state", "Technical supplement state", RUN/"review-supplements.json")
        # Verify pinned Git text directly against its saved source snapshots.
        catalog=json.loads(saved["legacy-source-catalog.json"])
        for i,item in enumerate(catalog["sources"]):
            raw=subprocess.check_output(["git","show",f"{item['revision']}:{item['path_at_revision']}"],cwd=REPO)
            if digest(raw)!=item["sha256"]: raise ValueError("Historical Git text differs from saved catalogue")
            self.archive_source(f"git-legacy-{i}",item["path_at_revision"],f"git:{item['revision']}:{item['path_at_revision']}",raw,"Verified against Git object and saved source-catalog hash.")
        self.checks["git_legacy_sources_verified"] = len(catalog["sources"])

    def data_tables(self, saved, comparison):
        phase_names={"late_training":"Late training 55–64","early_testing":"Early testing 65–74","late_testing":"Late testing 85–94"}
        self.tokens["table:phases"] = table(["Analysis / uncertainty",*phase_names.values(),"Resamples accepted / requested"],[
            ("Current G adjusted model; derived pointwise Gaussian CI", *[f"{number(x['old_saved_model_estimate'])} [{number(x['old_derived_pointwise_gaussian_interval'][0])}, {number(x['old_derived_pointwise_gaussian_interval'][1])}]" for x in comparison['old_model_derived_phase_summaries']],"No new fit; frozen covariance"),
            *[(label,*[interval(x) for x in rows(saved[target+"/planned-phase-contrasts.csv"])],f"{json.loads(saved[target+'/result.json'])['bootstrap_successful']} / 999") for target,label in [("adjusted_response","Adjusted response LMM; joint CI"),("difference","Direct-change LMM; joint CI")]],
            ("Unsmoothed fish-change mean; joint CI",*[interval(x) for x in rows(saved["unsmoothed-fish-change-phases.csv"])],"999 / 999")
        ],"All effects use control − Delay change from Pre; positive is the conditioned suppression direction. Log units. Joint families contain 90 trials + 3 phase averages, separately for each target. Current-G phase intervals were derived in the October 15 comparison and were not originally displayed in G.")
        self.tokens["table:fish-pointwise"] = table(["Phase","Fish-level pointwise interval","Contributing control / Delay"],[(phase_names[x["contrast"]],interval(x,"pointwise"),f"{x['control_fish']} / {x['delay_fish']}") for x in rows(saved["fish-phase-robustness.csv"])],"A separate pointwise fish-phase sensitivity, not a joint family. No cohort member is discarded because it cannot contribute to one phase.")
        timing=[]
        for target,label in [("difference","Direct change"),("adjusted_response","Adjusted response"),("unsmoothed","Unsmoothed")]:
            data=json.loads(saved["unsmoothed-fish-change-result.json"] if target=="unsmoothed" else saved[target+"/result.json"])
            for x in data['timing']:
                timing.append((label,f"{100*x['threshold_fraction']:.0f}%",number(x['estimate_trial'],0),number(x['earliest_compatible_trial'],0),number(x['latest_bound_trial'],0),number(x['first_supported_trial'],0),x['status'].replace('_',' ')))
        self.tokens["table:timing"] = table(["Estimator","Threshold","Point crossing","Earliest compatible","Latest bound","First supported run","Saved status"],timing,"K = 3 scheduled training trials; all threshold choices and timing inference are exploratory. A dash is unbounded/not localized, never a substituted final trial. Nominal bounds require calibrated bands before scientific interpretation.")
        recipes=rows(saved['legacy-cohort-phase-results.csv'])
        names={"all_selected_G":"All selected G","legacy_any_block_6":"ANY block ≥6","legacy_baseline_3_projection":"Baseline ≥3/QC block","legacy_CR_3_projection":"CR ≥3/QC block","legacy_baseline_and_CR_projection":"Baseline + CR","legacy_train_US_projection":"All training-US responses","legacy_combined_with_viability_proxy":"Combined + last-US proxy","all_required_blocks_6_alternative":"ALL required blocks ≥6"}
        summary=[]
        for recipe,label in names.items():
            a=next(x for x in recipes if x['recipe']==recipe and x['target']=='difference' and x['contrast']=='late_training')
            b=next(x for x in recipes if x['recipe']==recipe and x['target']=='adjusted_response' and x['contrast']=='late_training')
            summary.append((label,f"{a['control_fish']} / {a['delay_fish']}",a['eligible_rows'],interval(a),interval(b),f"{a['bootstrap_successful']}, {b['bootstrap_successful']} / 199"))
        self.tokens['table:cohorts']=table(["Recipe","Control / Delay","Eligible rows","Late training: direct change","Late training: adjusted","Accepted: direct, adjusted"],summary,"Nominal simultaneous intervals from separate 199-resample cohort sensitivities. All-selected row is this smaller-resample comparison, not the main 999-resample result.")
        membership=rows(saved['legacy-sensitivity-membership.csv'])
        self.tokens['table:excluded']=table(["Recipe","Excluded controls","Excluded Delay"],[(label,', '.join(x['fish_id'] for x in membership if x['recipe']==recipe and x['condition_id']=='control' and x['included']=='False') or 'None',', '.join(x['fish_id'] for x in membership if x['recipe']==recipe and x['condition_id']=='delay' and x['included']=='False') or 'None') for recipe,label in names.items()],"Saved reconstructed exclusions, not a historical run roster. Full per-rule reasons are linked in Sources.")
        self.tokens['table:technical']=table(["Target","Phase","Effect [joint CI]","Control / Delay","Eligible rows","Accepted / requested"],[(x['target'],phase_names[x['contrast']],interval(x),f"{x['control_fish']} / {x['delay_fish']}",x['eligible_rows'],f"{x['bootstrap_successful']} / {x['bootstrap_requested']}") for x in rows(saved['technical-phase-results.csv'])],"Source-only protocol-flag sensitivity: removes Delay 20221122_03, without selecting on its behavior or label.")
        sens=[]
        for target in ['difference','adjusted_response']:
            lof=rows(saved[target+'/leave-one-fish-out.csv'])
            reff=json.loads(saved[target+'/random-effects-sensitivity.json'])[0]
            diag=json.loads(saved[target+'/day-cluster-diagnostics.json'])
            for x in rows(saved[target+'/day-cluster-phase-sensitivity.csv']):
                sens.append((target,'Recording-day resampling',phase_names[x['contrast']],interval(x),'199 requested; see diagnostic source'))
            sens.append((target,'Leave one fish out','57 fitted omissions',f"{sum(x['status']=='ok' for x in lof)} / {len(lof)} accepted",f"Max curve shift {number(max(float(x['max_contrast_shift']) for x in lof))}"))
            sens.append((target,'Random intercept only','Same phase mean family',reff['diagnostic']['diagnostic_status'],f"Optimizer {reff['diagnostic']['optimizer']}; max curve shift {number(reff['max_contrast_shift'])}"))
        self.tokens['table:sensitivities']=table(['Target','Sensitivity','Scope','Saved result','Details'],sens,"These are sensitivity results, not independent confirmation. Day clustering changes the resampling unit; RI-only changes the covariance structure.")
        pilots=rows(saved['calibration-pilot.csv']);summary=[]
        for scenario in dict.fromkeys(x['scenario'] for x in pilots):
            for target in ['difference','adjusted_response']:
                rr=[x for x in pilots if x['scenario']==scenario and x['target']==target]
                accept=[x for x in rr if x['simultaneous_trial_coverage'] in ('True','False')]
                summary.append((scenario,target,len(rr),len(accept),f"{sum(x['simultaneous_trial_coverage']=='True' for x in accept)} / {len(accept)}",f"{sum(x['simultaneous_phase_coverage']=='True' for x in accept)} / {len(accept)}",sum(x['false_supported_run']=='True' for x in rr)))
        self.tokens['table:lmm-calibration']=table(['Scenario','Target','Datasets','Accepted bands','Full trial curve covered','All phase means covered','False supported run count'],summary,"110 independent datasets, each used for both targets: 220 model cases. 10 per scenario, 99 resamples. Counts among accepted bands expose failures separately; this is a pilot, not a paper calibration gate.")
        robust=rows(saved['unsmoothed-calibration-summary.csv'])
        self.tokens['table:unsmoothed-calibration']=table(['Scenario','Full-curve coverage','Binomial 95% interval','Phase coverage','False supported runs','Onset bound contains truth'],[(x['scenario'],f"{x['successes']} / {x['independent_experiments']}",f"{number(x['binomial_lower'],3)}–{number(x['binomial_upper'],3)}",next(y['successes'] for y in robust if y['scenario']==x['scenario'] and y['metric']=='simultaneous_phase_coverage')+' / 100',next(y['successes'] for y in robust if y['scenario']==x['scenario'] and y['metric']=='false_supported_run')+' / 100',next(y['successes'] for y in robust if y['scenario']==x['scenario'] and y['metric']=='onset_interval_contains_truth')+' / 100') for x in robust if x['metric']=='simultaneous_trial_coverage'],"1,100 independent datasets (100 × 11 scenarios), 499 resamples each. Broad or unbounded timing bounds can contain truth without useful precision.")
        assert len(pilots)==220
        self.checks['LMM_calibration_model_cases']=len(pilots)
        self.checks['unsmoothed_calibration_experiments']=sum(int(x['independent_experiments']) for x in robust if x['metric']=='simultaneous_trial_coverage')
        self.tokens['table:heldout']=table(['Method','Condition','Fish','RMSE','Predictive coverage','Spearman'],[(x['method'],x['condition'],x['fish'],number(x['rmse']),number(x['predictive_coverage']),number(x['spearman'])) for x in rows(saved['individual-heldout-summary.csv'])],"Exploratory held-out early-test 65–69 comparison after training-only fitting. These aggregate predictive summaries do not assign or validate binary learner labels.")

    def substitute(self, text):
        def replacement(m):
            token=m[1]
            if token.startswith('sources:'):
                keys=token.split(':',1)[1].split(',')
                return "<p class='source-line'>Evidence: "+" · ".join(f"<a href='#source-{k}'>{html.escape(self.sources[k]['title'])}</a>" for k in keys)+"</p>"
            if token.startswith('link:'):
                key=token.split(':',1)[1];s=self.sources[key]
                return f"<a href='{html.escape(s['copy'] or Path(s['path']).as_uri())}' target='_blank' rel='noopener'>{html.escape(s['title'])}</a>"
            if token not in self.tokens:raise ValueError('Unknown content token '+token)
            return self.tokens[token]
        return re.sub(r'\{\{([^{}]+)\}\}',replacement,text)

    def write(self, node, node_modules):
        from classical_conditioning.reviews.figure2_g_sources import bundle_markdown
        chunks=re.split(r'^## (.+)$',bundle_markdown(self.manuscript.read_text()),flags=re.M)
        chapters=[]
        for i in range(1,len(chunks),2):
            title=chunks[i];slug=re.sub(r'[^a-z0-9]+','-',title.lower()).strip('-')
            chapters.append((slug,title,self.substitute(chunks[i+1])))
        sources=[]
        for key,s in self.sources.items():
            link=f"<a href='{html.escape(s['copy'])}' target='_blank' rel='noopener'>Open saved source</a>" if s['copy'] else "Existing detailed atlas retained at its recorded drive path."
            sources.append(f"<div class='source-card' id='source-{key}'><h4>{html.escape(s['title'])}</h4><p>{link} <span class='source-badge'>{html.escape(s['verification'])}</span></p><p>{html.escape(s['note'])}</p><code>{html.escape(s['path'])}</code><br><code>SHA-256 {s['sha256']}</code></div>")
        self.tokens['sources:inventory']=''.join(sources)
        # No JavaScript executes in Markdown. marked is a build-time dependency.
        env=dict(os.environ)
        if node_modules: env['NODE_PATH']=str(node_modules)
        js="const {marked}=require('marked');const fs=require('fs');const parts=JSON.parse(fs.readFileSync(0,'utf8'));process.stdout.write(JSON.stringify(parts.map(p=>marked.parse(p,{gfm:true}))));"
        rendered=json.loads(subprocess.check_output([node,'-e',js],input=json.dumps([c[2] for c in chapters]).encode(),env=env))
        fingerprint=digest(json.dumps(self.sources,sort_keys=True).encode())
        config={'schema':'figure2-g-review-notebook/1','guide_id':'figure2-g-analysis-guide-20261017-v01','source_fingerprint':fingerprint,'sections':[{'id':s,'title':t} for s,t,_ in chapters]}
        body=[]
        for (slug,title,_),content in zip(chapters,rendered):
            # marked emits plain tables for prose; saved-result tables already
            # carry their own wrapper/caption. Both must scroll within a chapter.
            content=re.sub(r'<table>\s*<thead>.*?</table>',lambda m:"<div class='table-wrap' tabindex='0'>"+m[0]+"</div>",content,flags=re.S)
            note=f"""<details class='notebook no-print' data-section='{slug}'><summary>Your review notebook — {html.escape(title)}</summary><p>These are your provisional notes. They do not approve an analysis or change any figure.</p><label class='check-row'><input type='checkbox' data-field='understood'> I can explain this section without AI</label><label>Review position <select data-field='position'><option value='unreviewed'>Not reviewed</option><option value='questions'>Questions remain</option><option value='tentative'>Provisional preference</option><option value='ready'>Ready for discussion</option></select></label><label>My understanding<textarea data-field='understanding' placeholder='Explain the result and its limits in your own words.'></textarea></label><label>Open questions<textarea data-field='questions' placeholder='What do I still need to check?'></textarea></label><label>Provisional decision and reason<textarea data-field='decision' placeholder='State the choice, evidence and remaining conditions.'></textarea></label><span class='save-status' data-local-status>No notes entered.</span></details>"""
            body.append(f"<section class='chapter' id='{slug}'><h2>{html.escape(title)}</h2>{content}{note}</section>")
        body.append("<section class='chapter' id='source-inventory'><h2>Source inventory and identities</h2><p class='small'>A dated evidence snapshot. Source copies retain original bytes. Embedded historical illustrations retain their original bytes; the current G excerpt is an HTML-only isolated view. This guide does not certify a new scientific freeze.</p>"+''.join(sources)+"</section>")
        nav=''.join(f"<a href='#{s}'>{html.escape(t)}</a>" for s,t,_ in chapters)+"<a href='#source-inventory'>Source inventory</a>"
        stamp=datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
        document=f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Figure 2 G — analysis guide and review notebook</title><style>{CSS}</style></head><body><a class='skip-link' href='#main'>Skip to guide</a><div class='shell'><aside class='rail'><div class='brand'>Classical conditioning</div><h2>Read, compare, decide</h2><p class='small'>Figure 2 G · Delay<br>Evidence guide v01 · 17 October 2026</p><label class='small' for='guide-search'>Find a topic</label><input id='guide-search' type='search' placeholder='e.g. baseline, onset, WIP'><div class='search-results' id='search-results' aria-live='polite'></div><details class='chapter-menu' id='chapter-menu' open><summary>Chapters and source inventory</summary><nav aria-label='Guide chapters'>{nav}</nav></details><div class='read-progress' aria-hidden='true'><span id='progress-fill'></span></div><div class='tools'><button id='export-notes'>Export notes</button><button id='import-notes'>Import notes</button><button id='print-guide'>Print guide</button></div><input id='import-file' type='file' accept='.json,application/json' hidden><p class='save-status' id='save-status' role='status'>Checking browser note storage…</p><p class='small' id='import-message' role='status'></p><details class='note-help'><summary>Where are my notes saved?</summary><p class='small'>Notes belong to this guide version and browser origin. Export them before changing devices, server ports or browser profiles.</p></details></aside><main id='main'><header class='hero'><div class='eyebrow'>A guide for independent scientific review</div><h1>What Figure 2 G shows.<br>What its analyses establish.</h1><p>Follow the evidence from bout frames to population changes, acquisition timing, exclusion rules and learner counts. The final scientific choices remain yours.</p><div class='meta'>Version 01 · Evidence captured {stamp}<br>57 fish · 90 scheduled CS trials · 4,811 eligible conditional-vigor outcomes<br>Current scientific G: scoped October 9 freeze. Alternatives: exploratory.<br>Drive-based guide. No AI or network service is needed to read the saved explanations and outputs.</div></header>{''.join(body)}<footer>Built from authenticated saved outputs and source snapshots. No model was refitted, no fish membership changed, and no figure selection or freeze record was updated. <a href='evidence-manifest.json'>Guide evidence manifest</a> · <a href='topic-inventory.json'>Topic coverage inventory</a>.</footer></main></div><dialog id='import-dialog'><h2>Import review notes</h2><p id='import-warning'></p><pre id='import-preview'></pre><p>Only review notes in this browser will change. Existing notes will be backed up for export.</p><button id='confirm-import'>Replace notes with this import</button> <button id='cancel-import'>Cancel</button></dialog><script type='application/json' id='guide-config'>{json.dumps(config).replace('<','\\u003c')}</script><script>{NOTEBOOK.read_text()}</script></body></html>"""
        (self.out/'review.html').write_text(document)
        topics=[{'section':s,'title':t,'has_notebook':True} for s,t,_ in chapters]
        coverage={
            'Two distinct scientific claims':2,'Trial schedule and repeated-measure units':2,
            'LMM versus LLM':3,'Worked outcome and contrast examples':3,
            'Pointwise, simultaneous, IQR and multiplicity':3,
            'Selected scientific G versus current presentation':4,
            'Frame-to-trial-to-population aggregation':4,'D/M/R and BH90 hypotheses':4,
            'Unavailable Test3 and heavy tails':4,'Outcome alternatives and zero handling':5,
            'Global versus phase-aware models':5,'Archived plots and CI/IQR versions':5,
            'Independent direct and adjusted targets':6,'Planned phase numerical results':6,
            'Unsmoothed and pointwise fish comparators':6,'Covariance, day, overlap and omission sensitivities':6,
            'First review and subsequent implementation repairs':6,'Proposed versus executed statistical methods':6,
            'Meaningful thresholds and scheduled persistence':7,'Timing outputs and unbounded results':7,
            'Individual spline/state-space and held-out checks':7,'LMM calibration limitations':8,
            'Unsmoothed calibration and precision limits':8,'Missingness imbalance and conditional intensity':9,
            'Four distinct eligibility/membership/label concepts':9,'February and March Git rules':9,
            'ANY versus ALL and last-US proxy':9,'Every reconstructed cohort recipe and its output':9,
            'Protocol corruption and technical-only exclusion':9,'Four legacy classifier definitions':10,
            'September and bout-only counts with denominators':10,'No approved learner count and circular validation':10,
            'No-star, dense-star and bracket versions':11,'Why the figures look different':11,
            'Neuroscience reporting examples':11,'Supported conclusions and final decision checklist':12,
            'Acquisition, learners and extinction kept separate':12,'Evidence gaps':12,'Manual source review and glossary':13,
            'Later author intensity clarification and frame weighting':14,
            'Configured versus fitted parameters and model teaching':14,
            'GAMM proposal, boundary argument and exploration plan':14,
            'Figure 3 acquisition score versus legacy recovery':14,
            'Clean-chat handover and remaining questions':14,
        }
        (self.out/'topic-inventory.json').write_text(json.dumps({'chapters':topics,'coverage':[{'topic':topic,'section':chapters[index-1][0]} for topic,index in coverage.items()],'source_ids':list(self.sources),'scope':'Delay and Figure 2 G; minimal individual-fish material','known_gaps':['Actual original historical discard rosters not established','Original segmented Bout/700 FPS inputs unavailable','Exact selected MR-local-block-tests.csv not recovered; displayed entries authenticated via sidecar'],'no_models_refitted':True},indent=2)+'\n')
        self.checks.update({'all_tokens_resolved':not re.search(r'\{\{[^{}]+\}\}',document),'source_count':len(self.sources),'chapter_count':len(chapters),'figure_count':len(self.figure_hashes)+1})
        # Recheck the complete read set after assembling, before publishing.
        changed=[p for p,h in self.protected.items() if digest(Path(p).read_bytes())!=h]
        if changed:raise ValueError('Source changed during build: '+str(changed))
        self.checks['all_read_sources_unchanged_before_publication']=True
        manifest={'schema':'figure2-g-analysis-guide/1','guide':config,'built_utc':stamp,'sources':self.sources,'protected_source_hashes':self.protected,'figures':self.figure_hashes,'checks':self.checks,'scientific_status':'Teaching and review only; no new inference or freeze','artifacts':[]}
        for p in sorted(self.out.rglob('*')):
            if p.is_file():manifest['artifacts'].append({'path':str(p.relative_to(self.out)),'sha256':digest(p.read_bytes())})
        (self.out/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--manuscript',type=Path,default=CONTENT)
    p.add_argument('--manuscript-sha256')
    p.add_argument('--node',default=shutil.which('node'))
    p.add_argument('--node-modules',type=Path)
    args=p.parse_args()
    if not Path('/Volumes/JOAQUIM').is_mount():raise SystemExit('JOAQUIM is not mounted; no substitute output location is allowed.')
    output=args.output_dir.resolve()
    if not output.is_relative_to(OUTPUTS.resolve()):raise SystemExit('Guide output must be on JOAQUIM ClassicalConditioning Outputs.')
    if output.exists():raise SystemExit('Output already exists; use a new version/draft directory to preserve it.')
    if not args.node:raise SystemExit('Provide --node and, if necessary, --node-modules for marked.')
    output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.g-guide-build-',dir=output.parent))
    try:
        (stage/'assets').mkdir()
        guide=Guide(stage,args.manuscript,args.manuscript_sha256);guide.capture();guide.write(args.node,args.node_modules)
        # Atomic publication of a NEW guide directory, never a figure pointer.
        os.rename(stage,output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    print(json.dumps({'html':str(output/'review.html'),'manifest':str(output/'evidence-manifest.json'),'checks':guide.checks},indent=2))


if __name__=='__main__':main()
