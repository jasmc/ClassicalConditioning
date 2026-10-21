"""Audit and publish the separate Figure 2 presentation through scoped guards."""
from pathlib import Path
import argparse
import base64
import copy
import io
import json
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib as mpl
from matplotlib.colors import Normalize

import explore_figure2_columns as build
from classical_conditioning.figure_workspace import Snapshot, publish_state, sha, tag, measured_presentation
from classical_conditioning.figures.layout_frames import assess
from main_paper_figures import render

OUT, STATE = build.OUT, build.STATE
load, ref, write, encode = build.load, build.ref, build.write, build.encode

LEGEND = """Figure 2 | Population conditioning measures with exploratory display limits.

(a–c) Control/conditioned population heatmaps: Delay, 28/29; 3sTrace, 90/89; 10sTrace, 32/29 fish. Direct 1 s median ln positive detected-bout vigor, centred within fish/trial by finite baseline-bin medians [−15,0) s, then averaged with equal fish weight. Colours saturate at ±0.1; black denotes missing values. Green: CS onset/offset; violet: expected training US (10sTrace: 20 s). Display trials 1–90 correspond to acquisition trials 5–94.

(d–f) Fish/block median response-minus-baseline median ln positive valid moving-frame vigor. Baseline [−15,0) s; Delay/3sTrace/10sTrace responses [0,9)/[0,13)/[0,20) s. PT/ET/LT: trials 10–14/65–69/90–94; minimum three eligible trials per five-trial block. Thin lines: paired fish; black: median/IQR. Two-sided exact sign tests (zero/paired changes) and Brunner–Munzel t tests (independent blocks/paired changes), separate Holm18 per panel. Dagger: versus zero; stars: adjusted p<0.05/0.01/0.001/0.0001.

(g–i) Across-fish median log-vigor differences; pointwise 95% intervals from 5,000 whole-fish resamples (seed 10). Shared statistical rows: D, baseline-adjusted block interaction (Holm8); M, local condition mean (BH9); R raw/BH, local slope (raw/BH9); LMM, phase-aware random-intercept/slope change from average Pre5–14 (BH90). Families are separate within panels; H/I show at most three stars; n/a denotes unavailable fits. Trial stars denote BH90 p<0.05. Exploratory significance does not establish learning onset.

D–I axes span ±0.1: out-of-range observations, lines and uncertainty are clipped only for display. All source values and statistics remain unchanged. C/F/I share 32 Control/29 Trace fish; excluded 20230315_04 (timing) and 20250715_08 (short recording) remain in audit/sensitivity records. These 32 controls share historical 3sTrace acquisitions and are not independent across assays.
"""


def inputs_guard():
    ticket = build.baseline()
    extra = Snapshot([])
    extra.hashes = load(OUT / "build-ticket.json")
    extra.check()
    ticket.hashes.update(extra.hashes)
    return ticket


def prepare():
    ticket = inputs_guard()
    candidate = load(OUT / "candidate.json")
    for key in ["svg", "html", "legend", "render_registry", "numerical_evidence", "generator"]:
        assert ref(candidate[key]["path"])["sha256"] == candidate[key]["sha256"], key
    original = ET.parse(OUT / "previous-selected/figure2.svg").getroot()
    current = ET.parse(OUT / "figure2.svg").getroot()
    ids = {n.get("id"): n for n in current.iter() if n.get("id")}
    old = {n.get("id"): n for n in original.iter() if n.get("id")}
    numeric = load(OUT / "numerical-display-evidence.json")
    registry = load(OUT / "figure2-render.figure.json")["artist_registry"]
    inherited = load(OUT / "previous-selected/semantic-registry.json")
    cmap = mpl.colormaps["managua_r"].copy()
    cmap.set_bad("black")
    pixels = []
    for p, condition in [("A", "delay"), ("B", "trace"), ("C", "trace")]:
        path = Path(numeric["panels"][p]["source"]["path"])
        data = pd.read_parquet(path)
        for i, c in enumerate(["control", condition]):
            matrix = data.loc[data.condition_id.eq(c)].pivot(index="acquisition_trial", columns="bin_index", values="mean_delta_log_vigor").reindex(index=range(5, 95), columns=range(40)).to_numpy()
            assert sha(np.asarray(matrix, dtype="<f8").tobytes()) == numeric["panels"][p]["conditions"][c]["matrix_sha256"]
            for label, node, lim, flip in [("selected", old[f"fig2__{p.lower()}{i}__heatmap__vigor-cells"], .25, p != "C"),
                                            ("exploratory", ids[p + "-heatmap-" + c], .1, True)]:
                image = next(n for n in node.iter() if n.tag == tag("image"))
                uri = image.get("{http://www.w3.org/1999/xlink}href", image.get("href"))
                rgba = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGBA")
                sampled = np.asarray(rgba.resize((40, 90), Image.Resampling.NEAREST))
                expected = np.asarray(cmap(Normalize(-lim, lim)(np.ma.masked_invalid(matrix))) * 255, dtype=np.uint8)
                if flip:
                    expected = expected[::-1]
                assert np.array_equal(sampled, expected), (p, c, label)
                pixels.append({"panel": p, "condition": c, "version": label, "source": ref(path),
                               "image_size": list(rgba.size), "exact_cell_colours_and_orientation": True,
                               "display_limits": [-lim, lim], "maximum_channel_difference": 0})
    strings = []
    def text(node):
        return "".join(node.itertext()).strip()
    for p in "DEF":
        tests = pd.read_csv(numeric["panels"][p]["statistics"]["path"])
        prefix = "trace10-F__" if p == "F" else "matched-" + p + "__"
        # The selected renderer displays zero/within/between marks; the full
        # 18-test family, including paired-change contrasts, stays in its CSV.
        for row in tests.loc[tests.p_holm18.lt(.05) & tests.family.isin(["zero", "within", "between"])].itertuples():
            left, right = str(int(row.left)), str(int(row.right))
            if row.family == "within":
                old_key = prefix + f"{p}-{row.condition}-within-label-{left}-{right}"
                new_key = f"{p}-{row.condition}-within-{left}-{right}-label"
                glyph = row.stars
            elif row.family == "zero":
                old_key = prefix + f"{p}-{row.condition}-zero-label-{left}"
                new_key = f"{p}-{row.condition}-zero-label-{left}"
                glyph = "†" + row.stars
            else:
                old_key = prefix + f"{p}-between-label-{left}"
                new_key = f"{p}-between-{left}-label"
                glyph = row.stars
            assert text(old[old_key]) == text(ids[new_key]) == glyph, (old_key, new_key)
            strings.append({"old_element_id": old_key, "new_element_id": new_key, "glyph": glyph,
                            "p_holm18": row.p_holm18, "statistics": numeric["panels"][p]["statistics"]})
    for p in "GHI":
        for record in numeric["statistical_annotations"][p]:
            key = record["key"]
            if "phase-header" in key or key.endswith("trial-stars"):
                continue
            old_key = "edit-G-" + key if p == "G" else ("matched-H__" if p == "H" else "trace10-I__") + key
            assert text(old[old_key]) == text(ids[key]) == record["glyph"], key
            strings.append({"old_element_id": old_key, "new_element_id": key, "glyph": record["glyph"]})
    masks = {}
    for p in "HI":
        old_registry = load(inherited["matched_panel_registries"][p]["path"])["artist_registry"]
        before = old_registry[p + "-trial-stars"]["source"]["selected_trials"]
        after = registry[p + "-trial-stars"]["source_mapping"]["selected_trials"]
        assert before == after, p
        masks[p] = after
    assert numeric["panels"]["E"]["eligible_blocks"] == 530
    # Preserve scientific relationships and replace only obsolete presentation geometry.
    axes = []
    for col, p in enumerate("ABC"):
        for i, c in enumerate(["control", "delay" if p == "A" else "trace"]):
            axes.extend([{"id": p + str(i) + "-x", "panel": p, "orientation": "x", "quantity": "time from CS onset", "units": "s", "limits": [-20, 20], "tick_values": [-20, 0, 20], "source": numeric["panels"][p]["source"]},
                         {"id": p + str(i) + "-y", "panel": p, "orientation": "y", "quantity": "displayed trial", "units": "trial", "limits": [90.5, .5], "mapping": "acquisition trial minus 4", "source": numeric["panels"][p]["source"]}])
    axes.append({"id": "shared-colourbar", "orientation": "color", "panels": list("ABC"), "quantity": "equal-fish mean of baseline-centred median ln positive vigor", "units": "dimensionless", "limits": [-.1, .1], "ticks": [-.1, 0, .1], "normalization": "linear", "colormap": "managua_r", "outside": "saturate", "missing": "black"})
    for p in "DEF":
        axes.extend([{"id": p + "-x", "panel": p, "orientation": "x", "quantity": "selected acquisition-trial block", "units": "categorical", "categories": ["PT", "ET", "LT"], "trial_spans": [[10, 14], [65, 69], [90, 94]]},
                     {"id": p + "-y", "panel": p, "orientation": "y", "quantity": "fish/block median response-minus-baseline median ln positive valid moving-frame vigor", "units": "dimensionless", "limits": [-.1, .1], "ticks": [-.1, 0, .1], "baseline_s": [-15, 0], "response_s": [0, {"D": 9, "E": 13, "F": 20}[p]], "source": numeric["panels"][p]["source"]}])
    for p in "GHI":
        axes.extend([{"id": p + "-observed-x", "panel": p, "orientation": "x", "quantity": "global CS acquisition trial", "units": "trial", "limits": [4, 95], "ticks": [5, 25, 45, 65, 85, 94]},
                     {"id": p + "-observed-y", "panel": p, "orientation": "y", "quantity": "across-fish median of response-minus-baseline median ln positive valid moving-frame vigor", "units": "dimensionless", "limits": [-.1, .1], "ticks": [-.1, 0, .1], "baseline_s": [-15, 0], "response_s": [0, {"G": 9, "H": 13, "I": 20}[p]], "source": numeric["panels"][p]["source"]},
                     {"id": p + "-statistics", "panel": p, "coordinate_system": "statistical rows sharing global CS trial x", "outcome_axis": False, "rows": ["LMM BH90", "R BH9", "R raw", "M BH9", "D Holm8"], "row_pitch_mm": 2., "font_pt": 5.}])
    audit = {"status": "passed presentation-only source and saved-artwork audit", "svg": candidate["svg"],
             "heatmap_pixels": pixels, "statistical_glyphs_exactly_preserved": strings, "trial_star_masks": masks,
             "E_eligible_blocks": 530, "scientific_analyses_restarted": False, "scientific_changes": False,
             "source_value_hashes_rechecked": True, "learner_outputs_touched": False, "generator": ref(Path(__file__))}
    write(OUT / "source-and-artwork-audit.json", audit)
    write(OUT / "axis-inventory.json", {"artifact_sha256": candidate["svg"]["sha256"], "axes": axes,
                                      "sharing": "Numerical y labels shared within each row; local spines/ticks retained. Layout columns do not imply independent shared controls."})
    write(OUT / "legend-exploratory.md", LEGEND.encode())
    page = render([{"id": "fig2", "svg_bytes": (OUT / "figure2.svg").read_bytes(), "legend_text": LEGEND,
                    "svg": candidate["svg"], "legend": ref(OUT / "legend-exploratory.md"), "whole_figure_frozen": False}])[0]
    write(OUT / "figure2-review.html", page)
    candidate.update(html=ref(OUT / "figure2-review.html"), legend=ref(OUT / "legend-exploratory.md"),
                     artwork_audit=ref(OUT / "source-and-artwork-audit.json"), axis_inventory=ref(OUT / "axis-inventory.json"),
                     review_generator=ref(Path(__file__)), legend_word_count=len(LEGEND.split()))
    write(OUT / "review-candidate.json", candidate)
    ticket.check()
    print("AUDITED", len(strings), "statistical labels; 12 heatmap comparisons; legend words", len(LEGEND.split()))


def verify(temporary):
    ticket = inputs_guard()
    candidate = load(OUT / "review-candidate.json")
    svg = (OUT / "figure2.svg").read_bytes()
    browser, columns = load(temporary / "browser.json"), load(temporary / "column-measurements.json")
    frame = assess(svg, browser)
    assert frame["status"] == "passed", frame["issues"]
    assert columns["status"] == "passed" and columns["artifact_sha256"] == sha(svg)
    assert columns["visible_scope"]["figure_count"] == columns["visible_scope"]["svg_count"] == 1
    assert columns["visible_scope"]["legend"].strip() == "\n\n".join(p.strip() for p in LEGEND.strip().split("\n\n"))
    font_rows = [r for r in columns["text_measurements"]["fonts"] if r["id"].startswith(("fig2__g__stats__", "H-D-", "H-M-", "I-D-", "I-M-"))]
    assert font_rows and all(abs(r["size_pt"] - 5) < .01 for r in font_rows), font_rows[:2]
    assert all("Arial" in r["family"] for r in columns["text_measurements"]["fonts"])
    for name, obj in [("browser.json", browser), ("column-measurements.json", columns), ("layout-evidence.json", frame)]:
        write(OUT / name, obj)
    write(OUT / "effective-presentation.json", {**measured_presentation(svg, width_mm=156.),
          "browser_physical_text": ref(OUT / "column-measurements.json"), "profile_inputs": candidate["profile_inputs"],
          "coverage": "Final SVG physical typography and strokes plus saved 600dpi frame/column measurements; production PDF/font embedding and full geometric axis-gap certification remain separate"})
    ticket.check()
    print("SAVED RENDER PASSED", {"frame_issues": frame["issues"], "columns": columns["status"], "stats_font_pt": 5})


def publish(note):
    ticket = inputs_guard()
    candidate = load(OUT / "review-candidate.json")
    for key in ["svg", "html", "legend", "render_registry", "numerical_evidence", "generator", "artwork_audit", "axis_inventory", "review_generator"]:
        assert ref(candidate[key]["path"])["sha256"] == candidate[key]["sha256"], key
    frame = assess((OUT / "figure2.svg").read_bytes(), load(OUT / "browser.json"))
    assert frame["status"] == "passed"
    columns = load(OUT / "column-measurements.json")
    assert columns["status"] == "passed" and columns["artifact_sha256"] == candidate["svg"]["sha256"]
    implementation = columns["measurement_implementation"]
    assert ref(implementation["path"])["sha256"] == implementation["sha256"]
    registry = load(OUT / "previous-selected/semantic-registry.json")
    previous_relations = registry.get("panel_relationships", [])
    frames = json.loads(ET.parse(OUT / "figure2.svg").getroot().find(tag("metadata") + "[@id='layout-frames']").text)
    registry["inherited_previous_registry"] = {"source": candidate["source"]["semantic_registry"], "status": "Scientific context retained; previous SVG artists/layout measurements are historical"}
    registry["element_registry"] = load(OUT / "figure2-render.figure.json")["artist_registry"]
    registry["axis_inventory"] = load(OUT / "axis-inventory.json")
    registry["presentation_layout"] = frames
    registry["panel_relationships"] = [r for r in previous_relations if r.get("type") != "layout"] + frames["panel_relationships"]
    registry["historical_presentation_relationships"] = [r for r in previous_relations if r.get("type") == "layout"]
    registry["panel_relationships"].append({"id": "fig2-CFI-common61", "type": "scientific", "panels": list("CFI"),
        "roster": candidate["source"]["trace10_scientific_draft"]["roster"], "condition_counts": {"control": 32, "trace": 29},
        "shared_controls": "Historical 3sTrace acquisition identities; not independent new acquisitions across assays"})
    registry["whole_figure_frozen"] = False
    evidence = {"status": "verified separate exploratory presentation; unfrozen", "candidate": candidate,
        "layout_frames": ref(OUT / "layout-evidence.json"), "columns": ref(OUT / "column-measurements.json"),
        "source_and_artwork_audit": candidate["artwork_audit"], "effective_presentation": ref(OUT / "effective-presentation.json"),
        "visual_review": note, "coverage": "Exact saved SVG 600dpi painted/unclipped frames, three assay-column centres, panel letters, source matrices, statistical glyphs/trial masks and physical font sizes. No recomputation of science. Production PDF/font embedding and full geometric axis-gap certification remain separate.",
        "precedence": "Only this exploratory presentation supersedes earlier row outer-content gaps and display limits; scientific selections and original draft files are preserved"}
    registry["exploratory_columns_and_limits_20261020"] = evidence
    state = load(STATE)
    old = copy.deepcopy(state["draft"])
    state.setdefault("preserved_drafts", []).append({"id": "figure2-before-exploratory-range-20261020",
        "state_snapshot": ref(OUT / "previous-selected/state.json"),
        "working_svg": ref(OUT / "previous-selected/figure2.svg"), "review": ref(OUT / "previous-selected/figure2.html"),
        "legend": ref(OUT / "previous-selected/legend.md"), "semantic_registry": ref(OUT / "previous-selected/semantic-registry.json"),
        "original_paths": {k: old[k] for k in ["working_svg", "review", "legend", "semantic_registry"]}})
    draft = state["draft"]
    draft.update(working_svg=candidate["svg"], review=candidate["html"], legend={**candidate["legend"], "panel_ids": list("ABCDEFGHI")},
                 semantic_registry={"path": str(OUT / "semantic-registry.json"), "sha256": sha(encode(registry))},
                 layout_frames=ref(OUT / "layout-evidence.json"), effective_presentation=ref(OUT / "effective-presentation.json"),
                 assembly_scale={"final_width_mm": 156., "final_height_mm": 138., "plot_body_sizes_preserved": True},
                 status="Separate exploratory presentation: aligned ADG/BEH/CFI assay columns; display −0.1 to +0.1; scientific selections unchanged; no freeze")
    draft["exploratory_columns_and_limits"] = evidence
    draft["presentation_profile"] = {"id": "nature-main", "version": "1.0.1", "inputs": candidate["profile_inputs"],
        "evidence": {"path": str(OUT / "presentation-evidence.json"), "sha256": sha(encode(evidence))},
        "status": "Verified named display/layout/physical-font properties; remaining production coverage explicit"}
    draft["verification"] = {"layout_frames": ref(OUT / "layout-evidence.json"), "columns": ref(OUT / "column-measurements.json"),
        "source_and_artwork": candidate["artwork_audit"], "visual_review": {"status": "performed", "note": note}, "coverage": evidence["coverage"]}
    draft.setdefault("consistency_requests", []).append({"id": "fig2-exploratory-columns-limits-20261020", "scope": list("ABCDEFGHI"),
        "status": "verified", "properties": {"columns": ["ADG", "BEH", "CFI"], "display_limits": [-.1, .1],
        "statistics_font_pt": 5., "statistical_row_pitch_mm": 2.}, "evidence": ref(OUT / "column-measurements.json"), "scientific_changes": False})
    draft["reproduction"]["last_scoped_edit"] = {"generator": candidate["generator"], "review_generator": ref(Path(__file__)),
        "command": ".venv-trace/Scripts/python.exe scripts/explore_figure2_columns.py build",
        "starting_ticket": ref(OUT / "starting-ticket.json"), "source_snapshot": ref(OUT / "previous-selected/state.json")}
    bound = [OUT / n for n in ["review-candidate.json", "layout-evidence.json", "column-measurements.json", "browser.json", "effective-presentation.json"]]
    bound += [Path(candidate[k]["path"]) for k in ["svg", "html", "legend", "render_registry", "numerical_evidence", "artwork_audit", "axis_inventory"]]
    bound += [Path(__file__), Path(implementation["path"])]
    ticket.hashes.update(Snapshot(bound).hashes)
    publish_state(STATE, state, ticket, artifacts=[(OUT / "semantic-registry.json", encode(registry)),
                                                  (OUT / "presentation-evidence.json", encode(evidence))])
    write(OUT / "publication.json", {"status": "published separate exploratory draft", "state": ref(STATE),
          "svg": candidate["svg"], "html": candidate["html"], "legend": candidate["legend"], "no_freeze": True,
          "previous_selected_preserved": state["preserved_drafts"][-1], "visual_review": note})
    print(candidate["html"]["path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "verify", "publish"])
    parser.add_argument("--temporary", type=Path)
    parser.add_argument("--visual-review-note")
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare()
    elif args.operation == "verify" and args.temporary:
        verify(args.temporary)
    elif args.operation == "publish" and args.visual_review_note:
        publish(args.visual_review_note)
    else:
        parser.error("verify requires --temporary; publish requires --visual-review-note")
