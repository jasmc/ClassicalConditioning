"""Scoped Figure 2 draft: median across saved, individually centered fish bins.

Reuse the selected SVG, replacing only six heatmap rasters and the quantity
label. Preserve all per-fish calculations, rosters, lower rows and freezes.
"""
from pathlib import Path
import argparse
import base64
import copy
import io
import json
import sys
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib as mpl
from matplotlib.colors import Normalize
from classical_conditioning.figure_workspace import Snapshot, publish_state, sha, tag
from classical_conditioning.artifacts import sha256_file
from classical_conditioning.figures.layout_frames import assess
from classical_conditioning.figures.presentation_profiles import load_presentation_profile
from main_paper_figures import render

STATE = REPO / "docs/paper/figures/figure-02/current-state.json"
BASE = Path("J:/ClassicalConditioning Outputs/current/reviews")
OUT = BASE / "figure-02-median-pooling-20261020"
XLINK = "{http://www.w3.org/1999/xlink}href"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def enc(x):
    return (json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def ref(p):
    return {"path": str(Path(p).resolve()), "sha256": sha256_file(Path(p))}


def write(p, value):
    with Path(p).open("xb") as f:
        f.write(value if isinstance(value, bytes) else enc(value))


def check(r):
    assert ref(r["path"])["sha256"] == r["sha256"], r["path"]


def guard():
    s = Snapshot([])
    s.hashes = load(OUT / "starting-ticket.json")
    s.check()
    return s


def build():
    state = load(STATE)
    d = state["draft"]
    for k in ["working_svg", "review", "legend", "semantic_registry"]:
        check(d[k])
    profile = load_presentation_profile("nature-main")
    assert profile.inputs == d["presentation_profile"]["inputs"]
    assert not OUT.exists(), "Preserve an existing median candidate; do not overwrite"
    assert BASE.is_dir(), "JOAQUIM required"
    sources = [STATE, Path(__file__), *[Path(d[k]["path"]) for k in ["working_svg", "review", "legend", "semantic_registry"]]]
    sources += [REPO / p for p in profile.inputs]
    # Read only the authenticated saved bin calculations, never raw recordings.
    ap = BASE / "figure-02-a-pooled"
    ac = load(ap / "calculation.json")
    for r in [ac["fish_bins"], ac["pooled_bins"]]:
        p = Path(r["path"].replace("/Volumes/JOAQUIM/", "J:/"))
        assert ref(p)["sha256"] == r["sha256"]
        sources.append(p)
    cohort = Path("J:/Digested Data/allDelay-full-v1/Processed data/Cohorts/allDelay-full-v1/cohort-manifest-v1_review.csv")
    expected = next(r for r in ac["cohort_inputs"] if r["path"].endswith(".csv"))
    assert ref(cohort)["sha256"] == expected["sha256"]
    ar = pd.read_csv(cohort)
    ar = ar.loc[ar.primary_included.eq(True)]
    assert len(ar) == 57 and ar.review_status.eq("approved").all()
    mapping_a = ar.set_index("recording_id").condition_id.to_dict()
    sources += [cohort, ap / "calculation.json"]
    parts = {"A": pd.read_parquet(ap / "fish-bins.parquet")}
    mappings = {"A": mapping_a}
    bindings = {"A": [ref(ap / "fish-bins.parquet"), ref(cohort), ref(ap / "calculation.json")]}
    for panel, plan_ref, rows_key, cache, record_bindings in [
        ("B", d["matched_trace3_scientific_draft"]["data"], "selected", BASE / "figure-02-3strace/fish", None),
        ("C", d["trace10_scientific_draft"]["roster"], "selected", BASE / "figure-02-10strace-20261020/fish", d["trace10_scientific_draft"]["data"]),
    ]:
        check(plan_ref)
        plan = load(plan_ref["path"])
        sources.append(Path(plan_ref["path"]))
        if record_bindings:
            check(record_bindings)
            records = load(record_bindings["path"])["fish_records"]
            sources.append(Path(record_bindings["path"]))
        else:
            records = plan["inputs"]
        rows = plan[rows_key]
        mappings[panel] = {r["recording_id"]: r["condition"] for r in rows}
        tables = []
        bindings[panel] = [plan_ref]
        for rid in mappings[panel]:
            rp = cache / (rid + ".json")
            bound = next(r for r in records if Path(r["path"]) == rp)
            check(bound)
            record = load(rp)
            assert record["heatmap"] == "complete"
            bins = next(r for r in record["outputs"] if r["path"].endswith("-bins.parquet"))
            check(bins)
            sources += [rp, Path(bins["path"])]
            bindings[panel] += [bound, bins]
            tables.append(pd.read_parquet(bins["path"]))
        parts[panel] = pd.concat(tables, ignore_index=True)
    ticket = Snapshot(sources)
    OUT.mkdir()
    write(OUT / "starting-ticket.json", ticket.hashes)
    write(OUT / "previous-state.json", STATE.read_bytes())
    root = ET.parse(d["working_svg"]["path"]).getroot()
    ids = {n.get("id"): n for n in root.iter() if n.get("id")}
    registry = load(d["semantic_registry"]["path"])
    cmap = mpl.colormaps["managua_r"].copy()
    cmap.set_bad("black")
    results, modified = {}, []
    for panel in "ABC":
        fish = parts[panel].copy()
        assert not fish.duplicated(["fish", "acquisition_trial", "bin_index"]).any()
        assert set(fish.fish) == set(mappings[panel])
        fish["condition_id"] = fish.fish.map(mappings[panel])
        pooled = fish.groupby(["condition_id", "acquisition_trial", "bin_index"]).delta_log_vigor.agg(
            median_delta_log_vigor="median", contributing_fish="count").reset_index()
        totals = pd.Series(mappings[panel]).value_counts()
        pooled["cohort_fish"] = pooled.condition_id.map(totals)
        pooled["coverage_fraction"] = pooled.contributing_fish / pooled.cohort_fish
        target = OUT / (panel + "-median-pooled-bins.parquet")
        pooled.to_parquet(target, index=False)
        results[panel] = {"data": ref(target), "inputs": bindings[panel], "conditions": {}}
        for condition in totals.index:
            key = panel + "-heatmap-" + condition
            row = registry["element_registry"][key]
            old_map = row["source_mapping"]
            limits = old_map["limits"]
            assert limits == [-.1, .1], "Retain the selected colour range"
            matrix = pooled.loc[pooled.condition_id.eq(condition)].pivot(index="acquisition_trial", columns="bin_index", values="median_delta_log_vigor").reindex(index=range(5,95), columns=range(40)).to_numpy()
            # Independent matrix calculation from the fish-by-cell array.
            sub = fish.loc[fish.condition_id.eq(condition)]
            wide = sub.pivot(index=["acquisition_trial", "bin_index"], columns="fish", values="delta_log_vigor").reindex(pd.MultiIndex.from_product([range(5,95),range(40)],names=["acquisition_trial","bin_index"]))
            np.testing.assert_allclose(np.nanmedian(wide.to_numpy(), axis=1).reshape(90,40), matrix, rtol=0, atol=1e-12, equal_nan=True)
            old_data = pd.read_parquet(old_map["artifact"]["path"])
            old = old_data.loc[old_data.condition_id.eq(condition)].pivot(index="acquisition_trial", columns="bin_index", values="mean_delta_log_vigor").reindex(index=range(5,95), columns=range(40)).to_numpy()
            assert sha(np.asarray(old,dtype="<f8").tobytes()) == old_map["matrix_sha256"]
            t = pooled.loc[pooled.condition_id.eq(condition)]
            np.testing.assert_array_equal(t.contributing_fish, old_data.loc[old_data.condition_id.eq(condition)].contributing_fish)
            image = next(n for n in ids[key].iter() if n.tag == tag("image"))
            original = Image.open(io.BytesIO(base64.b64decode(image.get(XLINK).split(",",1)[1]))).convert("RGBA")
            expected_old = np.asarray(cmap(Normalize(*limits)(np.ma.masked_invalid(old)))*255,dtype=np.uint8)[::-1]
            assert np.array_equal(np.asarray(original.resize((40,90),Image.Resampling.NEAREST)),expected_old)
            colors = np.asarray(cmap(Normalize(*limits)(np.ma.masked_invalid(matrix)))*255,dtype=np.uint8)[::-1]
            new = Image.fromarray(colors).resize(original.size,Image.Resampling.NEAREST)
            assert np.array_equal(np.asarray(new.resize((40,90),Image.Resampling.NEAREST)),colors)
            buf = io.BytesIO()
            new.save(buf,format="PNG")
            image.set(XLINK,"data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode())
            modified.append(image.get("id"))
            mapping = {**old_map,"artifact":ref(target),"field":"median_delta_log_vigor","matrix_sha256":sha(np.asarray(matrix,dtype="<f8").tobytes()),"across_fish_estimator":"median of finite per-fish centered values","coverage_field":"contributing_fish","minimum_contributors":1,"scientific_status":"unfrozen scientific draft; author-requested median pooling"}
            row["source_mapping"] = mapping
            row["superseded_mean_mapping"] = old_map
            results[panel]["conditions"][condition] = {"fish":int(totals[condition]),"minimum_contributors":int(t.contributing_fish.min()),"maximum_contributors":int(t.contributing_fish.max()),"matrix_sha256":mapping["matrix_sha256"],"exact_raster_cell_colours":True,"changed_cells":int(np.sum(~np.isclose(matrix,old,equal_nan=True))),"maximum_absolute_change":float(np.nanmax(np.abs(matrix-old))),"image_size":list(new.size)}
    labels = [n for n in root.iter(tag("text")) if n.text == "Mean Δ ln vigor"]
    assert len(labels)==1
    labels[0].text = "Median Δ ln vigor"
    # Prove all other SVG artists, geometry, text and lower-row results unchanged.
    original_root = ET.parse(d["working_svg"]["path"]).getroot()
    source_images = {n.get("id"):n for n in original_root.iter(tag("image"))}
    comparison = copy.deepcopy(root)
    for n in comparison.iter(tag("image")):
        if n.get("id") in modified:
            n.set(XLINK,source_images[n.get("id")].get(XLINK))
    next(n for n in comparison.iter(tag("text")) if n.text=="Median Δ ln vigor").text="Mean Δ ln vigor"
    assert ET.tostring(comparison)==ET.tostring(original_root)
    svg = ET.tostring(root,encoding="utf-8",xml_declaration=True)
    write(OUT / "figure2.svg",svg)
    legend = Path(d["legend"]["path"]).read_text(encoding="utf-8")
    assert "then averaged with equal fish weight" in legend
    legend = legend.replace("then averaged with equal fish weight","then summarized by the median across available fish (one centered value per fish)")
    legend += "\nThis unfrozen A–C draft replaces the across-fish mean with a median; per-fish preprocessing, contributing-fish counts and D–I results are unchanged.\n"
    write(OUT / "legend.md",legend.encode())
    summary={"status":"numerically verified median-pooling draft; saved-render review pending","panels":results,"changed_scope":"Six A/B/C heatmap rasters and shared quantity label only","all_other_svg_content_identical":True,"baseline_and_missing_policy":"unchanged; finite contributors only; no zero imputation","no_freeze":True,"generator":ref(Path(__file__)),"profile_inputs":profile.inputs}
    write(OUT / "numerical-evidence.json",summary)
    registry["median_pooling_draft"]={"evidence":ref(OUT / "numerical-evidence.json"),"scope":list("ABC"),"supersedes":"Across-fish mean in selected presentation only; accepted A mean freeze retained","source_registry":d["semantic_registry"]}
    registry["axis_inventory"]["artifact_sha256"]=sha(svg)
    registry["whole_figure_frozen"]=False
    write(OUT / "semantic-registry.json",registry)
    page=render([{"id":"fig2","svg_bytes":svg,"legend_text":legend,"svg":ref(OUT / "figure2.svg"),"legend":ref(OUT / "legend.md"),"whole_figure_frozen":False}])[0]
    write(OUT / "figure2.html",page)
    ticket.check()
    print(json.dumps({"html":str(OUT / "figure2.html"),"results":{p:r["conditions"] for p,r in results.items()}}))


def publish(temporary, note):
    ticket=guard()
    svg=(OUT / "figure2.svg").read_bytes()
    browser=load(temporary / "browser.json")
    frame=assess(svg,browser)
    assert frame["status"]=="passed",frame["issues"]
    columns=load(temporary / "column-measurements.json")
    assert columns["status"]=="passed" and columns["artifact_sha256"]==sha(svg)
    assert columns["visible_scope"]["figure_count"]==columns["visible_scope"]["svg_count"]==1
    assert "then summarized by the median" in columns["visible_scope"]["legend"]
    for name,obj in [("browser.json",browser),("layout-evidence.json",frame),("column-measurements.json",columns)]:
        write(OUT / name,obj)
    previous=load(OUT / "previous-state.json")
    state=copy.deepcopy(previous)
    evidence={"status":"verified scoped scientific draft; no freeze","numerical":ref(OUT / "numerical-evidence.json"),"layout_frames":ref(OUT / "layout-evidence.json"),"columns":ref(OUT / "column-measurements.json"),"visual_review":note,"source_state":ref(OUT / "previous-state.json"),"generator":ref(Path(__file__))}
    write(OUT / "evidence.json",evidence)
    state.setdefault("preserved_drafts",[]).append({"id":"figure2-before-median-pooling-20261020","state_snapshot":ref(OUT / "previous-state.json"),**{k:previous["draft"][k] for k in ["working_svg","review","legend","semantic_registry"]}})
    d=state["draft"]
    d.update(working_svg=ref(OUT / "figure2.svg"),review=ref(OUT / "figure2.html"),legend={**ref(OUT / "legend.md"),"panel_ids":list("ABCDEFGHI")},semantic_registry=ref(OUT / "semantic-registry.json"),layout_frames=ref(OUT / "layout-evidence.json"),status="Unfrozen A/B/C across-fish median draft; retained selected +/-0.1 presentation and D-I results")
    d["median_pooling"]=evidence
    d["verification"]={"layout_frames":ref(OUT / "layout-evidence.json"),"columns":ref(OUT / "column-measurements.json"),"scientific_data":ref(OUT / "numerical-evidence.json"),"visual_review":{"status":"performed","note":note},"coverage":"Independent median and unchanged counts; exact saved raster cells; unchanged other SVG content; saved 600dpi frame/column checks. No freeze or broader production claim."}
    d.setdefault("consistency_requests",[]).append({"id":"fig2-abc-median-pooling-20261020","scope":list("ABC"),"type":"scientific","author_direction":"Try median instead of mean across fish and show Figure 2","status":"verified scientific draft","evidence":ref(OUT / "evidence.json")})
    d.setdefault("reproduction",{})["last_scoped_edit"]={"generator":ref(Path(__file__)),"command":".venv-trace/Scripts/python.exe scripts/figure2_median_pooling.py build","starting_ticket":ref(OUT / "starting-ticket.json")}
    ticket.hashes.update(Snapshot([p for p in OUT.iterdir() if p.is_file()]).hashes)
    publish_state(STATE,state,ticket)
    write(OUT / "publication.json",{"state":ref(STATE),"html":d["review"],"svg":d["working_svg"],"no_freeze":True})
    print(d["review"]["path"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("operation",choices=["build","publish"])
    p.add_argument("--temporary",type=Path)
    p.add_argument("--visual-review-note")
    a=p.parse_args()
    if a.operation=="build":build()
    elif a.temporary and a.visual_review_note:publish(a.temporary,a.visual_review_note)
    else:p.error("publish requires saved browser evidence and an actual visual-review note")
