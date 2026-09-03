"""Semantic, provenance-rich figure export."""

from __future__ import annotations

import json
import re
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from classical_conditioning.artifacts import (
    artifact_staging,
    publish_transaction,
    sha256_file,
    write_json_atomic,
)

SVG_NAMESPACE = "http://www.w3.org/2000/svg"
PROVENANCE_NAMESPACE = "https://classical-conditioning.local/provenance/v1"
ET.register_namespace("", SVG_NAMESPACE)
ET.register_namespace("cc", PROVENANCE_NAMESPACE)


class FigureMode(str, Enum):
    PUBLICATION = "publication"
    STATIC = "static"
    INTERACTIVE = "interactive"


@dataclass(frozen=True)
class FigureProvenance:
    figure_id: str
    analysis_recipe: str
    source_file: str
    source_symbol: str
    source_hash: str
    reproduction_snippet: str
    input_artifacts: tuple[dict[str, Any], ...]
    cohort_hash: str | None = None
    figure_spec_version: str = "1.0"
    artist_mappings: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class FigureExportResult:
    outputs: tuple[Path, ...]
    sidecar: Path


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-")
    return normalized.lower() or "unnamed"


def assign_axes_semantic_ids(
    figure: Figure,
    panel_ids: list[str] | None = None,
    artist_mappings: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Assign stable structural IDs and return an artist registry."""
    panel_ids = panel_ids or [
        chr(ord("A") + index) for index in range(len(figure.axes))
    ]
    if len(panel_ids) != len(figure.axes):
        raise ValueError("Panel ID count must match the number of axes.")
    figure.canvas.draw()
    figure.set_gid("figure")
    registry: dict[str, dict[str, Any]] = {
        "figure": {
            "role": "figure",
            "artist_type": type(figure).__name__,
            "required_in_svg": "true",
        }
    }
    for axis, panel_id in zip(figure.axes, panel_ids):
        panel = _slug(panel_id)
        axis_id = f"axes__{panel}__main"
        axis.set_gid(axis_id)
        is_colorbar = "colorbar" in panel
        registry[axis_id] = {
            "role": "axes",
            "panel": panel,
            "required_in_svg": "true",
        }
        for dimension, matplotlib_axis in (("x", axis.xaxis), ("y", axis.yaxis)):
            axis_component_id = f"axis__{panel}__{dimension}"
            matplotlib_axis.set_gid(axis_component_id)
            registry[axis_component_id] = {
                "role": "axis",
                "panel": panel,
                "dimension": dimension,
                "required_in_svg": "true",
            }
            label = matplotlib_axis.label
            label_id = f"axis-title__{panel}__{dimension}"
            label.set_gid(label_id)
            registry[label_id] = {
                "role": "axis-title",
                "panel": panel,
                "dimension": dimension,
                "required_in_svg": str(
                    bool(label.get_visible() and label.get_text())
                ).lower(),
            }
            for index, tick in enumerate(matplotlib_axis.get_major_ticks()):
                lower, upper = matplotlib_axis.get_view_interval()
                location = tick.get_loc()
                if not (min(lower, upper) <= location <= max(lower, upper)):
                    continue
                tick_line = (
                    tick.tick1line
                    if tick.tick1line.get_visible()
                    else tick.tick2line
                )
                tick_label = (
                    tick.label1
                    if tick.label1.get_visible() and tick.label1.get_text()
                    else tick.label2
                )
                if not tick_line.get_visible():
                    continue
                tick_id = f"tick__{panel}__{dimension}__{index:03d}"
                tick_line.set_gid(tick_id)
                registry[tick_id] = {
                    "role": "tick",
                    "panel": panel,
                    "dimension": dimension,
                    "required_in_svg": "true",
                }
                tick_label_id = f"tick-label__{panel}__{dimension}__{index:03d}"
                if tick_label.get_visible() and tick_label.get_text():
                    tick_label.set_gid(tick_label_id)
                    registry[tick_label_id] = {
                        "role": "tick-label",
                        "panel": panel,
                        "dimension": dimension,
                        "required_in_svg": "true",
                    }
        for name, spine in axis.spines.items():
            spine_id = f"spine__{panel}__{_slug(name)}"
            spine.set_gid(spine_id)
            registry[spine_id] = {
                "role": "spine",
                "panel": panel,
                "side": name,
                "required_in_svg": str(
                    bool(
                        spine.get_visible()
                        and spine.get_linewidth() > 0
                        and not is_colorbar
                    )
                ).lower(),
            }
        legend = axis.get_legend()
        if legend is not None:
            legend_id = f"legend__{panel}__main"
            legend.set_gid(legend_id)
            registry[legend_id] = {
                "role": "legend",
                "panel": panel,
                "required_in_svg": "true",
            }
    for artist in figure.findobj():
        artist_id = artist.get_gid()
        if not artist_id or artist_id in registry:
            continue
        registry[artist_id] = {
            "role": artist_id.split("__", 1)[0],
            "artist_type": type(artist).__name__,
            "required_in_svg": "true",
        }
    for artist_id, mapping in (artist_mappings or {}).items():
        registry_id = artist_id
        if artist_id == "colorbar" and "axes__colorbar__main" in registry:
            registry_id = "axes__colorbar__main"
        if registry_id not in registry:
            raise ValueError(
                f"Artist mapping references an unknown semantic ID: {artist_id}"
            )
        registry[registry_id].update(mapping)
    return registry


def _git_commit(source_file: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_file.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _git_dirty(source_file: Path) -> bool:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=source_file.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode != 0 or bool(completed.stdout.strip())


def _embed_svg_metadata(
    svg_path: Path,
    compact_provenance: dict[str, Any],
) -> None:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    metadata = root.find(f"{{{SVG_NAMESPACE}}}metadata")
    if metadata is None:
        metadata = ET.Element(f"{{{SVG_NAMESPACE}}}metadata")
        root.insert(0, metadata)
    node = ET.SubElement(
        metadata,
        f"{{{PROVENANCE_NAMESPACE}}}analysis-provenance",
        {"id": "classical-conditioning-provenance"},
    )
    node.text = json.dumps(compact_provenance, separators=(",", ":"))
    tree.write(svg_path, encoding="utf-8", xml_declaration=True)


def _validate_svg_registry(
    svg_path: Path,
    registry: dict[str, dict[str, Any]],
) -> None:
    root = ET.parse(svg_path).getroot()
    ids = [
        element.attrib["id"]
        for element in root.iter()
        if "id" in element.attrib
    ]
    if len(ids) != len(set(ids)):
        raise ValueError("Semantic SVG contains duplicate element IDs.")
    required_ids = {
        artist_id
        for artist_id, metadata in registry.items()
        if metadata.get("required_in_svg") == "true"
    }
    missing = sorted(required_ids.difference(ids))
    if missing:
        raise ValueError(f"Semantic SVG is missing registered artist IDs: {missing}")
    provenance = root.find(
        f".//{{{PROVENANCE_NAMESPACE}}}analysis-provenance"
    )
    if provenance is None or not provenance.text:
        raise ValueError("Semantic SVG lacks embedded analysis provenance.")


def export_matplotlib_figure(
    figure: Figure,
    output_base: Path,
    provenance: FigureProvenance,
    *,
    mode: FigureMode,
    panel_ids: list[str] | None = None,
    overwrite: bool = False,
    allow_dirty_publication: bool = False,
) -> FigureExportResult:
    """Export one Matplotlib figure in publication or static mode."""
    if mode == FigureMode.INTERACTIVE:
        raise ValueError("Interactive HTML requires a native interactive builder.")
    output_base = output_base.resolve()
    extensions = ("svg", "pdf") if mode == FigureMode.PUBLICATION else ("png",)
    final_outputs = tuple(
        output_base.with_suffix(f".{extension}") for extension in extensions
    )
    sidecar = output_base.with_suffix(".figure.json")
    existing = [path for path in (*final_outputs, sidecar) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Figure outputs already exist: {existing}")

    registry = assign_axes_semantic_ids(
        figure,
        panel_ids,
        provenance.artist_mappings,
    )
    source_file = Path(provenance.source_file).resolve()
    dirty = _git_dirty(source_file)
    if (
        mode == FigureMode.PUBLICATION
        and dirty
        and not allow_dirty_publication
    ):
        raise RuntimeError(
            "Publication figures require a clean Git worktree so the recorded "
            "commit contains all rendering code."
        )
    export_source = Path(__file__).resolve()
    sidecar_payload = {
        **asdict(provenance),
        "source_commit": _git_commit(source_file),
        "source_worktree_dirty": dirty,
        "code_dependencies": [
            {"path": str(source_file), "sha256": sha256_file(source_file)},
            {"path": str(export_source), "sha256": sha256_file(export_source)},
        ],
        "artist_registry": registry,
        "mode": mode.value,
    }

    output_base.parent.mkdir(parents=True, exist_ok=True)
    with artifact_staging(
        output_base.parent,
        prefix=f".{output_base.name}-figure-",
    ) as staging_root:
        staged_outputs: list[Path] = []
        output_records: list[dict[str, Any]] = []
        for extension in extensions:
            staged = staging_root / f"{output_base.name}.{extension}"
            if extension == "pdf":
                figure.savefig(
                    staged,
                    format="pdf",
                    metadata={
                        "Title": provenance.figure_id,
                        "Subject": provenance.reproduction_snippet,
                    },
                )
                output_records.append(
                    {
                        "path": str(
                            output_base.with_suffix(f".{extension}")
                        ),
                        "sha256": sha256_file(staged),
                    }
                )
            elif extension == "svg":
                output_records.append(
                    {
                        "path": str(output_base.with_suffix(".svg")),
                        "sha256": None,
                        "integrity": "SVG embeds the authoritative sidecar hash.",
                    }
                )
            else:
                figure.savefig(staged, format="png", dpi=300)
                output_records.append(
                    {
                        "path": str(
                            output_base.with_suffix(f".{extension}")
                        ),
                        "sha256": sha256_file(staged),
                    }
                )
            staged_outputs.append(staged)
        sidecar_payload["outputs"] = output_records
        staged_sidecar = staging_root / sidecar.name
        write_json_atomic(staged_sidecar, sidecar_payload)
        sidecar_hash = sha256_file(staged_sidecar)
        compact = {
            "figure_id": provenance.figure_id,
            "figure_spec_version": provenance.figure_spec_version,
            "analysis_recipe": provenance.analysis_recipe,
            "source_commit": sidecar_payload["source_commit"],
            "source_file": provenance.source_file,
            "source_symbol": provenance.source_symbol,
            "source_hash": provenance.source_hash,
            "reproduction_snippet": provenance.reproduction_snippet,
            "input_artifacts": provenance.input_artifacts,
            "cohort_hash": provenance.cohort_hash,
            "sidecar_sha256": sidecar_hash,
        }
        if mode == FigureMode.PUBLICATION:
            svg_index = extensions.index("svg")
            staged_svg = staged_outputs[svg_index]
            figure.savefig(
                staged_svg,
                format="svg",
                metadata={
                    "Title": provenance.figure_id,
                    "Description": json.dumps(compact, separators=(",", ":")),
                },
            )
            _embed_svg_metadata(staged_svg, compact)
            _validate_svg_registry(staged_svg, registry)
        publish_transaction(
            tuple(zip((*staged_outputs, staged_sidecar), (*final_outputs, sidecar))),
            staging_root,
            overwrite=overwrite,
        )
    return FigureExportResult(outputs=final_outputs, sidecar=sidecar)
