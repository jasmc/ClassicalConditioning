"""
Standalone test script: Discarded Fish Heatmap Grid
====================================================

Loads the discarded-fish IDs and discard reasons, finds their pre-rendered
scaled-vigor heatmap images, and assembles them into a grid figure with each
subplot annotated with the discard reason.

Usage:
    python test_discarded_heatmap_grid.py
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import figure_saving
import file_utils
import plotting_style
from experiment_configuration import ExperimentType, get_experiment_config

plotting_style.set_plot_style()

# ==============================================================================
# Parameters
# ==============================================================================
EXPERIMENT = ExperimentType.ALL_DELAY.value
FIG_DPI: int = 200
BORDER_COLOR_DISCARDED = "#CC3333"
BORDER_COLOR_INCLUDED = "#336699"


# ==============================================================================
# Resolve paths
# ==============================================================================
config = get_experiment_config(EXPERIMENT)

(
    _,
    _,
    _,
    path_processed_data,
    _,
    _,
    _,
    _,
    _,
    path_scaled_vigor_fig_cs,
    _,
    _,
    _,
    path_pooled_vigor_fig,
    _,
    path_orig_pkl,
    _,
    _,
) = file_utils.create_folders(config.path_save)


# ==============================================================================
# Load discarded fish IDs + reasons
# ==============================================================================
def load_discard_reasons(processed_data_dir: Path) -> dict[str, str]:
    """Parse ``Fish to discard.txt`` into a {fish_id: reason} mapping.

    File format written by 1_Preprocessing…::

        fish_name  reason text here          (two-space separator)

    Header / comment / indented lines are skipped.
    """
    reasons_file = processed_data_dir / "Fish to discard.txt"
    if not reasons_file.exists():
        print(f"  [WARN] Reasons file not found: {reasons_file}")
        return {}

    reasons: dict[str, str] = {}
    for raw_line in reasons_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if raw_line.startswith(" ") or raw_line.startswith("\t"):
            continue
        if line.lower().startswith("analysis parameters"):
            continue
        if line.lower().startswith("missing excluded"):
            continue

        parts = re.split(r"\s{2,}", line, maxsplit=1)
        if len(parts) == 2:
            reasons[parts[0].strip()] = parts[1].strip()
        elif len(parts) == 1:
            reasons[parts[0].strip()] = ""
    return reasons


discard_file = path_orig_pkl / "Excluded" / "excluded_fish_ids.txt"
fish_ids_to_discard: list[str] = []
discard_reasons: dict[str, str] = {}

if discard_file.exists():
    fish_ids_to_discard = file_utils.load_discarded_fish_ids(discard_file)
    discard_reasons = load_discard_reasons(path_processed_data)
    print(f"Loaded {len(fish_ids_to_discard)} discarded fish IDs from: {discard_file}")
    if discard_reasons:
        print(f"Loaded discard reasons for {len(discard_reasons)} fish.")
else:
    print(f"[WARN] Discard file not found: {discard_file}")


# ==============================================================================
# Heatmap grid builder
# ==============================================================================
def _crop_border(img: np.ndarray, frac: float = 0.02) -> np.ndarray:
    """Trim a thin border from an image array."""
    if img is None or not hasattr(img, "shape") or len(img.shape) < 2:
        return img
    h, w = int(img.shape[0]), int(img.shape[1])
    dy, dx = int(round(h * frac)), int(round(w * frac))
    if (h - 2 * dy) < 2 or (w - 2 * dx) < 2:
        return img
    return img[dy : h - dy, dx : w - dx, ...]


def _load_image(img_path: Path) -> np.ndarray | None:
    """Load a PNG or SVG image into a numpy array."""
    try:
        if img_path.suffix.lower() == ".svg":
            try:
                import importlib

                cairosvg = importlib.import_module("cairosvg")
                png_data = cairosvg.svg2png(url=str(img_path))
                img = Image.open(io.BytesIO(png_data))
                return _crop_border(np.array(img), frac=0.02)
            except ImportError:
                try:
                    import importlib

                    renderPM = importlib.import_module("reportlab.graphics.renderPM")
                    svglib = importlib.import_module("svglib.svglib")
                    drawing = svglib.svg2rlg(str(img_path))
                    png_data = renderPM.drawToString(drawing, fmt="PNG")
                    img = Image.open(io.BytesIO(png_data))
                    return _crop_border(np.array(img), frac=0.02)
                except ImportError:
                    print(f"    [WARN] Cannot load SVG (install cairosvg or svglib): {img_path.name}")
                    return None
        img = Image.open(img_path)
        return _crop_border(np.array(img), frac=0.02)
    except Exception as e:
        print(f"    [WARN] Failed to load image {img_path.name}: {e}")
        return None


def build_heatmap_lookup(heatmap_dir: Path) -> dict[str, Path]:
    """Map lowercased fish IDs (``day_fishno``) to their heatmap file path."""
    heatmap_files = list(heatmap_dir.glob("*.svg")) + list(heatmap_dir.glob("*.png"))
    fish_to_heatmap: dict[str, Path] = {}
    for fpath in heatmap_files:
        parts = fpath.stem.split("_")
        if len(parts) >= 2:
            fid = "_".join(parts[:2]).lower()
            # Prefer PNG over SVG when both exist
            if fid not in fish_to_heatmap or fpath.suffix.lower() == ".png":
                fish_to_heatmap[fid] = fpath
    return fish_to_heatmap


def save_heatmap_grid(
    fish_ids: list[str],
    fish_to_heatmap: dict[str, Path],
    output_dir: Path,
    title: str = "Scaled Vigor Heatmaps",
    filename: str = "Heatmap_Grid.png",
    border_color: str = "#333333",
    discard_reasons: dict[str, str] | None = None,
) -> None:
    """Assemble a grid of per-fish heatmap images.

    Parameters
    ----------
    fish_ids : list of fish IDs to include.
    fish_to_heatmap : lookup mapping lowercased fish ID -> image path.
    output_dir : directory where the grid PNG will be saved.
    title : figure suptitle.
    filename : output file name.
    border_color : colour for subplot borders and title text.
    discard_reasons : optional {fish_id: reason} dict.  When provided a
        semi-transparent text box with the reason is drawn in the top-right
        corner of each subplot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    n_fish = len(fish_ids)
    if n_fish == 0:
        print(f"  [WARN] No fish to plot for: {title}")
        return

    n_cols = 6
    n_rows = max(1, (n_fish + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows + 1.5),
        facecolor="white",
        squeeze=False,
    )
    fig.suptitle(title, fontsize=16, y=0.99)

    missing_fish: list[str] = []
    loaded_count = 0

    for plot_idx, fish in enumerate(fish_ids):
        row, col = divmod(plot_idx, n_cols)
        ax = axes[row, col]
        fish_lower = fish.lower()

        heatmap_path = fish_to_heatmap.get(fish_lower)
        if heatmap_path is not None:
            img_array = _load_image(heatmap_path)
            if img_array is not None:
                ax.imshow(img_array)
                ax.set_aspect("equal", adjustable="box")
                ax.axis("off")
                loaded_count += 1
            else:
                ax.text(
                    0.5, 0.5, f"Load Error\n{fish}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=13, color="red",
                )
                ax.axis("off")
        else:
            missing_fish.append(fish)
            ax.text(
                0.5, 0.5, f"No heatmap\n{fish}",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color="gray",
            )
            ax.axis("off")

        ax.set_title(fish, fontsize=14, color=border_color, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(border_color)
            spine.set_linewidth(1.5)

        # Show discard reason as a text box in the top-right corner
        if discard_reasons is not None:
            reason = discard_reasons.get(fish, discard_reasons.get(fish_lower, ""))
            if reason:
                ax.text(
                    0.98, 0.96, reason,
                    transform=ax.transAxes,
                    fontsize=7,
                    verticalalignment="top",
                    horizontalalignment="right",
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        facecolor="white",
                        alpha=0.75,
                        edgecolor=border_color,
                    ),
                    color="#333333",
                    wrap=True,
                )

    # Hide unused axes
    for plot_idx in range(n_fish, n_rows * n_cols):
        row, col = divmod(plot_idx, n_cols)
        axes[row, col].set_visible(False)

    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    plt.subplots_adjust(wspace=0.08, hspace=0.14)

    grid_path = output_dir / filename
    figure_saving.save_figure(fig, grid_path, frmt="png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved heatmap grid: {grid_path.name}")
    print(f"    Loaded: {loaded_count}/{n_fish} heatmaps")
    if missing_fish:
        print(f"    Missing heatmaps for: {missing_fish[:10]}{'...' if len(missing_fish) > 10 else ''}")


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    if not fish_ids_to_discard:
        print("No discarded fish IDs found – nothing to plot.")
        return

    fish_to_heatmap = build_heatmap_lookup(path_scaled_vigor_fig_cs)
    print(f"Found {len(fish_to_heatmap)} heatmap files in: {path_scaled_vigor_fig_cs}")

    output_dir = path_pooled_vigor_fig / "Fish grouping heatmaps"

    # --- Discarded fish grid ---
    print(f"\n--- Discarded fish ({len(fish_ids_to_discard)}) ---")
    save_heatmap_grid(
        sorted(fish_ids_to_discard),
        fish_to_heatmap,
        output_dir,
        title=f"Discarded Fish – Scaled Vigor Heatmaps (aligned to CS)",
        filename="Heatmap_Grid_Discarded.png",
        border_color=BORDER_COLOR_DISCARDED,
        discard_reasons=discard_reasons,
    )

    # --- Included fish grid (all heatmap files NOT in the discard list) ---
    discard_set_lower = {fid.lower() for fid in fish_ids_to_discard}
    included_ids = sorted(
        fid for fid in fish_to_heatmap if fid not in discard_set_lower
    )
    print(f"\n--- Included fish ({len(included_ids)}) ---")
    if included_ids:
        save_heatmap_grid(
            included_ids,
            fish_to_heatmap,
            output_dir,
            title=f"Included Fish – Scaled Vigor Heatmaps (aligned to CS)",
            filename="Heatmap_Grid_Included.png",
            border_color=BORDER_COLOR_INCLUDED,
        )
    else:
        print("  No included fish to plot.")

    print("\nDone.")


if __name__ == "__main__":
    main()
