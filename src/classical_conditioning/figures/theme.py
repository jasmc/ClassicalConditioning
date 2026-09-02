"""Immutable visual language for scientific Matplotlib figures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager

from classical_conditioning.config.domain import Alignment, ConditionSpec

MM_PER_INCH = 25.4
SINGLE_COLUMN_MM = 89.0
DOUBLE_COLUMN_MM = 183.0
DEFAULT_CS_DURATION_S = 10.0
DEFAULT_US_DURATION_S = 0.1
PREFERRED_SANS_SERIF = ("Arial", "Helvetica", "DejaVu Sans")

# Okabe–Ito, skipping black so traces can stay black.
COLORBLIND_QUALITATIVE = (
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#F0E442",
)


def mm_to_in(width_mm: float, height_mm: float | None = None) -> tuple[float, float]:
    """Convert millimetre figure size to inches."""
    width_in = width_mm / MM_PER_INCH
    if height_mm is None:
        return (width_in, width_in)
    return (width_in, height_mm / MM_PER_INCH)


def rgb_255_to_unit(color_rgb_255: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert 0–255 RGB triples with division by 255."""
    return tuple(channel / 255.0 for channel in color_rgb_255)


def condition_color(spec: ConditionSpec) -> tuple[float, float, float]:
    """Return a condition colour in Matplotlib unit RGB."""
    return rgb_255_to_unit(spec.color_rgb_255)


def resolve_sans_serif_fonts() -> tuple[str, ...]:
    """Return installed sans-serif names in preference order, with a fallback."""
    available = {font.name for font in fontManager.ttflist}
    resolved = [name for name in PREFERRED_SANS_SERIF if name in available]
    if "DejaVu Sans" not in resolved:
        resolved.append("DejaVu Sans")
    return tuple(resolved)


def stimulus_duration_s(
    alignment: Alignment | str,
    *,
    cs_duration_s: float = DEFAULT_CS_DURATION_S,
    us_duration_s: float = DEFAULT_US_DURATION_S,
) -> float:
    """Return CS or US duration in seconds."""
    value = alignment.value if isinstance(alignment, Alignment) else str(alignment)
    if value == Alignment.CS.value:
        return float(cs_duration_s)
    if value == Alignment.US.value:
        return float(us_duration_s)
    raise ValueError(f"Unknown stimulus alignment: {alignment!r}")


@dataclass(frozen=True)
class FigureTheme:
    font_size: float = 8.0
    axes_labelsize: float = 8.0
    axes_titlesize: float = 8.0
    tick_labelsize: float = 7.0
    legend_fontsize: float = 7.0
    figure_titlesize: float = 9.0
    axes_linewidth: float = 0.5
    lines_linewidth: float = 0.5
    tick_major_size: float = 2.0
    tick_major_width: float = 0.5
    tick_major_pad: float = 2.0
    axes_labelpad: float = 3.0
    axes_titlepad: float = 4.0
    figure_dpi: int = 300
    savefig_dpi: int = 600
    cs_color: tuple[float, float, float] = rgb_255_to_unit((13, 129, 54))
    us_color: tuple[float, float, float] = rgb_255_to_unit((112, 46, 120))
    baseline_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    missing_color: tuple[float, float, float] = (0.82, 0.82, 0.82)
    heatmap_bad_color: tuple[float, float, float] = (0.75, 0.75, 0.75)
    single_series_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    qualitative_colors: tuple[str, ...] = COLORBLIND_QUALITATIVE
    intensity_cmap: str = "magma"
    probability_cmap: str = "cividis"
    stimulus_span_alpha: float = 0.18
    constrained_h_pad: float = 0.04
    constrained_w_pad: float = 0.04
    left_spine_offset: float = 0.0
    bottom_spine_offset: float = 4.0


DEFAULT_THEME = FigureTheme()


def get_theme() -> FigureTheme:
    return DEFAULT_THEME


def apply_theme(theme: FigureTheme | None = None) -> FigureTheme:
    """Apply the scientific theme to Matplotlib rcParams."""
    theme = theme or DEFAULT_THEME
    sans = list(resolve_sans_serif_fonts())
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": sans,
            "font.size": theme.font_size,
            "axes.labelsize": theme.axes_labelsize,
            "axes.titlesize": theme.axes_titlesize,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.titlelocation": "left",
            "axes.titlepad": theme.axes_titlepad,
            "axes.labelpad": theme.axes_labelpad,
            "axes.linewidth": theme.axes_linewidth,
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelcolor": "black",
            "xtick.labelsize": theme.tick_labelsize,
            "ytick.labelsize": theme.tick_labelsize,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": theme.tick_major_size,
            "ytick.major.size": theme.tick_major_size,
            "xtick.major.width": theme.tick_major_width,
            "ytick.major.width": theme.tick_major_width,
            "xtick.major.pad": theme.tick_major_pad,
            "ytick.major.pad": theme.tick_major_pad,
            "xtick.top": False,
            "ytick.right": False,
            "legend.fontsize": theme.legend_fontsize,
            "legend.frameon": False,
            "legend.borderaxespad": 0.2,
            "figure.titlesize": theme.figure_titlesize,
            "figure.dpi": theme.figure_dpi,
            "figure.facecolor": "white",
            "figure.constrained_layout.use": True,
            "figure.constrained_layout.h_pad": theme.constrained_h_pad,
            "figure.constrained_layout.w_pad": theme.constrained_w_pad,
            "savefig.dpi": theme.savefig_dpi,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "lines.linewidth": theme.lines_linewidth,
            "text.usetex": False,
        }
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=list(theme.qualitative_colors))
    return theme


def style_axes(
    ax: Axes,
    *,
    theme: FigureTheme | None = None,
    show_xticks: bool = True,
    show_yticks: bool = True,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Hide unused spines and ticks on one axes."""
    theme = theme or DEFAULT_THEME
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(show_yticks)
    ax.spines["bottom"].set_visible(show_xticks)
    if show_yticks:
        ax.spines["left"].set_position(("outward", theme.left_spine_offset))
    if show_xticks:
        ax.spines["bottom"].set_position(("outward", theme.bottom_spine_offset))
    ax.tick_params(
        axis="both",
        which="both",
        bottom=show_xticks,
        labelbottom=show_xticks,
        left=show_yticks,
        labelleft=show_yticks,
        top=False,
        right=False,
        direction="out",
    )
    if xlabel is not None:
        ax.set_xlabel(xlabel if show_xticks else "")
    elif not show_xticks:
        ax.set_xlabel("")
    if ylabel is not None:
        ax.set_ylabel(ylabel)


def add_stimulus_window(
    ax: Axes,
    alignment: Alignment | str,
    *,
    onset_s: float = 0.0,
    duration_s: float | None = None,
    theme: FigureTheme | None = None,
    gid: str | None = None,
) -> tuple[object, object]:
    """Mark stimulus duration in data coordinates (span plus onset line)."""
    theme = theme or DEFAULT_THEME
    value = alignment.value if isinstance(alignment, Alignment) else str(alignment)
    if duration_s is None:
        duration_s = stimulus_duration_s(value)
    color = theme.cs_color if value == Alignment.CS.value else theme.us_color
    span = ax.axvspan(
        onset_s,
        onset_s + duration_s,
        facecolor=color,
        edgecolor="none",
        alpha=theme.stimulus_span_alpha,
        zorder=3,
        clip_on=True,
    )
    onset = ax.axvline(
        onset_s,
        color=color,
        linewidth=theme.lines_linewidth,
        zorder=4,
        clip_on=True,
    )
    if gid is not None:
        span.set_gid(gid)
    return span, onset


def qualitative_color(index: int, theme: FigureTheme | None = None) -> str:
    theme = theme or DEFAULT_THEME
    return theme.qualitative_colors[index % len(theme.qualitative_colors)]


def heatmap_cmap(name: str, theme: FigureTheme | None = None):
    """Return a colormap with a neutral colour for missing values."""
    theme = theme or DEFAULT_THEME
    try:
        cmap = mpl.colormaps[name].copy()
    except (KeyError, AttributeError, ValueError) as error:
        raise ValueError(f"Unknown scientific colormap: {name!r}") from error
    cmap.set_bad(theme.heatmap_bad_color)
    if not getattr(cmap, "name", None):
        cmap.name = name
    return cmap


def stacked_subplots(
    nrows: int,
    *,
    width_mm: float = DOUBLE_COLUMN_MM,
    row_height_mm: float = 28.0,
    sharex: bool = True,
    theme: FigureTheme | None = None,
) -> tuple[Figure, Sequence[Axes]]:
    """Create a vertical stack sized in millimetres."""
    theme = theme or DEFAULT_THEME
    apply_theme(theme)
    height_mm = max(row_height_mm * nrows, 40.0)
    figure, axes = plt.subplots(
        nrows,
        1,
        figsize=mm_to_in(width_mm, height_mm),
        sharex=sharex,
        constrained_layout=True,
    )
    if nrows == 1:
        axes = [axes]
    return figure, axes
