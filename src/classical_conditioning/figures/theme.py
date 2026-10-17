"""Shared Matplotlib visual parameters with legacy defaults and opt-in profiles.

Renderers must opt in explicitly; this module does not restyle saved artwork or
certify final glyph geometry, resource embedding or scientific identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager

from classical_conditioning.config.domain import Alignment, ConditionSpec

if TYPE_CHECKING:
    from classical_conditioning.figures.presentation_profiles import PresentationProfile

# Physical layout constants and bundled font for cross-platform consistency.
MM_PER_INCH = 25.4
SINGLE_COLUMN_MM = 89.0
DOUBLE_COLUMN_MM = 183.0
DEFAULT_CS_DURATION_S = 10.0
DEFAULT_US_DURATION_S = 0.1
PREFERRED_SANS_SERIF = ("DejaVu Sans",)

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
    # Matplotlib expects inches; omitted height intentionally returns a square.
    width_in = width_mm / MM_PER_INCH
    if height_mm is None:
        return (width_in, width_in)
    return (width_in, height_mm / MM_PER_INCH)


def rgb_255_to_unit(color_rgb_255: tuple[int, int, int]) -> tuple[float, float, float]:
    """Convert 0–255 RGB triples with division by 255."""
    return tuple(channel / 255.0 for channel in color_rgb_255)


def condition_color(spec: ConditionSpec) -> tuple[float, float, float]:
    """Return a condition colour in Matplotlib unit RGB."""
    # Keep condition rendering tied to immutable experiment configuration.
    return rgb_255_to_unit(spec.color_rgb_255)


def resolve_sans_serif_fonts(
    preferred: tuple[str, ...] = PREFERRED_SANS_SERIF,
) -> tuple[str, ...]:
    """Resolve one declared face; never silently substitute an undeclared font."""
    available = {font.name for font in fontManager.ttflist}
    for family in preferred:
        if family in available:
            return (family,)
    raise RuntimeError(f"Declared figure font is unavailable: {', '.join(preferred)}")


def stimulus_duration_s(
    alignment: Alignment | str,
    *,
    cs_duration_s: float = DEFAULT_CS_DURATION_S,
    us_duration_s: float = DEFAULT_US_DURATION_S,
) -> float:
    """Return CS or US duration in seconds."""
    # Accept enum/string callers while constraining output to known CS/US timing.
    value = alignment.value if isinstance(alignment, Alignment) else str(alignment)
    if value == Alignment.CS.value:
        return float(cs_duration_s)
    if value == Alignment.US.value:
        return float(us_duration_s)
    raise ValueError(f"Unknown stimulus alignment: {alignment!r}")


# Immutable visual parameters: callers can override a complete theme explicitly.
@dataclass(frozen=True)
class FigureTheme:
    # Gather publication-facing typography, line, colour, and raster settings
    # so every figure recipe starts from the same visual contract.
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
    # All distances here are typographic points, independent of figure size/DPI.
    tick_major_pad: float = 3.0
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
    intensity_cmap: str = "managua_r"
    # March 24 single-fish, baseline-centred log-vigor figure (signed scale).
    single_fish_scaled_vigor_cmap: str = "managua_r"
    single_fish_scaled_vigor_vmin: float = -0.25
    single_fish_scaled_vigor_vmax: float = 0.25
    probability_cmap: str = "managua_r"
    stimulus_span_alpha: float = 0.18
    constrained_h_pad: float = 0.04
    constrained_w_pad: float = 0.04
    left_spine_offset: float = 0.0
    bottom_spine_offset: float = 0.0
    # Explicit profiles may choose another declared face; legacy output stays bundled.
    font_sans_serif: tuple[str, ...] = PREFERRED_SANS_SERIF
    font_style: str | None = None
    font_weight: str | None = None
    text_color: str | None = None
    tick_color: str | None = None
    spine_color: str | None = None
    tick_capstyle: str | None = None
    cs_linewidth: float | None = None
    us_linewidth: float | None = None
    stimulus_span_zorder: float = 3
    axes_titlelocation: str = "left"


# Default theme is a reusable immutable value, not mutable global plotting state.
DEFAULT_THEME = FigureTheme()


# Share one immutable default theme across figure builders.
def get_theme(*, profile: str | PresentationProfile | None = None) -> FigureTheme:
    """Return legacy defaults or adapt one named/already-resolved policy snapshot."""
    if profile is None:
        return DEFAULT_THEME
    from classical_conditioning.figures.presentation_profiles import (
        PresentationProfile,
        load_presentation_profile,
    )
    if isinstance(profile, str):
        profile = load_presentation_profile(profile)
    elif not isinstance(profile, PresentationProfile):
        raise TypeError("profile must be a name or resolved PresentationProfile")
    return FigureTheme(**profile.theme_parameters())


def apply_theme(
    theme: FigureTheme | None = None,
    *,
    profile: str | PresentationProfile | None = None,
) -> FigureTheme:
    """Apply one explicit theme/profile; callers can isolate it with mpl.rc_context."""
    if theme is not None and profile is not None:
        raise ValueError("Pass either theme or profile, not both")
    # Resolve optional override and installed fonts before updating global rcParams.
    theme = theme if theme is not None else get_theme(profile=profile)
    sans = list(resolve_sans_serif_fonts(theme.font_sans_serif))
    # Apply typography, axes, layout, export, and no-LaTeX portability settings together.
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": sans,
            **({"font.style": theme.font_style} if theme.font_style is not None else {}),
            **({"font.weight": theme.font_weight} if theme.font_weight is not None else {}),
            **({"text.color": theme.text_color} if theme.text_color is not None else {}),
            **({key: theme.tick_color for key in (
                "xtick.color", "ytick.color", "xtick.labelcolor", "ytick.labelcolor",
            )} if theme.tick_color is not None else {}),
            "font.size": theme.font_size,
            "axes.labelsize": theme.axes_labelsize,
            "axes.titlesize": theme.axes_titlesize,
            "axes.titleweight": "normal",
            "axes.labelweight": "normal",
            "axes.titlelocation": theme.axes_titlelocation,
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
            "xtick.minor.pad": theme.tick_major_pad,
            "ytick.minor.pad": theme.tick_major_pad,
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
    show_xticklabels: bool | None = None,
    show_yticklabels: bool | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Style ordinary left/bottom axes, preserving data coordinates and tick values.

    Pass the build's resolved theme explicitly. Boxed/shared scientific families
    retain their own registered visibility rules; this is not their layout engine.
    """
    # Set visibility/spacing first, then label behaviour based on requested tick axes.
    theme = theme or DEFAULT_THEME
    # Resolve declared profile text before changing any existing artist.
    profile_text = theme.font_style is not None or theme.font_weight is not None
    family = resolve_sans_serif_fonts(theme.font_sans_serif) if profile_text else None
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(show_yticks)
    ax.spines["bottom"].set_visible(show_xticks)
    if show_yticks:
        ax.spines["left"].set_position(("outward", theme.left_spine_offset))
    if show_xticks:
        ax.spines["bottom"].set_position(("outward", theme.bottom_spine_offset))
    if theme.spine_color is not None:
        for spine in ax.spines.values():
            spine.set_color(theme.spine_color)
            spine.set_linewidth(theme.axes_linewidth)
    ax.tick_params(
        axis="both",
        which="major",
        bottom=show_xticks,
        labelbottom=show_xticks if show_xticklabels is None else show_xticklabels,
        left=show_yticks,
        labelleft=show_yticks if show_yticklabels is None else show_yticklabels,
        top=False,
        right=False,
        direction="out",
        length=theme.tick_major_size,
        width=theme.tick_major_width,
        pad=theme.tick_major_pad,
        labelsize=theme.tick_labelsize,
        **({"colors": theme.tick_color} if theme.tick_color is not None else {}),
    )
    ax.tick_params(
        axis="both", which="minor", bottom=False, left=False,
        top=False, right=False,
    )
    ax.xaxis.label.set_fontsize(theme.axes_labelsize)
    ax.yaxis.label.set_fontsize(theme.axes_labelsize)
    ax.xaxis.labelpad = theme.axes_labelpad
    ax.yaxis.labelpad = theme.axes_labelpad
    text_artists = [(ax.xaxis.label, theme.text_color), (ax.yaxis.label, theme.text_color)]
    for axis in (ax.xaxis, ax.yaxis):
        for tick in axis.get_major_ticks():
            label_color = theme.tick_color if theme.tick_color is not None else theme.text_color
            text_artists.extend(((tick.label1, label_color), (tick.label2, label_color)))
            if theme.tick_capstyle is not None:
                for line in (tick.tick1line, tick.tick2line):
                    line.set_solid_capstyle(theme.tick_capstyle)
                    line.set_dash_capstyle(theme.tick_capstyle)
    for text, color in text_artists:
        if family is not None:
            text.set_fontfamily(family)
        if theme.font_style is not None:
            text.set_fontstyle(theme.font_style)
        if theme.font_weight is not None:
            text.set_fontweight(theme.font_weight)
        if color is not None:
            text.set_color(color)
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
    # Resolve timing/colour from alignment unless an assay-specific duration is supplied.
    theme = theme or DEFAULT_THEME
    value = alignment.value if isinstance(alignment, Alignment) else str(alignment)
    if duration_s is None:
        duration_s = stimulus_duration_s(value)
    color = theme.cs_color if value == Alignment.CS.value else theme.us_color
    # Return both artists so figure builders can register semantic provenance IDs.
    span = ax.axvspan(
        onset_s,
        onset_s + duration_s,
        facecolor=color,
        edgecolor="none",
        alpha=theme.stimulus_span_alpha,
        zorder=theme.stimulus_span_zorder,
        clip_on=True,
    )
    onset_width = theme.cs_linewidth if value == Alignment.CS.value else theme.us_linewidth
    onset = ax.axvline(
        onset_s,
        color=color,
        linewidth=theme.lines_linewidth if onset_width is None else onset_width,
        zorder=4,
        clip_on=True,
    )
    if gid is not None:
        span.set_gid(gid)
    return span, onset


# Cycle through the theme palette for arbitrary category counts.
def qualitative_color(index: int, theme: FigureTheme | None = None) -> str:
    theme = theme or DEFAULT_THEME
    return theme.qualitative_colors[index % len(theme.qualitative_colors)]


def heatmap_cmap(name: str, theme: FigureTheme | None = None):
    """Return a colormap with a neutral colour for missing values."""
    # Copy before changing the bad colour so registered colormaps stay untouched.
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
    # Apply shared theme and convert physical row layout into a Matplotlib figure.
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
