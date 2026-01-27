"""Backward-compatible facade pointing to :mod:`plotting_style_new`."""

__all__ = [
    "set_plot_style",
    "configure_axes_for_pipeline",
    "configure_trace_axes",
    "get_plot_config",
    "SAVEFIG_KW",
    "HEATMAP_FIGSIZE",
    "HEATMAP_GRIDSPACE",
    "HEATMAP_STYLE_KW",
    "HEATMAP_TICK_KW",
    "LINEPLOT_FIG_KW",
    "LINEPLOT_STYLE_KW",
    "LEGEND_KW",
]



"""Centralized matplotlib style configuration used across plots."""
from dataclasses import dataclass, field
from typing import List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

# ==============================================================================
# PLOTTING CONSTANTS
# ==============================================================================
pass # Removed constants, moved to PlotStyleConfig



@dataclass
class PlotStyleConfig:
    # Custom (for analysis_utils)
    fontsize_title: int = 12
    fontsize_axis_label: int = 11

    # Positioning
    pad_supylabel: Tuple[float, float] = (-5, 0)
    pad_supxlabel: Tuple[float, float] = (0, -5)
    pad_fig_title: Tuple[float, float] = (0, 5)
    pad_panel_label: Tuple[float, float] = (0, 5)

    # Text / Fonts
    font_family: str = 'sans-serif'
    font_size: float = 10
    axes_labelsize: float = 10
    axes_titlesize: float = 10
    xtick_labelsize: float = 9
    ytick_labelsize: float = 9
    legend_fontsize: float = 10
    axes_titleweight: str = 'normal'
    axes_labelweight: str = 'bold'

    # Lines
    lines_linewidth: float = 0.5
    lines_markersize: float = 5
    lines_markeredgewidth: float = 1.0

    # Patch
    patch_linewidth: float = 1.0

    # Boxplot
    boxplot_flier_markersize: float = 6
    boxplot_linewidth: float = 1.0

    # Axes
    axes_linewidth: float = 0.5
    axes_titlepad: float = 6.0
    axes_labelpad: float = 4.0
    axes_prop_cycle_colors: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])

    # Ticks
    xtick_major_size: float = 2
    xtick_minor_size: float = 2
    xtick_major_width: float = 0.5
    xtick_minor_width: float = 0.5
    xtick_major_pad: float = 2
    xtick_minor_pad: float = 2
    
    ytick_major_size: float = 2
    ytick_minor_size: float = 2
    ytick_major_width: float = 0.5
    ytick_minor_width: float = 0.25
    ytick_major_pad: float = 2
    ytick_minor_pad: float = 2

    # Figure
    figure_titlesize: float = 11
    figure_labelsize: float = 11
    figure_figsize: Tuple[float, float] = (5/2.54, 8/2.54)
    figure_dpi: int = 300
    
    # Savefig
    savefig_dpi: int = 600

_CURRENT_PLOT_CONFIG = PlotStyleConfig()

def get_plot_config() -> PlotStyleConfig:
    return _CURRENT_PLOT_CONFIG

# Import plotting config if needed, or just set defaults here
# from general_configuration import config



def configure_axes_for_pipeline(
    ax,
    *,
    x_ticks=None,
    y_ticks=None,
    xlim=None,
    ylim=None,
    show_xticks=True,
    show_yticks=True,
    hide_spines=("top", "right"),
    left_spine_offset=0,
    bottom_spine_offset=6,
):
    """
    Apply the pipeline's shared axis formatting (ticks, spines, and limits).

    The helper keeps tick direction/dimensions consistent and allows callers
    to override whether specific spines or axis ticks should be visible.
    """
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if show_xticks:
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
    else:
        ax.set_xticks([])

    if show_yticks:
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
    else:
        ax.set_yticks([])

    ax.tick_params(
        axis="both",
        which="both",
        bottom=show_xticks,
        top=False,
        right=False,
        left=show_yticks,
        direction="out",
    )

    def _set_spine(name, offset, visible):
        spine = ax.spines.get(name)
        if spine is None:
            return
        spine.set_visible(visible and (name not in hide_spines))
        if visible and name not in hide_spines:
            spine.set_position(("outward", offset))

    _set_spine("left", left_spine_offset, show_yticks)
    _set_spine("bottom", bottom_spine_offset, show_xticks)

    for spine_name in hide_spines:
        spine = ax.spines.get(spine_name)
        if spine is not None:
            spine.set_visible(False)



def configure_trace_axes(
    ax,
    *,
    x_ticks=None,
    y_ticks=None,
    xlim=None,
    ylim=None,
    show_xticks=True,
    show_yticks=True,
    hide_spines=("top", "right"),
    left_spine_offset=0,
    bottom_spine_offset=6,
    x_spine_bounds=None,
    y_spine_bounds=None,
):
    """
    Apply trace-style axis formatting, including optional spine bounds.
    """
    configure_axes_for_pipeline(
        ax,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        xlim=xlim,
        ylim=ylim,
        show_xticks=show_xticks,
        show_yticks=show_yticks,
        hide_spines=hide_spines,
        left_spine_offset=left_spine_offset,
        bottom_spine_offset=bottom_spine_offset,
    )

    def _bounds_from_ticks(ticks):
        if ticks is None:
            return None
        try:
            return (min(ticks), max(ticks))
        except TypeError:
            return None

    if show_yticks:
        if y_spine_bounds is None:
            y_spine_bounds = _bounds_from_ticks(y_ticks)
        if y_spine_bounds is not None:
            spine = ax.spines.get("left")
            if spine is not None:
                spine.set_bounds(y_spine_bounds[0], y_spine_bounds[1])

    if show_xticks:
        if x_spine_bounds is None:
            x_spine_bounds = _bounds_from_ticks(x_ticks)
        if x_spine_bounds is not None:
            spine = ax.spines.get("bottom")
            if spine is not None:
                spine.set_bounds(x_spine_bounds[0], x_spine_bounds[1])


def set_plot_style(
    context="paper",
    style="ticks",
    rc_overrides=None,
    use_constrained_layout=True,
    label_pad_pts=None,
    tick_label_pad_pts=None,
):
    """
    Configures matplotlib to produce high-quality scientific plots.
    This function sets default parameters for font size, line width, colors, etc.

    Args:
        context: Seaborn context (e.g., "paper", "talk").
        style: Seaborn style (e.g., "ticks", "whitegrid").
        rc_overrides: Optional dict of matplotlib rcParams to override defaults.
        use_constrained_layout: Toggle matplotlib constrained layout globally.
    """
    # rcParams changes are global for the current Python process.
    config = get_plot_config()

    sns.set_context(context)
    sns.set_style(style)

    # Enable LaTeX rendering for text in the figure
    plt.rcParams['text.usetex'] = False
    # Set SVG font type to 'none' to export text as text
    plt.rcParams['svg.fonttype'] = 'none'

    # General font settings
    plt.rcParams['font.family'] = config.font_family
    plt.rcParams['font.size'] = config.font_size
    plt.rcParams['axes.labelsize'] = config.axes_labelsize
    plt.rcParams['axes.titlesize'] = config.axes_titlesize
    plt.rcParams['xtick.labelsize'] = config.xtick_labelsize
    plt.rcParams['ytick.labelsize'] = config.ytick_labelsize
    plt.rcParams['legend.fontsize'] = config.legend_fontsize
    plt.rcParams['axes.titleweight'] = config.axes_titleweight
    plt.rcParams['axes.labelweight'] = config.axes_labelweight

    # Line settings
    plt.rcParams['lines.linewidth'] = config.lines_linewidth
    plt.rcParams['lines.linestyle'] = '-'
    plt.rcParams['lines.color'] = 'C0'
    plt.rcParams['lines.marker'] = 'None'
    plt.rcParams['lines.markerfacecolor'] = 'black'
    plt.rcParams['lines.markeredgecolor'] = 'black'
    plt.rcParams['lines.markeredgewidth'] = config.lines_markeredgewidth
    plt.rcParams['lines.markersize'] = config.lines_markersize
    plt.rcParams['lines.dash_joinstyle'] = 'round'
    plt.rcParams['lines.dash_capstyle'] = 'butt'
    plt.rcParams['lines.solid_joinstyle'] = 'round'
    plt.rcParams['lines.solid_capstyle'] = 'projecting'
    plt.rcParams['lines.antialiased'] = True

    # Patch settings
    plt.rcParams['patch.linewidth'] = config.patch_linewidth
    plt.rcParams['patch.facecolor'] = 'none'
    plt.rcParams['patch.edgecolor'] = 'black'
    plt.rcParams['patch.force_edgecolor'] = False
    plt.rcParams['patch.antialiased'] = True

    # Hatch settings
    plt.rcParams['hatch.color'] = 'black'
    plt.rcParams['hatch.linewidth'] = 1.0

    # Boxplot settings
    plt.rcParams['boxplot.notch'] = False
    plt.rcParams['boxplot.vertical'] = True
    plt.rcParams['boxplot.whiskers'] = 1.5
    plt.rcParams['boxplot.bootstrap'] = None
    plt.rcParams['boxplot.patchartist'] = False
    plt.rcParams['boxplot.showmeans'] = False
    plt.rcParams['boxplot.showcaps'] = True
    plt.rcParams['boxplot.showbox'] = True
    plt.rcParams['boxplot.showfliers'] = True
    plt.rcParams['boxplot.meanline'] = False

    plt.rcParams['boxplot.flierprops.color'] = 'black'
    plt.rcParams['boxplot.flierprops.marker'] = 'o'
    plt.rcParams['boxplot.flierprops.markerfacecolor'] = 'none'
    plt.rcParams['boxplot.flierprops.markeredgecolor'] = 'none'
    plt.rcParams['boxplot.flierprops.markeredgewidth'] = 1.0
    plt.rcParams['boxplot.flierprops.markersize'] = config.boxplot_flier_markersize
    plt.rcParams['boxplot.flierprops.linestyle'] = 'none'
    plt.rcParams['boxplot.flierprops.linewidth'] = 1.0

    plt.rcParams['boxplot.boxprops.color'] = 'none'
    plt.rcParams['boxplot.boxprops.linewidth'] = config.boxplot_linewidth
    plt.rcParams['boxplot.boxprops.linestyle'] = '-'

    plt.rcParams['boxplot.whiskerprops.color'] = 'black'
    plt.rcParams['boxplot.whiskerprops.linewidth'] = config.boxplot_linewidth
    plt.rcParams['boxplot.whiskerprops.linestyle'] = '-'

    plt.rcParams['boxplot.capprops.color'] = 'black'
    plt.rcParams['boxplot.capprops.linewidth'] = config.boxplot_linewidth
    plt.rcParams['boxplot.capprops.linestyle'] = '-'

    plt.rcParams['boxplot.medianprops.color'] = 'black'
    plt.rcParams['boxplot.medianprops.linewidth'] = config.boxplot_linewidth
    plt.rcParams['boxplot.medianprops.linestyle'] = '-'

    plt.rcParams['boxplot.meanprops.color'] = 'C2'
    plt.rcParams['boxplot.meanprops.marker'] = '^'
    plt.rcParams['boxplot.meanprops.markerfacecolor'] = 'C2'
    plt.rcParams['boxplot.meanprops.markeredgecolor'] = 'C2'
    plt.rcParams['boxplot.meanprops.markersize'] = 6
    plt.rcParams['boxplot.meanprops.linestyle'] = '--'
    plt.rcParams['boxplot.meanprops.linewidth'] = 1.0

    # Axes settings
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = 'black'
    plt.rcParams['axes.linewidth'] = config.axes_linewidth
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.grid.axis'] = 'both'
    plt.rcParams['axes.grid.which'] = 'major'
    plt.rcParams['axes.titlelocation'] = 'left'
    plt.rcParams['axes.titlecolor'] = 'black'
    plt.rcParams['axes.titlepad'] = config.axes_titlepad
    plt.rcParams['axes.labelpad'] = config.axes_labelpad
    plt.rcParams['axes.labelcolor'] = 'black'
    plt.rcParams['axes.axisbelow'] = 'line'
    plt.rcParams['axes.formatter.limits'] = (-4, 4)
    plt.rcParams['axes.formatter.use_locale'] = False
    plt.rcParams['axes.formatter.use_mathtext'] = False
    plt.rcParams['axes.formatter.min_exponent'] = 0
    plt.rcParams['axes.formatter.useoffset'] = True
    plt.rcParams['axes.formatter.offset_threshold'] = 4
    plt.rcParams['axes.spines.left'] = True
    plt.rcParams['axes.spines.bottom'] = True
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.unicode_minus'] = True
    plt.rcParams['axes.prop_cycle'] = cycler('color', config.axes_prop_cycle_colors)
    plt.rcParams['axes.xmargin'] = 0.05
    plt.rcParams['axes.ymargin'] = 0.05
    plt.rcParams['axes.zmargin'] = 0.05
    plt.rcParams['axes.autolimit_mode'] = 'data'

    # Ticks settings
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.major.size'] = config.xtick_major_size
    plt.rcParams['xtick.minor.size'] = config.xtick_minor_size
    plt.rcParams['xtick.major.width'] = config.xtick_major_width
    plt.rcParams['xtick.minor.width'] = config.xtick_minor_width
    plt.rcParams['xtick.major.pad'] = config.xtick_major_pad
    plt.rcParams['xtick.minor.pad'] = config.xtick_minor_pad
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['xtick.labelcolor'] = 'inherit'
    plt.rcParams['xtick.direction'] = 'out'
    plt.rcParams['xtick.minor.visible'] = False
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.major.bottom'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.right'] = False
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.labelright'] = False
    plt.rcParams['ytick.major.size'] = config.ytick_major_size
    plt.rcParams['ytick.minor.size'] = config.ytick_minor_size
    plt.rcParams['ytick.major.width'] = config.ytick_major_width
    plt.rcParams['ytick.minor.width'] = config.ytick_minor_width
    plt.rcParams['ytick.major.pad'] = config.ytick_major_pad
    plt.rcParams['ytick.minor.pad'] = config.ytick_minor_pad
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['ytick.labelcolor'] = 'inherit'
    plt.rcParams['ytick.direction'] = 'out'
    plt.rcParams['ytick.minor.visible'] = False
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.minor.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.alignment'] = 'center_baseline'

    # Legend settings
    plt.rcParams['legend.loc'] = 'upper right'
    plt.rcParams['legend.frameon'] = False
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.facecolor'] = 'inherit'
    plt.rcParams['legend.edgecolor'] = 'black'
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.shadow'] = False
    plt.rcParams['legend.numpoints'] = 1
    plt.rcParams['legend.scatterpoints'] = 1
    plt.rcParams['legend.markerscale'] = 1.0
    plt.rcParams['legend.labelcolor'] = 'black'
    plt.rcParams['legend.title_fontsize'] = 10
    plt.rcParams['legend.borderpad'] = 0.4
    plt.rcParams['legend.labelspacing'] = 0.5
    plt.rcParams['legend.handlelength'] = 2.0
    plt.rcParams['legend.handleheight'] = 0.7
    plt.rcParams['legend.handletextpad'] = 0.8
    plt.rcParams['legend.borderaxespad'] = 0.1
    plt.rcParams['legend.columnspacing'] = 2.0

    # Figure settings
    plt.rcParams['figure.titlesize'] = config.figure_titlesize
    plt.rcParams['figure.titleweight'] = 'normal'
    plt.rcParams['figure.labelsize'] = config.figure_labelsize
    plt.rcParams['figure.labelweight'] = 'normal'
    plt.rcParams['figure.figsize'] = config.figure_figsize
    plt.rcParams['figure.dpi'] = config.figure_dpi
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['figure.edgecolor'] = 'white'
    plt.rcParams['figure.frameon'] = False
    plt.rcParams['figure.subplot.left'] = 0.125
    plt.rcParams['figure.subplot.right'] = 0.9
    plt.rcParams['figure.subplot.bottom'] = 0.11
    plt.rcParams['figure.subplot.top'] = 0.88
    plt.rcParams['figure.subplot.wspace'] = 0.2
    plt.rcParams['figure.subplot.hspace'] = 0.2
    plt.rcParams['figure.autolayout'] = False
    plt.rcParams['figure.constrained_layout.use'] = use_constrained_layout
    plt.rcParams['figure.constrained_layout.h_pad'] = 0.02
    plt.rcParams['figure.constrained_layout.w_pad'] = 0.02
    plt.rcParams['figure.constrained_layout.hspace'] = 0.02
    plt.rcParams['figure.constrained_layout.wspace'] = 0.02

    # Savefig settings
    plt.rcParams['savefig.dpi'] = config.savefig_dpi
    plt.rcParams['savefig.facecolor'] = 'white'
    plt.rcParams['savefig.edgecolor'] = 'white'
    plt.rcParams['savefig.format'] = 'svg'
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.02
    plt.rcParams['savefig.transparent'] = False
    plt.rcParams['savefig.orientation'] = 'portrait'

    # PS backend settings
    plt.rcParams['ps.papersize'] = 'letter'
    plt.rcParams['ps.useafm'] = False
    plt.rcParams['ps.usedistiller'] = False
    plt.rcParams['ps.distiller.res'] = 6000
    plt.rcParams['ps.fonttype'] = 3

    # PDF backend settings
    plt.rcParams['pdf.compression'] = 6
    plt.rcParams['pdf.fonttype'] = 3
    plt.rcParams['pdf.use14corefonts'] = False
    plt.rcParams['pdf.inheritcolor'] = False

    # SVG backend settings
    # plt.rcParams['svg.image_inline'] = True
    # plt.rcParams['svg.fonttype'] = 'path'
    # plt.rcParams['svg.hashsalt'] = None

    if label_pad_pts is not None:
        plt.rcParams['axes.labelpad'] = float(label_pad_pts)

    if tick_label_pad_pts is not None:
        tick_pad = float(tick_label_pad_pts)
        plt.rcParams['xtick.major.pad'] = tick_pad
        plt.rcParams['xtick.minor.pad'] = tick_pad
        plt.rcParams['ytick.major.pad'] = tick_pad
        plt.rcParams['ytick.minor.pad'] = tick_pad


# ==============================================================================
# SPECIFIC PLOT CONFIGURATIONS (Moved from SV heatmaps lineplots.py)
# ==============================================================================
SAVEFIG_KW = {"dpi": 600, "transparent": False, "bbox_inches": "tight"}

HEATMAP_FIGSIZE = (2.5 / 2.54, 6 / 2.54)
HEATMAP_GRIDSPACE = 0.025
HEATMAP_STYLE_KW = {
    "cbar": False,
    "robust": False,
    "yticklabels": False,
    "clip_on": False,
    "vmin": 0,
    "vmax": 1,
    "rasterized": True,
}
HEATMAP_TICK_KW = {
    "axis": "both",
    "which": "both",
    "bottom": True,
    "top": False,
    "right": False,
    "direction": "out",
}

LINEPLOT_FIG_KW = {"facecolor": "white", "clip_on": False}
LINEPLOT_STYLE_KW = {
    "alpha": 0.9,
    "linewidth": 0.5,
    "marker": ".",
    "markersize": 0,
    "markeredgecolor": "k",
    "markeredgewidth": 2,
    "estimator": "median",
    "err_style": "band",
    "err_kws": {"alpha": 0.3},
    "seed": 10,
    "clip_on": False,
}
LEGEND_KW = {
    "frameon": False,
    "bbox_to_anchor": (1, 1),
    "bbox_transform": plt.gcf().transFigure,
    "loc": "upper left",
    "borderaxespad": 0,
}
