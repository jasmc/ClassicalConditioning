"""Logical source bindings and portable Markdown links for the G study guide."""
from pathlib import Path
import re

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
        r"\]\(<(/Volumes/JOAQUIM/ClassicalConditioning Outputs/(?:current/reviews/figure-02-g-analysis-guide-20261017-v01|figures/figure-02/panel-g/studies/analysis-guide)/sources/[^>]+)>\)",
        lambda match: "](sources/" + Path(match[1]).name + ")",
        markdown,
    )
