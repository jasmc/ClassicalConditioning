# Supported figure build tools

Use these tools with authenticated scientific inputs. Capability does not establish scientific approval; current selections and presentation state remain separate.

| Tool | Implemented purpose |
|---|---|
| `classical-conditioning render-paper-panels` | Supported panel orchestration with explicit completed/failed/blocked outcomes |
| `scripts/assemble_svg_figure.py --plan` | Read-only source/hash/geometry preflight; use `--strict` when every panel source is required |
| `scripts/assemble_svg_figure.py --replace-panel-source ID=PATH` | A separate candidate panel replacement with baseline/provenance safeguards |
| `scripts/update_latest_figure_assemblies.py` | Existing Figure 1–2 scoped-selection assembly route |
| `scripts/figure_working_copy.py verify` | Authenticate the recorded frozen display, active draft and bound evidence |
| `scripts/figure_working_copy.py begin --ticket <temporary-ticket.json>` | Capture a pre-edit state/input ticket for guarded publication |
| `scripts/build_figure_legend_review.py --baseline <ticket> --output <external-review.html>` | Guarded consolidated review and editable draft publication; retain unchanged other-figure material |
| `scripts/freeze_figure.py` | Explicit author-requested freeze gate; run `--check-only` first |

Consult each command's `--help` for its actual interface. The generic assembler composes supplied sources; it does not select scientifically current panels. Figure-specific source routing is described with [Figure 1](../paper/figures/figure-01/assembly.md), [Figure 2](../paper/figures/figure-02/assembly.md) and [Figure 4](../paper/figures/figure-04/analysis-and-rendering.md).

Use the [assembly preflight/replacement guide](../paper/figures/shared/assembly-preflight.md) for replacements and the [iteration workflow](../paper/figures/shared/iteration-workflow.md) for current-state updates. The assembler's transaction is not the figure-state publication protocol. Missing sources and approvals must remain explicit; do not substitute a review precursor for a frozen selection.

Permanent outputs require an explicit external destination. A normal author review is one self-contained HTML with the recorded editable SVG copies permitted by the workflow. Extra PNG/PDF/SVG exports require author direction. Ordinary rendering is not a freeze.

[Plan 07](../../Plans/07_FIGURE_BUILD_TOOLS.md) owns remaining routing/reproduction integration. The [preserved tooling design](../../records/repository-history/tooling-designs/figure-build-tools.md) retains optional CLI/API/notebook proposals and historical pilot notes; those commands are not the current interface or a live progress board.
