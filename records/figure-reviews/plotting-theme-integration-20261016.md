# Shared Python plotting integration — 2026-10-16

Scope: the existing shared plotting module and its Nature profile adapter.
This is implementation evidence, not a scientific freeze or saved-figure
compliance report. Nature profile version: **1.0.1**.

## Implemented behavior

- A profile name or one resolved profile object can be passed directly to
  `apply_theme`; `get_theme` also reuses resolved objects. The build can use one
  policy snapshot for both styling and provenance, with no second file read or
  global profile cache. Conflicting theme/profile arguments fail before mutation.
- Ordinary axes receive declared fonts, text/tick colors, spine widths and tick
  caps when using the profile. Tick-label visibility is independent of local
  spine/tick visibility, supporting registered shared-label groups.
- New Matplotlib panel/condition headings default to the data-rectangle center.
  CS/US guide widths and stimulus-window layer order come from existing roles.
  Limits, tick values, plotted observations and event intervals are preserved.
- Calls without a profile retain legacy font/size defaults and stimulus behavior.
  Documentation uses `matplotlib.rc_context()` to contain global render settings.
- AGENTS.md names the existing shared module; the skill/guide explain loading
  once per build and passing the same theme to helpers. Exact role values remain
  in the machine profile and inherited contract, rather than per-panel copies.

## Validation

The `test_presentation_profiles.py` and `test_theme.py` suites pass **21 tests**.
New cases exercise snapshot reuse without file reads, conflicting arguments,
restyling existing axes while preserving numerical coordinates, local ticks with
shared labels, and source-defined stimulus widths/timing/layering. Existing tests
retain checks for legacy defaults, invalid rules/fonts and policy provenance.
The skill validator, targeted link checks and scoped whitespace checks pass.

Protected repository specification/contract, state/legend and unrelated working
files were checked against the captured starting hashes. No selected figure SVG,
HTML or current-state pointer was changed. JOAQUIM was not mounted during this
code-only task; no current external artwork inspection or publication is claimed.

## Bound code and policy

| Input | SHA-256 |
|---|---|
| `configs/paper-figures/nature-main.json` | `a2941821c8873f1559efe200b72467f84376e53dc08232c6438e496ef6b39732` |
| `configs/paper-figures/figure-elements.json` | `61c9407cdf18b052798fba99ba3935615c8cb0bf4fc4ca7f40a80d210e66bf24` |
| `configs/paper-figures/layout-frames.json` | `d77e8b94880f60d255ecd2772043b67414bad0ce7b08a297aeee01e6c632db3c` |
| `src/classical_conditioning/figures/theme.py` | `12dcf95debf20f626e413ab326efb055b35c5f93ad87a938befc71a4433dbe2d` |
| `src/classical_conditioning/figures/presentation_profiles.py` | `e440c45ae7482ad3fd50c38445616097edc7bebea14bf87de7f29e3009f79395` |
| `tests/test_theme.py` | `3954edb8b705dec3d1328eb1f5c90ffc0987196e5e8dfd2c73c5c711a971b43c` |
| `tests/test_presentation_profiles.py` | `fb692d0843ff2443ff6f2de5902c687c2d291fd6fdc4a717ecebd8c1baabea37` |

## Remaining scope

Nominal renderer pads do not establish geometric glyph gaps. Exact saved-SVG
measurements, related-panel assembly, font embedding, conditional plot-family
layouts and publication/freeze integration remain owned by Plan 07 and the
existing guarded workflow. This change does not certify those checks.
