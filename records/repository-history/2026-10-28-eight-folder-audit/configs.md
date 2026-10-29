# Independent audit: configs

Audit date: 28 October 2026. Baseline: `702193b9591b62bf12820bc2b01ca5e8928f9528`.
Focused review of the guide, three root run settings, assembly metadata and their
consumers. Findings are recommendations, not executed dispositions.

## Ownership

Keep reusable executable settings, presentation profiles and scoped scientific
selection pointers here. Exclude run outputs, numerical payloads, audit reports
and scientific approval inferred merely from a setting's presence. Selection
records and ordinary editable examples have different preservation obligations.

## Findings and dispositions

- **C1 — Clarify a retired execution route.** [The guide](../../../configs/README.md)
  line 13 names absent `scripts/update_latest_figure_assemblies.py`; both assembly
  JSON files repeat it as `current_assembly_tool`. The command-relocation record
  maps this identity to an archived historical assembly driver, not a current
  executable. `scripts/paper/figure_02/assembly/build.py` still consumes the
  Figure 2 layout. Replace the guide's instruction with navigation to scoped
  figure state and the supported build-tools guide; use
  `scripts/main_paper_figures.py build` only for the ordered collection. Retain
  historical JSON metadata until its hash/source bindings are checked; do not
  replace its value with a command having a different assembly contract.
- **C2 — Retain technical settings; clarify portability.** The two Windows
  configurations deliberately combine `J:` raw roots and `F:` processed roots;
  the trace settings retain 59 recording IDs and the acquisition spelling
  `all3sTtrace`. Windows stages and `scripts/workflows/trace/prepare_runs.py`
  consume these paths/settings. Copy and configure settings for a new host;
  neither normalize acquisition names nor change the recording selection as
  cleanup. All three root JSON files use only fields accepted by the strict
  run-settings parser. This field check does not validate mounted input bytes.
- **C3 — Retain selection layout and names.** Current, history and lineage
  pointers have distinct documented roles. Mixed historical naming is not
  evidence that immutable pointers should be renamed. No selection rename or
  unused-field removal is justified by this focused review.

## Proposed instruction wording — not active

Owner: `configs/README.md`, with preservation procedures linked to the central
structure/storage owners.

> Describe each maintained configuration's consumer and whether it is an example,
> host-specific run setting, presentation policy or protected selection pointer.
> Examples require explicit input/output paths before use. Configure host paths
> in a copy; preserve acquisition names, recording selections and scientific
> settings. Historical metadata is not a current command recommendation.
> Before retiring or renaming a configuration, trace its executable consumers,
> selection bindings and exact recovery requirements.

## Verification and limits

Evidence: external `config-field-check.json`, static import/text evidence and
the command-relocation record. Proposed guide changes need local link checking;
any later layout edit also needs its consumer tests and binding review. No
configuration was executed. Individual scientific values and every profile
field were not revalidated; no claim of scientific equivalence is made.
