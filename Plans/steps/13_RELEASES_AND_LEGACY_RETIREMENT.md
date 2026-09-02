# Step 13 — Reproduction Releases, Documentation, and Legacy Retirement

**Status:** Not started  
**Change class:** Release/cleanup; no new scientific behavior  
**Depends on:** All steps required by the selected release  
**Unlocks:** Stable paper reproduction and safe removal of ambiguity

## Objective

Produce immutable, independently verifiable analysis releases and retire
legacy execution paths only after their replacements pass their acceptance
gates.

## Release types

### R1 — Legacy-equivalent release

Contains:

- current behavior under the pinned environment;
- canonical Parquet/JSON mirrors;
- temporary legacy outputs where required;
- explicit known-scientific-issues report;
- evidence of stage-by-stage equivalence.

Scientific status:

```text
legacy_reproduction
```

It is not automatically approved for final paper inference.

### R2 — Candidate-development release

Contains:

- common tail representation;
- all feasible candidate metrics;
- movement-detector candidates;
- synthetic/video/positive-control validation;
- robustness grid;
- selection scorecard;
- frozen confirmation recipe.

R2 requires the Step 12 panel-data builder, candidate-comparison renderer, and
figure provenance/QC subset. It does not require the complete final CLI,
notebook suite, learner figures, or major-paper composition system.

Scientific status:

```text
candidate_development
```

### R3 — Corrected paper-candidate release

Contains:

- frozen corrected preprocessing recipe;
- complete paper-scope reprocessing;
- approved cohort;
- canonical outcomes;
- primary and sensitivity statistics;
- classifier results where used;
- regenerated figures;
- legacy-versus-corrected impact assessment.

Scientific status:

```text
corrected_candidate
```

### R4 — Approved paper release

Created only after scientific owner approval.

Scientific status:

```text
paper_approved
```

## Release contents

```text
release/
    release.json
    README.txt
    source-inventory.json
    code-identity.json
    resolved-config.json
    environment/
        lockfile
        environment-report.json
    decisions/
    artifacts/
        artifact-manifest.json
        canonical tables or approved external references
    cohort/
    statistics/
        model-inputs/
        model-results/
        diagnostics/
        sensitivity/
    classification/
    panel-data/
    figures/
    comparisons/
        legacy-versus-new/
    manuscript/
        methods-parameters.csv
        claim-to-artifact.csv
    logs/
        reproduction.log
        validation-report.json
```

Large data may be referenced in protected storage rather than duplicated, but
the release must include immutable identifiers, hashes, access expectations,
and verification instructions.

## Work packages

### 13.1 Implement reproduction command

```powershell
python -m classical_conditioning reproduce --release <release-id>
```

It:

1. validates code/environment/input identity;
2. resolves the immutable recipe;
3. verifies or builds required artifacts;
4. runs tests and release-specific checks;
5. creates a temporary release directory;
6. verifies the release manifest;
7. atomically publishes the immutable release.

### 13.2 Produce impact assessment

For corrected releases, compare with R1:

- recording success/failure;
- frame and trial coverage;
- cohort additions/removals and reasons;
- metric and bout distributions;
- total/probability/intensity outcomes;
- effect sizes and intervals;
- statistical conclusions;
- learner labels;
- figure changes;
- manuscript claims and methods.

### 13.3 Create claim-to-artifact index

Every paper claim links to:

- result ID;
- model/contrast ID;
- outcome and cohort;
- panel/table;
- source artifacts;
- code/config/environment.

### 13.4 Reconcile manuscript

Verify:

- mathematical methods match executed versions;
- units and windows match artifacts;
- sample sizes reconcile;
- exclusions and flow are reported;
- bootstrap and model descriptions match;
- learner analysis is described by its validation mode;
- exploratory analyses are labeled;
- figure legends identify outcomes and uncertainty units.

### 13.5 Retire legacy execution

Only after R1 and required corrected replacements:

1. confirm active imports;
2. convert numbered scripts to wrappers or move archival copies under
   `legacy/`;
3. archive alternate learner scripts with version mapping;
4. remove ambiguous first/latest artifact discovery;
5. prevent new imports of retired modules;
6. stop canonical pickle writing;
7. keep read-only pickle conversion for historical artifacts;
8. document replacement commands.

Never remove a legacy implementation in the same change that first introduces
its replacement.

Before final release, run a separate repository-organization and stale-file
pass. Inventory tracked and ignored files, remove generated caches/build
outputs and user-specific editor artifacts, and ensure every retained root file
has a documented role. Do not infer that an old-looking scientific script is
stale: moving or deleting executable analysis still requires replacement
equivalence, migrated consumers or a tested wrapper, and the relevant
scientific gate. Raw data and generated scientific artifacts outside the
repository are never cleanup targets.

### 13.6 Archive completed implementation plans

After each step passes its exit gate:

1. move its plan from `Plans/steps/` to `Plans/Archive/steps/`;
2. record completion evidence and relevant commit IDs in the archived file;
3. update `Plans/README.md`, `Plans/Archive/README.md`, and the implementation
   index;
4. update the repository README when commands, formats, or active versions
   changed;
5. commit those documentation changes atomically with a semantic message.

### 13.7 Final validation

- Full unit, contract, integration, statistics, figure, and end-to-end tests
- Complete paper-scope recording accounting
- Artifact hash verification
- Reproduction from a clean environment
- Independent review of release contents
- Scientific owner sign-off

## Definition of release immutability

- Release ID is unique.
- Files are not overwritten.
- Any correction creates a new release ID.
- Recipe, code, environment, source inventory, and cohort are frozen.
- Artifact hashes are verified on load.
- Timestamps do not substitute for content identity.

## Exit gate

One documented command reproduces or verifies the approved paper release from
explicit inputs; all paper claims and figures are traceable; canonical analysis
does not depend on pickle; legacy paths are wrappers or clearly archived.

## Post-release changes

Any later change is classified as:

- documentation-only;
- rendering-only;
- software correction;
- scientific correction;
- new exploratory analysis.

Changes that affect results create a new recipe and release with an explicit
impact report.
