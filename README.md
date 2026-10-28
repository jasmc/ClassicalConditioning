# Classical Conditioning

Analysis of tail tracking from a head-fixed larval-zebrafish conditioning assay.

Start with the [paper figures](docs/paper/figures/README.md) for selected panels, scientific purpose, legends and review. The [documentation guide](docs/README.md) separates paper work, running the analysis and project maintenance.

| Need | Canonical home |
|---|---|
| Figures and panel work | [Paper figures](docs/paper/figures/README.md) |
| Compare discussions and approaches for Figure 2 G/H/I and Figures 3/4 | [Review library and SSD entry point](docs/paper/figures/review-library.md) |
| Shared scientific definitions | [Methods](docs/paper/methods/README.md) |
| Accepted choices and open paper questions | [Decisions](docs/paper/decisions.md) and [paper review](docs/paper/review.md) |
| Run, resume or troubleshoot the code | [Running the analysis](docs/running-analysis/README.md) |
| Configure a run and understand its outputs | [Parameters](docs/running-analysis/parameters.md) and [outputs](docs/running-analysis/outputs-and-provenance.md) |
| Folder rules, governance and SSD storage | [Project maintenance](docs/project-maintenance/README.md) |
| Exact frozen versions and supporting evidence | [Records](records/README.md) |
| Future implementation and progress | [Plans](Plans/README.md) and [implementation index](Plans/IMPLEMENTATION_STEP_INDEX.md) |

## Set up and run

Use CPython 3.12 or 3.13 and the locked environment:

```sh
uv sync --frozen --all-extras
uv run python -m unittest discover -s tests
```

Copy [the strict JSON example](configs/example-run.json), set the raw and derived data paths, and run:

```sh
uv run classical-conditioning run-pipeline --config configs/example-run.json
```

The routine pipeline inventories recordings, verifies or creates lossless intake, runs corrected analysis and schedules every ready descriptive figure. Failures and blocked outputs are recorded in the run ledger. Follow the [run/resume guide](docs/running-analysis/run-pipeline.md) for recovery; use [parameters](docs/running-analysis/parameters.md) instead of an independent option table here.

## Scientific authority and preservation

Accepted scoped freezes govern their panels. An older whole assembly does not override a newer selection, and a presentation draft does not replace frozen science. Use the figure workflow and scoped current state to resume the recorded frozen display or active draft. Whole-figure and release approval remain separate.

Keep raw data, accepted frozen bytes and hashes intact. Supported code lives in `src/`, reusable tools in `scripts/`, and tests in `tests/`; inherited reference code remains in `legacy/`. Permanent media and numerical payloads belong on JOAQUIM. Use [storage guidance](docs/project-maintenance/storage.md) and exact relocation/hash resolution for historical artifacts.
