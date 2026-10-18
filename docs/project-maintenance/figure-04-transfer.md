# Transfer Figure 4 work to another computer

Use the reviewed repository revision and locked environment on the receiving
computer. Record its exact commit and retain the source machine's inputs and
manifests until the transferred work is authenticated. Do not apply the integrated
historical handover patch to a current checkout.

The [Figure 4 operational manual](../running-analysis/figure4.md) owns input
schemas, assay scope, analyze/render commands, output contracts and delivery
checks. Approved paper methods still come from scoped scientific selections and
Gate L; transferring an exploratory run does not approve it.

## Transfer inputs and paths

Transfer the reviewed cohorts, matching classifier table and manifest, selection
assessments and their authenticated constituent artifacts. Keep inputs available
through analysis. The operational manual specifies their required identities.

Source manifests and saved analyses can bind absolute paths. Use the same paths
on the receiving machine, or rerun authenticated intake/derived stages and
reissue the cohort/classifier manifests through their owning workflow. Do not
hand-edit hashed manifests. Run the analysis where its final project paths will
remain, and preserve saved tables if rendering fails.

## Prepare the receiving environment

Use CPython 3.12 or 3.13 with the repository lock:

```sh
uv sync --frozen --all-extras
uv run python -m unittest discover -s tests -p 'test_figure4.py'
```

These synthetic tests check software behavior. Authenticate the transferred
inputs separately, then follow the operational manual's run and delivery checks.

The [preserved handover](../../records/repository-history/2026-10-17-organization-reconciliation/original-documents/docs/project-maintenance/figure-04-transfer.md)
records the dated 6 October state and earlier exploratory run references. It is
historical evidence, not a current machine-availability or paper-approval claim.
