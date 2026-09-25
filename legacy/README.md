# Original analysis code

This directory keeps the original code that preceded the supported package
refactor for quick source reference and descriptive comparisons.

| Directory | Original source |
| --- | --- |
| `scripts/` | Numbered preprocessing, plotting, grouping, and learner scripts, including their original learner variants. |
| `helpers/` | Shared helpers imported by those scripts. |
| `modules/` | Earlier `my_*` helper modules. |

The supported implementation is under [`../src/classical_conditioning/`](../src/classical_conditioning/).
The `compare-legacy-learners` command runs the original learner scripts in
isolated processes for descriptive comparison. Other original scripts are
reference source and have no supported execution entry point.

Intermediate package source, old plan files, and dated status snapshots remain
only in Git history.
