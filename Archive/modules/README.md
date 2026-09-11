# Archived Legacy Modules

These modules are retained as read-only implementation history:

- `my_general_variables.py`
- `my_experiment_specific_variables.py`
- `my_functions.py`

They are not part of the installable `classical_conditioning` package and must
not be imported by active analysis code. Their modern responsibility mappings
are documented in `docs/analysis/ANALYSIS_FILES_INDEX.md`.

Git history preserves their original root locations. Do not modify these files
to implement new behavior; migrate required behavior into tested package
modules or explicit compatibility adapters.
