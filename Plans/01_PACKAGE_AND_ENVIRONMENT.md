# Step 01 — Python Package, Test, and Environment Foundation

**Status:** In progress — foundation implemented; Step 00 reference scope is blocked
**Change class:** Behavior-preserving  
**Depends on:** Step 00 governance decisions  
**Unlocks:** Structured implementation and automated validation

## Objective

Create the smallest installable Python foundation needed to migrate real
analysis behavior. Existing scripts must continue to run.

## Target structure

```text
pyproject.toml
src/
    classical_conditioning/
        __init__.py
        exceptions.py
        cli/
tests/
    unit/
    contracts/
    characterization/
    integration/
    regression/
    statistics/
    figures/
    end_to_end/
    fixtures/
```

Create additional modules only when a later step migrates real behavior.

## Required decisions

- [x] Python version and supported platform(s): CPython 3.12, 64-bit Windows
- [x] Dependency manager and lockfile: `uv` with committed `uv.lock`
- [x] Test runner: standard-library `unittest`
- [x] Package versioning scheme: semantic `0.x` versions during migration
- [x] Formatting/linting policy: no new tool yet; tests plus `git diff --check`
- [x] CI scope: no CI is currently configured; local frozen validation is required

## Work packages

### 01.1 Add package metadata

Define:

- package name and version;
- Python requirement;
- runtime dependencies;
- optional development/test dependencies;
- CLI entry point;
- test configuration.

Do not include libraries solely because they might be useful later.

### 01.2 Build a pinned environment

Cover the complete pipeline, not only learner analysis:

- NumPy
- pandas
- SciPy
- Matplotlib
- seaborn
- statsmodels
- scikit-learn
- optional pyarrow/Numba only where used
- figure and annotation dependencies used by active scripts

Record:

- exact versions;
- hashes where the lock mechanism supports them;
- BLAS/LAPACK implementation;
- default thread counts;
- font and Matplotlib information needed for figure regression.

### 01.3 Add one validation command

Examples:

```powershell
uv run python -m unittest discover -s tests
```

Use the project's `unittest` discovery command. Avoid introducing a second
parallel testing framework.

### 01.4 Add package import and environment tests

Test:

- package imports;
- package version is available;
- critical scientific packages import;
- non-interactive figure backend is supported;
- test fixtures can be located without machine-specific paths.

### 01.5 Define exception categories

Start with only useful boundaries:

```text
ConfigurationError
SchemaValidationError
ArtifactNotFoundError
ArtifactIntegrityError
AmbiguousArtifactError
ScientificValidationError
ModelDiagnosticError
```

Do not broadly catch these and return success-shaped empty outputs.

### 01.6 Preserve current entry points

At this stage:

- do not move numbered scripts;
- do not rename current modules;
- do not alter scientific defaults;
- do not introduce import-time side effects into the new package.

### 01.7 Regenerate the numerical reference baseline

After the environment is pinned, rerun the accepted Step 00 representative
baseline in this exact environment. That pinned numerical reference is required
for the final Step 01 exit gate and for paper-scope equivalence work. Fixture-
scoped Step 05 already completed under local fixtures without waiting for this
item; it does not replace the Step 00–scoped baseline.

## Deliverables

- Installable local package
- Locked development/paper environment
- One documented test command
- Initial test layout
- Package version
- Minimal exception module
- Optional CI validation

## Validation

- [x] Clean environment install succeeds.
- [x] Package imports from the repository and installed environment.
- [x] Tests run without experimental data.
- [x] Current numbered scripts still import through direct or characterization
      harnesses. Stage 4 retains its characterized import-time output-directory
      creation and therefore requires the harness on machines without its
      historical `F:\` path.
- [x] No scientific output changes.
- [x] The lockfile is committed; virtual environments are not.
- [ ] A Step 00–scoped pinned-environment numerical reference is available
      (blocks final Step 01 exit, not fixture-scoped Step 05).

## Implementation evidence

The frozen command:

```powershell
uv sync --frozen --all-extras
uv run python -m unittest discover -s tests
```

created an isolated CPython 3.12.10 environment with 30 locked packages and
completed 113 data-free tests. Package import, the installed
`classical-conditioning` entry point, Agg rendering, legacy scientific-package
imports, and environment-report JSON publication passed.

The lockfile SHA-256 is:

```text
4da0f6b387736cf02b0c70cefb7df79da5b43794fc55f0f370757c3dc536b070
```

The locked Windows environment includes NumPy 2.3.5, pandas 2.3.3, SciPy
1.18.1, Matplotlib 3.10.9, PyArrow 21.0.0, and scipy-openblas 0.3.30.
Thread-count variables are intentionally not set by the project and are
captured by `environment-report`.

Step 01 cannot pass its final exit gate until Step 00 defines and approves the
representative baseline scope. Re-running or overwriting scientific pilot
artifacts under a changed numerical environment before that decision would
violate provenance and behavior-preservation requirements. The existing pilot
artifacts remain unchanged.

## Exit gate

A new contributor can create the documented environment, install the package,
and run the test command without editing source code or accessing private data.

## Downstream invalidation

Package infrastructure changes do not invalidate scientific artifacts unless
they change a numerical dependency. Numerical dependency changes require a new
environment identity and targeted regression validation.
