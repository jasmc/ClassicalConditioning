"""Typed failure boundaries for the analysis package.

Review note: callers catch these types to distinguish invalid user input,
missing/corrupt derived files, scientific gating failures, and model failures.
No class adds behaviour; the type itself is the contract.
"""

# Package-level base class: applications can catch one type for any expected
# domain failure without swallowing unrelated Python/runtime exceptions.
class ClassicalConditioningError(Exception):
    """Base class for package-defined failures."""

# Input/configuration validation failures are recoverable by correcting settings.
class ConfigurationError(ClassicalConditioningError):
    """Configuration is missing, inconsistent, or unsupported."""

# Schema failures identify structurally invalid raw or derived tables.
class SchemaValidationError(ClassicalConditioningError):
    """An input or derived table violates its declared schema."""

# This preserves FileNotFoundError compatibility for code that checks files.
class ArtifactNotFoundError(ClassicalConditioningError, FileNotFoundError):
    """A required analysis artifact could not be found."""

# Hashes, markers, or provenance no longer authenticate the requested artifact.
class ArtifactIntegrityError(ClassicalConditioningError):
    """Artifact content or lineage does not match its authenticated metadata."""

# Artifact discovery found multiple valid matches, so choosing would be unsafe.
class AmbiguousArtifactError(ClassicalConditioningError):
    """Artifact discovery produced more than one valid candidate."""

# A data-dependent scientific gate failed even though software execution worked.
class ScientificValidationError(ClassicalConditioningError):
    """Scientific validation requirements were not met."""

# Model fitting completed or began, but required diagnostic criteria were not met.
class ModelDiagnosticError(ClassicalConditioningError):
    """A fitted model failed a required diagnostic."""
