"""Typed failure boundaries for the analysis package."""


class ClassicalConditioningError(Exception):
    """Base class for package-defined failures."""


class ConfigurationError(ClassicalConditioningError):
    """Configuration is missing, inconsistent, or unsupported."""


class SchemaValidationError(ClassicalConditioningError):
    """An input or derived table violates its declared schema."""


class ArtifactNotFoundError(ClassicalConditioningError, FileNotFoundError):
    """A required analysis artifact could not be found."""


class ArtifactIntegrityError(ClassicalConditioningError):
    """Artifact content or lineage does not match its authenticated metadata."""


class AmbiguousArtifactError(ClassicalConditioningError):
    """Artifact discovery produced more than one valid candidate."""


class ScientificValidationError(ClassicalConditioningError):
    """Scientific validation requirements were not met."""


class ModelDiagnosticError(ClassicalConditioningError):
    """A fitted model failed a required diagnostic."""
