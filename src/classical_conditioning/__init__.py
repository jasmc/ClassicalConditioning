"""Classical conditioning analysis package."""

from classical_conditioning.exceptions import (
    AmbiguousArtifactError,
    ArtifactIntegrityError,
    ArtifactNotFoundError,
    ClassicalConditioningError,
    ConfigurationError,
    ModelDiagnosticError,
    SchemaValidationError,
    ScientificValidationError,
)
from classical_conditioning.intake import (
    IntakeResult,
    RecordingSources,
    discover_recording,
    inspect_table_structure,
    intake_recording,
)

__all__ = [
    "AmbiguousArtifactError",
    "ArtifactIntegrityError",
    "ArtifactNotFoundError",
    "ClassicalConditioningError",
    "ConfigurationError",
    "IntakeResult",
    "ModelDiagnosticError",
    "RecordingSources",
    "SchemaValidationError",
    "ScientificValidationError",
    "discover_recording",
    "inspect_table_structure",
    "intake_recording",
]

__version__ = "0.1.0"
