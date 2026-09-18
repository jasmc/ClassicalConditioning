"""Public, stable imports for the classical-conditioning analysis package.

Review note: implementation modules remain importable, but this file defines
the compact public API that users and downstream code should prefer.
"""

# Re-export expected domain failures from one convenient package-level location.
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
# Re-export the primary raw-recording discovery and intake interface.
from classical_conditioning.intake import (
    IntakeResult,
    RecordingSources,
    discover_recording,
    inspect_table_structure,
    intake_recording,
)

# Explicitly document and constrain what ``from classical_conditioning import *``
# exposes; internal helpers are intentionally omitted.
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

# Package version used by callers that need to report the installed API release.
__version__ = "0.1.0"
