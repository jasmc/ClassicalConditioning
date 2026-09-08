"""Raw acquisition readers and audits for local ingestion."""

from classical_conditioning.ingestion.frame_sequence import (
    FrameSequenceReport,
    validate_frame_sequence,
)
from classical_conditioning.ingestion.readers import (
    CameraReadResult,
    ProtocolReadResult,
    TrackingReadResult,
    read_camera,
    read_protocol,
    read_tracking,
)
from classical_conditioning.ingestion.tracking_audit import (
    GATE_T0_LEGACY_EVIDENCE,
    audit_tracking_file,
    classify_tracking_columns,
    write_tracking_audit,
)

__all__ = [
    "CameraReadResult",
    "FrameSequenceReport",
    "GATE_T0_LEGACY_EVIDENCE",
    "ProtocolReadResult",
    "TrackingReadResult",
    "audit_tracking_file",
    "classify_tracking_columns",
    "read_camera",
    "read_protocol",
    "read_tracking",
    "validate_frame_sequence",
    "validate_raw_triplet",
    "write_tracking_audit",
]


def __getattr__(name: str):
    if name in {"validate_raw_triplet", "RawValidationResult"}:
        from classical_conditioning.ingestion import validate_raw as _validate_raw

        return getattr(_validate_raw, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
