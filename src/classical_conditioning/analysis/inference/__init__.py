"""Optional statistical inference built from candidate analysis artifacts."""

__all__ = [
    "LearningOnsetConfig",
    "LearningOnsetResult",
    "build_learning_onset_analysis",
    "load_learning_onset_analysis",
    "localize_learning_onset",
]


def __getattr__(name: str):
    if name in set(__all__):
        from classical_conditioning.analysis.inference import learning_onset

        return getattr(learning_onset, name)
    raise AttributeError(name)
