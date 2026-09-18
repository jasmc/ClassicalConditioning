"""Optional statistical inference built from candidate analysis artifacts.

Review note: the public inference façade currently lazy-loads learning-onset
analysis, preventing optional statistics dependencies from loading at import.
"""

# Names supplied by learning_onset.py and intentionally exposed by this package.
__all__ = [
    "LearningOnsetConfig",
    "LearningOnsetResult",
    "build_learning_onset_analysis",
    "load_learning_onset_analysis",
    "localize_learning_onset",
]


def __getattr__(name: str):
    # All listed inference exports currently live in the learning-onset module.
    if name in set(__all__):
        from classical_conditioning.analysis.inference import learning_onset

        return getattr(learning_onset, name)
    raise AttributeError(name)
