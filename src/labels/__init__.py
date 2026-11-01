"""Label data access and processing utilities."""

from .api import (
    CONTEXT_NAMES,
    DAILY_LABELS,
    LABEL_NAMES,
    LABEL_TYPES,
    MULTI_CATEGORICAL_NAMES,
    TARGET_NAMES,
    LabelResult,
    LabelTypeError,
    LabelValueError,
    LabelWindowResult,
    configure,
    get_labels,
    get_labels_contains,
    get_labels_count,
    get_labels_one_hot,
    get_labels_statistics,
    get_labels_windowed,
    print_labels_statistics,
    years_between_birth_year,
)

__all__ = [
    "CONTEXT_NAMES",
    "DAILY_LABELS",
    "LABEL_NAMES",
    "LABEL_TYPES",
    "LABEL_TYPES_PATH",
    "MULTI_CATEGORICAL_NAMES",
    "ORDINAL_DICTIONARY_PATH",
    "TARGET_NAMES",
    "LabelResult",
    "LabelTypeError",
    "LabelValueError",
    "LabelWindowResult",
    "configure",
    "get_labels",
    "get_labels_contains",
    "get_labels_count",
    "get_labels_one_hot",
    "get_labels_statistics",
    "get_labels_windowed",
    "print_labels_statistics",
    "years_between_birth_year",
]


def __getattr__(name: str):
    """Expose current schema paths after explicit reconfiguration."""
    if name in {"LABEL_TYPES_PATH", "ORDINAL_DICTIONARY_PATH"}:
        from . import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
