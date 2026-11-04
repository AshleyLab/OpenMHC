"""API for accessing and processing MHC benchmark labels."""

from __future__ import annotations

import json
import logging
import math
import statistics
from bisect import bisect_left, bisect_right
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from openmhc._label_paths import LabelPaths

logger = logging.getLogger(__name__)


def _require_dataset_payload_path(
    path: Path | None,
    *,
    env_var: str,
    filename: str,
) -> Path:
    if path is not None:
        return path
    raise ValueError(
        f"OpenMHC dataset root is required to resolve labels/{filename}. "
        f"Expected this file under <dataset_root>/labels/{filename}. "
        "Call `labels.configure(data_dir=...)` with a dataset containing all three "
        "schema files before querying. Alternatively set `MHC_DATA_DIR` and call "
        f"`labels.configure()`. `{env_var}` overrides this payload only; it does "
        "not select the required dataset schemas."
    )


_PATHS = LabelPaths.resolve()


def _load_label_types(path: Path) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Load label type mappings from label_types.json.

    Handles both old flat format (``{"Diabetes": "binary"}``) and new
    object format (``{"Diabetes": {"type": "binary", "target": true}}``).

    Returns:
        Tuple of (types_dict, meta_dict) where types_dict maps label→type
        string for backward compatibility, and meta_dict maps label→full
        metadata dict.
    """
    with path.open("r") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"Label types must be a nonempty JSON object: {path}")
    types: dict[str, str] = {}
    meta: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            # Old flat format: {"Diabetes": "binary"}
            types[key] = value
            meta[key] = {"type": value, "target": True}
        elif isinstance(value, dict) and "type" in value:
            # New object format: {"Diabetes": {"type": "binary", "target": true}}
            types[key] = value["type"]
            meta[key] = value
        else:
            raise ValueError(f"Invalid label type metadata for {key!r}: {path}")
        if types[key] not in (
            "binary",
            "ordinal",
            "categorical",
            "multi_categorical",
            "continuous",
        ):
            raise ValueError(f"Unknown label type for {key!r}: {path}")
    return types, meta


# Metadata is populated by configure() or the first label query.
LABEL_TYPES: dict[str, str] = {}
_LABEL_META: dict[str, dict[str, Any]] = {}
_SCHEMAS_CONFIGURED = False


def _ensure_configured() -> None:
    """Configure lazily so imports work before an explicit dataset is selected."""
    if not _SCHEMAS_CONFIGURED:
        configure()


class LabelTypeError(Exception):
    """Raised when a label value cannot be converted to its expected type."""

    pass


class LabelValueError(Exception):
    """Raised when a label value is NaN or None."""

    pass


def _enforce_type(label: str, value: Any) -> bool | int | float:
    """Enforce the expected type for a label value.

    Rules:
    - binary: must be bool
    - ordinal: can be string, int or float
    - categorical: must be string
    - continuous: must be float

    Raises:
        LabelValueError: If value is None or NaN
        LabelTypeError: If value cannot be converted to expected type
    """
    if value is None:
        raise LabelValueError(f"Label '{label}' has None value")

    # Check for NaN (works for float NaN)
    if isinstance(value, float) and math.isnan(value):
        raise LabelValueError(f"Label '{label}' has NaN value")

    # Check string "nan" or "NaN"
    if isinstance(value, str) and value.lower() == "nan":
        raise LabelValueError(f"Label '{label}' has NaN string value")

    label_type = LABEL_TYPES.get(label)
    if label_type is None:
        raise LabelTypeError(f"Unknown label type for '{label}'")

    try:
        if label_type == "binary":
            return _to_bool(value, label)
        elif label_type == "ordinal":
            return _to_int(value, label)
        elif label_type == "categorical":
            return _to_int(value, label)
        elif label_type == "multi_categorical":
            return _to_tuple_of_int(value, label)
        elif label_type == "continuous":
            return _to_float(value, label)
        else:
            raise LabelTypeError(f"Unknown label type '{label_type}' for label '{label}'")
    except (ValueError, TypeError, KeyError) as e:
        raise LabelTypeError(
            f"Cannot convert value '{value}' to {label_type} for label '{label}': {e}"
        )


def _to_bool(value: Any, label: str) -> bool:
    """Convert value to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1, 0.0, 1.0):
            return bool(value)
        raise LabelTypeError(f"Cannot convert numeric value '{value}' to bool for label '{label}'")
    if isinstance(value, str):
        # Handle common string representations
        lower = value.lower().strip()
        if lower in ("true", "1", "1.0", "yes", "male"):
            return True
        if lower in ("false", "0", "0.0", "no", "female"):
            return False
        raise LabelTypeError(f"Cannot convert string '{value}' to bool for label '{label}'")
    raise LabelTypeError(f"Cannot convert {type(value).__name__} to bool for label '{label}'")


def _to_int(value: Any, label: str) -> int:
    """Convert value to int."""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        raise LabelTypeError(
            f"Cannot convert non-integer float '{value}' to int for label '{label}'"
        )
    if isinstance(value, str):
        try:
            # Try to parse as float first, then convert to int
            parsed = float(value)
            if parsed.is_integer():
                return int(parsed)
        except (ValueError, TypeError):  # noqa: BLE001
            # if not directly convertible into int, then search in ordinal dictionary
            ord_dict = _load_ordinal_dictionary(ordinal_dictionary_path=_PATHS.ordinal_dictionary)
            return int(ord_dict[label][value])
        # search in dictionary if not directly convertible
        raise LabelTypeError(
            f"Cannot convert non-integer string '{value}' to int for label '{label}'"
        )
    raise LabelTypeError(f"Cannot convert {type(value).__name__} to int for label '{label}'")


def _to_float(value: Any, label: str) -> float:
    """Convert value to float."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        # date times have been converted to categories
        # # Handle datetime strings for GoSleepTime/WakeUpTime by extracting hour as decimal
        # if label in ("GoSleepTime", "WakeUpTime"):
        #     try:
        #         ts = pd.Timestamp(value)
        #         # Convert to decimal hours (e.g., 6:30 AM = 6.5)
        #         return float(ts.hour + ts.minute / 60.0 + ts.second / 3600.0)
        #     except Exception:
        #         pass
        return float(value)
    raise LabelTypeError(f"Cannot convert {type(value).__name__} to float for label '{label}'")


def _to_tuple_of_int(value: Any, label: str) -> tuple[int, ...]:
    """Convert a list/tuple of values to a sorted tuple of ints.

    Used for the ``multi_categorical`` label type. Sorted ascending so that
    ``{1, 3}`` and ``{3, 1}`` compare equal regardless of source order.

    Args:
        value: The raw value from storage; must be a list or tuple.
        label: Label name (only used in error messages).

    Returns:
        Sorted tuple of integer option codes.

    Raises:
        LabelTypeError: If ``value`` is not a list/tuple, is empty, contains
            a bool, contains a non-integer float, or contains a non-numeric.
    """
    if not isinstance(value, (list, tuple)):
        raise LabelTypeError(
            f"multi_categorical label '{label}' expects list/tuple, got {type(value).__name__}"
        )
    if len(value) == 0:
        raise LabelTypeError(f"multi_categorical label '{label}' has empty selection")
    coerced: list[int] = []
    for elem in value:
        if isinstance(elem, bool):
            raise LabelTypeError(f"multi_categorical '{label}' element is bool: {elem!r}")
        if isinstance(elem, int):
            coerced.append(elem)
        elif isinstance(elem, float):
            if not elem.is_integer():
                raise LabelTypeError(
                    f"multi_categorical '{label}' element is non-integer float: {elem}"
                )
            coerced.append(int(elem))
        else:
            raise LabelTypeError(
                f"multi_categorical '{label}' element has unsupported type "
                f"{type(elem).__name__}: {elem!r}"
            )
    return tuple(sorted(coerced))


# Derive label names from label_types.json (insertion order preserved).
LABEL_NAMES: list[str] = []
TARGET_NAMES: list[str] = []
CONTEXT_NAMES: list[str] = []
MULTI_CATEGORICAL_NAMES: list[str] = []


def _apply_metadata(types: dict, meta: dict) -> None:
    """Refresh the imported metadata containers without replacing their identities."""
    LABEL_TYPES.clear()
    LABEL_TYPES.update(types)
    _LABEL_META.clear()
    _LABEL_META.update(meta)
    LABEL_NAMES[:] = types
    TARGET_NAMES[:] = [k for k, v in meta.items() if v.get("target", True)]
    CONTEXT_NAMES[:] = [k for k, v in meta.items() if not v.get("target", True)]
    MULTI_CATEGORICAL_NAMES[:] = [
        k for k, v in meta.items() if v.get("type") == "multi_categorical"
    ]


_apply_metadata(LABEL_TYPES.copy(), _LABEL_META.copy())


_VALID_AGGREGATIONS = ("mean", "median", "min", "max", "std", "last", "first")


@dataclass(frozen=True)
class LabelResult:
    """Result of a label lookup operation."""

    matched_timestamp: pd.Timestamp
    value: Any


@dataclass(frozen=True)
class LabelWindowResult:
    """Result of a windowed label query."""

    value: Any
    n_points: int
    window_start: pd.Timestamp
    window_end: pd.Timestamp


class LabelSeries:
    """A single time series for a healthCode/label pair."""

    __slots__ = ("timestamps_ns", "values", "valid")

    def __init__(self, timestamps_ns: list[int], values: list[Any]) -> None:
        """Initialize a LabelSeries with timestamps and values."""
        self.timestamps_ns = timestamps_ns
        self.values = values
        self.valid: list[bool] | None = None

    def nearest(self, target: pd.Timestamp) -> LabelResult:
        """Find the nearest timestamp to the target timestamp.

        Args:
            target: Target timestamp to find nearest match for.

        Returns:
            LabelResult containing the matched timestamp and corresponding value.

        Raises:
            LookupError: If no timestamps are available.
        """
        if not self.timestamps_ns:
            return LabelResult(matched_timestamp=None, value=self.values[0])
        if len(self.timestamps_ns) == 1:
            matched = pd.Timestamp(self.timestamps_ns[0])
            return LabelResult(matched_timestamp=matched, value=self.values[0])

        target_ns = int(target.value)

        insert_at = bisect_left(self.timestamps_ns, target_ns)
        index = _select_nearest_index(self.timestamps_ns, target_ns, insert_at)
        matched = pd.Timestamp(self.timestamps_ns[index])
        return LabelResult(matched_timestamp=matched, value=self.values[index])

    def nearest_valid(self, target: pd.Timestamp) -> LabelResult:
        """Find the nearest timestamp among valid measurements only.

        Scans all measurements marked valid and returns the one closest to
        *target*.  Raises ``KeyError`` when no valid measurements exist.
        """
        if not self.timestamps_ns:
            # Static label with a single validity flag
            if self.valid and self.valid[0]:
                return LabelResult(matched_timestamp=None, value=self.values[0])
            raise KeyError("No valid measurements for this label/healthCode")
        target_ns = int(target.value)
        best_idx: int | None = None
        best_dist = float("inf")
        for i, (ts_ns, v) in enumerate(zip(self.timestamps_ns, self.valid)):
            if v:
                dist = abs(ts_ns - target_ns)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
        if best_idx is None:
            raise KeyError("No valid measurements for this label/healthCode")
        matched = pd.Timestamp(self.timestamps_ns[best_idx])
        return LabelResult(matched_timestamp=matched, value=self.values[best_idx])

    def windowed(
        self,
        target: pd.Timestamp,
        window_days: int,
        aggregation: str = "median",
    ) -> LabelWindowResult:
        """Aggregate all measurements within ±window_days of *target*.

        Args:
            target: Center of the query window.
            window_days: Half-width in days.  0 means same calendar day only.
            aggregation: One of "mean", "median", "min", "max", "std",
                "last", "first".

        Raises:
            KeyError: If no measurements fall within the window.
            ValueError: If aggregation method is unknown.
        """
        if aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f"Unknown aggregation '{aggregation}', expected one of {_VALID_AGGREGATIONS}"
            )
        if not self.timestamps_ns:
            raise KeyError("No timestamps available for windowed query")

        ns_per_day = 86_400_000_000_000
        target_ns = int(target.value)
        half_window_ns = window_days * ns_per_day + (ns_per_day - 1)

        lo_ns = target_ns - half_window_ns
        hi_ns = target_ns + half_window_ns

        lo_idx = bisect_left(self.timestamps_ns, lo_ns)
        hi_idx = bisect_right(self.timestamps_ns, hi_ns)

        if lo_idx >= hi_idx:
            raise KeyError("No measurements in window")

        window_vals = [
            float(v)
            for v in self.values[lo_idx:hi_idx]
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ]
        if not window_vals:
            raise KeyError("No non-null measurements in window")

        if aggregation == "median":
            agg_val = statistics.median(window_vals)
        elif aggregation == "mean":
            agg_val = statistics.mean(window_vals)
        elif aggregation == "min":
            agg_val = min(window_vals)
        elif aggregation == "max":
            agg_val = max(window_vals)
        elif aggregation == "std":
            if len(window_vals) < 2:
                raise KeyError("Need >=2 points for std aggregation")
            agg_val = statistics.stdev(window_vals)
        elif aggregation == "last":
            agg_val = window_vals[-1]
        elif aggregation == "first":
            agg_val = window_vals[0]

        return LabelWindowResult(
            value=agg_val,
            n_points=len(window_vals),
            window_start=pd.Timestamp(self.timestamps_ns[lo_idx]),
            window_end=pd.Timestamp(self.timestamps_ns[hi_idx - 1]),
        )


class LabelsIndex:
    """Immutable index mapping labels -> healthCode -> series."""

    def __init__(self, index: dict[str, dict[str, LabelSeries]]) -> None:
        """Initialize the labels index."""
        self._index = index

    def get_series(self, label: str, health_code: str) -> LabelSeries:
        """Get the label series for a specific label and health code.

        Args:
            label: Name of the label.
            health_code: Health code identifier.

        Returns:
            LabelSeries containing timestamps and values.

        Raises:
            KeyError: If label or health_code is not found.
        """
        per_label = self._index.get(label)
        if per_label is None:
            raise KeyError(f"Unknown label: {label}")

        series = per_label.get(health_code)
        if series is None:
            raise KeyError(f"Unknown healthCode for label {label}: {health_code}")

        return series


class EnrollmentIndex:
    """Thin wrapper around enrollment data for future use."""

    def __init__(self, enrollment: dict[str, dict[str, Any]]) -> None:
        """Initialize the enrollment index."""
        if not isinstance(enrollment, dict):
            raise ValueError("Enrollment metadata must be a JSON object keyed by user ID")
        self._enrollment = enrollment
        self._has_birth_years = any(
            isinstance(record, dict)
            and (record.get("birth_year") is not None or record.get("birthdate") is not None)
            for record in enrollment.values()
        )

    def get(self, health_code: str) -> dict[str, Any] | None:
        """Get enrollment information for a health code.

        Args:
            health_code: Health code identifier.

        Returns:
            Dictionary with enrollment information, or None if not found.
        """
        return self._enrollment.get(health_code)

    def get_birth_year(self, health_code: str) -> int:
        """Get birth year for a health code.

        Args:
            health_code: Health code identifier.

        Returns:
            Birth year as an integer.

        Raises:
            KeyError: If health code not found or no birth year available.
        """
        user = self.get(health_code)
        if user is None:
            raise KeyError(f"Unknown healthCode in enrollment: {health_code}")
        birth_year = user.get("birth_year")
        if birth_year is None:
            # Transition fallback for pre-migration enrollment fixtures/files.
            birthdate = user.get("birthdate")
            if birthdate is None:
                if self._has_birth_years:
                    raise KeyError(
                        f"Enrollment record lacks birth_year for healthCode: {health_code}. "
                        "Other records in this file have it, so this is a gap in this record "
                        "rather than an outdated release."
                    )
                raise KeyError(
                    f"Enrollment metadata lacks birth_year for healthCode: {health_code}. "
                    "Download a dataset release containing birth_year into a new directory "
                    "and call labels.configure(data_dir=...). Stored age cannot replace birth_year."
                )
            return int(pd.Timestamp(birthdate).year)
        return int(birth_year)

    def age_at(self, health_code: str, date: pd.Timestamp) -> int:
        """Return calendar-year age at an explicit date from enrolment birth year.

        The supplied timestamp's year is used without timezone conversion.
        Missing participants or birth years raise KeyError. Invalid timestamps,
        invalid birth years and dates before the birth year raise ValueError.
        """
        if not isinstance(date, pd.Timestamp) or pd.isna(date):
            raise ValueError("date must be a valid pandas.Timestamp")
        record = self.get(health_code)
        if record is not None and not isinstance(record, dict):
            raise ValueError("Invalid enrollment record")
        raw_year = record.get("birth_year") if record is not None else None
        if isinstance(raw_year, bool) or (
            isinstance(raw_year, float) and not raw_year.is_integer()
        ):
            raise ValueError("Birth year must be an integer")
        try:
            birth_year = self.get_birth_year(health_code)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid birth year in enrollment metadata") from exc
        if not 1 <= birth_year <= 9999:
            raise ValueError("Birth year must be between 1 and 9999")
        if date.year < birth_year:
            raise ValueError("date is before the participant's birth year")
        return years_between_birth_year(birth_year, date)

    def age_at_enrollment(self, health_code: str) -> int:
        """Return calendar-year age at the participant's enrollment createdOn.

        Missing records or createdOn raise KeyError; malformed dates raise
        ValueError. Stored age is never used as a fallback.
        """
        record = self.get(health_code)
        if record is not None and not isinstance(record, dict):
            raise ValueError("Invalid enrollment record")
        created_on = record.get("createdOn") if record is not None else None
        if created_on is None:
            raise KeyError(f"No enrollment createdOn for healthCode: {health_code}")
        if not isinstance(created_on, (str, pd.Timestamp)):
            raise ValueError("Enrollment createdOn must be a timestamp")
        try:
            date = pd.Timestamp(created_on)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("Invalid enrollment createdOn") from exc
        return self.age_at(health_code, date)

    def get_birthdate(self, health_code: str) -> pd.Timestamp:
        """Get a synthetic birthdate for compatibility.

        Enrollment data stores only birth year.  This method returns January 1
        of that year so legacy consumers that expect a Timestamp keep running;
        the returned month/day is not the user's real birth month/day.
        """
        return pd.Timestamp(year=self.get_birth_year(health_code), month=1, day=1)


class LabelsStore:
    """Lazily loads label and enrollment data and provides nearest-time lookups."""

    def __init__(
        self,
        paths: LabelPaths,
    ) -> None:
        """Initialize the labels store with the resolved configuration."""
        self.paths = paths
        self._labels_index: LabelsIndex | None = None
        self._enrollment: EnrollmentIndex | None = None
        self._validity_attached: bool = False
        self._daily_index: LabelsIndex | None = None

    def configure(self, paths: LabelPaths) -> None:
        """Replace paths and discard all cached payloads, keeping this store."""
        self.paths = paths
        self._labels_index = None
        self._enrollment = None
        self._validity_attached = False
        self._daily_index = None

    def get_labels(self, health_code: str, timestamp: pd.Timestamp, label: str) -> LabelResult:
        """Get label value for a health code at a specific timestamp.

        Args:
            health_code: Health code identifier.
            timestamp: Timestamp to query.
            label: Label name to retrieve.

        Returns:
            LabelResult with matched timestamp and value.
        """
        series = self.labels_index.get_series(label, health_code)
        return series.nearest(timestamp)

    @property
    def labels_index(self) -> LabelsIndex:
        """Lazy-loaded labels index.

        Merges ``last_labels.json`` (targets) with ``context_labels.json``
        (context variables) if the latter exists.  For overlapping labels,
        ``last_labels.json`` takes precedence.
        """
        if self._labels_index is None:
            index = _load_labels(
                _require_dataset_payload_path(
                    self.paths.labels,
                    env_var="LABELS_DATA_PATH",
                    filename="last_labels.json",
                )
            )
            if self.paths.context is not None and self.paths.context.exists():
                ctx = _load_labels(self.paths.context)
                for label, per_user in ctx.items():
                    if label not in index:
                        index[label] = per_user
                    else:
                        # Merge users, targets take precedence per-user
                        for user, series in per_user.items():
                            if user not in index[label]:
                                index[label][user] = series
            self._labels_index = LabelsIndex(index)
        return self._labels_index

    @property
    def enrollment(self) -> EnrollmentIndex:
        """Lazy-loaded enrollment index.

        Returns:
            EnrollmentIndex containing enrollment data.
        """
        if self._enrollment is None:
            self._enrollment = EnrollmentIndex(
                _load_enrollment(
                    _require_dataset_payload_path(
                        self.paths.enrollment,
                        env_var="ENROLLMENT_DATA_PATH",
                        filename="enrollment_info.json",
                    )
                )
            )
        return self._enrollment

    @property
    def daily_index(self) -> LabelsIndex | None:
        """Lazy-loaded daily HealthKit/happiness index.

        Returns None if healthkit_daily.json does not exist.
        """
        if self._daily_index is None:
            if self.paths.daily is not None and self.paths.daily.exists():
                self._daily_index = LabelsIndex(_load_labels(self.paths.daily))
        return self._daily_index

    def _ensure_validity_loaded(self) -> None:
        """Load label_validity.json and attach masks to LabelSeries instances.

        Called once on first ``return_valid_only=True`` query.  If the
        validity file does not exist, all series keep ``valid=None``
        (graceful degradation — behaves like ``return_valid_only=False``).
        """
        if self._validity_attached:
            return
        self._validity_attached = True
        if self.paths.validity is None or not self.paths.validity.exists():
            return
        validity_data = _load_validity(self.paths.validity)
        raw_index = self.labels_index._index  # noqa: SLF001
        for label, per_user in validity_data.items():
            label_series = raw_index.get(label, {})
            for user, mask in per_user.items():
                series = label_series.get(user)
                if series is not None and (
                    len(mask) == len(series.timestamps_ns)
                    or (not series.timestamps_ns and len(mask) == 1)
                ):
                    series.valid = mask


def _select_nearest_index(timestamps: list[int], target: int, insert_at: int) -> int:
    previous_idx = insert_at - 1 if insert_at > 0 else None
    next_idx = insert_at if insert_at < len(timestamps) else None

    if previous_idx is None:
        return next_idx  # type: ignore[return-value]
    if next_idx is None:
        return previous_idx

    prev_diff = target - timestamps[previous_idx]
    next_diff = timestamps[next_idx] - target

    if next_diff < prev_diff:
        return next_idx
    return previous_idx


def _load_labels(labels_path: Path) -> dict[str, dict[str, LabelSeries]]:
    with labels_path.open("r") as handle:
        raw: dict[str, dict[str, dict[str, Any]]] = json.load(handle)

    return {label: _build_health_series(per_label) for label, per_label in raw.items()}


def _load_ordinal_dictionary(ordinal_dictionary_path: Path) -> dict[str, dict[str, LabelSeries]]:
    with ordinal_dictionary_path.open("r") as handle:
        raw = json.load(handle)
    # The release contains both value-to-code and code-to-description mappings.
    if not isinstance(raw, dict) or any(
        not isinstance(mapping, dict)
        or any(not isinstance(value, (str, int)) for value in mapping.values())
        for mapping in raw.values()
    ):
        raise ValueError(
            f"Ordinal dictionary must contain JSON mappings: {ordinal_dictionary_path}"
        )
    return raw


def _load_enrollment(enrollment_path: Path) -> dict[str, dict[str, Any]]:
    with enrollment_path.open("r") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid enrollment JSON: {enrollment_path}") from exc


def _load_validity(validity_path: Path) -> dict[str, dict[str, list[bool]]]:
    with validity_path.open("r") as handle:
        return json.load(handle)


def _build_health_series(per_label: dict[str, dict[str, Any]]) -> dict[str, LabelSeries]:
    return {health_code: _build_series(payload) for health_code, payload in per_label.items()}


def _build_series(entry: dict[str, Any]) -> LabelSeries:
    paired = _pair_sorted(entry.get("timestamps", []), entry.get("values", []))
    if not paired:
        return LabelSeries([], [])

    ts_ns, vals = zip(*paired)
    return LabelSeries(list(ts_ns), list(vals))


def _pair_sorted(timestamps: Iterable[str], values: Iterable[Any]) -> list[tuple[int, Any]]:
    ts_ns = [_to_epoch_ns(ts) for ts in timestamps]
    paired = list(zip(ts_ns, values))
    return sorted(paired, key=lambda item: item[0])


def _to_epoch_ns(ts: str) -> int:
    return int(pd.Timestamp(ts).value)


def years_between(birthdate: pd.Timestamp, at: pd.Timestamp) -> int:
    """Return whole years elapsed between birthdate and a reference timestamp."""
    years = at.year - birthdate.year
    has_had_birthday = (at.month, at.day) >= (birthdate.month, birthdate.day)
    return years if has_had_birthday else years - 1


def years_between_birth_year(birth_year: int, at: pd.Timestamp) -> int:
    """Return calendar-year age from a de-identified birth year."""
    return at.year - int(birth_year)


STORE = LabelsStore(_PATHS)


def _load_schemas(resolved: LabelPaths) -> tuple[dict, dict]:
    """Validate all small schemas without mutating live state or payload caches."""
    resolved.require_schemas()
    try:
        types, meta = _load_label_types(resolved.label_types)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid label schema {resolved.label_types}: {exc}") from exc
    try:
        _load_ordinal_dictionary(resolved.ordinal_dictionary)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid label schema {resolved.ordinal_dictionary}: {exc}") from exc
    try:
        with resolved.validity_config.open() as handle:
            validity = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid label schema {resolved.validity_config}: {exc}") from exc
    if not isinstance(validity, dict) or any(
        value is not None
        and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        )
        for value in validity.values()
    ):
        raise ValueError(
            f"Validity config must map labels to nonnegative days or null: {resolved.validity_config}"
        )
    return types, meta


def configure(data_dir: str | Path | None = None, *, paths: LabelPaths | None = None) -> None:
    """Select a dataset for labels and devices; refresh metadata and lazy caches.

    All three schemas come from the selected dataset's ``labels/`` directory.
    Conflicting schema environment paths raise an error. Payload overrides
    are resolved by ``LabelPaths``.
    Call before querying/evaluating a new dataset. References imported with
    ``from labels.api import STORE, LABEL_TYPES, CONTEXT_NAMES`` stay valid.
    Configuration is process-global; concurrent evaluations of different
    datasets require separate processes.
    """
    if paths is not None and data_dir is not None:
        raise ValueError("Pass data_dir or paths, not both")
    resolved = paths if paths is not None else LabelPaths.resolve(data_dir)
    # Validate before changing metadata or either store.
    types, meta = _load_schemas(resolved)

    global _PATHS, _SCHEMAS_CONFIGURED
    _PATHS = resolved
    _apply_metadata(types, meta)
    STORE.configure(resolved)
    from devices import api as devices_api

    devices_api.configure(paths=resolved)
    _SCHEMAS_CONFIGURED = True


def age_at(health_code: str, date: pd.Timestamp) -> int:
    """Return calendar-year age at an explicitly supplied pandas.Timestamp.

    Uses the configured dataset's enrolment birth year and returns
    ``date.year - birth_year``; this is not birthday-based age. The enrolment
    reader also accepts legacy records containing birthdate instead of birth_year.
    Missing participants or birth years raise KeyError. Invalid dates or birth
    years raise ValueError. Stored age labels are not read or modified.
    """
    _ensure_configured()
    return STORE.enrollment.age_at(health_code, date)


def age_at_enrollment(health_code: str) -> int:
    """Return age at enrollment createdOn using the configured birth year.

    Missing metadata raises KeyError and invalid metadata raises ValueError.
    """
    _ensure_configured()
    return STORE.enrollment.age_at_enrollment(health_code)


AGE_GROUP_EDGES = (18, 30, 40, 50, 60)


def _age_group(age: int | None) -> str:
    """Return the group for an age; missing and under-18 ages are unknown."""
    if age is None or age < AGE_GROUP_EDGES[0]:
        return "unknown"
    for low, high in zip(AGE_GROUP_EDGES, AGE_GROUP_EDGES[1:]):
        if age < high:
            return f"{low}-{high - 1}"
    return f"{AGE_GROUP_EDGES[-1]}+"


def _sex_group(value: Any) -> str:
    """Return male or female for a BiologicalSex value, otherwise unknown."""
    if isinstance(value, bool):
        return "male" if value else "female"
    if isinstance(value, str) and value.strip().lower() in ("male", "female"):
        return value.strip().lower()
    return "unknown"


def user_demographics(
    user_ids: Iterable[str], *, paths: LabelPaths | None = None
) -> dict[str, dict[str, str]]:
    """Return ``{user_id: {"age_group": ..., "sex": ...}}``, one group per participant.

    This is the single definition of the reporting groups for all tracks. Age
    group uses calendar-year age at enrollment; sex is the latest BiologicalSex
    label. Missing or invalid values, including malformed records, give ``unknown``
    and keep the user. The unknown age count is logged, as is a cohort with no usable
    sex label. Unreadable or malformed files raise.

    Args:
        user_ids: Users to look up.
        paths: Read these files instead of the configured dataset, leaving the
            configured dataset unchanged.
    """
    if paths is None:
        _ensure_configured()
        store = STORE
    else:
        store = LabelsStore(paths)
    # Load outside the per-user tries: a broken file is not a missing value.
    enrollment = store.enrollment
    labels_index = store.labels_index

    groups = {}
    for user_id in sorted({str(user) for user in user_ids}):
        try:
            age = enrollment.age_at_enrollment(user_id)
        except (KeyError, ValueError):
            age = None
        try:
            sex_values = labels_index.get_series("BiologicalSex", user_id).values
        except KeyError:
            sex_values = []
        groups[user_id] = {
            "age_group": _age_group(age),
            "sex": _sex_group(sex_values[-1] if sex_values else None),
        }
    unknown = sum(row["age_group"] == "unknown" for row in groups.values())
    if unknown:
        logger.warning(
            "%d/%d users have unknown age groups. Check birth_year and createdOn "
            "in enrollment_info.json; ages below 18 are also unknown.%s",
            unknown,
            len(groups),
            " The cohort has no usable age groups." if unknown == len(groups) else "",
        )
    if groups and all(row["sex"] == "unknown" for row in groups.values()):
        logger.warning(
            "No user has a usable BiologicalSex label. Check last_labels.json; "
            "sex cohorts will be empty."
        )
    return groups


def get_labels(
    health_code: str,
    timestamp: pd.Timestamp,
    label: str,
    enforce_type: bool = True,
    return_valid_only: bool = True,
) -> LabelResult:
    """Return the nearest-in-time label value for a healthCode.

    The closest timestamp is selected, breaking ties in favor of the earlier
    observation.

    When *return_valid_only* is ``True`` (default) and a ``label_validity.json``
    file is available, only measurements that have co-located wearable data
    (within the threshold defined in ``validity_config.json``) are considered.
    If no valid measurement exists for the requested user+label, a ``KeyError``
    is raised — which existing consumers already handle by excluding the user.

    Args:
        health_code: The user's health code identifier
        timestamp: The target timestamp for nearest-match lookup
        label: The label name to retrieve
        enforce_type: If True (default), enforce type conversion based on label_types.json:
            - binary -> bool
            - ordinal -> int
            - categorical -> str
            - continuous -> float
        return_valid_only: If True (default), only return measurements with
            nearby wearable data.  Pass False to get the old behaviour
            (nearest measurement regardless of validity).

    Raises:
        ValueError: If label is unknown
        KeyError: If healthCode not found, or no valid measurements exist
        LabelValueError: If value is None or NaN
        LabelTypeError: If value cannot be converted to expected type
    """
    _ensure_configured()
    if label not in LABEL_NAMES:
        raise ValueError(f"Unknown label: {label}")

    if return_valid_only:
        STORE._ensure_validity_loaded()

    series = STORE.labels_index.get_series(label, health_code)

    if return_valid_only and series.valid is not None:
        result = series.nearest_valid(timestamp)
    else:
        result = series.nearest(timestamp)

    if enforce_type:
        enforced_value = _enforce_type(label, result.value)
        return LabelResult(matched_timestamp=result.matched_timestamp, value=enforced_value)
    return result


# Labels that have daily-resolution data in healthkit_daily.json
DAILY_LABELS: set[str] = {
    "Watch_RestingHeartRate",
    "Watch_VO2Max",
    "Watch_HeartRateVariabilitySDNN",
    "Watch_WalkingHeartRateAverage",
    "Watch_StandTime",
    "Watch_BasalEnergyBurned",
    "Watch_RespiratoryRate",
    "happiness",
}


def get_labels_windowed(
    health_code: str,
    timestamp: pd.Timestamp,
    label: str,
    window_days: int = 90,
    aggregation: str = "median",
) -> LabelWindowResult:
    """Return an aggregated label value over a time window.

    For labels with daily-resolution data (Watch_* HealthKit metrics and
    happiness), queries ``healthkit_daily.json`` which has one value per day.
    For other labels, falls back to the standard ``last_labels.json`` series.

    Args:
        health_code: The user's health code identifier.
        timestamp: Center of the query window.
        label: The label name to retrieve.
        window_days: Half-width of the window in days.  ``0`` means
            same calendar day only.
        aggregation: Aggregation method — one of ``"mean"``, ``"median"``,
            ``"min"``, ``"max"``, ``"std"``, ``"last"``, ``"first"``.

    Returns:
        LabelWindowResult with the aggregated value and metadata.

    Raises:
        ValueError: If label is unknown or aggregation method is invalid.
        KeyError: If healthCode not found or no measurements in window.
    """
    _ensure_configured()
    if label not in LABEL_NAMES and label not in DAILY_LABELS:
        raise ValueError(f"Unknown label: {label}")

    if LABEL_TYPES.get(label) == "multi_categorical":
        raise ValueError(
            f"get_labels_windowed does not support multi_categorical labels "
            f"(got '{label}'); use get_labels_one_hot/count/contains instead."
        )

    # Prefer daily-resolution index for supported labels
    series = None
    if label in DAILY_LABELS and STORE.daily_index is not None:
        try:
            series = STORE.daily_index.get_series(label, health_code)
        except KeyError:
            pass

    # Fall back to standard labels index
    if series is None:
        series = STORE.labels_index.get_series(label, health_code)

    return series.windowed(timestamp, window_days, aggregation)


def get_labels_one_hot(
    health_code: str,
    timestamp: pd.Timestamp,
    label: str,
    *,
    options: list[int] | None = None,
    return_valid_only: bool = True,
) -> dict[int, bool]:
    """Decode a multi_categorical label as ``{option_code: was_selected}``.

    Args:
        health_code: User identifier.
        timestamp: Query timestamp (forwarded to nearest-match lookup).
        label: A label of type ``multi_categorical``.
        options: Explicit option codes for deterministic key order. If
            ``None``, defaults to the integer keys of
            ``ORDINAL_DICTIONARY[label]``.
        return_valid_only: Forwarded to :func:`get_labels`.

    Returns:
        Dictionary mapping each option code to ``True`` if the user's
        selection contained that code, else ``False``.

    Raises:
        ValueError: If ``label`` is not a ``multi_categorical`` label.
    """
    _ensure_configured()
    if LABEL_TYPES.get(label) != "multi_categorical":
        raise ValueError(
            f"get_labels_one_hot only supports multi_categorical labels, "
            f"got '{label}' (type={LABEL_TYPES.get(label)!r})"
        )
    selected = get_labels(
        health_code,
        timestamp,
        label,
        enforce_type=True,
        return_valid_only=return_valid_only,
    ).value
    if options is None:
        ord_dict = _load_ordinal_dictionary(_PATHS.ordinal_dictionary)
        options = sorted(int(k) for k in ord_dict.get(label, {}).keys())
    s = set(selected)
    return {opt: (opt in s) for opt in options}


def get_labels_count(
    health_code: str,
    timestamp: pd.Timestamp,
    label: str,
    *,
    return_valid_only: bool = True,
) -> int:
    """Return the number of options selected for a multi_categorical label.

    Args:
        health_code: User identifier.
        timestamp: Query timestamp (forwarded to nearest-match lookup).
        label: A label of type ``multi_categorical``.
        return_valid_only: Forwarded to :func:`get_labels`.

    Returns:
        Length of the selected tuple.

    Raises:
        ValueError: If ``label`` is not a ``multi_categorical`` label.
    """
    _ensure_configured()
    if LABEL_TYPES.get(label) != "multi_categorical":
        raise ValueError(
            f"get_labels_count only supports multi_categorical labels, "
            f"got '{label}' (type={LABEL_TYPES.get(label)!r})"
        )
    return len(
        get_labels(
            health_code,
            timestamp,
            label,
            enforce_type=True,
            return_valid_only=return_valid_only,
        ).value
    )


def get_labels_contains(
    health_code: str,
    timestamp: pd.Timestamp,
    label: str,
    option: int,
    *,
    return_valid_only: bool = True,
) -> bool:
    """Return True iff the given option was selected for a multi_categorical label.

    Args:
        health_code: User identifier.
        timestamp: Query timestamp (forwarded to nearest-match lookup).
        label: A label of type ``multi_categorical``.
        option: Option code to test for membership.
        return_valid_only: Forwarded to :func:`get_labels`.

    Returns:
        ``True`` if ``option`` is in the user's selected tuple.

    Raises:
        ValueError: If ``label`` is not a ``multi_categorical`` label.
    """
    _ensure_configured()
    if LABEL_TYPES.get(label) != "multi_categorical":
        raise ValueError(
            f"get_labels_contains only supports multi_categorical labels, "
            f"got '{label}' (type={LABEL_TYPES.get(label)!r})"
        )
    selected = get_labels(
        health_code,
        timestamp,
        label,
        enforce_type=True,
        return_valid_only=return_valid_only,
    ).value
    return option in selected


def get_labels_statistics() -> pd.DataFrame:
    """Return a DataFrame with statistics for all available labels.

    For each label, includes: data type, min value, max value, median value,
    and count of unique values.
    """
    _ensure_configured()
    labels_data = _load_labels(
        _require_dataset_payload_path(
            _PATHS.labels,
            env_var="LABELS_DATA_PATH",
            filename="last_labels.json",
        )
    )
    stats_data = []

    for label in sorted(LABEL_NAMES):
        # Numeric statistics are meaningless for multi-select tuple values;
        # use get_labels_one_hot/count/contains instead.
        if LABEL_TYPES.get(label) == "multi_categorical":
            continue
        all_values = _collect_all_values(labels_data, label)
        if not all_values:
            stats_data.append(
                {
                    "label": label,
                    "type": "N/A",
                    "min": pd.NA,
                    "max": pd.NA,
                    "median": pd.NA,
                    "unique": 0,
                }
            )
            continue

        stats = _calculate_label_statistics(all_values)
        stats_data.append(
            {
                "label": label,
                "type": stats["type"],
                "min": pd.to_numeric(stats["min"], errors="coerce")
                if stats["min"] != "N/A"
                else pd.NA,
                "max": pd.to_numeric(stats["max"], errors="coerce")
                if stats["max"] != "N/A"
                else pd.NA,
                "median": pd.to_numeric(stats["median"], errors="coerce")
                if stats["median"] != "N/A"
                else pd.NA,
                "unique": int(stats["unique"]),
            }
        )

    return pd.DataFrame(stats_data)


def print_labels_statistics() -> None:
    """Print a table showing statistics for all available labels.

    For each label, displays: data type, min value, max value, median value,
    and count of unique values.
    """
    df = get_labels_statistics()

    print(f"{'Label':<25} {'Type':<12} {'Min':<12} {'Max':<12} {'Median':<12} {'Unique':<8}")
    print("-" * 85)

    for _, row in df.iterrows():
        min_val = f"{row['min']:.2f}" if pd.notna(row["min"]) else "N/A"
        max_val = f"{row['max']:.2f}" if pd.notna(row["max"]) else "N/A"
        median_val = f"{row['median']:.2f}" if pd.notna(row["median"]) else "N/A"

        print(
            f"{row['label']:<25} {row['type']:<12} {min_val:<12} {max_val:<12} {median_val:<12} {row['unique']:<8}"
        )


def _collect_all_values(labels_data: dict[str, dict[str, LabelSeries]], label: str) -> list[Any]:
    """Collect all values for a given label across all health codes."""
    all_values = []
    per_label = labels_data.get(label, {})

    for health_code_data in per_label.values():
        all_values.extend(health_code_data.values)

    return all_values


def _calculate_label_statistics(values: list[Any]) -> dict[str, str]:
    """Calculate statistics for a list of label values."""
    if not values:
        return {"type": "N/A", "min": "N/A", "max": "N/A", "median": "N/A", "unique": "0"}

    # Determine data type
    types = [type(v).__name__ for v in values]
    most_common_type = Counter(types).most_common(1)[0][0]
    data_type = most_common_type if len(set(types)) == 1 else "mixed"

    # Count unique values
    unique_count = len(set(str(v) for v in values))

    # For numeric statistics, try to convert to numeric
    try:
        numeric_series = pd.Series(pd.to_numeric(values, errors="coerce"))
        numeric_values = numeric_series.dropna()

        if len(numeric_values) == 0:
            return {
                "type": data_type,
                "min": "N/A",
                "max": "N/A",
                "median": "N/A",
                "unique": str(unique_count),
            }

        min_val = f"{numeric_values.min():.2f}"
        max_val = f"{numeric_values.max():.2f}"
        median_val = f"{numeric_values.median():.2f}"

    except (ValueError, TypeError):
        min_val = "N/A"
        max_val = "N/A"
        median_val = "N/A"

    return {
        "type": data_type,
        "min": min_val,
        "max": max_val,
        "median": median_val,
        "unique": str(unique_count),
    }
