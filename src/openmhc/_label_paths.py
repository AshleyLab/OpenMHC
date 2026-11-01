"""Resolve dataset schemas and label/device payload paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from openmhc._dataset import data_dir


@dataclass(frozen=True)
class LabelPaths:
    """Label inputs resolved once for a dataset/configuration."""

    labels: Path | None
    context: Path | None
    enrollment: Path | None
    validity: Path | None
    daily: Path | None
    devices: Path | None
    schema_dir: Path | None

    @property
    def label_types(self) -> Path | None:
        """Path to the selected dataset's label types."""
        return self.schema_dir / "label_types.json" if self.schema_dir is not None else None

    @property
    def ordinal_dictionary(self) -> Path | None:
        """Path to the selected dataset's category dictionary."""
        return self.schema_dir / "ordinal_dictionary.json" if self.schema_dir is not None else None

    @property
    def validity_config(self) -> Path | None:
        """Path to the selected dataset's validity configuration."""
        return self.schema_dir / "validity_config.json" if self.schema_dir is not None else None

    @classmethod
    def resolve(
        cls,
        root: str | Path | None = None,
        *,
        labels: str | Path | None = None,
        enrollment: str | Path | None = None,
    ) -> LabelPaths:
        """Resolve schemas from the root and apply payload overrides.

        ``root`` takes precedence over ``MHC_DATA_DIR``. A missing root leaves
        paths unset, allowing imports before explicit API configuration. ``labels`` and
        ``enrollment`` support the forecasting fairness command's file flags.
        """
        try:
            labels_dir = data_dir(root) / "labels"
        except ValueError:
            labels_dir = None

        def resolve_file(
            env: str, filename: str, base: Path | None, explicit: str | Path | None = None
        ) -> Path | None:
            override = explicit if explicit is not None else os.getenv(env)
            if override:
                return Path(override).expanduser().resolve()
            return base / filename if base is not None else None

        return cls(
            labels=resolve_file("LABELS_DATA_PATH", "last_labels.json", labels_dir, labels),
            context=resolve_file("CONTEXT_LABELS_PATH", "context_labels.json", labels_dir),
            enrollment=resolve_file(
                "ENROLLMENT_DATA_PATH", "enrollment_info.json", labels_dir, enrollment
            ),
            validity=resolve_file("LABEL_VALIDITY_PATH", "label_validity.json", labels_dir),
            daily=resolve_file("HEALTHKIT_DAILY_PATH", "healthkit_daily.json", labels_dir),
            devices=resolve_file("USER_DEVICE_INFO_PATH", "user_device_info.json", labels_dir),
            schema_dir=labels_dir,
        )

    def require_schemas(self) -> tuple[Path, Path, Path]:
        """Require all dataset schemas and reject conflicting environment paths."""
        paths = (self.label_types, self.ordinal_dictionary, self.validity_config)
        if any(path is None for path in paths):
            raise ValueError(
                "A complete label schema set is required: label_types.json, "
                "ordinal_dictionary.json and validity_config.json. "
                "Call labels.configure(data_dir=...) / pass data_dir= to evaluation "
                "or set MHC_DATA_DIR. Individual schema paths do not select a dataset."
            )
        for env, path in zip(
            ("LABEL_TYPES_PATH", "ORDINAL_DICTIONARY_PATH", "VALIDITY_CONFIG_PATH"), paths
        ):
            override = os.getenv(env)
            if override and Path(override).expanduser().resolve() != path.resolve():
                raise ValueError(
                    f"{env}={override!r} conflicts with the selected dataset schema {path}. "
                    f"Unset {env}; select a complete dataset with data_dir= or MHC_DATA_DIR. "
                    "Schema files are loaded together from <dataset_root>/labels/."
                )
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Incomplete dataset label schemas; missing required file(s): "
                + ", ".join(missing)
                + ". Supply all three schemas in <dataset_root>/labels/; no repo fallback is used."
            )
        return paths
