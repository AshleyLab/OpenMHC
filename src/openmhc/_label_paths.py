"""Shared resolution of label and device paths.

Schema defaults stay repo-owned during the resolver migration. Participant
payloads come from the selected dataset; explicit per-file overrides win.
Resolution does not read payloads, so optional files remain lazily loaded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from openmhc._dataset import bundled_metadata_dir, data_dir


@dataclass(frozen=True)
class LabelPaths:
    """Label inputs resolved once for a dataset/configuration."""

    labels: Path | None
    context: Path | None
    enrollment: Path | None
    validity: Path | None
    daily: Path | None
    devices: Path | None
    label_types: Path
    ordinal_dictionary: Path
    validity_config: Path

    @classmethod
    def resolve(
        cls,
        root: str | Path | None = None,
        *,
        labels: str | Path | None = None,
        enrollment: str | Path | None = None,
    ) -> LabelPaths:
        """Resolve explicit paths, then file env overrides, then defaults.

        ``root`` takes precedence over ``MHC_DATA_DIR``. A missing root leaves
        participant paths unset, allowing metadata-only imports. ``labels`` and
        ``enrollment`` support the forecasting fairness command's file flags.
        This is the only place that reads label-path environment overrides.
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

        schema_dir = bundled_metadata_dir()
        return cls(
            labels=resolve_file("LABELS_DATA_PATH", "last_labels.json", labels_dir, labels),
            context=resolve_file("CONTEXT_LABELS_PATH", "context_labels.json", labels_dir),
            enrollment=resolve_file(
                "ENROLLMENT_DATA_PATH", "enrollment_info.json", labels_dir, enrollment
            ),
            validity=resolve_file("LABEL_VALIDITY_PATH", "label_validity.json", labels_dir),
            daily=resolve_file("HEALTHKIT_DAILY_PATH", "healthkit_daily.json", labels_dir),
            devices=resolve_file("USER_DEVICE_INFO_PATH", "user_device_info.json", labels_dir),
            label_types=resolve_file("LABEL_TYPES_PATH", "label_types.json", schema_dir),
            ordinal_dictionary=resolve_file(
                "ORDINAL_DICTIONARY_PATH", "ordinal_dictionary.json", schema_dir
            ),
            validity_config=resolve_file(
                "VALIDITY_CONFIG_PATH", "validity_config.json", schema_dir
            ),
        )
