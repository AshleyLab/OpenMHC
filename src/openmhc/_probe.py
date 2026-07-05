"""The benchmark's standard linear probe.

Turns a model's per-participant embeddings into predictions using the *same* fixed
head every submission shares — PCA-50 followed by logistic regression (binary /
multiclass), a K-1 ordinal logistic decomposition (ordinal), or ordinary least
squares (regression). Because the head is fixed, a model's score reflects its
representation rather than its choice of classifier.

An encoder-style submission calls this inside ``predict`` so it returns predictions
(not embeddings) while staying comparable to every other encoder::

    class MyEncoder:
        def fit(self, data, labels, task_type):
            emb = np.stack([self._encode(x) for x in data])
            self._probe = openmhc.LinearProbe(task_type).fit(emb, labels)

        def predict(self, data):
            return self._probe.predict(np.stack([self._encode(x) for x in data]))
"""

from __future__ import annotations

import numpy as np


class LinearProbe:
    """PCA (up to 50 components) + a fixed linear head, selected by task type."""

    def __init__(self, task_type: str, n_components: int | None = 50, seed: int = 42) -> None:
        """Build the probe for ``task_type``.

        Args:
            task_type: one of ``"binary"``, ``"multiclass"``, ``"ordinal"``,
                ``"regression"``.
            n_components: PCA dimensionality before the head, capped at the data rank
                so small embeddings reduce gracefully (``None`` disables PCA). Defaults to 50.
            seed: random_state pinning PCA's randomized SVD solver and the
                classifier, so the probe is reproducible.
        """
        # Imported lazily so ``import openmhc`` stays light — the engine (config,
        # sklearn/xgboost) loads only when a probe is actually built.
        from downstream_evaluation.config import PROBE_BY_TASK_TYPE, ClassifierConfig
        from downstream_evaluation.models.registry import create_model

        if task_type not in PROBE_BY_TASK_TYPE:
            raise ValueError(
                f"task_type must be one of {sorted(PROBE_BY_TASK_TYPE)}, got {task_type!r}"
            )

        self.task_type = task_type
        config = ClassifierConfig(
            type=PROBE_BY_TASK_TYPE[task_type],
            use_scaler=False,  # encoders are probed on PCA features only; no scaler
            pca_n_components=n_components,
        )
        self._clf = create_model(config, random_state=seed, task_type=task_type)

    @staticmethod
    def _finite_rows(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return the float32 matrix and a mask of the rows that are entirely finite.

        A row with any NaN/±inf marks a participant the encoder could not represent. Such
        rows are excluded from fitting and predicted as NaN, so the harness can substitute
        the Linear baseline for them — the same "emit NaN where you cannot predict"
        convention the Forecaster protocol uses.
        """
        x = np.asarray(emb, dtype=np.float32)
        return x, np.isfinite(x).all(axis=1)

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> LinearProbe:
        """Fit the probe on training embeddings ``(n, D)`` and their labels ``(n,)``.

        Participants whose embedding is non-finite are dropped from the fit.
        """
        x, finite = self._finite_rows(embeddings)
        self._clf.fit(x[finite], np.asarray(labels)[finite])
        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Predict for ``(n, D)`` embeddings.

        Returns the class-1 probability for binary tasks and the point prediction
        otherwise (ordinal labels are integer-valued; regression is continuous). A
        participant whose embedding is non-finite is returned as NaN, so the harness scores
        it with the fallback rather than a fabricated value.
        """
        x, finite = self._finite_rows(embeddings)
        out = np.full(len(x), np.nan, dtype=np.float64)
        if finite.any():
            xv = x[finite]
            if self.task_type == "binary":
                out[finite] = self._clf.predict_proba(xv)[:, 1]
            else:
                out[finite] = self._clf.predict(xv)
        return out
