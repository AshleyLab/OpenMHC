"""Fail-loud alignment of per-user features to a cohort.

Each embedding/feature model aligns a ``{user_id: vector}`` store to the cohort's
``user_ids``. A cohort user absent from the store would otherwise be scored from a
zero-filled row — a fabricated prediction that bypasses the Linear fallback — so the
models stop instead. This is the single tested implementation of that guard.
"""

from __future__ import annotations

import numpy as np


def raise_if_missing(model_name: str, missing, store_desc: str) -> None:
    """Raise if ``missing`` is non-empty, naming the model, the count, and the store.

    ``missing`` is the list of cohort user ids that have no entry in the feature store;
    ``store_desc`` names that store for the message (e.g. ``"embedding cache"``).
    """
    if missing:
        raise ValueError(
            f"{model_name}: {len(missing)} cohort user(s) are missing from the {store_desc} "
            f"(e.g. {list(missing)[:5]}); the cohort lookup and the {store_desc} are out of "
            "sync."
        )


def aligned_feature_matrix(model_name: str, user_ids, by_user, store_desc: str) -> np.ndarray:
    """Per-user feature matrix aligned to ``user_ids``; fails loud on a missing user.

    ``by_user`` maps ``str(user_id) -> feature vector`` (all the same length). Returns a
    ``(len(user_ids), dim)`` float32 matrix whose rows follow ``user_ids``.
    """
    missing = [str(u) for u in user_ids if str(u) not in by_user]
    raise_if_missing(model_name, missing, store_desc)
    dim = len(next(iter(by_user.values()))) if by_user else 0
    X = np.zeros((len(user_ids), dim), dtype=np.float32)
    for i, uid in enumerate(user_ids):
        X[i] = by_user[str(uid)]
    return X
