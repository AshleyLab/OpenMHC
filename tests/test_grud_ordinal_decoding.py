"""Check decoder parity against the existing public wrappers, including crossings."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from downstream_evaluation.models.grud import _ordinal_class_prediction, _Trainer
from downstream_evaluation.models.registry import LogRegOrdinalWrapper, XGBOrdinalWrapper


class FixedProbability:
    """Supply fixed threshold probabilities to the public ordinal wrappers."""

    def __init__(self, p):
        """Store the positive-class probabilities."""
        self.p = p

    def predict_proba(self, X):
        """Return both binary class probabilities for each supplied row."""
        return np.column_stack([1 - self.p, self.p])


class OrdinalDecodingTests(unittest.TestCase):
    """Check decoder consistency and preserve binary/regression predictions."""

    def test_matches_both_public_wrappers_for_monotone_crossed_and_tied_inputs(self):
        """Match both wrappers even when cumulative probabilities cross or tie."""
        q = np.array(
            [
                [0.9, 0.8, 0.7],
                [0.8, 0.2, 0.1],
                [0.3, 0.2, 0.1],
                [0.5, 0.5, 0.5],
                [0.2, 0.8, 0.1],
                [0.1, 0.2, 0.9],
                [1, 1, 1],
                [0, 0, 0],
            ],
            dtype=np.float32,
        )
        for wrapper in [LogRegOrdinalWrapper({}), XGBOrdinalWrapper({})]:
            wrapper.levels = np.arange(4)
            wrapper.clfs = [FixedProbability(q[:, i]) for i in range(3)]
            np.testing.assert_array_equal(
                _ordinal_class_prediction(q), wrapper.predict(np.zeros((len(q), 1)))
            )

    def test_two_level_case(self):
        """Decode one threshold, including the tie-breaking case."""
        np.testing.assert_array_equal(
            _ordinal_class_prediction(np.array([[0.2], [0.8], [0.5]])), [0, 1, 0]
        )

    def _predict(self, logits, task_type):
        class FixedModel:
            """Return predetermined logits without training or loading weights."""

            def eval(self):
                """Accept the trainer's evaluation-mode call."""
                pass

            def __call__(self, batch):
                return {"task": logits}

        tr = _Trainer({}, [], {"task": logits.shape[1] + 1}, "cpu")
        tr.model = FixedModel()
        ds = SimpleNamespace(user_ids=[f"synthetic_{i}" for i in range(len(logits))])
        with patch("torch.utils.data.DataLoader", return_value=[{"n_users": len(logits)}]):
            result = tr.predict(ds, "task", task_type)
        return result, tr

    def test_actual_predict_branch_and_same_weights_diagnostic(self):
        """Exercise ordinal prediction and retain its threshold diagnostics."""
        import torch

        logits = torch.tensor([[2.0, 1.0, -0.2], [-2.0, -3.0, -4.0], [0.2, 1.0, -0.3]])
        result, tr = self._predict(logits, "ordinal")
        q = torch.sigmoid(logits)
        np.testing.assert_array_equal(result, _ordinal_class_prediction(q.numpy()))
        np.testing.assert_array_equal(
            tr.ordinal_prediction_audit["task"]["expected_level"], q.sum(dim=1).numpy()
        )
        np.testing.assert_array_equal(
            tr.ordinal_prediction_audit["task"]["threshold_probs"], q.numpy()
        )

    def test_binary_prediction_unchanged(self):
        """Keep binary outputs equal to positive-class softmax probabilities."""
        import torch

        logits = torch.tensor([[1.0, 2.0], [-1.0, -2.0]])
        result, _ = self._predict(logits, "binary")
        np.testing.assert_array_equal(result, torch.softmax(logits, dim=1)[:, 1].numpy())

    def test_regression_prediction_unchanged(self):
        """Keep regression outputs equal to the model's raw values."""
        import torch

        logits = torch.tensor([[2.3], [-4.5]])
        result, _ = self._predict(logits, "regression")
        np.testing.assert_array_equal(result, logits.squeeze(-1).numpy())


if __name__ == "__main__":
    unittest.main(verbosity=2)
