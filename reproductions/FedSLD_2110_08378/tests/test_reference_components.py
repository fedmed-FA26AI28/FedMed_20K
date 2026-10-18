import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.nn.functional as F

from fedsld_reproduction.federated import aggregate_states
from fedsld_reproduction.loss import federation_prior, fedsld_loss
from fedsld_reproduction.model import PaperCNN
from fedsld_reproduction.partition import (
    partition_indices,
    pathological_plan,
    practical_plan,
)
from fedsld_reproduction.run import (
    _load_progress_checkpoint,
    _run_signature,
    _save_progress_checkpoint,
)


class FedSLDLossTests(unittest.TestCase):
    def test_prior_sums_client_counts(self):
        counts = torch.tensor([[2, 1], [0, 1]])
        prior = federation_prior(counts)
        torch.testing.assert_close(prior, torch.tensor([0.5, 0.5]))

    def test_loss_matches_equation(self):
        logits = torch.tensor([[2.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
        targets = torch.tensor([0, 0, 1])
        prior = torch.tensor([0.25, 0.75])
        batch_prior = torch.tensor([2 / 3, 1 / 3])
        weights = batch_prior[targets] / prior[targets]
        expected = torch.mean(F.cross_entropy(logits, targets, reduction="none") * weights)
        torch.testing.assert_close(fedsld_loss(logits, targets, prior), expected)


class PartitionTests(unittest.TestCase):
    def test_practical_plan_has_paper_shards(self):
        plan = practical_plan(8, seed=7)
        for row in plan.fractions:
            self.assertEqual(10, int(np.isclose(row, 0.01).sum()))
            self.assertEqual(1, int(np.isclose(row, 0.10).sum()))
            self.assertEqual(1, int(np.isclose(row, 0.80).sum()))

    def test_pathological_clients_have_two_classes(self):
        labels = np.repeat(np.arange(8), 120)
        plan = pathological_plan(8, seed=3)
        partitions = partition_indices(labels, plan, seed=3)
        observed = []
        for indices in partitions:
            observed.extend(indices)
            self.assertEqual(2, len(np.unique(labels[indices])))
        self.assertEqual(list(range(len(labels))), sorted(observed))

    def test_practical_partition_is_complete_and_disjoint(self):
        labels = np.repeat(np.arange(3), 1000)
        partitions = partition_indices(labels, practical_plan(3, seed=9), seed=9)
        observed = [index for values in partitions for index in values]
        self.assertEqual(list(range(len(labels))), sorted(observed))
        for values in partitions:
            self.assertEqual({0, 1, 2}, set(labels[values]))


class ModelAndAggregationTests(unittest.TestCase):
    def test_model_shape_for_gray_and_rgb(self):
        self.assertEqual((4, 11), tuple(PaperCNN(1, 11)(torch.randn(4, 1, 28, 28)).shape))
        self.assertEqual((4, 9), tuple(PaperCNN(3, 9)(torch.randn(4, 3, 28, 28)).shape))

    def test_sample_weighted_aggregation(self):
        states = [
            {"weight": torch.tensor([1.0, 3.0])},
            {"weight": torch.tensor([5.0, 7.0])},
        ]
        result = aggregate_states(states, [1, 3])
        torch.testing.assert_close(result["weight"], torch.tensor([4.0, 6.0]))


class ResumeCheckpointTests(unittest.TestCase):
    @staticmethod
    def _args():
        return Namespace(
            dataset="organamnist",
            algorithm="fedsld",
            partition="practical",
            num_clients=12,
            local_epochs=5,
            batch_size=256,
            learning_rate=0.01,
            proximal_mu=0.01,
            seed=42,
            exclude_official_val=False,
        )

    def test_checkpoint_roundtrip_uses_cpu_state(self):
        model = torch.nn.Linear(2, 2)
        signature = _run_signature(self._args(), rounds=80)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            _save_progress_checkpoint(
                path, signature, 3, model, 0.75, 2,
                {key: value.detach().clone() for key, value in model.state_dict().items()},
                [{"round": 1}, {"round": 2}, {"round": 3}], 12.5,
            )
            payload = _load_progress_checkpoint(path, signature)
        self.assertEqual(3, payload["completed_round"])
        self.assertEqual(3, len(payload["history"]))
        self.assertTrue(all(value.device.type == "cpu" for value in payload["model_state"].values()))

    def test_checkpoint_rejects_different_seed(self):
        model = torch.nn.Linear(2, 2)
        signature = _run_signature(self._args(), rounds=80)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            _save_progress_checkpoint(
                path, signature, 1, model, 0.5, 1, model.state_dict(),
                [{"round": 1}], 1.0,
            )
            changed = dict(signature)
            changed["seed"] = 43
            with self.assertRaisesRegex(ValueError, "configuration mismatch"):
                _load_progress_checkpoint(path, changed)


if __name__ == "__main__":
    unittest.main()

