"""SCR is active through the Flower strategy and local training path."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import FitIns, ndarrays_to_parameters
from torch.utils.data import DataLoader, TensorDataset

from algorithms.base import FedStrategyBase
from algorithms.scr import SCRStrategy, training_counts_from_partition
from client.client import _local_train, _training_class_counts
from models.cnn import get_parameters


class FlatClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 3)

    def forward(self, x):
        return self.fc(x)


class FlowerSCRTests(unittest.TestCase):
    def test_strategy_uses_training_histograms_and_broadcasts_aggregates(self):
        partition = {"num_clients": 2, "total_samples": 7, "clients": {
            "0": {"num_samples": 3, "indices": [0, 1, 2],
                  "class_distribution": {"0": 2, "1": 0, "2": 1},
                  "test_class_distribution": {"0": 100, "1": 100, "2": 100}},
            "1": {"num_samples": 4, "indices": [3, 4, 5, 6],
                  "class_distribution": {"0": 0, "1": 3, "2": 1}},
        }}
        with tempfile.TemporaryDirectory() as root:
            file = Path(root) / "partition.json"
            file.write_text(json.dumps(partition), encoding="utf-8")
            np.testing.assert_array_equal(training_counts_from_partition(file, 3, 2),
                                          [[2, 0, 1], [0, 3, 1]])
            model = FlatClassifier()
            strategy = SCRStrategy(model=model, partition_path=file,
                                   num_total_clients=2, scr_gamma=1,
                                   scr_head_lambda=.01)
            fit = FitIns(ndarrays_to_parameters(get_parameters(model)),
                         {"strategy": "scr"})
            with patch.object(FedStrategyBase, "configure_fit", return_value=[("client", fit)]):
                sent = strategy.configure_fit(1, fit.parameters, None)
            config = sent[0][1].config
            self.assertEqual(config["strategy"], "scr")
            self.assertEqual(json.loads(config["scr_global_counts"]), [2, 3, 2])
            self.assertEqual(config["scr_total_samples"], 7)
            self.assertEqual(config["scr_num_clients"], 2)
            self.assertTrue(all(value > 0 for value in json.loads(config["scr_coverage_sum"])))
            partition["clients"]["0"]["class_distribution"]["0"] = 3
            file.write_text(json.dumps(partition), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                training_counts_from_partition(file, 3, 2)

    def test_local_training_uses_mean_weighted_ce_and_head_anchor(self):
        torch.manual_seed(7)
        model = FlatClassifier()
        images = torch.tensor([[1., 0.], [0., 1.]])
        labels = torch.tensor([0, 2])
        loader = DataLoader(TensorDataset(images, labels), batch_size=2)
        weights = torch.tensor([2., 0., 4.])
        anchor = torch.tensor([.2, 1., .8])
        reference = (model.fc.weight.detach().clone(), model.fc.bias.detach().clone())
        with torch.no_grad():
            model.fc.bias.add_(.1)
        logits = model(images)
        ce = F.cross_entropy(logits, labels, reduction="none")
        drift = (model.fc.weight - reference[0]).square().sum(1)
        drift += (model.fc.bias - reference[1]).square()
        expected = (weights[labels] * ce).mean() + .5 * .1 * (anchor * drift).sum() / 3
        metrics = _local_train(
            model, loader, torch.optim.SGD(model.parameters(), lr=0),
            torch.device("cpu"), epochs=1, scr_sample_weights=weights,
            scr_anchor=anchor, scr_reference=reference, scr_head_lambda=.1,
        )
        self.assertAlmostEqual(metrics["train_loss"], expected.item(), places=6)

    def test_training_counts_are_from_local_subset(self):
        dataset = TensorDataset(torch.randn(5, 2), torch.tensor([0, 1, 1, 2, 2]))
        subset = torch.utils.data.Subset(dataset, [0, 2, 4])
        loader = DataLoader(subset, batch_size=2)
        np.testing.assert_array_equal(_training_class_counts(loader, 3), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
