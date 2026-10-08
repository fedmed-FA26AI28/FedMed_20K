import tempfile
import time
import unittest
import sys
import types
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np
import cloudpickle
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Resource monitoring is not exercised here. Keep the regression suite runnable
# in minimal CPU-only environments that have not installed optional monitors.
try:
    import psutil  # noqa: F401
except ImportError:
    sys.modules["psutil"] = types.SimpleNamespace()

from algorithms.fedavg import FedAvgStrategy, _weighted_average_metrics
from algorithms.coverage import coverage_weights, logit_adjustment
from client.client import _local_train
from datasets.partition import (
    dirichlet_partition,
    load_partition_metadata,
    stratified_holdout_indices,
    stratified_subsample_indices,
    stratified_validation_partition,
)
from datasets.medmnist_code import build_transform, fit_train_normalization, get_bloodmnist_dataset
from datasets.sampling import balanced_loader
from models.cnn import build_model, set_parameters
from monitoring.research_validation import evaluate_validation
from experiments.run_simulation import _make_client_fn, run_simulation
from models.cnn import CNN, get_parameters
from server.server import (
    _evaluate_final_global_model,
    _evaluate_metrics_aggregation_fn,
)


class LabelDataset(Dataset):
    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.zeros(1), self.labels[index]


class ClientProxyStub:
    cid = "0"


class FederatedDataFlowTests(unittest.TestCase):
    def test_ray_client_factory_contains_indices_not_raw_images(self):
        factory = _make_client_fn([[0, 1], [2, 3]], [[0], [1]],
                                  num_classes=8, local_epochs=1,
                                  learning_rate=0.001, batch_size=2,
                                  client_cpus=1)
        self.assertLess(len(cloudpickle.dumps(factory)), 100_000)

    def test_final_run_requires_locked_development_spec(self):
        args = SimpleNamespace(num_clients=2, client_fraction=1.0,
                               client_gpus=0.0, calibration_fraction=0.5,
                               final_test=True, locked_config=None)
        with self.assertRaises(ValueError):
            run_simulation(args)

    def test_fedavg_aggregates_sample_weighted_parameters_not_metrics(self):
        strategy = FedAvgStrategy(min_fit_clients=1, min_evaluate_clients=1,
                                  min_available_clients=1)
        strategy._round_start = time.time()
        fit = []
        for count, weight, accuracy in ((2, 3.0, 1.0), (8, 7.0, 0.0)):
            fit.append((ClientProxyStub(), FitRes(
                Status(Code.OK, "ok"),
                ndarrays_to_parameters([np.array([weight], dtype=np.float32)]),
                count, {"train_accuracy": accuracy},
            )))
        parameters, _ = strategy.aggregate_fit(1, fit, [])
        from flwr.common import parameters_to_ndarrays
        self.assertAlmostEqual(6.2, float(parameters_to_ndarrays(parameters)[0][0]), places=5)

    def test_exact_global_validation_uses_only_supplied_indices(self):
        dataset = TensorDataset(torch.randn(16, 3, 28, 28),
                                torch.arange(16) % 8)
        metrics = evaluate_validation(get_parameters(build_model("tiny_cnn")),
                                      "tiny_cnn", dataset, list(range(8)), batch_size=4)
        self.assertEqual("val_monitor", metrics["split"])
        self.assertEqual(8, metrics["num_samples"])
        self.assertIn("f1_macro", metrics)

    def test_64_loader_uses_native_medmnist_plus_size(self):
        with mock.patch("medmnist.BloodMNIST") as blood:
            get_bloodmnist_dataset("train", download=False, size=64)
        self.assertEqual(64, blood.call_args.kwargs["size"])
        self.assertEqual("train", blood.call_args.kwargs["split"])

    def test_validation_transform_is_deterministic_and_train_only(self):
        with self.assertRaises(ValueError):
            build_transform("test", augment=True)
        self.assertEqual(2, len(build_transform("val").transforms))
        self.assertGreater(len(build_transform("train", augment=True).transforms), 2)

    def test_normalization_is_fitted_on_permitted_train_indices(self):
        dataset = mock.Mock(split="train")
        dataset.imgs = np.stack([np.zeros((2, 2, 3), dtype=np.uint8),
                                 np.full((2, 2, 3), 255, dtype=np.uint8)])
        mean, _ = fit_train_normalization(dataset, [0])
        self.assertEqual((0.0, 0.0, 0.0), mean)
        dataset.split = "test"
        with self.assertRaises(ValueError):
            fit_train_normalization(dataset, [0])

    def test_model_parameter_round_trip_and_balanced_budget(self):
        for name in ("tiny_cnn", "mobilenet_v3_small"):
            model = build_model(name)
            restored = build_model(name)
            set_parameters(restored, get_parameters(model))
            self.assertEqual(8, restored.fc.out_features)
            with torch.no_grad():
                self.assertEqual((2, 8), tuple(restored.eval()(torch.zeros(2, 3, 64, 64)).shape))
        labels = LabelDataset([0] * 9 + [1])
        loader = DataLoader(labels, batch_size=3)
        balanced = balanced_loader(loader, torch.tensor([9.0, 1.0]), seed=42)
        self.assertEqual(10, len(list(balanced.sampler)))
        self.assertEqual(len(loader), len(balanced))

    def test_train_and_validation_partitions_are_complete_and_disjoint(self):
        train_dataset = LabelDataset([0] * 15 + [1] * 15 + [2] * 15)
        val_dataset = LabelDataset([0] * 9 + [1] * 9 + [2] * 9)

        train_parts = dirichlet_partition(
            train_dataset, num_clients=3, alpha=0.3, seed=7
        )
        val_parts = stratified_validation_partition(
            val_dataset, num_clients=3, seed=7
        )

        for parts, dataset in (
            (train_parts, train_dataset),
            (val_parts, val_dataset),
        ):
            flattened = [index for part in parts for index in part]
            self.assertEqual(len(dataset), len(flattened))
            self.assertEqual(len(flattened), len(set(flattened)))
            self.assertEqual(set(flattened), set(range(len(dataset))))

        # Validation is stratified rather than using the non-IID train policy.
        for client_indices in val_parts:
            client_labels = [val_dataset.labels[i] for i in client_indices]
            self.assertEqual([3, 3, 3], [client_labels.count(c) for c in range(3)])

    def test_training_budget_and_calibration_holdout_are_stratified(self):
        dataset = LabelDataset([0] * 20 + [1] * 20 + [2] * 20)
        selected = stratified_subsample_indices(dataset, num_samples=30, seed=4)
        self.assertEqual(30, len(selected))
        self.assertEqual(30, len(set(selected)))
        self.assertEqual(
            [10, 10, 10],
            [[dataset.labels[i] for i in selected].count(c) for c in range(3)],
        )

        monitor, calibration = stratified_holdout_indices(
            dataset, holdout_fraction=0.25, seed=4
        )
        self.assertFalse(set(monitor) & set(calibration))
        self.assertEqual(set(range(len(dataset))), set(monitor) | set(calibration))

    def test_coverage_objective_strengthens_missing_classes(self):
        counts = torch.tensor([0.0, 8.0, 32.0, 320.0])
        weights = coverage_weights(counts, kappa=32.0)
        self.assertTrue(torch.all(weights[:-1] > weights[1:]))
        self.assertAlmostEqual(1.0, float(weights[0]))
        adjustment = logit_adjustment(counts, tau=1.0, smoothing=1.0)
        self.assertTrue(torch.isfinite(adjustment).all())

    def test_partition_metadata_rejects_non_training_indices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "partition.json"
            path.write_text(
                '{"split":"test","num_clients":1,"clients":{"0":{"indices":[0]}}}',
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_partition_metadata(str(path))

    def test_local_training_uses_validation_and_reports_train_count(self):
        torch.manual_seed(3)
        train_loader = DataLoader(
            TensorDataset(torch.randn(12, 4), torch.randint(0, 2, (12,))),
            batch_size=4,
            shuffle=False,
        )
        val_loader = DataLoader(
            TensorDataset(torch.randn(6, 4), torch.randint(0, 2, (6,))),
            batch_size=3,
            shuffle=False,
        )
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

        metrics = _local_train(
            model,
            train_loader,
            optimizer,
            torch.device("cpu"),
            epochs=2,
            val_loader=val_loader,
        )

        self.assertEqual(12, metrics["num_samples"])
        self.assertEqual(6, metrics["num_val_samples"])
        self.assertIn("val_loss", metrics)
        self.assertIn("val_accuracy", metrics)

    def test_metrics_do_not_replace_sample_weighted_parameter_aggregation(self):
        aggregated = _weighted_average_metrics(
            [
                (2, {"train_accuracy": 0.0, "val_accuracy": 1.0, "num_val_samples": 1}),
                (8, {"train_accuracy": 1.0, "val_accuracy": 0.0, "num_val_samples": 3}),
            ]
        )
        self.assertAlmostEqual(0.8, aggregated["train_accuracy"])
        self.assertAlmostEqual(0.25, aggregated["val_accuracy"])

        strategy = FedAvgStrategy(
            min_fit_clients=1,
            min_evaluate_clients=1,
            min_available_clients=1,
        )
        strategy._round_start = time.time()
        fit_res = FitRes(
            status=Status(code=Code.OK, message="ok"),
            parameters=ndarrays_to_parameters([np.array([3.0], dtype=np.float32)]),
            num_examples=5,
            metrics={"client_id": 0, "device_type": "pc"},
        )
        parameters, _ = strategy.aggregate_fit(
            1, [(ClientProxyStub(), fit_res)], []
        )
        self.assertIs(parameters, strategy.latest_parameters)

    def test_validation_reports_worst_client_metrics(self):
        metrics = _evaluate_metrics_aggregation_fn(
            [
                (10, {"accuracy": 0.9, "f1_macro": 0.8}),
                (10, {"accuracy": 0.4, "f1_macro": 0.3}),
            ]
        )
        self.assertAlmostEqual(0.65, metrics["accuracy"])
        self.assertAlmostEqual(0.4, metrics["worst_client_accuracy"])
        self.assertAlmostEqual(0.3, metrics["worst_client_f1_macro"])

    def test_final_evaluator_loads_only_the_test_split(self):
        dataset = TensorDataset(
            torch.randn(4, 3, 28, 28), torch.tensor([0, 1, 2, 3])
        )
        parameters = get_parameters(CNN(num_classes=8))

        with mock.patch(
            "datasets.medmnist_code.get_bloodmnist_dataset",
            return_value=(dataset, 8),
        ) as load_split:
            metrics = _evaluate_final_global_model(parameters, num_classes=8)

        load_split.assert_called_once_with("test", download=True)
        self.assertEqual(4, metrics["num_samples"])
        self.assertIn("f1_macro", metrics)
        self.assertIn("uncalibrated", metrics)

    def test_final_evaluator_calibrates_without_using_test_for_fitting(self):
        calibration_dataset = TensorDataset(
            torch.randn(8, 3, 28, 28), torch.arange(8)
        )
        test_dataset = TensorDataset(
            torch.randn(8, 3, 28, 28), torch.arange(8)
        )
        parameters = get_parameters(CNN(num_classes=8))

        with mock.patch(
            "datasets.medmnist_code.get_bloodmnist_dataset",
            side_effect=[(calibration_dataset, 8), (test_dataset, 8)],
        ) as load_split:
            metrics = _evaluate_final_global_model(
                parameters,
                num_classes=8,
                calibration_indices=list(range(8)),
            )

        self.assertEqual(
            [mock.call("val", download=True), mock.call("test", download=True)],
            load_split.call_args_list,
        )
        self.assertIn("calibrated", metrics)
        self.assertIn("conformal", metrics)


if __name__ == "__main__":
    unittest.main()
