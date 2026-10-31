"""Checks for the research risk identity and split isolation."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from research import analyze, compare_models, compare_norm, data as research_data, finalize, plots, run, select
from research.fast_data import FastImageDataset, prepare_batch
from research.support_calibrated import (
    TinyCNNGN, classifier_head, client_support_parameters, fedswitch_oracle_weights, make_model,
    head_penalty, local_objective, support_statistics,
)


class ToyDataset:
    def __init__(self, labels):
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1, 1)
        self.imgs = np.full((len(labels), 28, 28, 3), 127, dtype=np.uint8)
        self.transform = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.imgs[index])
        return self.transform(image), int(self.labels[index, 0])


class SupportRiskTests(unittest.TestCase):
    def test_validation_plots_use_completed_development_results(self):
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            files = []
            for model, score in (("baseline", .4), ("baseline_bn", .5)):
                folder = root_path / model
                folder.mkdir()
                result = {"phase": "development", "configuration": {
                    "dataset": "pathmnist", "partition": "practical", "norm": model,
                    "method": "support_calibrated", "seed": 42, "lr": .001,
                    "gamma": 1., "head_lambda": .01, "limit_per_client": 0},
                    "best_validation_round": 2,
                    "best_validation": {"macro_f1": score, "accuracy": score + .1,
                                        "worst_class_recall": .2},
                    "history": [{"round": number, "validation": {
                        "macro_f1": score - .1 + number * .05,
                        "accuracy": score + number * .05,
                        "worst_class_recall": .1 * number},
                        "mean_client_loss": 1 / number} for number in (1, 2)]}
                file = folder / "development.json"
                file.write_text(json.dumps(result), encoding="utf-8")
                files.append(file)
                self.assertTrue(plots.plot_run(result, folder / "validation_curves.png").exists())
            figures = plots.plot_matrix(files, root_path, "test")
            self.assertEqual(len(figures), 1)
            self.assertTrue(figures[0].exists())
            self.assertEqual(len((figures[0].parent / "validation_summary.csv")
                                 .read_text(encoding="utf-8").splitlines()), 3)

    def test_team_baseline_and_bn_support_scr_with_matched_initialization(self):
        torch.manual_seed(42)
        baseline = make_model(3, "baseline")
        torch.manual_seed(42)
        baseline_bn = make_model(3, "baseline_bn")
        for name in ("conv1.weight", "conv2.weight", "fc1.weight", "fc2.weight"):
            torch.testing.assert_close(baseline.state_dict()[name], baseline_bn.state_dict()[name])
        for model in (baseline, baseline_bn):
            head = classifier_head(model)
            reference = (head.weight.detach().clone(), head.bias.detach().clone())
            with torch.no_grad():
                head.bias.add_(0.1)
            penalty = head_penalty(model, reference, torch.ones(3), 0.01)
            self.assertGreater(penalty.item(), 0)
            logits = model(torch.randn(2, 3, 28, 28))
            self.assertEqual(tuple(logits.shape), (2, 3))

    def test_gn_bn_share_architecture_and_initial_parameters(self):
        torch.manual_seed(42)
        gn = make_model(3, "gn")
        torch.manual_seed(42)
        bn = make_model(3, "bn")
        for name in ("encoder.0.weight", "encoder.3.weight", "fc.weight"):
            torch.testing.assert_close(gn.state_dict()[name], bn.state_dict()[name])
        self.assertEqual(gn(torch.randn(2, 3, 28, 28)).shape, (2, 3))
        self.assertEqual(bn(torch.randn(2, 3, 28, 28)).shape, (2, 3))
        self.assertEqual(bn.encoder[1].num_batches_tracked.dtype, torch.int64)

    def test_absent_classes_and_limits(self):
        counts = np.array([[2, 0, 1, 0], [0, 3, 1, 0], [1, 1, 0, 0]])
        moderate = support_statistics(counts, 1.0)
        self.assertTrue(np.all(np.isfinite(moderate.sample_weights)))
        self.assertEqual(moderate.sample_weights[0, 1], 0)
        self.assertEqual(moderate.head_anchor[0, 1], 1)
        np.testing.assert_allclose(moderate.class_client_share.sum(axis=0), [1, 1, 1, 0])
        np.testing.assert_array_equal(moderate.sample_weights[:, 3], 0)
        small = support_statistics(counts, 1e-8)
        for cls in range(3):
            positive = counts[:, cls] > 0
            np.testing.assert_allclose(small.class_client_share[positive, cls],
                                       1 / positive.sum(), rtol=1e-7)
        large = support_statistics(counts, math.inf)
        finite_large = support_statistics(counts, 1e8)
        np.testing.assert_allclose(large.class_client_share,
                                   np.divide(counts, counts.sum(axis=0),
                                             out=np.zeros_like(counts, dtype=float),
                                             where=counts.sum(axis=0) > 0))
        np.testing.assert_allclose(finite_large.class_client_share,
                                   large.class_client_share, atol=1e-8)
        for cls in range(3):
            positive = counts[:, cls] > 0
            self.assertAlmostEqual(np.ptp(large.sample_weights[positive, cls]), 0)
        pooled_variance = 1 / counts.sum(axis=0)[:3]
        np.testing.assert_allclose(large.variance_factor[:3], pooled_variance)
        self.assertTrue(np.all(moderate.variance_factor[:3] >= pooled_variance - 1e-12))

    def test_client_configuration_uses_only_local_and_aggregate_counts(self):
        counts = np.array([[2, 0, 1, 0], [0, 3, 1, 0], [1, 1, 0, 0]])
        for gamma in (0.1, 1.0, 10.0, math.inf):
            stats = support_statistics(counts, gamma)
            for client_id in range(len(counts)):
                weights, anchor = client_support_parameters(
                    counts[client_id], counts.sum(axis=0), stats.coverage.sum(axis=0),
                    int(counts.sum()), len(counts), gamma)
                np.testing.assert_allclose(weights, stats.sample_weights[client_id])
                np.testing.assert_allclose(anchor, stats.head_anchor[client_id])

    def test_weighted_fedavg_gradient_matches_declared_risk(self):
        labels = [[0, 0, 2], [1, 1, 2], [0, 1]]
        counts = np.array([[2, 0, 1], [0, 2, 1], [1, 1, 0]])
        features = [torch.tensor([[1.0, .1], [0.7, .2], [.2, .8]]),
                    torch.tensor([[.1, 1.0], [.3, .9], [.2, .5]]),
                    torch.tensor([[.8, .3], [.2, .7]])]
        stats = support_statistics(counts, .7)
        weight = torch.randn(2, 3, requires_grad=True)
        local_objectives = []
        target = weight.new_zeros(())
        for client_id, client_labels in enumerate(labels):
            y = torch.tensor(client_labels)
            ce = F.cross_entropy(features[client_id] @ weight, y, reduction="none")
            sample_w = torch.tensor(stats.sample_weights[client_id], dtype=ce.dtype)
            local_objectives.append((sample_w[y] * ce).mean())
            for cls in range(3):
                if counts[client_id, cls]:
                    target = target + stats.class_client_share[client_id, cls] * ce[y == cls].mean() / 3
        fedavg = sum(len(labels[k]) / 8 * local_objectives[k] for k in range(3))
        torch.testing.assert_close(fedavg, target)
        torch.testing.assert_close(torch.autograd.grad(fedavg, weight, retain_graph=True)[0],
                                   torch.autograd.grad(target, weight)[0])

    def test_absent_row_gets_largest_anchor(self):
        counts = np.array([[0, 1, 20], [10, 10, 10]])
        stats = support_statistics(counts, 1)
        model = TinyCNNGN(3)
        reference = (model.fc.weight.detach().clone(), model.fc.bias.detach().clone())
        with torch.no_grad():
            model.fc.bias.add_(1)
        full = head_penalty(model, reference,
                            torch.tensor(stats.head_anchor[0], dtype=torch.float32), 0.1)
        self.assertGreater(full.item(), 0)
        self.assertGreater(stats.head_anchor[0, 0], stats.head_anchor[0, 1])
        self.assertGreater(stats.head_anchor[0, 1], stats.head_anchor[0, 2])
        self.assertEqual(head_penalty(model, reference, torch.ones(3), 0).item(), 0)

    def test_development_runner_never_loads_test(self):
        train = ToyDataset([0, 1, 1, 2, 0, 2])
        val = ToyDataset([0, 1, 2])
        with tempfile.TemporaryDirectory() as root:
            cfg = argparse.Namespace(dataset="bloodmnist", partition="dirichlet",
                method="support_calibrated", seed=42, clients=3, alpha=.1,
                rounds=1, epochs=1, batch_size=2, resolution=28, lr=.001,
                norm="bn", optimizer="adam", momentum=.9, weight_decay=0.,
                gamma=1., head_lambda=.01, coverage_kappa=10.,
                fedlc_tau=.5, fedvls_lambda=1.,
                client_shift_strength=0., limit_per_client=0, output_root=Path(root),
                download=False, device="cpu", fast_data=False)
            def loader(*args, **kwargs):
                self.assertFalse(kwargs["load_test"])
                return train, val, None, 3
            with patch.object(run, "arguments", return_value=cfg), \
                 patch.object(run, "get_medmnist_datasets", side_effect=loader), \
                 patch.object(run, "partition_indices", return_value=[[0, 1], [2, 3], [4, 5]]):
                run.main()
            result = json.loads(next(Path(root).glob("*/development.json")).read_text())
            self.assertEqual(result["dataset_split"], "official MedMNIST train/val; test never loaded")
            self.assertEqual(len(result["history"]), 1)
            self.assertTrue(next(Path(root).glob("*/validation_curves.png")).exists())

    def test_selection_uses_complete_validation_seeds_only(self):
        with tempfile.TemporaryDirectory() as root:
            for lr, seed, score, resolution, gamma, head_lambda in [
                (.001, 42, .4, 28, 1., 0.), (.001, 43, .5, 28, 1., 0.),
                (.0001, 42, .95, 28, 1., 0.),
                (.001, 42, .99, 64, 1., 0.), (.001, 43, .99, 64, 1., 0.),
                (.001, 42, .99, 28, 100., 0.), (.001, 43, .99, 28, 100., 0.),
                (.001, 42, .99, 28, 1., 1.), (.001, 43, .99, 28, 1., 1.),
            ]:
                folder = Path(root) / f"{lr}_{seed}_{resolution}_{gamma}_{head_lambda}"
                folder.mkdir()
                payload = {"configuration": {"dataset": "bloodmnist", "partition": "dirichlet",
                    "method": "support_calibrated", "seed": seed, "rounds": 30, "lr": lr,
                    "clients": 12, "alpha": .1, "epochs": 1, "batch_size": 64,
                    "resolution": resolution, "device": "auto", "client_shift_strength": 0.,
                    "coverage_kappa": 10., "fast_data": True, "limit_per_client": 0,
                    "gamma": gamma, "head_lambda": head_lambda},
                    "best_validation": {"macro_f1": score},
                    "partition_sha256": "same"}
                (folder / "development.json").write_text(json.dumps(payload))
                (folder / "best.pt").write_bytes(b"mock checkpoint")
            lock_file = Path(root) / "lock.json"
            argv = ["select", "--dataset", "bloodmnist", "--partition", "dirichlet",
                    "--seeds", "42", "43", "--methods", "support_calibrated", "--rounds", "30",
                    "--lrs", ".001", "--gammas", "1", "--head-lambdas", "0",
                    "--output-root", root, "--lock-file", str(lock_file)]
            with patch("sys.argv", argv[:argv.index("--lrs") + 1] + [".0001", ".001"]
                       + argv[argv.index("--gammas"):]):
                with self.assertRaisesRegex(RuntimeError, "incomplete validation settings"):
                    select.main()
            with patch("sys.argv", argv):
                select.main()
            lock = json.loads(lock_file.read_text())
            self.assertEqual(lock["methods"]["support_calibrated"]["settings"]["lr"], .001)
            self.assertEqual(lock["methods"]["support_calibrated"]["mean_validation_macro_f1"], .45)

    def test_selection_does_not_filter_inapplicable_method_grids(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / "fedavg"
            folder.mkdir()
            payload = {"configuration": {"dataset": "bloodmnist", "partition": "dirichlet",
                "method": "fedavg", "seed": 42, "rounds": 30, "lr": .001,
                "clients": 12, "alpha": .1, "epochs": 1, "batch_size": 64,
                "resolution": 28, "device": "auto", "client_shift_strength": 0.,
                "coverage_kappa": 10., "fast_data": True, "limit_per_client": 0,
                "gamma": 1., "head_lambda": 0., "fedlc_tau": .5,
                "fedvls_lambda": .1}, "best_validation": {"macro_f1": .4},
                "partition_sha256": "same"}
            (folder / "development.json").write_text(json.dumps(payload))
            (folder / "best.pt").write_bytes(b"mock checkpoint")
            lock_file = Path(root) / "lock.json"
            argv = ["select", "--dataset", "bloodmnist", "--partition", "dirichlet",
                    "--seeds", "42", "--methods", "fedavg", "--lrs", ".001",
                    "--gammas", ".1", "--head-lambdas", ".01",
                    "--fedlc-taus", ".1", "--fedvls-lambdas", ".01",
                    "--output-root", root, "--lock-file", str(lock_file)]
            with patch("sys.argv", argv):
                select.main()
            self.assertTrue(lock_file.exists())

    def test_finalizer_loads_test_only_and_skips_repeat(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            development = folder / "development.json"
            development.write_text(json.dumps({"configuration": {
                "dataset": "bloodmnist", "resolution": 28, "batch_size": 2,
                "fast_data": True, "norm": "gn"}}))
            checkpoint = folder / "best.pt"
            torch.save({"state_dict": TinyCNNGN(3).state_dict(), "round": 1}, checkpoint)
            lock_file = folder / "lock.json"
            lock_file.write_text(json.dumps({"selection_metric": "validation macro-F1",
                "methods": {"fedavg": {"runs": [{"seed": 42,
                    "development": str(development), "checkpoint": str(checkpoint)}]}}}))
            test_data = ToyDataset([0, 1, 2])
            def loader(*args, **kwargs):
                self.assertFalse(kwargs["load_train"])
                self.assertFalse(kwargs["load_val"])
                self.assertTrue(kwargs["load_test"])
                return None, None, test_data, 3
            argv = ["finalize", "--lock-file", str(lock_file), "--device", "cpu"]
            with patch("sys.argv", argv), \
                 patch.object(finalize, "get_medmnist_datasets", side_effect=loader), \
                 patch.object(finalize, "evaluate", return_value={"macro_f1": .5, "accuracy": .6}) as evaluator:
                finalize.main()
                finalize.main()
                self.assertEqual(evaluator.call_count, 1)
            result = json.loads((folder / "lock_test.json").read_text())
            self.assertEqual(result["test_evaluations"], 1)

    def test_paired_analysis_uses_matching_seeds(self):
        values = {"fedavg": {42: .3, 43: .4},
                  "support_calibrated": {42: .5, 43: .45}}
        comparison = analyze.paired_summary(values, "fedavg")["support_calibrated"]
        np.testing.assert_allclose(comparison["per_seed_difference"], [.2, .05])
        self.assertAlmostEqual(comparison["mean_difference"], .125)
        with self.assertRaises(ValueError):
            analyze.paired_summary({"fedavg": {42: .3}, "method": {43: .4}}, "fedavg")

    def test_norm_comparison_separates_method_and_bn_effects(self):
        def arm(norm, fedavg, scr):
            lock = {"dataset": "pathmnist", "partition": "practical",
                    "seeds": [42, 43], "protocol": {"norm": norm, "clients": 12},
                    "methods": {method: {"runs": [
                        {"seed": seed, "partition_sha256": f"partition-{seed}"}
                        for seed in (42, 43)]} for method in ("fedavg", "support_calibrated")}}
            result = {"methods": {}}
            for method, values in (("fedavg", fedavg), ("support_calibrated", scr)):
                result["methods"][method] = [
                    {"seed": seed, "metrics": {name: value for name in compare_norm.METRICS}}
                    for seed, value in zip((42, 43), values)]
            return lock, result
        gn = arm("gn", (.40, .50), (.50, .60))
        bn = arm("bn", (.45, .55), (.60, .70))
        report = compare_norm.compare(*gn, *bn)
        contrasts = report["paired_differences"]["macro_f1"]
        self.assertAlmostEqual(contrasts["scr_vs_fedavg_gn"]["mean_difference"], .10)
        self.assertAlmostEqual(contrasts["scr_vs_fedavg_bn"]["mean_difference"], .15)
        self.assertAlmostEqual(contrasts["bn_vs_gn_under_scr"]["mean_difference"], .10)
        self.assertAlmostEqual(
            contrasts["normalization_by_method_interaction"]["mean_difference"], .05)

    def test_team_baseline_comparison_requires_matched_protocol(self):
        def arm(model, fedavg, scr):
            lock = {"dataset": "pathmnist", "partition": "practical",
                    "seeds": [42, 43],
                    "protocol": {"norm": model, "clients": 12, "weight_decay": 0.0},
                    "methods": {method: {"runs": [
                        {"seed": seed, "partition_sha256": f"partition-{seed}"}
                        for seed in (42, 43)]}
                        for method in ("fedavg", "support_calibrated")}}
            test = {"methods": {}}
            for method, values in (("fedavg", fedavg), ("support_calibrated", scr)):
                test["methods"][method] = [
                    {"seed": seed, "metrics": {metric: value for metric in compare_models.METRICS}}
                    for seed, value in zip((42, 43), values)]
            return lock, test
        baseline = arm("baseline", (.4, .5), (.5, .6))
        baseline_bn = arm("baseline_bn", (.45, .55), (.6, .7))
        report = compare_models.compare(*baseline, *baseline_bn)
        self.assertIn("scr_minus_fedavg/baseline", compare_models.markdown(report))
        with tempfile.TemporaryDirectory() as root:
            self.assertTrue(plots.plot_test_report(report, Path(root) / "test.png").exists())
        self.assertAlmostEqual(report["paired_differences"]["macro_f1"][
            "scr_minus_fedavg/baseline"]["mean"], .1)
        self.assertAlmostEqual(report["paired_differences"]["macro_f1"][
            "method_by_model_interaction"]["mean"], .05)
        baseline_bn[0]["protocol"]["weight_decay"] = 1e-4
        with self.assertRaisesRegex(ValueError, "protocol field weight_decay"):
            compare_models.compare(*baseline, *baseline_bn)

    def test_fedswitch_oracle_aggregation_weights(self):
        counts = np.array([[50, 0], [50, 1], [0, 9]])
        expected = np.array([.5, .6, .9])
        expected /= expected.sum()
        np.testing.assert_allclose(fedswitch_oracle_weights(counts), expected)

    def test_uniform_head_ablation_has_same_data_loss(self):
        model = TinyCNNGN(3)
        images = torch.randn(2, 3, 28, 28)
        labels = torch.tensor([0, 2])
        counts = torch.tensor([1., 0., 1.])
        weights = torch.tensor([2., 0., 3.])
        reference = (model.fc.weight.detach().clone(), model.fc.bias.detach().clone())
        losses = [local_objective(method, model, images, labels, counts,
                  torch.tensor([4., 4., 4.]), weights, anchor, reference, 0.)
                  for method, anchor in (("support_uniform_head", torch.ones(3)),
                                         ("support_calibrated", torch.tensor([.5, 1., .8])))]
        torch.testing.assert_close(losses[0], losses[1])

    def test_synthetic_client_shift_is_consistent(self):
        data = ToyDataset([0, 1])
        data.transform = run.evaluation_transform(data, 28)
        shifted = run.ClientShiftDataset(data, [[0], [1]], seed=42, amount=.2)
        raw, _ = data[1]
        gain, offset = run.client_shift_parameters(42, 1, .2)
        torch.testing.assert_close(shifted[1][0], (raw * gain + offset).clamp(-1, 1))
        with self.assertRaises(ValueError):
            run.ClientShiftDataset(data, [[0, 1], [1]], seed=42, amount=.2)

    def test_fedlc_ignores_vacant_class_logit(self):
        model = TinyCNNGN(3)
        images = torch.randn(2, 3, 28, 28)
        labels = torch.tensor([0, 1])
        counts = torch.tensor([10., 10., 0.])
        reference = (model.fc.weight.detach().clone(), model.fc.bias.detach().clone())
        loss = local_objective("fedlc", model, images, labels, counts,
                               counts, torch.ones(3), torch.zeros(3), reference)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        torch.testing.assert_close(model.fc.bias.grad[2], torch.tensor(0.))

    def test_coverage_iw_includes_linked_head_penalty(self):
        model = TinyCNNGN(3)
        reference = (model.fc.weight.detach().clone(), model.fc.bias.detach().clone())
        with torch.no_grad():
            model.fc.bias.add_(.2)
        images = torch.randn(2, 3, 28, 28)
        labels = torch.tensor([0, 1])
        counts = torch.tensor([10., 10., 0.])
        anchor = torch.tensor([.5, .5, 1.])
        args = ("coverage_iw", model, images, labels, counts, counts,
                torch.ones(3), anchor, reference)
        no_head = local_objective(*args, head_lambda=0.)
        with_head = local_objective(*args, head_lambda=.1)
        expected = head_penalty(model, reference, anchor, .1 * len(anchor))
        torch.testing.assert_close(with_head - no_head, expected)

    def test_fast_batch_matches_native_evaluation_transform(self):
        data = ToyDataset([0, 1])
        data.transform = run.evaluation_transform(data, 28)
        fast = FastImageDataset(data, 28)
        raw = torch.stack([fast[0][0], fast[1][0]])
        prepared = prepare_batch(raw, 28, training=False)
        torch.testing.assert_close(prepared[0], data[0][0], atol=1e-6, rtol=0)
        augmented = prepare_batch(raw, 28, training=True)
        self.assertEqual(tuple(augmented.shape), (2, 3, 28, 28))
        self.assertTrue(torch.isfinite(augmented).all())

    def test_fast_crop_flip_matches_reference_coordinates(self):
        import torch.nn.functional as functional
        pixels = torch.arange(28 * 28, dtype=torch.int64).remainder(256).byte()
        raw = pixels.reshape(1, 1, 28, 28).repeat(2, 3, 1, 1)
        torch.manual_seed(732)
        tops = torch.randint(0, 5, (2,))
        lefts = torch.randint(0, 5, (2,))
        flips = torch.rand(2) < .5
        torch.manual_seed(732)
        actual = prepare_batch(raw, 28, training=True)
        padded = functional.pad(raw.float() / 255, (2, 2, 2, 2), mode="reflect")
        expected = []
        for index in range(2):
            crop = padded[index, :, tops[index]:tops[index]+28,
                          lefts[index]:lefts[index]+28]
            if flips[index]:
                crop = crop.flip(-1)
            expected.append(crop * 2 - 1)
        torch.testing.assert_close(actual, torch.stack(expected))

    def test_research_loader_does_not_construct_unrequested_test(self):
        requested = []
        class StubDataset:
            def __init__(self, split, **kwargs):
                requested.append(split)
        with tempfile.TemporaryDirectory() as root, \
             patch.object(research_data, "INFO", {"toy": {"python_class": "StubDataset",
                                                      "label": {"0": "a"}}}), \
             patch.object(research_data.medmnist, "StubDataset", StubDataset, create=True), \
             patch.object(research_data, "DEFAULT_ROOT", root), \
             patch.object(research_data, "WORKSPACE_CACHE", Path(root)):
            research_data.get_medmnist_datasets("toy", load_train=True,
                                                load_val=True, load_test=False)
        self.assertEqual(requested, ["train", "val"])


if __name__ == "__main__":
    unittest.main()
