"""Regression tests for the optional vacant-class experiment."""

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithms import get_strategy
from algorithms.vacant_distillation import (
    frozen_global_teacher,
    vacant_class_distillation_loss,
)
from client.client import _local_train
from server.server import _make_on_fit_config_fn


class VacantDistillationTests(unittest.TestCase):
    def test_single_vacant_class_has_nonzero_student_only_gradient(self):
        student = torch.tensor([[0.0, -2.0, 0.0]], requires_grad=True)
        teacher = torch.tensor([[0.0, 2.0, 0.0]], requires_grad=True)
        loss = vacant_class_distillation_loss(
            student, teacher, torch.tensor([4, 0, 4]), temperature=2.0,
        )
        self.assertGreater(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())
        self.assertGreater(float(student.grad.abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)

    def test_no_vacancy_is_exact_noop(self):
        student = torch.randn(2, 3, requires_grad=True)
        teacher = torch.randn(2, 3)
        loss = vacant_class_distillation_loss(student, teacher, torch.ones(3))
        self.assertEqual(0.0, float(loss.detach()))
        loss.backward()
        self.assertEqual(0.0, float(student.grad.abs().sum()))

    def test_teacher_is_frozen_and_disabled_during_warmup(self):
        model = torch.nn.Linear(4, 3)
        counts = torch.tensor([8, 0, 0])
        self.assertIsNone(frozen_global_teacher(model, counts, 0.1, 1))
        self.assertIsNone(frozen_global_teacher(model, torch.ones(3), 0.1, 2))
        teacher = frozen_global_teacher(model, counts, 0.1, 2)
        self.assertFalse(teacher.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))
        original = teacher.weight.detach().clone()
        with torch.no_grad():
            model.weight.add_(1.0)
        self.assertTrue(torch.equal(teacher.weight, original))

    def test_fedavg_strategy_and_round_config_preserve_parameter_flow(self):
        self.assertIsNotNone(get_strategy("vacant_distill"))
        self.assertIsNotNone(get_strategy("coverage_distill"))
        config = _make_on_fit_config_fn(
            1, 0.001, distill_mu=0.1, distill_temperature=2.0,
            distill_max_count=0, distill_warmup_rounds=1,
        )(2)
        self.assertEqual(2, config["server_round"])
        self.assertEqual(0.1, config["distill_mu"])
        self.assertNotIn("distill_mu", _make_on_fit_config_fn(1, 0.001)(2))

    def test_local_training_uses_teacher_without_updating_it(self):
        torch.manual_seed(17)
        loader = DataLoader(
            TensorDataset(torch.randn(8, 4), torch.zeros(8, dtype=torch.long)),
            batch_size=4,
        )
        model = torch.nn.Linear(4, 3)
        teacher = frozen_global_teacher(model, torch.tensor([8, 0, 0]), 0.1, 2)
        teacher_weight = teacher.weight.detach().clone()
        metrics = _local_train(
            model, loader, torch.optim.SGD(model.parameters(), lr=0.1),
            torch.device("cpu"), epochs=1,
            class_counts=torch.tensor([8, 0, 0]),
            global_teacher=teacher, distill_mu=0.1,
        )
        self.assertEqual(8, metrics["num_samples"])
        self.assertGreaterEqual(metrics["distill_loss"], 0.0)
        self.assertTrue(torch.equal(teacher.weight, teacher_weight))


if __name__ == "__main__":
    unittest.main()
