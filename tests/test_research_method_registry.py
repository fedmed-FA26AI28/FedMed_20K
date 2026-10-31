"""The shared experiment selects consistent algorithm policies and tuning axes."""

from __future__ import annotations

import argparse
import unittest

import numpy as np
import torch

from algorithms.research_methods import (
    ClientContext, METHODS, TUNING_GRIDS, TinyCNNGN, get_method,
    local_objective, support_statistics,
)
from research import select


class ResearchMethodRegistryTests(unittest.TestCase):
    def test_matrix_and_validation_selection_use_the_same_tuning_axes(self):
        args = argparse.Namespace(lrs=[.001], gammas=[.1, 1],
                                  head_lambdas=[0, .01], fedlc_taus=[.5],
                                  fedvls_lambdas=[.1])
        for name in METHODS:
            spec = get_method(name)
            expected = ["lr", *spec.tuning_parameters]
            self.assertEqual([key for key, _ in select.tuning_axes(name, args)], expected)
            self.assertTrue(all(key in TUNING_GRIDS for key in spec.tuning_parameters))
        self.assertEqual(get_method("support_calibrated").tuning_parameters,
                         ("gamma", "head_lambda"))
        self.assertTrue(get_method("fedvls").needs_teacher)
        self.assertTrue(get_method("fedswitch_oracle").oracle_aggregation)
        with self.assertRaisesRegex(ValueError, "unknown research method"):
            get_method("not_registered")

    def test_each_registered_method_has_finite_loss_and_client_parameters(self):
        count_matrix = np.array([[3, 2, 0, 0], [2, 0, 3, 0], [0, 3, 0, 2]])
        stats = support_statistics(count_matrix, 1)
        pooled = support_statistics(count_matrix, float("inf"))
        context = ClientContext(
            count_matrix[0], count_matrix.sum(axis=0), stats.coverage.sum(axis=0),
            pooled.coverage.sum(axis=0), int(count_matrix.sum()), 3, 1., 10.,
        )
        model = TinyCNNGN(4)
        teacher = TinyCNNGN(4)
        teacher.load_state_dict(model.state_dict())
        teacher.eval()
        images = torch.randn(2, 3, 28, 28)
        labels = torch.tensor([0, 1])
        reference = (model.fc.weight.detach().clone(), model.fc.bias.detach().clone())
        for name in METHODS:
            with self.subTest(method=name):
                spec = get_method(name)
                weights, anchor = spec.client_parameters(context)
                self.assertEqual(weights.shape, (4,))
                self.assertEqual(anchor.shape, (4,))
                self.assertTrue(np.isfinite(weights).all())
                self.assertTrue(np.isfinite(anchor).all())
                model.zero_grad(set_to_none=True)
                loss = local_objective(
                    name, model, images, labels,
                    torch.tensor(count_matrix[0], dtype=torch.float32),
                    torch.tensor(context.global_counts, dtype=torch.float32),
                    torch.tensor(weights, dtype=torch.float32),
                    torch.tensor(anchor, dtype=torch.float32), reference,
                    head_lambda=.01, teacher=teacher if spec.needs_teacher else None,
                )
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(torch.isfinite(model.fc.weight.grad).all())


if __name__ == "__main__":
    unittest.main()
