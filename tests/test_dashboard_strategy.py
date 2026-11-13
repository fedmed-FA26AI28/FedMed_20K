"""Check the real Flower strategy-to-dashboard event contract without training data."""
import contextlib
import io
import os
import unittest
from unittest.mock import patch

from flwr.common import Code, EvaluateRes, FitRes, Status
from algorithms.fedavg import FedAvgStrategy
from dashboard.metrics import MetricStream


class StrategyContractTests(unittest.TestCase):
    def test_round_and_node_events(self):
        strategy = FedAvgStrategy(min_fit_clients=1, min_available_clients=2,
                                  fraction_fit=.5, early_stop_patience=0, local_test=True,
                                  evaluate_metrics_aggregation_fn=lambda values: {
                                      'accuracy': sum(n*m['accuracy'] for n,m in values) / sum(n for n,_ in values)})
        metrics = {'client_id': 0, 'train_loss': .4, 'train_accuracy': .7,
                   'device_ipv4': '192.168.1.10', 'cpu_percent': 20.,
                   'ram_percent': 40., 'gpu_memory_mb': 128., 'ping_ms': 2.}
        # Finalize only: no parameters or model data need to be synthesized.
        result = FitRes(Status(Code.OK, ''), parameters=None, num_examples=10, metrics=metrics)
        output = io.StringIO()
        with patch.dict(os.environ, FEDMED_DASHBOARD_STREAM='1'), contextlib.redirect_stdout(output):
            strategy._finalize_fit_aggregation(1, None, {'train_loss': .4, 'train_accuracy': .7}, [(None, result)], [])
            strategy.aggregate_evaluate(1, [(None, EvaluateRes(Status(Code.OK, ''), .3, 10,
                {'accuracy': .8, 'client_id': 0}))], [])
        stream = MetricStream()
        for line in output.getvalue().splitlines():
            stream.feed(line)
        self.assertEqual(stream.rows[1]['train_accuracy'], .7)
        self.assertEqual(stream.rows[1]['test_loss'], .3)
        self.assertEqual(stream.rows[1]['test_accuracy'], .8)
        self.assertEqual(stream.rows[1]['bta'], .8)
        self.assertEqual(stream.nodes['0']['gpu_memory_mb'], 128.)
        self.assertEqual(stream.nodes['0']['test_accuracy'], .8)
        self.assertEqual(len(stream.client_records('0')), 1)
        self.assertEqual(strategy.num_fit_clients(10), (5, 2))


if __name__ == '__main__':
    unittest.main()
