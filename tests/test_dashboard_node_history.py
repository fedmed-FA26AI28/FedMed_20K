import json
import unittest

from dashboard.metrics import MetricStream


class NodeHistoryTests(unittest.TestCase):
    def test_merge_round_reports_and_preserve_other_clients(self):
        stream = MetricStream()
        def feed(round_number, **metrics):
            stream.feed('[METRIC] ' + json.dumps({'kind': 'node', 'round': round_number, 'metrics': metrics}))
        feed(1, client_id=0, train_loss=.4, train_accuracy=.7, device_ipv4='192.0.2.1', training_time=4)
        feed(1, client_id=0, test_loss=.5, test_accuracy=.6)
        feed(2, client_id=0, train_loss=.3, train_accuracy=.8)
        feed(2, client_id=1, train_loss=.9)
        feed(1, client_id=0, test_accuracy=.65)
        history = stream.client_records('0')
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]['train_accuracy'], .7)
        self.assertEqual(history[0]['test_accuracy'], .65)
        self.assertNotIn('test_accuracy', history[1])
        self.assertEqual(stream.nodes['0']['round'], 2)
        self.assertEqual(stream.client_records('1')[0]['train_loss'], .9)
        self.assertEqual(stream.round, 2)
        self.assertEqual(stream.records(), [])
