import unittest

from dashboard.client_setup import remote_client_setup


class RemoteClientSetupTests(unittest.TestCase):
    def test_portable_config_matches_server(self):
        source = dict(num_clients=3, dataset='pathmnist', model='baseline',
                      local_epochs=5, seed=42, partition_path='D:/server/partition.json',
                      output_dir='D:/server/results', device_type='pc', active_clients=[0, 1, 2])
        cfg, command = remote_client_setup(source, '192.168.1.2:8081', 2)
        self.assertEqual(cfg['num_clients'], 3)
        self.assertEqual(cfg['seed'], 42)
        self.assertEqual(cfg['server_address'], '192.168.1.2:8081')
        self.assertEqual(cfg['client_id'], 2)
        self.assertNotIn('partition_path', cfg)
        self.assertNotIn('device_type', cfg)
        self.assertIn('--client_id 2', command)
        self.assertIn('client.client', command)
        self.assertNotIn('nicegui', command)
        self.assertIn('partition_path', source)

    def test_invalid_remote_ids_and_bind_address(self):
        for value in [-1, 3, 1.5, float('nan')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                remote_client_setup({'num_clients': 3}, '192.168.1.2:8081', value)
        with self.assertRaises(ValueError):
            remote_client_setup({'num_clients': 3}, '0.0.0.0:8081', 0)
