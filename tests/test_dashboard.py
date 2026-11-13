import asyncio
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import psutil
from dashboard.controller import Controller, command, prepare_config
from dashboard.metrics import MetricStream, log_level
from dashboard.system import endpoint, ping, partition_data

VALUES = dict(dataset='pathmnist', strategy='fedavg', partition_mode='practical', model='cnn',
              optimizer='sgd', device='cpu', rounds=2, num_clients=2, local_epochs=1,
              batch_size=16, learning_rate=0.01, seed=42, client_id=0,
              server_host='127.0.0.1', server_port=8081)


class ConfigTests(unittest.TestCase):
    def test_config_and_commands(self):
        for mode, module in [('simulation', 'experiments.run_simulation'), ('server', 'server.server'), ('client', 'client.client')]:
            cfg = prepare_config(mode, {**VALUES, 'min_fit_clients': 1, 'fraction_fit': .5})
            cmd = command(mode, Path('config.yaml'), Path('out'), cfg)
            self.assertIn(module, cmd)
            self.assertEqual(cfg['min_fit_clients'], 2 if mode == 'simulation' else 1)
            self.assertEqual(cfg['min_available_clients'], 2)
            self.assertEqual(cfg['fl_rounds'], 2)
            if mode == 'client':
                self.assertIn('127.0.0.1:8081', cmd)
            if mode == 'simulation':
                self.assertEqual(cfg['strategies'], ['fedavg'])

    def test_validation(self):
        for key, value in [('rounds', 0), ('rounds', 101), ('num_clients', 1.5), ('learning_rate', float('nan')),
                           ('client_id', 2), ('fraction_fit', 1.1), ('min_fit_clients', 3), ('server_port', 65536)]:
            with self.subTest(key=key, value=value), self.assertRaises((ValueError, TypeError)):
                prepare_config('server', {**VALUES, key: value})
        with self.assertRaises(ValueError):
            prepare_config('server', VALUES, web_port=8081)

    def test_metrics_regex_and_structured(self):
        stream = MetricStream()
        stream.feed('\x1b[32mRound 1/40 | Test Accuracy: 0.5% | Loss: 1.2e-3\x1b[0m')
        self.assertEqual(stream.total, 40)
        self.assertEqual(stream.rows[1]['test_accuracy'], .005)
        self.assertEqual(stream.rows[1]['test_loss'], .0012)
        stream.feed('Round 2 | accuracy: 0.8 | Loss: 0.1 | BMCTA: 79% | BTA: 82%')
        self.assertEqual(stream.rows[2]['bta'], .82)
        self.assertEqual(stream.rows[2]['bmcta'], .79)
        stream.feed('[METRIC] ' + json.dumps({'kind': 'node', 'round': 2, 'metrics': {'client_id': 0, 'train_loss': .2, 'device_ipv4': '192.168.1.9'}}))
        self.assertEqual(stream.nodes['0']['device_ipv4'], '192.168.1.9')
        self.assertNotIn('train_loss', stream.rows[2])
        stream.feed('[Client 0] fit done | Round 2 | loss=99 acc=0.01')
        self.assertNotIn('train_loss', stream.rows[2])
        stream.feed('[METRIC] {bad}')
        stream.feed('[METRIC] []')
        self.assertEqual(log_level('[METRIC] {}'), 'METRIC')
        client = MetricStream(client=True)
        client.feed('[METRIC] {"kind":"node","round":1,"metrics":{"client_id":0,"train_loss":0.5}}')
        self.assertEqual(client.rows[1]['train_loss'], .5)

    def test_partitions(self):
        with tempfile.TemporaryDirectory() as tmp:
            part = Path(tmp) / 'part.json'
            part.write_text(json.dumps({'clients': {'0': {'class_distribution': {'0': 3, '1': 7}}, '1': {'class_distribution': {'0': 5}}}}))
            _, labels, series = partition_data(part)
            self.assertEqual(labels, ['C0', 'C1'])
            self.assertEqual(series[1]['data'], [7, 0])

    def test_network(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            sock.listen()
            port = sock.getsockname()[1]
            self.assertTrue(ping(f'127.0.0.1:{port}')['ok'])
        self.assertFalse(ping(f'127.0.0.1:{port}')['ok'])
        with self.assertRaises(ValueError):
            endpoint('host:99999')


class AsyncProcessTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ctl = Controller(Path(self.tmp.name))

    async def asyncTearDown(self):
        await self.ctl.shutdown()
        self.tmp.cleanup()

    async def start_stub(self, code, mode='simulation', values=None):
        with patch('dashboard.controller.command', return_value=[sys.executable, '-u', '-c', code]), \
             patch('dashboard.controller.check_port_in_use', return_value=False):
            return await self.ctl.start(mode, values or VALUES)

    async def test_stream_success_failure_and_saved_summary(self):
        for code in [0, 7]:
            job = await self.start_stub(f"print('Round 1/2 | Test Accuracy: 75% | Loss: 0.2'); raise SystemExit({code})")
            await asyncio.wait_for(job.task, 10)
            self.assertEqual(job.state, 'completed' if code == 0 else 'failed')
            self.assertEqual(job.exit_code, code)
            self.assertEqual(job.metrics.rows[1]['test_accuracy'], .75)
            summary = json.loads((job.folder / 'summary.json').read_text(encoding='utf-8'))
            self.assertEqual(summary['exit_code'], code)
            self.assertEqual(summary['rounds'][0]['test_accuracy'], .75)
            self.assertTrue((job.folder / 'config.yaml').is_file())
            self.assertIn('Round 1/2', (job.folder / 'console.log').read_text(encoding='utf-8'))

    async def test_concurrent_nodes_duplicate_and_stop(self):
        code = "import time; print('ready',flush=True); time.sleep(60)"
        server = await self.start_stub(code, 'server')
        client = await self.start_stub(code, 'client')
        with self.assertRaises(ValueError):
            await self.start_stub(code, 'client')
        with self.assertRaises(ValueError):
            await self.start_stub(code, 'simulation')
        await self.ctl.stop(client.id)
        await asyncio.wait_for(client.task, 15)
        self.assertEqual(client.state, 'stopped')
        self.assertIsNone(server.process.returncode)
        await self.ctl.stop(server.id)
        await asyncio.wait_for(server.task, 15)
        self.assertEqual(server.state, 'stopped')

    async def test_stop_descendant(self):
        code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print('CHILD',p.pid,flush=True); time.sleep(60)"
        job = await self.start_stub(code)
        child_pid = None
        for _ in range(100):
            for line in self.ctl.logs:
                if line['text'].startswith('CHILD '):
                    child_pid = int(line['text'].split()[1])
            if child_pid:
                break
            await asyncio.sleep(.05)
        self.assertIsNotNone(child_pid)
        await self.ctl.stop(job.id)
        await asyncio.wait_for(job.task, 15)
        self.assertFalse(psutil.pid_exists(child_pid))
        self.assertEqual(job.state, 'stopped')

    async def test_spawn_failure(self):
        with patch('dashboard.controller.command', return_value=['fedmedai-no-such-executable']):
            with self.assertRaises(OSError):
                await self.ctl.start('simulation', VALUES)
        self.assertEqual(next(iter(self.ctl.jobs.values())).state, 'failed')


if __name__ == '__main__':
    unittest.main()
