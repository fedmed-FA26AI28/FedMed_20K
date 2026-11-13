"""Merge structured events and legacy text without mixing clients and server curves."""
import json
import math
import re
import time

NUMBER = r'([-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?)'
ANSI = re.compile(r'\x1b\[[0-?]*[ -/]*[@-~]')


def finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


class MetricStream:
    def __init__(self, client=False):
        self.client = client
        self.round = 0
        self.total = 0
        self.rows = {}
        self.nodes = {}
        self.node_history = {}
        self.revision = 0

    def feed(self, line: str):
        line = ANSI.sub('', line)
        if '[METRIC]' in line:
            try:
                event = json.loads(line.split('[METRIC]', 1)[1].strip())
                if not isinstance(event, dict):
                    return
                round_value = finite(event.get('round'))
                if round_value is None or round_value < 0:
                    return
                self.round = int(round_value)
                kind, metrics = event.get('kind'), event.get('metrics', {})
                if not isinstance(metrics, dict):
                    return
                if kind == 'node' and metrics.get('client_id') is not None:
                    cid = str(metrics['client_id'])
                    history = self.node_history.setdefault(cid, {})
                    record = history.setdefault(self.round, {'round': self.round})
                    record.update({k: v for k, v in metrics.items() if v is not None})
                    record['last_seen'] = time.time()
                    if self.round >= self.nodes.get(cid, {}).get('round', -1):
                        self.nodes[cid] = dict(record)
                    self.round = max(self.round, max(self.rows, default=0),
                                     max((n['round'] for n in self.nodes.values()), default=0))
                    if self.client:
                        self._merge({k: metrics[k] for k in ('train_loss', 'train_accuracy') if k in metrics})
                    # A client job contains only node events; its own train curve is useful.
                    self.revision += 1
                    return
                if kind in ('fit', 'evaluation', 'global'):
                    self._merge(metrics)
                return
            except (ValueError, TypeError, OverflowError):
                return
        match = re.search(r'\bRound\s*\[?\s*(\d+)(?:\s*/\s*(\d+))?', line, re.I)
        if match:
            self.round = int(match[1])
            if match[2]:
                self.total = int(match[2])
        metrics = {}
        train = bool(re.search(r'fit done|Epoch\s*\[|train[_ ]', line, re.I))
        patterns = {
            'train_accuracy' if train else 'test_accuracy': r'(?<![\w])(?:Test Accuracy|Global Test Acc|accuracy|acc)\s*[:=]\s*' + NUMBER + r'\s*(%)?',
            'train_loss' if train else 'test_loss': r'(?<![\w])(?:test[_ ]loss|loss)\s*[:=]\s*' + NUMBER,
            'bta': r'\bBTA\s*[:=]\s*' + NUMBER + r'\s*(%)?',
            'bmcta': r'\bB(?:MC|M)TA\s*[:=]\s*' + NUMBER + r'\s*(%)?',
            'mcta': r'\bMCTA\s*[:=]\s*' + NUMBER + r'\s*(%)?',
        }
        # Structured node messages supersede text fit metrics; do not turn a single
        # client's fit log into a server aggregate.
        if '[Client' in line and not self.client:
            return
        for key, pattern in patterns.items():
            match = re.search(pattern, line, re.I)
            if match:
                value = finite(match[1])
                if value is not None:
                    percent = match.lastindex == 2 and match[2] == '%'
                    metrics[key] = value / 100 if percent else value
        if metrics:
            self._merge(metrics)

    def _merge(self, metrics):
        row = self.rows.setdefault(self.round, {'round': self.round})
        for key, value in metrics.items():
            value = finite(value)
            if value is not None:
                row[key] = value
        # Recompute chronological best: late updates must not corrupt prior rounds.
        best = None
        for _, item in sorted(self.rows.items()):
            value = item.get('test_accuracy')
            if value is not None:
                best = max(best, value) if best is not None else value
            if best is not None and 'bta' not in item:
                item['bta'] = best
        self.revision += 1

    def records(self):
        return [dict(row) for _, row in sorted(self.rows.items())]

    def client_records(self, client_id):
        return [dict(row) for _, row in sorted(self.node_history.get(str(client_id), {}).items())]


def log_level(line):
    if '[METRIC]' in line:
        return 'METRIC'
    if re.search(r'\b(error|exception|traceback|failed)\b', line, re.I):
        return 'ERROR'
    if re.search(r'\bwarn(?:ing)?\b', line, re.I):
        return 'WARNING'
    return 'INFO'
