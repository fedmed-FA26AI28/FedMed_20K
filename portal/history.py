"""Read-only adapter for old result folders; never repairs source artifacts."""
import csv
import hashlib
import json
import math
from pathlib import Path


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def unique_rounds(rows):
    merged = {}
    for row in rows:
        key = int(float(row.get('round', 0)))
        if key <= 0:
            continue
        if key not in merged:
            merged[key] = {}
        # Evaluation records have precedence; blank duplicate records cannot erase values.
        for name, value in row.items():
            if value not in ('', None) and (name not in merged[key] or name.startswith(('eval_', 'mcta', 'wcta', 'bta', 'bmcta'))):
                merged[key][name] = value
    return [merged[key] for key in sorted(merged)]


def scan(roots):
    entries, seen = [], set()
    for root in map(Path, roots):
        if not root.exists():
            continue
        for summary in root.rglob('simulation_summary.csv'):
            try:
                with summary.open(encoding='utf-8-sig', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                for row in rows:
                    local = summary.parent / row['dataset'] / row['partition_mode'] / row['strategy']
                    resolved = str(local.resolve())
                    if resolved in seen:
                        continue
                    seen.add(resolved)
                    metrics = local / 'round_metrics.csv'
                    curves = []
                    if metrics.exists():
                        with metrics.open(encoding='utf-8-sig', newline='') as stream:
                            curves = unique_rounds(list(csv.DictReader(stream)))
                    partition_hash = row.get('partition_hash') or None
                    entry = {key: row.get(key) for key in ('dataset', 'strategy', 'model', 'seed', 'status', 'optimizer')}
                    entry.update(id=hashlib.sha256(resolved.encode()).hexdigest()[:24], name=summary.parent.name,
                                 partition=row.get('partition_mode'), partition_hash=partition_hash,
                                 num_clients=number(row.get('num_clients')), source='recorded',
                                 bta=number(row.get('bta')), bmcta=number(row.get('bmcta')),
                                 final_accuracy=number(curves[-1].get('eval_accuracy')) if curves else None,
                                 loss=number(curves[-1].get('eval_loss')) if curves else None,
                                 time_seconds=number(row.get('total_time_seconds')), rounds=len(curves),
                                 requested_rounds=number(row.get('rounds_requested')),
                                 model_payload_mb=sum(number(r.get('total_weight_size_kb')) or 0 for r in curves)/1024 if curves else None,
                                 curves=[{k: number(v) for k, v in r.items()} for r in curves],
                                 configuration={k: row.get(k) for k in ('local_epochs','batch_size','learning_rate','momentum','weight_decay','augment','local_test')},
                                 comparison_identity='unknown' if not partition_hash else 'recorded_hash',
                                 artifact_dir=resolved)
                    entries.append(entry)
            except (OSError, ValueError, KeyError):
                continue
    return entries
