"""Read-only run diagnostics from durable events; no training dependencies required."""
import csv
import io
import math


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def all_events(store, run_id):
    after = 0
    while True:
        page = store.events(run_id, after=after, limit=5000)
        if not page:
            break
        yield from page
        after = page[-1]['seq']


def classification(matrix, labels):
    if not isinstance(matrix, list) or not matrix or len(matrix) > 32 or any(not isinstance(row, list) or len(row) != len(matrix) for row in matrix):
        return None
    if any(not finite(v) or v < 0 or int(v) != v for row in matrix for v in row):
        return None
    support = [sum(row) for row in matrix]
    predicted = [sum(row[c] for row in matrix) for c in range(len(matrix))]
    total = sum(support)
    if not total:
        return None
    classes = []
    for c, count in enumerate(support):
        tp = matrix[c][c]
        classes.append({'class_id': c, 'label': labels.get(str(c), f'Class {c}'), 'support': count,
                        'precision': tp / predicted[c] if predicted[c] else None,
                        'recall': tp / count if count else None,
                        'f1': 2 * tp / (count + predicted[c]) if count + predicted[c] else None})
    represented = [c for c in classes if c['support']]
    return {'matrix': matrix, 'classes': classes, 'samples': total,
            'accuracy': sum(matrix[c][c] for c in range(len(matrix))) / total,
            'balanced_accuracy': sum(c['recall'] for c in represented) / len(represented),
            'macro_f1': sum(c['f1'] for c in represented) / len(represented),
            'weighted_f1': sum(c['f1'] * c['support'] for c in represented) / total}


def snapshot(store, run, mode, labels=None, client_id=None, full=False):
    labels = labels or {}
    rounds, clients, epochs, systems, logs, latest, support = {}, {}, {}, [], [], {}, {}
    event_count = 0
    for e in all_events(store, run['id']):
        event_count += 1
        cid, rnd, phase = e['client_id'], e['round'], e['phase']
        if phase not in ('process_log', 'system_sample'):
            latest[cid] = e
        if phase == 'round_completed' and cid == -1:
            rounds[rnd] = dict(e.get('metrics', {}), round=rnd)
        elif phase in ('fit_completed', 'evaluation_completed', 'evaluation_skipped') and cid >= 0:
            row = clients.setdefault((cid, rnd), {'client_id': cid, 'round': rnd})
            if phase == 'fit_completed':
                row.update(e.get('metrics', {}))
            else:
                row.update(eval_accuracy=e.get('accuracy'), eval_loss=e.get('loss'),
                           test_samples=e.get('num_samples', 0), evaluation_skipped=phase == 'evaluation_skipped',
                           evaluation_seconds=e.get('seconds'), confusion_matrix=e.get('confusion_matrix'))
        elif phase == 'epoch_completed':
            epochs[(cid, rnd, e.get('epoch', 0))] = e
        elif phase == 'system_sample':
            systems.append(e)
        elif phase == 'process_log':
            logs.append(e)
        elif phase == 'support_configuration':
            support[(cid, rnd)] = e
    # Client reports are optional for older runs. Never infer zero for missing measurements.
    for rnd in sorted(set(rounds) | {r for _, r in clients}):
        if rnd not in rounds and mode == 'coordinator':
            continue  # An unfinished round must not look like an aggregated round.
        local = [c for (cid, r), c in clients.items() if r == rnd]
        if mode == 'institution' and not any('evaluation_skipped' in c for c in local):
            continue
        row = rounds.setdefault(rnd, {'round': rnd})
        tests = [c for c in local if c.get('test_samples', 0) > 0 and finite(c.get('eval_accuracy'))]
        if mode == 'institution' and tests:
            row.update(eval_accuracy=tests[0]['eval_accuracy'], eval_loss=tests[0]['eval_loss'])
        row['evaluated_clients'] = len(tests)
        row['skipped_clients'] = sum(bool(c.get('evaluation_skipped')) for c in local)
        fit = [c for c in local if finite(c.get('train_loss')) and c.get('num_samples', 0) > 0]
        if fit:
            n = sum(c['num_samples'] for c in fit)
            row['train_loss'] = sum(c['train_loss'] * c['num_samples'] for c in fit) / n
            row['train_accuracy'] = sum(c['train_accuracy'] * c['num_samples'] for c in fit) / n
        for field in ('upload_payload_kb', 'download_payload_kb'):
            values = [c[field] for c in local if finite(c.get(field))]
            row[field] = sum(values) if values else None
            row[field + '_reporting'] = len(values)
    curve = [rounds[r] for r in sorted(rounds)]
    valid = [r for r in curve if finite(r.get('eval_accuracy'))]
    best = max(valid, key=lambda r: r['eval_accuracy']) if valid else {}
    mcta = [r['mcta'] for r in curve if finite(r.get('mcta'))]
    chosen = [row for key, row in sorted(clients.items(), key=lambda p: (p[0][1], p[0][0])) if client_id is None or key[0] == client_id]
    evaluations = []
    for rnd in sorted({r['round'] for r in chosen}):
        matrices = [r['confusion_matrix'] for r in chosen if r['round'] == rnd and classification(r.get('confusion_matrix'), labels)]
        if matrices and all(len(m) == len(matrices[0]) for m in matrices):
            pooled = [[sum(m[i][j] for m in matrices) for j in range(len(matrices[0]))] for i in range(len(matrices[0]))]
            evaluations.append(dict(classification(pooled, labels), round=rnd, reporting_clients=len(matrices)))
    picked_epochs = [e for _, e in sorted(epochs.items(), key=lambda item: (item[0][1], item[0][2], item[0][0]))
                     if client_id is None or e['client_id'] == client_id]
    picked_systems = [e for e in systems if client_id is None or e['client_id'] == client_id]
    picked_logs = [e for e in logs if client_id is None or e['client_id'] == client_id]
    return {'run_id': run['id'], 'status': run['status'], 'configuration': run['spec'],
            'partition_hash': run.get('partition_hash'), 'dataset_hash': run.get('dataset_hash'),
            'event_count': event_count, 'rounds': curve, 'client_rounds': chosen,
            'epochs': picked_epochs if full else picked_epochs[-2000:],
            'systems': picked_systems if full else picked_systems[-2000:],
            'logs': picked_logs if full else picked_logs[-250:],
            'limits': {'epochs_total': len(picked_epochs), 'systems_total': len(picked_systems), 'logs_total': len(picked_logs)},
            'latest': list(latest.values()), 'support': [e for (cid, _), e in sorted(support.items()) if client_id is None or cid == client_id],
            'evaluations': evaluations, 'summary': {
                'current_accuracy': valid[-1]['eval_accuracy'] if valid else None,
                'final_accuracy': valid[-1]['eval_accuracy'] if valid and run['status'] == 'completed' else None,
                'bta': best.get('eval_accuracy'), 'bta_round': best.get('round'),
                'bmcta': max(mcta) if mcta else None,
                'completed_rounds': len(curve),
                'upload_payload_kb': sum(r['upload_payload_kb'] for r in curve if finite(r.get('upload_payload_kb'))) if any(finite(r.get('upload_payload_kb')) for r in curve) else None}}


def metrics_csv(data):
    records = []
    for kind, rows in [('round', data['rounds']), ('client_round', data['client_rounds']), ('epoch', data['epochs']), ('system', data['systems'])]:
        for row in rows:
            records.append({'record_type': kind, **{k: v for k, v in row.items() if v is None or isinstance(v, (int, float, bool, str))}})
    keys = ['record_type'] + sorted({k for row in records for k in row if k != 'record_type'})
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=keys)
    writer.writeheader()
    for row in records:
        # Spreadsheet exports must treat log/source strings as data, never formulas.
        writer.writerow({k: ("'" + v if isinstance(v, str) and v[:1] in ('=', '+', '-', '@') else v) for k, v in row.items()})
    return output.getvalue()
