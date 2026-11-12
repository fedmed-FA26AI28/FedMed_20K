"""Isolated training-environment tasks. Invoke through a configured Python executable."""
import argparse
import base64
import importlib
import io
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from portal.common import atomic_json, code_digest, digest

DATASETS = ('bloodmnist', 'organamnist', 'pathmnist')


def capabilities():
    result = {'python': platform.python_version(), 'platform': platform.platform(),
              'architecture': platform.machine(), 'code_hash': code_digest(), 'errors': [], 'cuda': False}
    for name in ('torch', 'torchvision', 'flwr', 'medmnist', 'numpy', 'yaml', 'psutil'):
        try:
            module = importlib.import_module(name)
            result[name] = str(getattr(module, '__version__', 'available'))
        except Exception as exc:
            result['errors'].append(f'{name}: {exc}')
    if not result['errors']:
        import torch
        result['cuda'] = torch.cuda.is_available()
        result['gpu'] = torch.cuda.get_device_name(0) if result['cuda'] else None
    release = Path('/etc/nv_tegra_release')
    result['jetpack_release'] = release.read_text().strip() if release.exists() else None
    result['ready'] = not result['errors']
    return result


def data(dataset, download=False):
    if dataset not in DATASETS:
        raise ValueError('Unsupported dataset')
    from datasets.medmnist_code import get_medmnist_datasets
    if download:
        Path(os.environ.get('FEDMED_DATA_ROOT', str(Path.home()/'.medmnist'))).mkdir(parents=True, exist_ok=True)
    return get_medmnist_datasets(dataset, download=download, augment=False, load_val=False)


def dataset_metadata(dataset, partition=None, client_id=None, download=False):
    import numpy as np
    from medmnist import INFO
    train, _, test, classes = data(dataset, download)
    indices, test_indices = list(range(len(train))), list(range(len(test)))
    if partition is not None:
        entry = partition['clients'][str(client_id)]
        indices, test_indices = entry['indices'], entry['test_indices']
        if not indices or min(indices) < 0 or max(indices) >= len(train):
            raise ValueError('Invalid or empty local training partition')
        if test_indices and (min(test_indices) < 0 or max(test_indices) >= len(test)):
            raise ValueError('Local test indices do not match the dataset')
        actual = np.bincount(np.asarray(train.labels).reshape(-1)[indices], minlength=classes)
        if actual.tolist() != [entry['class_distribution'].get(str(i), 0) for i in range(classes)]:
            raise ValueError('Training labels do not match the partition manifest')
    hist = np.bincount(np.asarray(train.labels).reshape(-1)[indices], minlength=classes).tolist()
    previews = []
    from PIL import Image
    # Preview several supported classes; this never changes sample assignments.
    preview_groups = {}
    labels = np.asarray(test.labels).reshape(-1)
    for index in test_indices:
        preview_groups.setdefault(int(labels[index]), []).append(index)
    preview_indices = []
    offset = 0
    while len(preview_indices) < min(12, len(test_indices)):
        for group in preview_groups.values():
            if offset < len(group) and len(preview_indices) < 12:
                preview_indices.append(group[offset])
        offset += 1
    for index in preview_indices:
        image = Image.fromarray(test.imgs[index])
        out = io.BytesIO()
        image.save(out, format='PNG')
        previews.append({'index': index, 'label': int(np.asarray(test.labels[index]).reshape(-1)[0]),
                         'image': 'data:image/png;base64,' + base64.b64encode(out.getvalue()).decode()})
    root = Path(os.environ.get('FEDMED_DATA_ROOT', str(Path.home() / '.medmnist')))
    return {'dataset': dataset, 'train_samples': len(indices), 'test_samples': len(test_indices),
            'classes': INFO[dataset]['label'], 'class_distribution': hist, 'previews': previews,
            'dataset_hash': digest(root / f'{dataset}.npz'), 'assigned': partition is not None}


def prepare_partition(request):
    from datasets.partition import dirichlet_partition, practical_partition, save_partition
    spec = request['spec']
    train, _, test, _ = data(spec['dataset'])
    n = len(spec['node_ids'])
    if spec['partition'].startswith('dirichlet_'):
        parts, tests = dirichlet_partition(train, n, float(spec['partition'].split('_')[1]), spec['seed'], test_dataset=test)
    else:
        parts, tests = practical_partition(train, n, spec['seed'], test_dataset=test)
    if any(len(p) < 2 for p in parts):
        raise ValueError('This partition gives a client fewer than two training images. Choose another seed or roster.')
    path = save_partition(parts, train, spec['dataset'], spec['partition'], spec['seed'], n,
                          test_dataset=test, test_indices=tests, save_dir=request['output_dir'])
    root = Path(os.environ.get('FEDMED_DATA_ROOT', str(Path.home() / '.medmnist')))
    return {'partition_path': str(path.resolve()), 'partition_hash': digest(path),
            'dataset_hash': digest(root / f"{spec['dataset']}.npz"), 'capabilities': capabilities()}


def model_manifest(request):
    import torch
    from medmnist import INFO
    from models import get_model
    path = Path(request['path'])
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = get_model(request['model'], num_classes=len(INFO[request['dataset']]['label']))
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 28, 28))
    if not torch.isfinite(output).all():
        raise ValueError('Checkpoint produces non-finite outputs')
    scoped = request['strategy'] in ('fedbn', 'fedbns')
    if scoped and not request.get('institution_id'):
        raise ValueError('FedBN/FedBNS needs a verified institution checkpoint with its local BatchNorm state.')
    checksum = digest(path)
    owner = request.get('institution_id') if scoped else None
    identity = [checksum, request['dataset'], request['model'], request['strategy'],
                request['source_run'], request['source_round'], request['selection_rule'], owner]
    evaluation = request.get('evaluation')
    if evaluation is None and (path.parent/'fl_results.json').is_file():
        from portal.history import unique_rounds
        content = json.loads((path.parent/'fl_results.json').read_text(encoding='utf-8'))
        rows = unique_rounds(content.get('rounds_history', []))
        record = next((r for r in rows if int(r['round']) == request['source_round']), {})
        evaluation = {k: record.get(k) for k in ('eval_accuracy', 'eval_loss', 'bta', 'bmcta')}
    return {'id': hashlib.sha256(json.dumps(identity).encode()).hexdigest(), 'path': str(path.resolve()), 'checksum': checksum,
            'dataset': request['dataset'], 'model': request['model'], 'strategy': request['strategy'],
            'classes': INFO[request['dataset']]['label'], 'preprocessing': {
                'size': [28, 28], 'channels': 3, 'mean': [0.5]*3, 'std': [0.5]*3,
                'grayscale_repeat': request['dataset'] == 'organamnist', 'augmentation': False},
            'source_run': request['source_run'], 'source_round': request['source_round'],
            'selection_rule': request['selection_rule'], 'scope': 'institution' if scoped else 'global',
            'institution_id': owner, 'created_at': time.time(), 'verified': True, 'evaluation': evaluation,
            'size_bytes': path.stat().st_size}


def predict(request):
    import torch
    from PIL import Image
    from models import get_model
    manifest = request['manifest']
    if digest(manifest['path']) != manifest['checksum']:
        raise ValueError('Checkpoint checksum has changed. Import it again before prediction.')
    model = get_model(manifest['model'], num_classes=len(manifest['classes']))
    model.load_state_dict(torch.load(manifest['path'], map_location='cpu', weights_only=True), strict=True)
    model.eval()
    if request.get('sample_index') is not None:
        _, _, test, _ = data(manifest['dataset'])
        index = int(request['sample_index'])
        allowed = request.get('allowed_indices')
        if allowed is not None and index not in allowed:
            raise ValueError('This sample is not assigned to this institution.')
        if index < 0 or index >= len(test):
            raise ValueError('Sample index is outside the dataset')
        tensor, _ = test[index]
    else:
        raw = base64.b64decode(request.get('image_base64', ''), validate=True)
        if len(raw) > 5_000_000:
            raise ValueError('Image must be smaller than 5 MB')
        with Image.open(io.BytesIO(raw)) as image:
            if image.format not in ('PNG', 'JPEG') or image.width * image.height > 16_000_000:
                raise ValueError('Use a PNG/JPEG image smaller than 16 megapixels')
            image = image.convert('L' if manifest['dataset'] == 'organamnist' else 'RGB')
            image = image.resize((28, 28), Image.Resampling.BILINEAR)
            from datasets.medmnist_code import get_evaluation_transform
            tensor = get_evaluation_transform(manifest['dataset'])(image)
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(tensor.unsqueeze(0))
        if not torch.isfinite(logits).all():
            raise ValueError('Model produced non-finite scores')
        scores = torch.softmax(logits, dim=1)[0]
        top = torch.topk(scores, min(3, len(scores)))
    return {'class_id': int(logits.argmax(1).item()),
            'label': manifest['classes'][str(int(logits.argmax(1).item()))],
            'top': [{'class_id': int(i), 'label': manifest['classes'][str(int(i))], 'score': float(s)}
                    for i, s in zip(top.indices, top.values)],
            'latency_ms': round((time.perf_counter() - started)*1000, 2),
            'model_id': manifest['id'], 'dataset': manifest['dataset'], 'scores': scores.tolist()}


def execute(request):
    action = request['action']
    if action == 'preflight':
        return capabilities()
    if action in ('dataset', 'download'):
        partition = json.loads(Path(request['partition_path']).read_text()) if request.get('partition_path') else None
        return dataset_metadata(request['dataset'], partition, request.get('client_id'), action == 'download')
    if action == 'partition':
        return prepare_partition(request)
    if action == 'verify':
        result = capabilities()
        if not result['ready']:
            raise ValueError('; '.join(result['errors']))
        if result['code_hash'] != request['expected_code_hash']:
            raise ValueError('Training code differs from the coordinator. Sync the repository first.')
        if result['flwr'] != request['expected_flower']:
            raise ValueError('Flower versions differ. Use the same tested version on both devices.')
        partition_path = Path(request['partition_path'])
        if digest(partition_path) != request['partition_hash']:
            raise ValueError('Partition checksum mismatch')
        metadata = dataset_metadata(request['dataset'], json.loads(partition_path.read_text()), request['client_id'])
        if metadata['dataset_hash'] != request['dataset_hash']:
            raise ValueError('Local dataset file differs from the coordinator')
        return {'capabilities': result, 'dataset': metadata, 'partition_hash': digest(partition_path)}
    if action == 'model':
        return model_manifest(request)
    if action == 'predict':
        return predict(request)
    raise ValueError(f'Unknown worker action: {action}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--response', required=True)
    args = parser.parse_args()
    try:
        request = json.loads(Path(args.request).read_text(encoding='utf-8'))
        if request['action'] != 'preflight':
            import torch
            torch.set_num_threads(int(os.environ.get('FEDMED_CPU_THREADS', '1')))
        result = execute(request)
        atomic_json(args.response, {'ok': True, 'result': result})
    except Exception as exc:
        traceback.print_exc()
        atomic_json(args.response, {'ok': False, 'error': str(exc) or type(exc).__name__})
        sys.exit(1)


if __name__ == '__main__':
    main()
