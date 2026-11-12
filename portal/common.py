import hashlib
import json
import os
import secrets
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    hasher = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def code_digest():
    h = hashlib.sha256()
    for folder in ['algorithms', 'client', 'datasets', 'models', 'server', 'utils']:
        for path in sorted((ROOT / folder).glob('*.py')):
            h.update(path.relative_to(ROOT).as_posix().encode())
            h.update(path.read_bytes().replace(b'\r\n', b'\n'))
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def token_hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def password_hash(password, salt=None):
    salt = salt or secrets.token_hex(16)
    value = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), n=16384, r=8, p=1).hex()
    return salt + ':' + value


def password_matches(password, saved):
    return secrets.compare_digest(password_hash(password, saved.split(':')[0]), saved)


def connectivity(last_seen):
    age = time.time() - (last_seen or 0)
    return 'offline' if age > 30 else 'stale' if age > 15 else 'online'


def contained_path(value, roots):
    path = Path(value).expanduser().resolve()
    if not any(path == r.resolve() or r.resolve() in path.parents for r in map(Path, roots)):
        raise ValueError('Path must be inside a configured artifact directory.')
    return path
