"""Small durable JSON document/event store with SQLite transactions."""
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.RLock()
        with self.connection() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    kind TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL,
                    PRIMARY KEY(kind,id));
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
                    run_id TEXT NOT NULL, data TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS events_run ON events(run_id,seq);
            ''')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=20)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, kind, key, default=None):
        with self.connection() as db:
            row = db.execute('SELECT data FROM documents WHERE kind=? AND id=?', (kind, key)).fetchone()
        return json.loads(row[0]) if row else default

    def all(self, kind):
        with self.connection() as db:
            rows = db.execute('SELECT data FROM documents WHERE kind=? ORDER BY rowid DESC', (kind,)).fetchall()
        return [json.loads(row[0]) for row in rows]

    def put(self, kind, value):
        with self.lock, self.connection() as db:
            db.execute('INSERT INTO documents VALUES(?,?,?) ON CONFLICT(kind,id) DO UPDATE SET data=excluded.data',
                       (kind, value['id'], json.dumps(value, allow_nan=False)))
        return value

    def update(self, kind, key, **changes):
        with self.lock:
            value = self.get(kind, key)
            if value is None:
                raise KeyError(key)
            value.update(changes)
            return self.put(kind, value)

    def delete(self, kind, key):
        with self.lock, self.connection() as db:
            db.execute('DELETE FROM documents WHERE kind=? AND id=?', (kind, key))

    def event(self, value):
        with self.lock, self.connection() as db:
            cursor = db.execute('INSERT OR IGNORE INTO events(id,run_id,data) VALUES(?,?,?)',
                                (value['id'], value['run_id'], json.dumps(value, allow_nan=False)))
            return cursor.rowcount == 1

    def events(self, run_id=None, after=0, limit=2000):
        with self.connection() as db:
            if run_id:
                rows = db.execute('SELECT seq,data FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?',
                                  (run_id, after, limit)).fetchall()
            else:
                rows = db.execute('SELECT seq,data FROM events WHERE seq>? ORDER BY seq LIMIT ?', (after, limit)).fetchall()
        return [dict(json.loads(data), seq=seq) for seq, data in rows]


def now():
    return time.time()
