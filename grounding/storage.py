"""SQLite metadata and local artifact storage. Each operation uses its own connection."""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get('GROUNDING_DATA_DIR', Path(__file__).resolve().parents[1] / 'data')).resolve()


def now():
    return datetime.now(timezone.utc).isoformat()


def uid():
    return uuid.uuid4().hex


def connection():
    ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ROOT / 'metadata.sqlite3', timeout=30)
    conn.execute('PRAGMA busy_timeout=30000')
    return conn


def init_db():
    for name in ('images', 'versions', 'runs', 'exports'):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    with connection() as conn:
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('CREATE TABLE IF NOT EXISTS documents (kind TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY(kind,id))')


def put(kind, id, data):
    encoded = json.dumps(data, allow_nan=False, ensure_ascii=False)
    with connection() as conn:
        conn.execute('INSERT INTO documents(kind,id,data,created_at) VALUES (?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET data=excluded.data', (kind, id, encoded, data.get('created_at', now())))
    return data


def get(kind, id):
    with connection() as conn:
        row = conn.execute('SELECT data FROM documents WHERE kind=? AND id=?', (kind, id)).fetchone()
    if row is None:
        raise KeyError(f'{kind} {id} does not exist')
    return json.loads(row[0])


def list_items(kind):
    with connection() as conn:
        rows = conn.execute('SELECT data FROM documents WHERE kind=? ORDER BY created_at DESC,id', (kind,)).fetchall()
    return [json.loads(row[0]) for row in rows]


def delete(kind, id):
    with connection() as conn:
        conn.execute('DELETE FROM documents WHERE kind=? AND id=?', (kind, id))


def patch(kind, id, updates):
    """Atomic read/update so worker progress cannot erase UI control requests."""
    with connection() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT data FROM documents WHERE kind=? AND id=?', (kind, id)).fetchone()
        if row is None:
            raise KeyError(f'{kind} {id} does not exist')
        data = json.loads(row[0])
        data.update(updates)
        conn.execute('UPDATE documents SET data=? WHERE kind=? AND id=?', (json.dumps(data, allow_nan=False), kind, id))
    return data


def safe_path(relative):
    resolved_root = ROOT.resolve()
    path = (resolved_root / relative).resolve()
    if not path.is_relative_to(resolved_root):
        raise ValueError('Path is outside the local data directory')
    return path


update = patch
