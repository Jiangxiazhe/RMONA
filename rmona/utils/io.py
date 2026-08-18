"""Experiment utilities: incremental CSV logging and completed-task tracking (for --resume)."""
from __future__ import annotations

import csv
import os
from collections.abc import Iterable

__all__ = ["append_rows", "load_existing", "ensure_dir"]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def append_rows(path: str, rows: Iterable[dict]) -> None:
    """Append rows to a CSV (write header if the file does not exist yet)."""
    rows = list(rows)
    if not rows:
        return
    exists = os.path.exists(path)
    with open(path, 'a') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)


def load_existing(path: str, key_cols: tuple = ('seed', 'method', 'lr')) -> set:
    """Read an existing CSV (for resume); returns the set of completed task keys.

    Args:
        path: CSV path (returns an empty set if missing).
        key_cols: column names forming the unique task key. Missing/unparsable
            values are stored as ``None``.
    """
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for row in csv.DictReader(f):
            key = []
            for c in key_cols:
                v = row.get(c)
                if c == 'lr' and v not in (None, ''):
                    try:
                        v = float(v)
                    except ValueError:
                        v = None
                key.append(v)
            done.add(tuple(key))
    return done


def summarize(rows: list, group: str, stat: str = 'mean') -> dict:
    """Group rows by a column and aggregate (mean/median) — experiment helper."""
    import collections
    agg: dict = collections.defaultdict(list)
    for r in rows:
        agg[r[group]].append(float(r[stat]))
    return {k: (sum(v) / len(v) if stat == 'mean' else sorted(v)[len(v) // 2])
            for k, v in agg.items()}
