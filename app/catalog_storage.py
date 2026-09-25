"""Revisioned catalog persistence for the two supported storage backends.

The host store supplies its existing lock and connection. Backend selection is
explicit in the store class; catalog operations never inspect backend attributes.
"""
from copy import deepcopy
import json


def revised_record(current, item_id, version, payload, expected_revision, immutable):
    """Apply identical revision/immutability rules inside each backend transaction."""
    if immutable and current:
        if current["payload"] != payload:
            raise ValueError("published version is immutable")
        return deepcopy(current)
    revision = current["revision"] if current else 0
    if expected_revision is not None and expected_revision != revision:
        raise ValueError("revision conflict; reload before saving")
    return {
        "id": item_id, "version": version,
        "revision": revision + 1, "payload": deepcopy(payload),
    }


class MemoryCatalogStorage:
    """Catalog operations under the in-memory store's lock."""

    def catalog_list(self, kind):
        with self._lock:
            return deepcopy([
                value for (record_kind, _, _), value in getattr(self, "_catalog", {}).items()
                if record_kind == kind
            ])

    def catalog_put(self, kind, item_id, version, payload, *, expected_revision=None, immutable=False):
        with self._lock:
            if not hasattr(self, "_catalog"):
                self._catalog = {}
            key = (kind, item_id, version)
            record = revised_record(
                self._catalog.get(key), item_id, version, payload, expected_revision, immutable,
            )
            self._catalog[key] = record
            return deepcopy(record)


class SQLiteCatalogStorage:
    """Catalog operations using SQLite transactions; schema is created at startup."""

    def catalog_list(self, kind):
        with self._lock:
            rows = self._conn.execute("SELECT payload FROM catalog WHERE kind = ?", (kind,))
            return [json.loads(row[0]) for row in rows.fetchall()]

    def catalog_put(self, kind, item_id, version, payload, *, expected_revision=None, immutable=False):
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    "SELECT payload FROM catalog WHERE kind = ? AND id = ? AND version = ?",
                    (kind, item_id, version),
                ).fetchone()
                record = revised_record(
                    json.loads(row[0]) if row else None,
                    item_id, version, payload, expected_revision, immutable,
                )
                self._conn.execute(
                    "INSERT OR REPLACE INTO catalog VALUES (?, ?, ?, ?)",
                    (kind, item_id, version, json.dumps(record, ensure_ascii=False)),
                )
                self._conn.commit()
                return record
            except Exception:
                self._conn.rollback()
                raise
