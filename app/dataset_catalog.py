"""Persistent, content-addressed catalog for shared source images and datasets."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1
NAMESPACE = uuid.UUID("a7dc197a-9f81-4f5e-9d83-8475b0653ee1")


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.executescript("""
      CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS source_images(
        image_id TEXT PRIMARY KEY, sha256 TEXT NOT NULL, byte_size INTEGER NOT NULL,
        width INTEGER, height INTEGER, format TEXT, first_imported_at TEXT NOT NULL,
        metadata_json TEXT NOT NULL DEFAULT '{}'
      );
      CREATE INDEX IF NOT EXISTS source_images_sha ON source_images(sha256);
      CREATE TABLE IF NOT EXISTS image_sources(
        source_id INTEGER PRIMARY KEY, image_id TEXT NOT NULL REFERENCES source_images(image_id),
        source_path TEXT NOT NULL UNIQUE, original_filename TEXT NOT NULL,
        dataset_source TEXT NOT NULL, imported_at TEXT NOT NULL, mtime_ns INTEGER NOT NULL,
        group_key TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1, UNIQUE(image_id, source_path)
      );
      CREATE INDEX IF NOT EXISTS image_sources_group ON image_sources(group_key);
      CREATE TABLE IF NOT EXISTS annotations(
        annotation_id INTEGER PRIMARY KEY, image_id TEXT NOT NULL REFERENCES source_images(image_id),
        task TEXT NOT NULL, model_key TEXT NOT NULL, version INTEGER NOT NULL,
        payload_json TEXT NOT NULL, source_path TEXT, created_at TEXT NOT NULL,
        UNIQUE(image_id, task, model_key, version)
      );
      CREATE TABLE IF NOT EXISTS split_assignments(
        group_key TEXT PRIMARY KEY, split TEXT NOT NULL CHECK(split IN ('train','validation','test')),
        strategy TEXT NOT NULL, assigned_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS dataset_versions(
        dataset_version_id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL, version INTEGER NOT NULL,
        model_key TEXT NOT NULL, created_at TEXT NOT NULL, annotation_versions_json TEXT NOT NULL,
        train_count INTEGER NOT NULL, validation_count INTEGER NOT NULL, test_count INTEGER NOT NULL,
        excluded_count INTEGER NOT NULL, derived_count INTEGER NOT NULL, manifest_json TEXT NOT NULL,
        UNIQUE(dataset_id, version)
      );
      CREATE TABLE IF NOT EXISTS dataset_version_images(
        dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(dataset_version_id),
        image_id TEXT NOT NULL REFERENCES source_images(image_id), split TEXT,
        excluded_reason TEXT, annotation_version INTEGER, PRIMARY KEY(dataset_version_id,image_id)
      );
      CREATE TABLE IF NOT EXISTS derived_samples(
        sample_id TEXT PRIMARY KEY, dataset_version_id TEXT REFERENCES dataset_versions(dataset_version_id),
        image_id TEXT NOT NULL REFERENCES source_images(image_id), model_key TEXT NOT NULL,
        sample_type TEXT NOT NULL, label_json TEXT NOT NULL, crop_json TEXT,
        preprocessing_version TEXT NOT NULL, cached_path TEXT, created_at TEXT NOT NULL
      );
      CREATE TABLE IF NOT EXISTS training_runs(
        run_id TEXT PRIMARY KEY, model_key TEXT NOT NULL,
        dataset_version_id TEXT REFERENCES dataset_versions(dataset_version_id),
        legacy_dataset_id TEXT, started_at TEXT
      );
      CREATE TABLE IF NOT EXISTS unreconciled_samples(
        sample_id TEXT PRIMARY KEY, dataset_version_id TEXT NOT NULL REFERENCES dataset_versions(dataset_version_id),
        legacy_dataset_id TEXT NOT NULL, sample_path TEXT NOT NULL, label_json TEXT NOT NULL,
        split TEXT NOT NULL, reason TEXT NOT NULL
      );
    """)
    db.execute("INSERT OR IGNORE INTO schema_migrations VALUES (?, datetime('now'))", (SCHEMA_VERSION,))
    source_columns={row[1] for row in db.execute("PRAGMA table_info(image_sources)")}
    if "active" not in source_columns:
        db.execute("ALTER TABLE image_sources ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
    db.commit()
    return db


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_image_id(digest: str) -> str:
    return uuid.uuid5(NAMESPACE, digest).hex


def assign_group_splits(db: sqlite3.Connection, group_keys: Iterable[str], now: str) -> None:
    existing = {row[0] for row in db.execute("SELECT group_key FROM split_assignments")}
    new_keys = sorted(set(group_keys) - existing)
    if not new_keys:
        return
    # Seeded order makes the first catalog build reproducible; existing groups never move.
    import random
    random.Random(42).shuffle(new_keys)
    split_names = ("train", "validation", "test")
    existing_counts = {name: db.execute("SELECT COUNT(*) FROM split_assignments WHERE split=?", (name,)).fetchone()[0]
                       for name in split_names}
    if not sum(existing_counts.values()):
        target_train = max(1, round(len(new_keys) * .8))
        target_validation = max(1, round(len(new_keys) * .1))
        target_test = max(1, round(len(new_keys) * .1))
        while target_train + target_validation + target_test > len(new_keys):
            target_train -= 1
        remaining = {"train": target_train, "validation": target_validation, "test": target_test}
        for key in new_keys:
            split = max(split_names, key=lambda name: remaining[name])
            db.execute("INSERT INTO split_assignments VALUES (?,?,?,?)", (key, split, "persistent-group-80-10-10-v1", now))
            remaining[split] -= 1
        return
    for key in new_keys:
        total = sum(existing_counts.values())
        targets = {"train": .8, "validation": .1, "test": .1}
        target_counts = {name: max(1, round(targets[name] * (total + 1))) for name in split_names}
        split = max(split_names, key=lambda name: (target_counts[name] - existing_counts[name], -split_names.index(name)))
        db.execute("INSERT INTO split_assignments VALUES (?,?,?,?)", (key, split, "persistent-group-80-10-10-v1", now))
        existing_counts[split] += 1


def next_dataset_version(db: sqlite3.Connection, dataset_id: str, model_key: str, now: str,
                         members: list[dict[str, Any]], annotation_versions: dict[str, int],
                         excluded: list[dict[str, str]] | None = None,
                         derived_count: int = 0,
                         split_counts_override: dict[str, int] | None = None,
                         manual_reconciliation: list[dict[str, Any]] | None = None) -> str:
    version = db.execute("SELECT COALESCE(MAX(version),0)+1 FROM dataset_versions WHERE dataset_id=?", (dataset_id,)).fetchone()[0]
    version_id = uuid.uuid4().hex
    split_counts = {name: sum(1 for row in members if row.get("split") == name)
                    for name in ("train", "validation", "test")}
    if split_counts_override:
        split_counts.update(split_counts_override)
    excluded = excluded or []
    manifest = {"datasetId": dataset_id, "version": version, "model": model_key,
                "includedImageIds": [row["image_id"] for row in members], "excluded": excluded,
                "manualReconciliation": manual_reconciliation or []}
    db.execute("INSERT INTO dataset_versions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (version_id, dataset_id, version, model_key, now, json.dumps(annotation_versions, sort_keys=True),
         split_counts["train"], split_counts["validation"], split_counts["test"], len(excluded), derived_count,
         json.dumps(manifest, sort_keys=True)))
    db.executemany("INSERT INTO dataset_version_images VALUES (?,?,?,?,?)",
        [(version_id, row["image_id"], row.get("split"), row.get("excluded_reason"), row.get("annotation_version")) for row in members])
    return version_id
