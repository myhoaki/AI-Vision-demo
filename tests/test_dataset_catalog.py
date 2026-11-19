import sqlite3
import tempfile
import unittest
from pathlib import Path

from app import main


class SharedDatasetCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.old_dataset = main.DATASET_ROOT
        self.old_runs = main.RUNS_DIR
        self.old_observability = main.OBSERVABILITY_ONLY
        self.old_labeler_db = main.LABELER_DB
        main.DATASET_ROOT = self.root / "source"
        main.RUNS_DIR = self.root / "runs"
        main.LABELER_DB = self.root / "labeler.sqlite3"
        main.OBSERVABILITY_ONLY = False
        (main.DATASET_ROOT / "images").mkdir(parents=True)
        (main.DATASET_ROOT / "labels").mkdir()
        db = sqlite3.connect(main.LABELER_DB)
        db.executescript("""
            CREATE TABLE text_sessions(id TEXT PRIMARY KEY, root_name TEXT);
            CREATE TABLE text_annotations(session_id TEXT, path TEXT, source_path TEXT, active INTEGER);
            CREATE TABLE container_counts(session_id TEXT, path TEXT, container_count INTEGER);
            INSERT INTO text_sessions VALUES ('verified', 'container_images_verified');
        """)
        db.commit()
        db.close()

    def tearDown(self):
        main.DATASET_ROOT = self.old_dataset
        main.RUNS_DIR = self.old_runs
        main.OBSERVABILITY_ONLY = self.old_observability
        main.LABELER_DB = self.old_labeler_db
        self.temp.cleanup()

    def add_image(self, name, data, label="0 0.5 0.5 0.4 0.4\n"):
        image = main.DATASET_ROOT / "images" / name
        image.write_bytes(data)
        (main.DATASET_ROOT / "labels" / f"{image.stem}.txt").write_text(label)
        return image

    def add_count_label(self, name, count, path=None):
        db = sqlite3.connect(main.LABELER_DB)
        annotation_path = path or f"crop-{name}"
        db.execute("INSERT INTO text_annotations VALUES ('verified',?,?,1)", (annotation_path, f"images/{name}"))
        db.execute("INSERT INTO container_counts VALUES ('verified',?,?)", (annotation_path, count))
        db.commit()
        db.close()

    def test_hash_catalog_deduplicates_bytes_and_keeps_group_splits(self):
        self.add_image("truck-a.jpg", b"same-bytes")
        self.add_image("truck-a_cropped.jpg", b"same-bytes")
        self.add_image("truck-b.jpg", b"different-b")
        first = main.sync_dataset_catalog()
        self.assertEqual(first["uniqueOriginalImages"], 2)
        self.assertEqual(first["duplicatePaths"], 1)

        db = sqlite3.connect(main.dataset_catalog_path())
        before = dict(db.execute("SELECT group_key,split FROM split_assignments"))
        db.close()
        self.add_image("truck-c.jpg", b"different-c")
        main.sync_dataset_catalog()
        db = sqlite3.connect(main.dataset_catalog_path())
        after = dict(db.execute("SELECT group_key,split FROM split_assignments"))
        db.close()
        self.assertTrue(all(after[key] == split for key, split in before.items()))
        self.assertIn("truck-c", after)

    def test_detector_snapshot_uses_original_symlinks_and_versioned_splits(self):
        for name, payload in (("truck-a.jpg", b"a"), ("truck-b.jpg", b"b"), ("truck-c.jpg", b"c"), ("truck-d.jpg", b"d")):
            self.add_image(name, payload)
        yaml, counts, _, skipped, diagnostics = main.build_training_dataset("a" * 32)
        self.assertEqual(skipped, 0)
        self.assertEqual(sum(counts.values()), 4)
        self.assertTrue(yaml.is_file())
        self.assertTrue(all(p.is_symlink() for p in (yaml.parent / "images").rglob("*.jpg")))
        self.assertTrue(diagnostics["datasetVersionId"])
        db = sqlite3.connect(main.dataset_catalog_path())
        version = db.execute("SELECT train_count,validation_count,test_count FROM dataset_versions WHERE dataset_version_id=?", (diagnostics["datasetVersionId"],)).fetchone()
        run_ref = db.execute("SELECT dataset_version_id FROM training_runs WHERE run_id=?", ("a" * 32,)).fetchone()
        db.close()
        self.assertEqual(sum(version), 4)
        self.assertEqual(run_ref[0], diagnostics["datasetVersionId"])

    def test_server_classifier_manifest_links_original_images_without_copying(self):
        self.add_image("one.jpg", b"one", "0 0.5 0.5 0.4 0.4\n")
        self.add_image("two.jpg", b"two", "0 0.3 0.3 0.2 0.2\n0 0.7 0.7 0.2 0.2\n")
        self.add_count_label("one.jpg", 2)
        self.add_count_label("two.jpg", 1)
        manifest = main.prepare_server_classifier_dataset()
        self.assertEqual(manifest["imageCount"], 2)
        self.assertEqual(manifest["classCounts"], {"1_container": 1, "2_containers": 1})
        self.assertEqual(manifest["labelSource"], "container_images_verified")
        paths = list(Path(manifest["path"]).glob("*/*.jpg"))
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(path.is_symlink() for path in paths))
        self.assertEqual({path.resolve() for path in paths}, set((main.DATASET_ROOT / "images").glob("*.jpg")))

    def test_annotation_revisions_and_inactive_source_paths_are_preserved(self):
        image = self.add_image("truck-a.jpg", b"a")
        first = main.sync_dataset_catalog()
        image_id = first["pathIds"]["images/truck-a.jpg"]
        (main.DATASET_ROOT / "labels" / "truck-a.txt").write_text("0 0.4 0.4 0.2 0.2\n")
        second = main.sync_dataset_catalog()
        self.assertEqual(second["uniqueOriginalImages"], 1)
        image.unlink()
        third = main.sync_dataset_catalog()
        self.assertEqual(third["uniqueOriginalImages"], 0)
        db = sqlite3.connect(main.dataset_catalog_path())
        versions = db.execute("SELECT version FROM annotations WHERE image_id=? ORDER BY version", (image_id,)).fetchall()
        retained = db.execute("SELECT COUNT(*) FROM source_images WHERE image_id=?", (image_id,)).fetchone()[0]
        db.close()
        self.assertEqual([row[0] for row in versions], [1, 2])
        self.assertEqual(retained, 1)

    def test_legacy_runs_get_an_explicit_unreconciled_dataset_version(self):
        run_id = "f" * 32
        job_path = main.RUNS_DIR / "jobs" / run_id / "job.json"
        job_path.parent.mkdir(parents=True)
        job_path.write_text('{"id":"' + run_id + '","createdAt":"2026-01-01T00:00:00Z","splitCounts":{"train":8,"val":1,"test":1}}')
        db = main.dataset_catalog.connect(main.dataset_catalog_path())
        main.migrate_legacy_training_records(db)
        row = db.execute("SELECT d.manifest_json,t.dataset_version_id FROM training_runs t JOIN dataset_versions d USING(dataset_version_id) WHERE t.run_id=?", (run_id,)).fetchone()
        db.close()
        self.assertTrue(row[1].startswith("legacy-"))
        self.assertFalse(__import__("json").loads(row[0])["reconciled"])

    def test_summary_keeps_originals_and_model_samples_separate(self):
        self.add_image("one.jpg", b"one", "0 0.5 0.5 0.3 0.3\n")
        self.add_image("two.jpg", b"two", "0 0.3 0.3 0.2 0.2\n0 0.7 0.7 0.2 0.2\n")
        self.add_image("many.jpg", b"many", "0 0.2 0.2 0.1 0.1\n0 0.4 0.4 0.1 0.1\n0 0.6 0.6 0.1 0.1\n")
        self.add_count_label("one.jpg", 2)
        self.add_count_label("two.jpg", 1)
        result = main.datasets_summary()
        self.assertEqual(result["sharedOriginalImages"], 3)
        self.assertEqual(result["models"]["containerDetector"]["trainingSamples"], 3)
        self.assertEqual(result["models"]["containerCountClassifier"]["trainingSamples"], 2)
        self.assertEqual(result["models"]["containerCountClassifier"]["excluded"], 1)
        self.assertEqual(result["models"]["containerCountClassifier"]["derivedSamples"], 0)

    def test_verified_source_count_overrides_conflicting_crop_counts(self):
        db = sqlite3.connect(main.LABELER_DB)
        db.execute("ALTER TABLE text_annotations ADD COLUMN cached_crop_path TEXT")
        db.execute("CREATE TABLE source_container_counts(source_path TEXT, container_count INTEGER)")
        db.execute("INSERT INTO text_annotations VALUES ('verified','crop-1','images/truck.jpg',1,'crop-1.png')")
        db.execute("INSERT INTO text_annotations VALUES ('verified','crop-2','images/truck.jpg',1,'crop-2.png')")
        db.execute("INSERT INTO container_counts VALUES ('verified','crop-1',1)")
        db.execute("INSERT INTO container_counts VALUES ('verified','crop-2',2)")
        db.execute("INSERT INTO source_container_counts VALUES ('images/truck.jpg',2)")
        db.commit()
        db.close()

        result = main.server_container_count_labels(required=True)
        self.assertEqual(result["labels"]["images/truck.jpg"], 2)
        self.assertNotIn("images/truck.jpg", result["conflicts"])


if __name__ == "__main__":
    unittest.main()
