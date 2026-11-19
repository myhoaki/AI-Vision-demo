import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from app import main


class DatasetExplorerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.original_dataset_root = main.DATASET_ROOT
        self.original_runs_dir = main.RUNS_DIR
        self.original_labeler_db = main.LABELER_DB
        main.DATASET_ROOT = self.root / "source"
        main.RUNS_DIR = self.root / "runs"
        main.LABELER_DB = self.root / "labeler.sqlite3"
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
        main.DATASET_ROOT = self.original_dataset_root
        main.RUNS_DIR = self.original_runs_dir
        main.LABELER_DB = self.original_labeler_db
        self.temp.cleanup()

    def image(self, relative):
        path = main.DATASET_ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
        return path

    def test_detection_view_reports_boxes_status_search_and_pagination(self):
        self.image("images/a.jpg")
        self.image("images/b.jpg")
        (main.DATASET_ROOT / "labels" / "a.txt").write_text("0 0.5 0.5 0.4 0.4\n0 0.2 0.2 0.1 0.1\n")
        (main.DATASET_ROOT / "labels" / "b.txt").write_text("")

        first = main.dataset_explorer_detection(page=1, pageSize=1)
        self.assertEqual(first["total"], 2)
        self.assertEqual(first["stats"], {"totalImages": 2, "annotatedImages": 1, "unannotatedImages": 1, "totalBoxes": 2})
        self.assertEqual(first["items"][0]["boxCount"], 2)
        second = main.dataset_explorer_detection(page=1, pageSize=1, annotation="unannotated")
        self.assertEqual(second["total"], 1)
        self.assertEqual(second["items"][0]["annotationStatus"], "unannotated")
        searched = main.dataset_explorer_detection(search="B.")
        self.assertEqual([row["filename"] for row in searched["items"]], ["b.jpg"])

    def test_classifier_view_uses_verified_server_labels_without_dataset_selection(self):
        one = self.image("images/one.jpg")
        two = self.image("images/two.jpg")
        (main.DATASET_ROOT / "labels" / "one.txt").write_text("0 0.5 0.5 0.2 0.2\n")
        (main.DATASET_ROOT / "labels" / "two.txt").write_text("0 0.3 0.3 0.2 0.2\n0 0.7 0.7 0.2 0.2\n")
        db = sqlite3.connect(main.LABELER_DB)
        db.execute("INSERT INTO text_annotations VALUES ('verified','one-crop','images/one.jpg',1)")
        db.execute("INSERT INTO container_counts VALUES ('verified','one-crop',2)")
        db.execute("INSERT INTO text_annotations VALUES ('verified','two-crop','images/two.jpg',1)")
        db.execute("INSERT INTO container_counts VALUES ('verified','two-crop',1)")
        db.commit()
        db.close()

        data = main.dataset_explorer_classification(label="2", pageSize=1)
        self.assertEqual(data["dataset"]["id"], "server-verified")
        self.assertEqual(data["stats"]["totalImages"], 2)
        self.assertEqual(data["stats"]["oneContainer"], 1)
        self.assertEqual(data["stats"]["twoContainers"], 1)
        self.assertEqual(data["items"][0]["path"], "images/one.jpg")
        self.assertEqual(Path(main.dataset_explorer_classification_file("server-verified", "images/one.jpg").path), one)

        # The classifier image-level count is independent of YOLO box count.
        detection = main.dataset_explorer_detection()
        self.assertEqual(detection["stats"]["totalBoxes"], 3)
        self.assertEqual(main.dataset_explorer_classification(label="1")["items"][0]["path"], "images/two.jpg")

    def test_filename_direction_filter_matches_left_right_and_top_case_insensitively(self):
        filenames = ["crate-LEFT.jpg", "crate-right.jpg", "crate-top.jpg"]
        for filename in filenames:
            self.image(f"images/{filename}")
        db = sqlite3.connect(main.LABELER_DB)
        for index, filename in enumerate(filenames):
            crop = f"crop-{index}"
            db.execute("INSERT INTO text_annotations VALUES ('verified',?,?,1)", (crop, f"images/{filename}"))
            db.execute("INSERT INTO container_counts VALUES ('verified',?,?)", (crop, index % 2 + 1))
        db.commit()
        db.close()

        for direction in ("left", "right", "top"):
            detection = main.dataset_explorer_detection(nameDirection=direction)
            classification = main.dataset_explorer_classification(nameDirection=direction)
            self.assertEqual(len(detection["items"]), 1)
            self.assertEqual(len(classification["items"]), 1)
            self.assertIn(direction, detection["items"][0]["filename"].casefold())
            self.assertIn(direction, classification["items"][0]["filename"].casefold())


if __name__ == "__main__":
    unittest.main()
