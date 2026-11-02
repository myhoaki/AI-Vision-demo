import json
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
        main.DATASET_ROOT = self.root / "source"
        main.RUNS_DIR = self.root / "runs"
        (main.DATASET_ROOT / "images").mkdir(parents=True)
        (main.DATASET_ROOT / "labels").mkdir()

    def tearDown(self):
        main.DATASET_ROOT = self.original_dataset_root
        main.RUNS_DIR = self.original_runs_dir
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

    def test_classifier_view_reads_class_folders_independently(self):
        dataset_id = "a" * 32
        root = main.RUNS_DIR / "classifiers" / "datasets" / dataset_id
        (root / "1_container").mkdir(parents=True)
        (root / "2_containers").mkdir()
        (root / "unclassified").mkdir()
        (root / "1_container" / "one.jpg").write_bytes(b"one")
        (root / "2_containers" / "two.jpg").write_bytes(b"two")
        (root / "unclassified" / "unknown.jpg").write_bytes(b"unknown")
        (root / "import.json").write_text(json.dumps({"source": "test import", "imageCount": 3}))
        self.image("images/one.jpg")
        (main.DATASET_ROOT / "labels" / "one.txt").write_text("0 0.5 0.5 0.2 0.2\n")

        data = main.dataset_explorer_classification(datasetId=dataset_id, label="2", pageSize=1)
        self.assertEqual(data["stats"]["totalImages"], 3)
        self.assertEqual(data["stats"]["onePercent"], 33.3)
        self.assertEqual(data["stats"]["unclassified"], 1)
        self.assertEqual(data["items"][0]["labelText"], "2 containers")
        self.assertEqual(data["total"], 1)

        # Classifier labels and YOLO boxes are separate records and routes.
        detection = main.dataset_explorer_detection()
        self.assertEqual(detection["stats"]["totalBoxes"], 1)
        self.assertEqual(main.dataset_explorer_classification(datasetId=dataset_id, label="1")["items"][0]["label"], "1")


if __name__ == "__main__":
    unittest.main()
