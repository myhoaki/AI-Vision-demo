import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from fastapi import HTTPException

from app import main


class FakeImageFolder:
    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.classes = ["1_container", "2_containers"]
        self.samples = [(str(self.root / "1_container" / "sample.jpg"), 0)]


class YoloClassifierDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "project" / "experiments" / main.CLASSIFIER_YOLO_EXPERIMENT
        self.root.mkdir(parents=True)
        self.rows = []
        for split in ("train", "val", "test"):
            for label in main.CLASSIFIER_CLASSES:
                folder = self.root / split / label
                folder.mkdir(parents=True)
                image_path = folder / f"{split}-{label}.jpg"
                Image.new("RGB", (8, 4), color=(20, 30, 40)).save(image_path)
                self.rows.append({"split": split, "label": label, "crop_path": f"{split}/{label}/{image_path.name}",
                                  "sample_id": f"{split}-{label}", "source_group": f"group-{split}", "status": "cropped"})
        (self.root / "metadata.json").write_text(json.dumps({
            "experiment": main.CLASSIFIER_YOLO_EXPERIMENT,
            "status": "complete",
            "output_counts": {"train": {label: 1 for label in main.CLASSIFIER_CLASSES},
                              "validation": {label: 1 for label in main.CLASSIFIER_CLASSES},
                              "test": {label: 1 for label in main.CLASSIFIER_CLASSES}},
        }))
        (self.root / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in self.rows))

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        with patch.object(main, "ROOT", self.root.parents[1] / "app"):
            return main.production_matched_classifier_dataset()

    def test_valid_dataset_counts_and_mapping_match_manifest(self):
        result = self.validate()
        self.assertEqual(result["imageCount"], 6)
        self.assertEqual(result["classToIndex"], {"1_container": 0, "2_containers": 1})
        self.assertEqual(result["splitCounts"]["val"], {"1_container": 1, "2_containers": 1})

    def test_missing_split_folder_has_clear_error(self):
        (self.root / "test").rename(self.root / "test-missing")
        with self.assertRaisesRegex(HTTPException, "missing test directory"):
            self.validate()

    def test_missing_class_has_clear_error(self):
        import shutil
        shutil.rmtree(self.root / "val" / "2_containers")
        with self.assertRaisesRegex(HTTPException, "val must contain exactly the class folders"):
            self.validate()

    def test_manifest_count_mismatch_is_rejected(self):
        (self.root / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in self.rows[:-1]))
        with self.assertRaisesRegex(HTTPException, "image paths do not exactly match manifest"):
            self.validate()

    def test_unreadable_image_is_rejected(self):
        target = self.root / "train" / "1_container" / "train-1_container.jpg"
        target.write_bytes(b"not an image")
        with self.assertRaisesRegex(HTTPException, "cannot identify image"):
            self.validate()

    def test_split_datasets_are_independent_and_use_eval_transform(self):
        train_transform, eval_transform = object(), object()
        loaded = main.load_yolo_classifier_splits(self.root, FakeImageFolder, train_transform, eval_transform)
        splits = loaded["datasets"]
        self.assertIs(splits["train"].transform, train_transform)
        self.assertIs(splits["validation"].transform, eval_transform)
        self.assertIs(splits["test"].transform, eval_transform)
        self.assertEqual(splits["train"].classes, splits["validation"].classes)
        self.assertEqual(splits["validation"].classes, splits["test"].classes)
        paths = [Path(dataset.samples[0][0]) for dataset in splits.values()]
        self.assertEqual(len(set(paths)), 3)
        self.assertEqual(loaded["classToIndex"], {"1_container": 0, "2_containers": 1})

    def test_mismatched_class_mapping_is_rejected(self):
        def mismatched(root, transform):
            dataset = FakeImageFolder(root, transform)
            if Path(root).name == "val":
                dataset.classes = ["2_containers", "1_container"]
            return dataset
        with self.assertRaisesRegex(ValueError, "validation class mapping differs"):
            main.load_yolo_classifier_splits(self.root, mismatched, None, None)

    def test_training_request_records_yolo_dataset_without_starting_training(self):
        info = self.validate()
        info["splitCounts"]["train"] = {"1_container": 6710, "2_containers": 1069}
        info["splitCounts"]["val"] = {"1_container": 799, "2_containers": 172}
        info["splitCounts"]["test"] = {"1_container": 833, "2_containers": 139}
        info["classCounts"] = {"1_container": 8342, "2_containers": 1380}
        info["imageCount"] = 9722
        test_runs = Path(self.temp.name) / "runs"
        with patch.object(main, "ROOT", self.root.parents[1] / "app"), \
             patch.object(main, "RUNS_DIR", test_runs), \
             patch.object(main, "production_matched_classifier_dataset") as validate_dataset, \
             patch.object(main.threading, "Thread") as worker:
            response = main.start_classifier_training({"datasetSource": "yolo_crops", "architecture": "mobilenet_v3_small",
                "epochs": 30, "batchSize": 16, "imageSize": 224, "learningRate": 0.001, "device": "cpu",
                "balancingStrategy": "class_weighted_loss", "targetDistribution": "70:30"})
        self.assertEqual(response["datasetSource"], "yolo_crops")
        self.assertEqual(response["status"], "queued")
        self.assertEqual(response["datasetPath"], info["datasetRoot"])
        self.assertEqual(response["architecture"], "mobilenet_v3_small")
        self.assertEqual(response["request"]["balancingStrategy"], "class_weighted_loss")
        persisted = json.loads((test_runs / "classifiers" / "jobs" / response["id"] / "job.json").read_text())
        self.assertEqual(persisted["status"], "queued")
        self.assertEqual(persisted["datasetPath"], info["datasetRoot"])
        validate_dataset.assert_not_called()
        worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
