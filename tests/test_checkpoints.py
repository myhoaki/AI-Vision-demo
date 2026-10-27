import json
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from app import main


class CheckpointRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.original_runs_dir = main.RUNS_DIR
        main.RUNS_DIR = Path(self.temp.name)

    def tearDown(self):
        main.RUNS_DIR = self.original_runs_dir
        self.temp.cleanup()

    def make_run(self, job_id, status, **extra):
        run_dir = main.RUNS_DIR / "jobs" / job_id
        weight_dir = run_dir / "results" / "container" / "weights"
        weight_dir.mkdir(parents=True)
        (weight_dir / "best.pt").write_bytes(b"best-checkpoint")
        (weight_dir / "last.pt").write_bytes(b"last-checkpoint")
        job = {
            "id": job_id,
            "status": status,
            "createdAt": "2026-10-27T01:00:00+00:00",
            "updatedAt": "2026-10-27T02:00:00+00:00",
            "startedAt": "2026-10-27T01:01:00+00:00",
            "epoch": 3,
            "epochs": 10,
            "baseModel": "yolo26n.pt",
            **extra,
        }
        (run_dir / "job.json").write_text(json.dumps(job))
        return job_id

    def test_active_run_groups_checkpoints_and_blocks_downloads(self):
        job_id = "a" * 32
        self.make_run(job_id, "running")

        runs = main.training_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["statusLabel"], "TRAINING")
        self.assertEqual((runs[0]["epoch"], runs[0]["totalEpochs"]), (3, 10))
        self.assertEqual({item["kind"] for item in runs[0]["checkpoints"]}, {"best", "last"})
        self.assertTrue(all(item["interim"] for item in runs[0]["checkpoints"]))
        self.assertTrue(all(not item["downloadAllowed"] for item in runs[0]["checkpoints"]))
        with self.assertRaises(HTTPException) as error:
            main.resolve_checkpoint(f"{job_id}-best")
        self.assertEqual(error.exception.status_code, 409)

    def test_completed_best_is_detection_ready_but_last_is_not(self):
        job_id = "b" * 32
        self.make_run(job_id, "completed")

        run = main.training_runs()[0]
        checkpoints = {item["kind"]: item for item in run["checkpoints"]}
        self.assertEqual(run["statusLabel"], "COMPLETED")
        self.assertTrue(checkpoints["best"]["readyForDetection"])
        self.assertFalse(checkpoints["last"]["readyForDetection"])
        self.assertFalse(checkpoints["best"]["interim"])
        self.assertTrue(checkpoints["best"]["downloadAllowed"])
        path, descriptor = main.resolve_checkpoint(f"{job_id}-best")
        self.assertTrue(path.is_file())
        self.assertEqual(descriptor["jobStatus"], "completed")

    def test_failed_run_is_never_marked_detection_ready(self):
        job_id = "c" * 32
        self.make_run(job_id, "failed")

        run = main.training_runs()[0]
        self.assertEqual(run["statusLabel"], "FAILED")
        self.assertFalse(any(item["readyForDetection"] for item in run["checkpoints"]))
        self.assertEqual(len(main.checkpoint_models()), 0)

    def test_unknown_status_fails_closed_for_checkpoint_download(self):
        job_id = "e" * 32
        self.make_run(job_id, "unknown-state")

        checkpoint = next(item for item in main.training_runs()[0]["checkpoints"] if item["kind"] == "best")
        self.assertFalse(checkpoint["downloadAllowed"])
        self.assertFalse(checkpoint["readyForDetection"])
        with self.assertRaises(HTTPException) as error:
            main.resolve_checkpoint(f"{job_id}-best")
        self.assertEqual(error.exception.status_code, 409)

    def test_legacy_models_response_keeps_flat_checkpoints_and_adds_runs(self):
        job_id = "d" * 32
        self.make_run(job_id, "completed")

        result = main.models()
        self.assertEqual(len(result["trainingRuns"]), 1)
        self.assertEqual(len(result["checkpoints"]), 2)
        self.assertEqual(len(result["models"]), 1)

    def test_batch_counter_advances_without_a_trainer_batch_i_attribute(self):
        current = 0
        current = main.advance_batch_count(current, 1934)
        self.assertEqual(current, 1)
        current = main.advance_batch_count(current, 1934)
        self.assertEqual(current, 2)
        self.assertEqual(main.advance_batch_count(1934, 1934), 1934)

    def test_resumed_run_uses_source_checkpoint_directory_without_duplicate_owner(self):
        source_id = "f" * 32
        resumed_id = "1" * 32
        self.make_run(source_id, "interrupted", resumedBy=resumed_id)
        resumed_dir = main.RUNS_DIR / "jobs" / resumed_id
        resumed_dir.mkdir(parents=True)
        resumed_job = {
            "id": resumed_id,
            "status": "running",
            "createdAt": "2026-10-27T03:00:00+00:00",
            "updatedAt": "2026-10-27T03:01:00+00:00",
            "initMode": "resume",
            "checkpointId": f"{source_id}-last",
            "baseModel": "yolo26n.pt",
            "epochs": 10,
        }
        (resumed_dir / "job.json").write_text(json.dumps(resumed_job))

        runs = {run["id"]: run for run in main.training_runs()}
        self.assertEqual(runs[source_id]["checkpoints"], [])
        self.assertEqual(len(runs[resumed_id]["checkpoints"]), 2)
        self.assertTrue(all(item["interim"] for item in runs[resumed_id]["checkpoints"]))
        self.assertEqual(len([item for item in main.list_checkpoints() if item["source"] == "training run"]), 2)


if __name__ == "__main__":
    unittest.main()
