import json
import csv
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

    def test_epoch_metrics_csv_parse_and_merge_deduplicates_resume_history(self):
        run_id = "2" * 32
        run_dir = main.RUNS_DIR / "jobs" / run_id
        results_dir = run_dir / "results" / "container"
        results_dir.mkdir(parents=True)
        csv_path = results_dir / "results.csv"
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["epoch", "time", "train/box_loss", "metrics/mAP50(B)"])
            writer.writeheader()
            writer.writerows([
                {"epoch": 1, "time": 8.5, "train/box_loss": 1.2, "metrics/mAP50(B)": 0.4},
                {"epoch": 2, "time": 18.0, "train/box_loss": 0.9, "metrics/mAP50(B)": 0.6},
            ])
        parsed = main.csv_epoch_metrics(csv_path)
        self.assertEqual(parsed[1]["epochDurationSeconds"], 9.5)
        job = {"parentRunId": "p" * 32, "resumeParentEpoch": 1, "epochMetrics": [parsed[0]]}
        merged = main.merge_epoch_metrics(job, parsed, "2026-10-27T00:00:00+00:00")
        self.assertEqual([point["epoch"] for point in merged], [1, 2])
        self.assertTrue(merged[0]["inherited"])
        self.assertFalse(merged[1].get("inherited", False))
        self.assertEqual(merged[1]["metrics"]["metrics/mAP50(B)"], 0.6)

    def test_metrics_endpoint_reads_real_csv_and_returns_one_point_per_epoch(self):
        job_id = "3" * 32
        run_dir = main.RUNS_DIR / "jobs" / job_id
        results_dir = run_dir / "results" / "container"
        results_dir.mkdir(parents=True)
        (run_dir / "job.json").write_text(json.dumps({"id": job_id, "status": "completed"}))
        (results_dir / "results.csv").write_text("epoch,time,metrics/mAP50(B)\n1,10,0.2\n2,20,0.3\n")
        response = main.training_metrics(job_id)
        self.assertEqual(response["source"], "Ultralytics results.csv")
        self.assertEqual(len(response["metrics"]), 2)
        self.assertEqual(response["metrics"][1]["metrics"]["metrics/mAP50(B)"], 0.3)

    def test_resources_and_event_helpers_keep_json_safe_bounded_history(self):
        job = {"events": [{"message": "old"}] * 201}
        main.append_job_event(job, "warning", "latest", "test")
        self.assertEqual(len(job["events"]), 200)
        self.assertEqual(job["events"][-1]["message"], "latest")
        self.assertIsNone(main.json_safe(float("nan")))

    def test_stale_warning_only_applies_to_old_active_job_updates(self):
        stale = {"status": "running", "updatedAt": "2020-01-01T00:00:00+00:00"}
        fresh = {"status": "running", "updatedAt": main.utc_now()}
        terminal = {"status": "failed", "updatedAt": "2020-01-01T00:00:00+00:00"}
        self.assertTrue(main.is_stale_job(stale))
        self.assertFalse(main.is_stale_job(fresh))
        self.assertFalse(main.is_stale_job(terminal))

    def test_csv_backfill_updates_terminal_jobs_but_never_active_jobs(self):
        for job_id, status in (("4" * 32, "completed"), ("5" * 32, "running")):
            run_dir = main.RUNS_DIR / "jobs" / job_id
            results_dir = run_dir / "results" / "container"
            results_dir.mkdir(parents=True)
            job = {"id": job_id, "status": status, "epochMetrics": []}
            (run_dir / "job.json").write_text(json.dumps(job))
            (results_dir / "results.csv").write_text("epoch,time,metrics/mAP50(B)\n1,10,0.2\n")
        main.backfill_job_metrics()
        terminal = json.loads((main.RUNS_DIR / "jobs" / ("4" * 32) / "job.json").read_text())
        active = json.loads((main.RUNS_DIR / "jobs" / ("5" * 32) / "job.json").read_text())
        self.assertEqual(len(terminal["epochMetrics"]), 1)
        self.assertEqual(active["epochMetrics"], [])

    def test_resume_metrics_mark_parent_epochs_using_parent_job_record(self):
        parent_id, child_id = "6" * 32, "7" * 32
        parent_dir = main.RUNS_DIR / "jobs" / parent_id
        parent_results = parent_dir / "results" / "container"
        parent_results.mkdir(parents=True)
        (parent_dir / "job.json").write_text(json.dumps({"id": parent_id, "status": "interrupted", "epoch": 1}))
        (parent_results / "results.csv").write_text("epoch,time,metrics/mAP50(B)\n1,10,0.2\n2,20,0.3\n")
        child_dir = main.RUNS_DIR / "jobs" / child_id
        (child_dir / "results" / "container").mkdir(parents=True)
        (child_dir / "job.json").write_text(json.dumps({"id": child_id, "status": "running", "initMode": "resume", "checkpointId": f"{parent_id}-last", "epoch": 2}))
        history = main.training_metrics(child_id)
        self.assertEqual([point.get("inherited", False) for point in history["metrics"]], [True, False])
        self.assertEqual(history["metrics"][0]["inheritedFromRunId"], parent_id)

    def test_epoch_timing_uses_real_durations_and_remaining_epochs(self):
        job = {"epoch": 3}
        main.update_epoch_timing(job, [{"epochDurationSeconds": 100}, {"epochDurationSeconds": 120}], 8)
        self.assertEqual(job["averageEpochDurationSeconds"], 110)
        self.assertEqual(job["estimatedRemainingSeconds"], 550)
        self.assertEqual(job["recentTrainingSpeed"]["imagesPerSecond"], None)

    def test_observability_instance_refuses_mutating_training_actions(self):
        previous = main.OBSERVABILITY_ONLY
        main.OBSERVABILITY_ONLY = True
        try:
            self.assertTrue(main.health()["readOnly"])
            with self.assertRaises(HTTPException) as error:
                main.start_training(main.TrainRequest(epochs=1, imageSize=640, batch=1, baseModel="yolo26n.pt", initMode="base"))
            self.assertEqual(error.exception.status_code, 403)
        finally:
            main.OBSERVABILITY_ONLY = previous


if __name__ == "__main__":
    unittest.main()
