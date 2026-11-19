from __future__ import annotations

import json
import os
import signal
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from app import job_queue
from app import main as training_runtime

RUNS_DIR = Path(os.getenv("RUNS_DIR", "/runs")).resolve()
WORKER_ID = job_queue.new_worker_id()
shutdown = threading.Event()
active_lock = threading.Lock()
active_jobs: dict[str, tuple[str, Path]] = {}
owners: dict[str, dict[str, Any]] = {}
registry_path = RUNS_DIR / "worker" / "registry" / f"{WORKER_ID}.json"
HEARTBEAT_SECONDS = max(1.0, float(os.getenv("WORKER_HEARTBEAT_SECONDS", "10")))


def read_job(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_job(path: Path, job: dict[str, Any]) -> None:
    job["updatedAt"] = training_runtime.utc_now()
    job_queue.atomic_json(path, job)


def heartbeat_loop() -> None:
    registry = {"workerId": WORKER_ID, "pid": os.getpid(), "containerId": os.getenv("HOSTNAME", "unknown"),
                "startedAt": training_runtime.utc_now(), "pipelines": ["yolo", "classifier"]}
    while not shutdown.wait(HEARTBEAT_SECONDS):
        registry["heartbeatAt"] = training_runtime.utc_now()
        job_queue.atomic_json(registry_path, registry)
        with active_lock:
            owner_snapshot = list(owners.items())
        for pipeline, owner in owner_snapshot:
            try:
                job_queue.update_claim(RUNS_DIR, pipeline, owner)
            except FileNotFoundError:
                if owners.get(pipeline) is owner:
                    shutdown.set()
                    return
            except (OSError, ValueError, RuntimeError):
                shutdown.set()
                return


def execute_job(pipeline: str, metadata: Path, owner: dict[str, Any]) -> None:
    job = read_job(metadata)
    job_id = str(job["id"])
    job_dir = metadata.parent
    with active_lock:
        active_jobs[pipeline] = (job_id, job_dir)
    try:
        job_queue.update_claim(RUNS_DIR, pipeline, owner, job_id)
        if job_queue.cancel_requested(job_dir):
            job.update(status="cancelled", finishedAt=training_runtime.utc_now(),
                       message="Queued job cancelled before execution.", workerId=WORKER_ID)
            save_job(metadata, job)
            job_queue.write_worker_log(job_dir, "Job cancelled while queued.")
            return
        job.update(status="preparing", workerId=WORKER_ID, workerPid=os.getpid(),
                   workerContainer=os.getenv("HOSTNAME", "unknown"), workerStartedAt=training_runtime.utc_now(),
                   message="Claimed by independent training worker.")
        save_job(metadata, job)
        job_queue.write_worker_log(job_dir, f"Worker {WORKER_ID} claimed {pipeline} job {job_id}.")
        if job.get("modelType") == "dummy":
            if os.getenv("WORKER_ALLOW_DUMMY_JOBS", "0").lower() not in {"1", "true", "yes"}:
                raise ValueError("Dummy worker jobs are disabled.")
            duration = max(1, min(1800, int(job.get("durationSeconds", 10))))
            started = time.monotonic()
            total_steps = duration
            job.update(status="running", totalSteps=total_steps, message="Running controlled worker lifecycle check.")
            save_job(metadata, job)
            while not shutdown.is_set() and time.monotonic() - started < duration:
                if job_queue.cancel_requested(job_dir):
                    job.update(status="cancelled", finishedAt=training_runtime.utc_now(), message="Dummy job cancelled.")
                    save_job(metadata, job)
                    return
                job["currentStep"] = min(total_steps, int(time.monotonic() - started) + 1)
                save_job(metadata, job)
                time.sleep(1)
            if shutdown.is_set():
                job.update(status="interrupted", finishedAt=training_runtime.utc_now(), message="Worker shutdown during dummy job.")
            else:
                job.update(status="completed", finishedAt=training_runtime.utc_now(), message="Dummy worker job completed.")
            save_job(metadata, job)
        elif pipeline == "yolo":
            request_data = job.get("request") or {key: job.get(key) for key in
                ("epochs", "imageSize", "batch", "baseModel", "initMode", "checkpointId")}
            request = training_runtime.TrainRequest(**request_data)
            training_runtime.train_worker(job_id, request)
        elif pipeline == "classifier":
            training_runtime.classifier_worker(job_id)
        else:
            raise ValueError(f"Unsupported pipeline {pipeline}")
        final = read_job(metadata)
        if final.get("status") == "stopped" and job_queue.cancel_requested(job_dir):
            final.update(status="cancelled", message=final.get("message") or "Training cancelled at a safe checkpoint boundary.")
            save_job(metadata, final)
        if final.get("status") in job_queue.ACTIVE:
            final.update(status="interrupted", finishedAt=training_runtime.utc_now(),
                         message="Worker returned before writing a terminal status; explicit recovery is required.")
            final["failureReason"] = "Training executor exited with a nonterminal job status."
            save_job(metadata, final)
        job_queue.write_worker_log(job_dir, f"Executor returned with status {read_job(metadata).get('status')}.")
    except BaseException as error:
        try:
            current = read_job(metadata)
            if current.get("status") not in {"completed", "failed", "cancelled", "stopped", "interrupted"}:
                current.update(status="failed", finishedAt=training_runtime.utc_now(),
                               message=str(error)[:1200], failureReason=traceback.format_exc()[-8000:])
                save_job(metadata, current)
            job_queue.write_worker_log(job_dir, f"Worker error: {error}")
        except Exception:
            pass
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        (job_dir / "cancel.request").unlink(missing_ok=True)
        with active_lock:
            active_jobs.pop(pipeline, None)
        try:
            job_queue.update_claim(RUNS_DIR, pipeline, owner, None)
        except (OSError, ValueError, RuntimeError):
            pass


def pipeline_loop(pipeline: str) -> None:
    while not shutdown.is_set():
        owner = job_queue.claim_pipeline(RUNS_DIR, pipeline, WORKER_ID)
        if owner is None:
            shutdown.wait(2)
            continue
        with active_lock:
            owners[pipeline] = owner
        try:
            jobs = job_queue.pending_jobs(RUNS_DIR, pipeline)
            if not jobs:
                with active_lock:
                    owners.pop(pipeline, None)
                job_queue.release_claim(RUNS_DIR, pipeline, WORKER_ID)
                shutdown.wait(2)
                continue
            for metadata in jobs:
                if shutdown.is_set():
                    break
                current = read_job(metadata)
                if current.get("status") != "queued":
                    continue
                execute_job(pipeline, metadata, owner)
        except BaseException as error:
            print(f"{pipeline} worker loop error: {error}", flush=True)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            shutdown.wait(2)
        finally:
            with active_lock:
                owners.pop(pipeline, None)
            try:
                job_queue.release_claim(RUNS_DIR, pipeline, WORKER_ID)
            except (OSError, ValueError, RuntimeError):
                pass


def request_shutdown(signum: int, _frame: Any) -> None:
    print(f"Received signal {signum}; requesting graceful training stop.", flush=True)
    shutdown.set()
    with active_lock:
        running = list(active_jobs.values())
    for _, job_dir in running:
        try:
            job_queue.request_cancel(job_dir)
        except OSError as error:
            print(f"Could not persist cancellation request for {job_dir.name}: {error}", flush=True)


def main() -> None:
    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    heartbeat = threading.Thread(target=heartbeat_loop, name="worker-heartbeat", daemon=True)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    job_queue.atomic_json(registry_path, {"workerId": WORKER_ID, "pid": os.getpid(),
        "containerId": os.getenv("HOSTNAME", "unknown"), "startedAt": training_runtime.utc_now(),
        "heartbeatAt": training_runtime.utc_now(), "pipelines": ["yolo", "classifier"]})
    heartbeat.start()
    threads = [threading.Thread(target=pipeline_loop, args=(name,), name=f"{name}-worker")
               for name in ("yolo", "classifier")]
    for thread in threads:
        thread.start()
    try:
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(timeout=1)
    except KeyboardInterrupt:
        request_shutdown(signal.SIGINT, None)
        for thread in threads:
            thread.join()
    print("Training worker exited.", flush=True)
    registry_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
