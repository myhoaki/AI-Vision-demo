import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from app import job_queue


def test_job_creation_and_pending_order(tmp_path: Path):
    queue = tmp_path / "runs"
    second = {"id": "b" * 32, "status": "queued", "createdAt": "2026-01-02T00:00:00Z"}
    first = {"id": "a" * 32, "status": "queued", "createdAt": "2026-01-01T00:00:00Z"}
    job_queue.create_job(queue, "yolo", second)
    job_queue.create_job(queue, "yolo", first)
    assert job_queue.pending_jobs(queue, "yolo") == [
        queue / "jobs" / first["id"] / "job.json",
        queue / "jobs" / second["id"] / "job.json",
    ]
    assert json.loads((queue / "jobs" / first["id"] / "job.json").read_text())["status"] == "queued"


def test_only_one_worker_can_claim_a_pipeline(tmp_path: Path):
    queue = tmp_path / "runs"
    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(lambda name: job_queue.claim_pipeline(queue, "classifier", name), ["w1", "w2"]))
    assert sum(claim is not None for claim in claims) == 1
    winner = next(claim for claim in claims if claim is not None)
    job_queue.release_claim(queue, "classifier", winner["workerId"])
    assert job_queue.claim_pipeline(queue, "classifier", "w3") is not None


def test_pipeline_claims_are_independent_and_cancel_is_separate(tmp_path: Path):
    queue = tmp_path / "runs"
    yolo = job_queue.claim_pipeline(queue, "yolo", "worker")
    classifier = job_queue.claim_pipeline(queue, "classifier", "worker")
    assert yolo and classifier
    job_dir = queue / "jobs" / ("a" * 32)
    job_dir.mkdir(parents=True)
    job_queue.request_cancel(job_dir)
    assert job_queue.cancel_requested(job_dir)
    assert not (job_dir / "job.json").exists()


def test_stale_heartbeat_alone_never_recovers_live_owner():
    old_claim = {"heartbeatAt": 1.0, "workerId": "worker"}
    assert not job_queue.claim_is_stale(old_claim, now=1000, max_age=30, owner_alive=lambda _claim: True)
    assert job_queue.claim_is_stale(old_claim, now=1000, max_age=30, owner_alive=lambda _claim: False)


def test_claim_is_published_as_complete_json(tmp_path: Path):
    queue = tmp_path / "runs"
    owner = job_queue.claim_pipeline(queue, "yolo", "worker")
    assert owner is not None
    claim = json.loads(job_queue.claim_path(queue, "yolo").read_text())
    assert claim["workerId"] == "worker"
    assert not list((queue / "worker" / "claims").glob("*.claim"))


def test_explicit_stale_recovery_marks_job_interrupted_and_never_requeues(tmp_path: Path):
    queue = tmp_path / "runs"
    job_id = "d" * 32
    job_dir = queue / "jobs" / job_id
    job_dir.mkdir(parents=True)
    metadata = job_dir / "job.json"
    metadata.write_text(json.dumps({"id": job_id, "status": "running", "epoch": 1}))
    owner = job_queue.claim_pipeline(queue, "yolo", "dead-worker")
    assert owner is not None
    job_queue.update_claim(queue, "yolo", owner, job_id)
    claim_path = job_queue.claim_path(queue, "yolo")
    claim = json.loads(claim_path.read_text())
    claim["heartbeatAt"] = 1
    job_queue.atomic_json(claim_path, claim)
    result = subprocess.run([
        sys.executable, "-m", "worker.control", "release-stale", "--pipeline", "yolo",
        "--runs-dir", str(queue), "--max-age-seconds", "1", "--confirm-owner-stopped",
    ], capture_output=True, text=True, check=True)
    updated = json.loads(metadata.read_text())
    assert updated["status"] == "interrupted"
    assert "no job was resumed" in result.stdout
    assert not claim_path.exists()
    assert len(list(claim_path.parent.glob("yolo.stale-*.json"))) == 1
    assert job_queue.pending_jobs(queue, "yolo") == []
