"""Compose lifecycle acceptance test against isolated temporary storage.

Run with `python3 tests/worker_web_isolation_smoke.py`. It creates unique Compose
projects and only removes those projects and their uniquely named test images.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from app.job_queue import create_job, request_cancel

SUFFIX = uuid.uuid4().hex[:8]
WORKER_PROJECT = f"aivision-test-worker-{SUFFIX}"
WEB_PROJECT = f"aivision-test-web-{SUFFIX}"
WORKER_IMAGE = f"ai-vision-worker:test-{SUFFIX}"
WEB_IMAGE = f"ai-vision-web:test-{SUFFIX}"


def command(args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if check and result.returncode:
        raise RuntimeError(result.stderr or result.stdout)
    return result.stdout.strip()


def docker(*args: str, check: bool = True) -> str:
    return command(["docker", *args], check=check)


def compose(project: str, file: Path, *args: str, check: bool = True) -> str:
    return command(["docker", "compose", "-p", project, "-f", str(file), *args], check=check)


def read_job(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for(predicate, timeout: int = 90):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(1)
    raise TimeoutError("Timed out waiting for lifecycle test condition")


def reserve_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def health(port: int) -> bool:
    try:
        return command(["curl", "-fsS", f"http://127.0.0.1:{port}/api/health"], check=False).startswith('{"status":"ok"')
    except Exception:
        return False


def write_compose_files(directory: Path, runs: Path, dataset: Path, port: int) -> tuple[Path, Path]:
    root = json.dumps(str(ROOT))
    image_worker, image_web = json.dumps(WORKER_IMAGE), json.dumps(WEB_IMAGE)
    runs_mount = json.dumps(f"{runs}:/runs")
    dataset_mount = json.dumps(f"{dataset}:/dataset:ro")
    inference_mount = json.dumps(f"{ROOT.parent / 'Container-Inference'}:/inference:ro")
    worker = directory / "training.yml"
    web = directory / "web.yml"
    worker.write_text(f"""services:
  training-worker:
    user: '{os.getuid()}:{os.getgid()}'
    build:
      context: {root}
      dockerfile: Dockerfile.worker
    image: {image_worker}
    restart: 'no'
    environment:
      RUNS_DIR: /runs
      DATASET_ROOT: /dataset
      WORKER_ALLOW_DUMMY_JOBS: '1'
    volumes:
      - {runs_mount}
""", encoding="utf-8")
    web.write_text(f"""services:
  web:
    user: '{os.getuid()}:{os.getgid()}'
    build:
      context: {root}
      dockerfile: Dockerfile
    image: {image_web}
    restart: 'no'
    ports:
      - '127.0.0.1:{port}:8768'
    environment:
      RUNS_DIR: /runs
      DATASET_ROOT: /dataset
      PORT: '8768'
      INFERENCE_MODEL_PATH: /inference/best.onnx
    volumes:
      - {runs_mount}
      - {dataset_mount}
      - {inference_mount}
""", encoding="utf-8")
    return worker, web


def queue_dummy(runs: Path, duration: int) -> Path:
    job_id = uuid.uuid4().hex
    return create_job(runs, "yolo", {
        "id": job_id, "modelType": "dummy", "status": "queued",
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "durationSeconds": duration, "message": "Isolated lifecycle dummy job queued.",
    })


def worker_identity(project: str, file: Path) -> tuple[str, str]:
    container = compose(project, file, "ps", "-q", "training-worker")
    if not container:
        raise RuntimeError("Training worker container is missing")
    inspect = json.loads(docker("inspect", container))[0]
    if not inspect["State"]["Running"]:
        raise RuntimeError("Training worker container is not running")
    # Worker runs as PID 1 in its isolated container. State.Pid is its host PID.
    worker_pid = str(inspect["State"]["Pid"])
    return inspect["Id"], worker_pid


def assert_worker_heartbeat(project: str, file: Path) -> dict:
    output = compose(project, file, "exec", "-T", "training-worker", "python", "-m", "worker.control", "status")
    line = next((item for item in output.splitlines() if item.startswith("worker ")), None)
    if not line:
        raise AssertionError(f"Worker registry is missing: {output}")
    registry = json.loads(line.split(" ", 1)[1])
    heartbeat = datetime.fromisoformat(registry["heartbeatAt"].replace("Z", "+00:00"))
    age = (datetime.now(timezone.utc) - heartbeat).total_seconds()
    assert age < 60, f"Worker heartbeat is stale ({age:.1f}s)"
    return registry


def main() -> None:
    if not shutil.which("docker"):
        raise SystemExit("Docker CLI is required")
    with tempfile.TemporaryDirectory(prefix=f"{SUFFIX}-", dir="/tmp") as temporary:
        root = Path(temporary)
        runs = root / "runs"
        runs.mkdir()
        dataset = root / "dataset"
        (dataset / "images").mkdir(parents=True)
        (dataset / "labels").mkdir()
        image_path = dataset / "images" / "lifecycle.jpg"
        image = Image.new("RGB", (320, 320), (75, 90, 105))
        ImageDraw.Draw(image).rectangle((80, 80, 240, 240), outline=(220, 220, 220), width=5)
        image.save(image_path, quality=85)
        (dataset / "labels" / "lifecycle.txt").write_text("0 0.5 0.5 0.5 0.5\n")
        upload_id = f"upload-{uuid.uuid4().hex}"
        checkpoint_dir = runs / "checkpoints"
        checkpoint_dir.mkdir()
        (checkpoint_dir / f"{upload_id}.pt").write_bytes(b"isolated checkpoint download fixture")
        (checkpoint_dir / f"{upload_id}.json").write_text(json.dumps({"name": "fixture.pt", "originalName": "fixture.pt"}))
        worker_file, web_file = write_compose_files(root, runs, dataset, reserve_port())
        port = int(web_file.read_text().split("127.0.0.1:", 1)[1].split(":", 1)[0])
        try:
            first = queue_dummy(runs, 3)
            compose(WORKER_PROJECT, worker_file, "up", "-d", "--build", "training-worker")
            wait_for(lambda: read_job(first).get("status") == "completed", timeout=60)
            assert (first.parent / "training.log").is_file()
            assert_worker_heartbeat(WORKER_PROJECT, worker_file)

            # Cancellation is exercised independently of natural completion.
            cancelled = queue_dummy(runs, 1800)
            wait_for(lambda: (job := read_job(cancelled)).get("status") == "running" and job.get("currentStep", 0) >= 1)
            request_cancel(cancelled.parent)
            wait_for(lambda: read_job(cancelled).get("status") == "cancelled", timeout=20)

            # Keep one job active throughout isolated web build and recreation.
            active = queue_dummy(runs, 1800)
            wait_for(lambda: (job := read_job(active)).get("status") == "running" and job.get("currentStep", 0) >= 2)
            worker_before = worker_identity(WORKER_PROJECT, worker_file)
            progress_before = int(read_job(active)["currentStep"])

            compose(WEB_PROJECT, web_file, "up", "-d", "--build", "web")
            wait_for(lambda: health(port))
            worker_status = json.loads(command(["curl", "-fsS", f"http://127.0.0.1:{port}/api/worker/status"]))
            assert worker_status["workers"] and worker_status["workers"][0]["healthy"]
            assert json.loads(command(["curl", "-fsS", f"http://127.0.0.1:{port}/api/training/{active.parent.name}"]))["status"] == "running"
            images = json.loads(command(["curl", "-fsS", f"http://127.0.0.1:{port}/api/dataset/images"]))
            assert images["total"] == 1 and images["items"] == ["images/lifecycle.jpg"]
            explorer = json.loads(command(["curl", "-fsS", f"http://127.0.0.1:{port}/api/dataset/explorer/detection"]))
            assert explorer["total"] == 1 and explorer["items"][0]["boxCount"] == 1
            inference = json.loads(command(["curl", "-sS", "-F", f"file=@{image_path}", "-F", "use_slicing=false",
                                            f"http://127.0.0.1:{port}/api/inference"]))
            if "runtime" not in inference:
                raise RuntimeError(f"Sliding-window inference compatibility check failed: {inference}")
            assert inference["model"] == "best.onnx" and inference["runtime"].startswith("Multi-scale sliding-window")
            logs = json.loads(command(["curl", "-fsS", f"http://127.0.0.1:{port}/api/training/{active.parent.name}/logs"]))
            assert logs["runId"] == active.parent.name and logs["files"]
            checkpoint = command(["curl", "-fsS", f"http://127.0.0.1:{port}/api/checkpoints/{upload_id}/download"])
            assert checkpoint == "isolated checkpoint download fixture"
            compose(WEB_PROJECT, web_file, "up", "-d", "--build", "--force-recreate", "--no-deps", "web")
            wait_for(lambda: health(port))

            progress_after = int(wait_for(lambda: (job["currentStep"] if (job := read_job(active)).get("currentStep", 0) > progress_before else None)))
            worker_after = worker_identity(WORKER_PROJECT, worker_file)
            assert worker_after == worker_before, (worker_before, worker_after)
            assert progress_after > progress_before
            api_job = json.loads(command(["curl", "-fsS", f"http://127.0.0.1:{port}/api/training/{active.parent.name}"]))
            assert api_job["id"] == active.parent.name and api_job["status"] == "running"
            assert (active.parent / "training.log").is_file()
            print(json.dumps({"result": "passed", "workerContainerId": worker_before[0],
                              "workerHostPid": worker_before[1], "progressBefore": progress_before,
                              "progressAfter": progress_after, "webRebuiltAndRecreated": True,
                              "existingJobVisibleThroughWeb": api_job["id"]}, indent=2))
            request_cancel(active.parent)
            wait_for(lambda: read_job(active).get("status") == "cancelled", timeout=20)
        finally:
            compose(WEB_PROJECT, web_file, "down", check=False)
            compose(WORKER_PROJECT, worker_file, "down", check=False)
            docker("image", "rm", WEB_IMAGE, WORKER_IMAGE, check=False)


if __name__ == "__main__":
    main()
