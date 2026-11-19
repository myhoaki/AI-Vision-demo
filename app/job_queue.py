"""Atomic filesystem queue and worker claims for persistent training jobs."""
from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import uuid
import fcntl
from pathlib import Path
from typing import Any, Callable

ACTIVE = {"queued", "preparing", "running", "evaluating"}
PIPELINES = {"yolo": "jobs", "classifier": "classifiers/jobs"}
_UNSET = object()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass


def create_job(runs_dir: Path, pipeline: str, job: dict[str, Any]) -> Path:
    if pipeline not in PIPELINES:
        raise ValueError(f"Unknown training pipeline: {pipeline}")
    root = runs_dir / PIPELINES[pipeline]
    directory = root / job["id"]
    directory.mkdir(parents=True, exist_ok=False)
    try:
        root_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
    except OSError:
        pass
    atomic_json(directory / "job.json", job)
    return directory / "job.json"


def request_cancel(job_dir: Path) -> None:
    marker = job_dir / "cancel.request"
    try:
        fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"requestedAt": time.time()}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        dir_fd = os.open(job_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def cancel_requested(job_dir: Path) -> bool:
    return (job_dir / "cancel.request").is_file()


def claim_path(runs_dir: Path, pipeline: str) -> Path:
    return runs_dir / "worker" / "claims" / f"{pipeline}.json"


def claim_pipeline(runs_dir: Path, pipeline: str, worker_id: str) -> dict[str, Any] | None:
    """Acquire a pipeline-wide exclusive claim without exposing partial JSON."""
    path = claim_path(runs_dir, pipeline)
    path.parent.mkdir(parents=True, exist_ok=True)
    owner = {"pipeline": pipeline, "workerId": worker_id, "pid": os.getpid(),
            "host": socket.gethostname(), "containerId": os.getenv("HOSTNAME", "unknown"),
            "startedAt": time.time(), "heartbeatAt": time.time(), "jobId": None}
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".claim", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(owner, stream)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # Hard-link creation is atomic and fails if another owner already claimed it.
            os.link(temporary, path)
        except FileExistsError:
            return None
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
        return owner
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def exclusive_recovery(runs_dir: Path, pipeline: str):
    """Serialize operator initiated stale recovery; process death releases flock."""
    path = claim_path(runs_dir, pipeline)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(".recovery.lock")
    stream = lock_path.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f"A stale-claim recovery is already running for {pipeline}")
    return stream


def update_claim(runs_dir: Path, pipeline: str, owner: dict[str, Any], job_id: Any = _UNSET) -> None:
    path = claim_path(runs_dir, pipeline)
    current = json.loads(path.read_text(encoding="utf-8"))
    if current.get("workerId") != owner["workerId"]:
        raise RuntimeError(f"Worker claim ownership changed for {pipeline}")
    owner["heartbeatAt"] = time.time()
    if job_id is not _UNSET:
        owner["jobId"] = job_id
    atomic_json(path, owner)


def release_claim(runs_dir: Path, pipeline: str, worker_id: str) -> None:
    path = claim_path(runs_dir, pipeline)
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return
    if current.get("workerId") != worker_id:
        raise RuntimeError(f"Refusing to release another worker's {pipeline} claim")
    path.unlink()
    try:
        dir_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError:
        pass


def claim_is_stale(claim: dict[str, Any], now: float, max_age: float,
                   owner_alive: Callable[[dict[str, Any]], bool]) -> bool:
    """A lease timeout is insufficient: caller must also establish owner death."""
    return now - float(claim.get("heartbeatAt", 0)) > max_age and not owner_alive(claim)


def pending_jobs(runs_dir: Path, pipeline: str) -> list[Path]:
    root = runs_dir / PIPELINES[pipeline]
    if not root.is_dir():
        return []
    entries = []
    for path in root.glob("*/job.json"):
        try:
            job = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if job.get("status") == "queued":
            entries.append((str(job.get("createdAt", "")), path))
    return [path for _, path in sorted(entries)]


def write_worker_log(job_dir: Path, message: str) -> None:
    path = job_dir / "training.log"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {message.rstrip()}\n")
        stream.flush()
        os.fsync(stream.fileno())


def new_worker_id() -> str:
    return f"{os.getenv('HOSTNAME', socket.gethostname())}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
