from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import platform
import random
import re
import resource as resource_usage
import shutil
import sqlite3
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from app import dataset_catalog
from app import job_queue

ROOT = Path(__file__).parent
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "/dataset")).resolve()
RUNS_DIR = Path(os.getenv("RUNS_DIR", "/runs")).resolve()
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_lock = threading.RLock()
_catalog_lock = threading.RLock()
_active_training: str | None = None
_model_lock = threading.Lock()
_loaded_model: tuple[str, Any] | None = None
_onnx_model_lock = threading.Lock()
_onnx_model: tuple[str, int, Any] | None = None
_export_lock = threading.Lock()
_active_export: str | None = None
_active_classifier: str | None = None
OBSERVABILITY_ONLY = os.getenv("OBSERVABILITY_ONLY", "0").lower() in {"1", "true", "yes"}
LABELER_DB = Path(os.getenv("LABELER_DB", "/labeler-data/container-labeler.sqlite3"))
app = FastAPI(title="AI Vision Demo", version="1.0.0")


def inference_model_path() -> Path:
    configured = os.getenv("INFERENCE_MODEL_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return (ROOT.parent.parent / "Container-Inference" / "best.onnx").resolve()


def load_inference_session() -> tuple[Path, Any]:
    global _onnx_model
    path = inference_model_path()
    if not path.is_file():
        raise HTTPException(503, f"ONNX inference model is unavailable: {path.name}")
    modified = path.stat().st_mtime_ns
    with _onnx_model_lock:
        if _onnx_model is None or _onnx_model[0] != str(path) or _onnx_model[1] != modified:
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            except Exception as error:
                raise HTTPException(503, f"Could not load ONNX inference model: {str(error)[:300]}") from error
            _onnx_model = (str(path), modified, session)
        return path, _onnx_model[2]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_images() -> list[Path]:
    if not DATASET_ROOT.is_dir():
        return []
    return sorted((path for path in DATASET_ROOT.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED), key=lambda path: path.relative_to(DATASET_ROOT).as_posix().casefold())


def safe_dataset_path(relative: str) -> Path:
    candidate = (DATASET_ROOT / relative).resolve()
    if not candidate.is_relative_to(DATASET_ROOT) or not candidate.is_file():
        raise HTTPException(404, "Dataset image not found")
    if candidate.suffix.lower() not in SUPPORTED:
        raise HTTPException(400, "Unsupported image type")
    return candidate


def job_path(job_id: str) -> Path:
    if not job_id.isalnum():
        raise HTTPException(400, "Invalid training job id")
    return RUNS_DIR / "jobs" / job_id / "job.json"


def read_job(job_id: str) -> dict[str, Any]:
    path = job_path(job_id)
    if not path.is_file():
        raise HTTPException(404, "Training job not found")
    job = json.loads(path.read_text(encoding="utf-8"))
    if job_queue.cancel_requested(path.parent):
        job["stopRequested"] = True
    return job


def save_job(job: dict[str, Any]) -> None:
    path = job_path(job["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with _lock:
        if path.is_file():
            try:
                if json.loads(path.read_text(encoding="utf-8")).get("stopRequested"):
                    job["stopRequested"] = True
            except (OSError, ValueError):
                pass
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(job, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass


def csv_epoch_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    points: list[dict[str, Any]] = []
    previous_elapsed = 0.0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                values: dict[str, float] = {}
                for key, raw in row.items():
                    if key is None or raw is None:
                        continue
                    try:
                        value = float(raw.strip())
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        values[key.strip()] = value
                epoch_value = values.get("epoch")
                if epoch_value is None:
                    continue
                epoch = int(epoch_value)
                elapsed = values.get("time")
                point: dict[str, Any] = {"epoch": epoch, "metrics": {k: v for k, v in values.items() if k not in {"epoch", "time"}}}
                if elapsed is not None:
                    point["elapsedTrainingSeconds"] = elapsed
                    point["epochDurationSeconds"] = max(0.0, elapsed - previous_elapsed)
                    previous_elapsed = elapsed
                points.append(point)
    except (OSError, csv.Error, UnicodeError):
        return []
    return points


def merge_epoch_metrics(job: dict[str, Any], rows: list[dict[str, Any]], recorded_at: str | None = None) -> list[dict[str, Any]]:
    existing = {int(point["epoch"]): point for point in job.get("epochMetrics", []) if isinstance(point, dict) and str(point.get("epoch", "")).isdigit()}
    inherited_through = int(job.get("resumeParentEpoch") or 0)
    inherited_from = job.get("parentRunId")
    for row in rows:
        epoch = int(row["epoch"])
        old = existing.get(epoch, {})
        point = {**old, **row}
        point.setdefault("recordedAt", recorded_at or utc_now())
        if inherited_from and epoch <= inherited_through:
            point["inherited"] = True
            point["inheritedFromRunId"] = inherited_from
        existing[epoch] = point
    return [existing[epoch] for epoch in sorted(existing)]


def append_job_event(job: dict[str, Any], level: str, message: str, source: str = "app") -> None:
    events = job.setdefault("events", [])
    events.append({"timestamp": utc_now(), "level": level.upper(), "source": source, "message": str(message)[:2000]})
    job["events"] = events[-200:]


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def resource_sample(previous: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    try:
        cpu = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
        current = {"total": sum(map(int, cpu[1:])), "idle": int(cpu[4]) + (int(cpu[5]) if len(cpu) > 5 else 0)}
        cpu_percent = None
        if previous and current["total"] > previous["total"]:
            total_delta = current["total"] - previous["total"]
            idle_delta = current["idle"] - previous["idle"]
            cpu_percent = round(max(0.0, min(100.0, 100 * (1 - idle_delta / total_delta))), 1)
        memory = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                memory[key] = int(value.strip().split()[0]) * 1024
        total_memory = memory.get("MemTotal")
        memory_used = total_memory - memory["MemAvailable"] if total_memory and "MemAvailable" in memory else None
        try:
            cgroup_used = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
            cgroup_limit_raw = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            cgroup_limit = int(cgroup_limit_raw) if cgroup_limit_raw != "max" else None
            if cgroup_limit:
                memory_used, total_memory = cgroup_used, cgroup_limit
        except (OSError, ValueError):
            pass
        process_cpu = resource_usage.getrusage(resource_usage.RUSAGE_SELF)
        cpu_seconds = process_cpu.ru_utime + process_cpu.ru_stime
        sampled_monotonic = time.monotonic()
        process_cpu_percent = None
        if previous and sampled_monotonic > previous.get("sampledMonotonic", sampled_monotonic):
            process_cpu_percent = round(max(0.0, (cpu_seconds - previous.get("processCpuSeconds", cpu_seconds)) / (sampled_monotonic - previous["sampledMonotonic"]) * 100), 1)
        current.update(processCpuSeconds=cpu_seconds, sampledMonotonic=sampled_monotonic)
        return {"cpuPercent": cpu_percent, "processCpuPercent": process_cpu_percent, "memoryTotalBytes": total_memory, "memoryAvailableBytes": memory.get("MemAvailable"), "memoryUsedBytes": memory_used, "memoryScope": "container cgroup limit/current" if Path("/sys/fs/cgroup/memory.current").is_file() else "host-visible /proc/meminfo"}, current
    except (OSError, IndexError, ValueError):
        return {}, previous


def results_csv_path(metadata: Path, job: dict[str, Any]) -> Path:
    return checkpoint_weight_dir(metadata, job).parent / "results.csv"


def metric_history(job_id: str) -> dict[str, Any]:
    metadata = job_path(job_id)
    if not metadata.is_file():
        raise HTTPException(404, "Training job not found")
    job = json.loads(metadata.read_text(encoding="utf-8"))
    history_job = dict(job)
    if not history_job.get("parentRunId") and history_job.get("initMode") == "resume":
        match = re.fullmatch(r"([a-f0-9]{32})-last", str(history_job.get("checkpointId", "")))
        if match:
            parent_id = match.group(1)
            parent_meta = RUNS_DIR / "jobs" / parent_id / "job.json"
            if parent_meta.is_file():
                parent_job = json.loads(parent_meta.read_text(encoding="utf-8"))
                parent_rows = csv_epoch_metrics(results_csv_path(parent_meta, parent_job))
                history_job["parentRunId"] = parent_id
                history_job["resumeParentEpoch"] = int(parent_job.get("epoch") or max((int(row["epoch"]) for row in parent_rows), default=0))
    points = merge_epoch_metrics(history_job, csv_epoch_metrics(results_csv_path(metadata, job)))
    if not points:
        points = job.get("epochMetrics", [])
    return {"runId": job_id, "metrics": points, "recordedAt": job.get("metricsUpdatedAt"), "source": "Ultralytics results.csv" if results_csv_path(metadata, job).is_file() else "job record"}


def backfill_job_metrics() -> None:
    jobs_dir = RUNS_DIR / "jobs"
    if not jobs_dir.is_dir():
        return
    for metadata in jobs_dir.glob("*/job.json"):
        try:
            job = json.loads(metadata.read_text(encoding="utf-8"))
            rows = csv_epoch_metrics(results_csv_path(metadata, job))
            if job.get("status") not in ACTIVE_TRAINING_STATUSES and rows and len(job.get("epochMetrics", [])) < len(rows):
                job["epochMetrics"] = merge_epoch_metrics(job, rows)
                job["metricsUpdatedAt"] = utc_now()
                save_job(job)
        except (OSError, ValueError, KeyError, TypeError):
            continue


def checkpoint_models() -> list[dict[str, Any]]:
    models = []
    jobs_dir = RUNS_DIR / "jobs"
    if not jobs_dir.is_dir():
        return models
    for metadata in jobs_dir.glob("*/job.json"):
        try:
            job = json.loads(metadata.read_text(encoding="utf-8"))
            if job.get("resumedBy"):
                continue
            best = checkpoint_weight_dir(metadata, job) / "best.pt"
            if best.is_file() and job.get("status") == "completed":
                source = job.get("datasetSource") or (job.get("request") or {}).get("datasetSource") or "container_detector"
                label = "Casting holes" if source == "casting_holes" else "Container"
                models.append({"id": job["id"], "datasetSource": source, "name": f"{label} {Path(job.get('baseModel', 'yolo26n.pt')).stem.upper()} · {job['id'][:8]}", "path": str(best), "createdAt": job.get("updatedAt"), "epochs": job.get("epochs"), "metrics": job.get("metrics", {})})
        except (OSError, ValueError, KeyError):
            continue
    return sorted(models, key=lambda model: model.get("createdAt") or "", reverse=True)


ACTIVE_TRAINING_STATUSES = {"queued", "preparing", "running", "evaluating"}
FINAL_TRAINING_STATUSES = {"completed", "failed", "cancelled", "canceled", "interrupted", "stopped"}


def checkpoint_weight_dir(metadata: Path, job: dict[str, Any]) -> Path:
    configured = job.get("checkpointOutputDir")
    if configured:
        candidate = Path(configured)
        candidate = candidate if candidate.is_absolute() else RUNS_DIR / candidate
        candidate = candidate.resolve()
        if candidate.is_relative_to(RUNS_DIR):
            return candidate

    if job.get("initMode") == "resume":
        source_id = re.fullmatch(r"([a-f0-9]{32})-last", str(job.get("checkpointId", "")))
        if source_id:
            source_job = RUNS_DIR / "jobs" / source_id.group(1)
            return source_job / "results" / "container" / "weights"

    return metadata.parent / "results" / "container" / "weights"


def checkpoint_timestamp(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()
    except OSError:
        return None


def is_stale_job(job: dict[str, Any], status: str | None = None) -> bool:
    if (status or str(job.get("status", ""))).lower() not in ACTIVE_TRAINING_STATUSES or not job.get("updatedAt"):
        return False
    try:
        updated = datetime.fromisoformat(str(job["updatedAt"]).replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - updated.astimezone(timezone.utc)).total_seconds() > 120
    except (ValueError, TypeError):
        return False


def recorded_training_config(metadata: Path, job: dict[str, Any]) -> dict[str, Any]:
    if isinstance(job.get("config"), dict) and job["config"]:
        return job["config"]
    args_path = results_csv_path(metadata, job).parent / "args.yaml"
    if args_path.is_file():
        try:
            import yaml
            value = yaml.safe_load(args_path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                value["source"] = "Ultralytics args.yaml"
                return json_safe(value)
        except (OSError, ValueError, ImportError):
            pass
    keys = ("initMode", "initialModel", "baseModel", "epochs", "totalEpochs", "imageSize", "batch", "datasetFingerprint")
    return {key: job[key] for key in keys if job.get(key) is not None}


def inherited_checkpoint_metadata(metadata: Path, job: dict[str, Any]) -> dict[str, Any]:
    recorded = job.get("inheritedCheckpoint")
    if isinstance(recorded, dict) and recorded:
        return recorded
    if job.get("initMode") != "resume" or not job.get("startedAt"):
        return {}
    weight = checkpoint_weight_dir(metadata, job) / "last.pt"
    try:
        modified = datetime.fromtimestamp(weight.stat().st_mtime, timezone.utc)
        started = datetime.fromisoformat(str(job["startedAt"]).replace("Z", "+00:00"))
        if modified <= started:
            match = re.fullmatch(r"([a-f0-9]{32})-last", str(job.get("checkpointId", "")))
            return {"runId": match.group(1) if match else job.get("resumedFromJobId"), "path": str(weight), "sizeBytes": weight.stat().st_size, "modifiedAt": modified.isoformat(), "source": "inferred from checkpoint file mtime before resume start"}
    except (OSError, ValueError, TypeError):
        return {}
    return {}


def resumed_parent(metadata: Path, job: dict[str, Any]) -> tuple[str | None, int | None]:
    parent_id = job.get("parentRunId") or job.get("resumedFromJobId")
    if not parent_id:
        match = re.fullmatch(r"([a-f0-9]{32})-last", str(job.get("checkpointId", "")))
        parent_id = match.group(1) if match else None
    if not parent_id:
        return None, None
    parent_path = RUNS_DIR / "jobs" / str(parent_id) / "job.json"
    try:
        parent_job = json.loads(parent_path.read_text(encoding="utf-8"))
        epoch = parent_job.get("epoch")
        if isinstance(epoch, int):
            return str(parent_id), epoch
        rows = csv_epoch_metrics(results_csv_path(parent_path, parent_job))
        return str(parent_id), max((int(row["epoch"]) for row in rows), default=0)
    except (OSError, ValueError, KeyError, TypeError):
        return str(parent_id), None


def training_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    jobs_dir = RUNS_DIR / "jobs"
    if not jobs_dir.is_dir():
        return runs

    ready_model_ids = {model["id"] for model in checkpoint_models()}
    for metadata in jobs_dir.glob("*/job.json"):
        try:
            job = json.loads(metadata.read_text(encoding="utf-8"))
            job_id = str(job["id"])
            status = str(job.get("status", "unknown")).lower()
            status_label = (
                "TRAINING" if status in ACTIVE_TRAINING_STATUSES else
                "COMPLETED" if status == "completed" else
                "CANCELLED" if status in {"cancelled", "canceled", "interrupted", "stopped"} else
                "FAILED" if status == "failed" else
                "FAILED"
            )
            weight_dir = checkpoint_weight_dir(metadata, job)
            checkpoints = []
            for kind in (() if job.get("resumedBy") else ("best", "last")):
                path = weight_dir / f"{kind}.pt"
                if not path.is_file():
                    continue
                ready = kind == "best" and status == "completed" and job_id in ready_model_ids
                checkpoints.append({
                    "id": f"{job_id}-{kind}",
                    "ownerRunId": job_id,
                    "kind": kind,
                    "filename": path.name,
                    "sizeBytes": path.stat().st_size,
                    "lastModifiedAt": checkpoint_timestamp(path),
                    "epoch": job.get("bestEpoch") if kind == "best" else job.get("epoch"),
                    "metrics": job.get("bestMetrics", {}) if kind == "best" else job.get("metrics", {}),
                    "selectionMetric": job.get("selectionMetric") if kind == "best" else None,
                    "inheritedCheckpointAt": inherited_checkpoint_metadata(metadata, job).get("modifiedAt") if kind == "last" else None,
                    "downloadUrl": f"/api/checkpoints/{job_id}-{kind}/download",
                    "downloadAllowed": status in FINAL_TRAINING_STATUSES,
                    "interim": status in ACTIVE_TRAINING_STATUSES,
                    "readyForDetection": ready,
                    "description": "Highest validation score so far; used for detection when the run passes readiness checks." if kind == "best" else "Most recent training state; useful for continuing an interrupted run.",
                })
            last_checkpoint_update = max((item["lastModifiedAt"] for item in checkpoints if item["lastModifiedAt"]), default=None)
            created_at = job.get("createdAt")
            display_time = job.get("startedAt") or created_at
            model_name = Path(str(job.get("baseModel", "yolo26n.pt"))).stem
            model_label = f"YOLO{model_name[4:]}" if model_name.lower().startswith("yolo") else model_name
            try:
                parsed_time = datetime.fromisoformat(str(display_time).replace("Z", "+00:00")) if display_time else None
                readable_time = parsed_time.astimezone().strftime("%b %d, %Y, %H:%M") if parsed_time else None
            except (ValueError, TypeError):
                readable_time = None
            runs.append({
                "id": job_id,
                "shortId": job_id[:8],
                "name": f"{model_label} Training — {readable_time}" if readable_time else f"{model_label} Training run",
                "baseModel": model_label,
                "datasetSource": job.get("datasetSource") or (job.get("request") or {}).get("datasetSource") or "container_detector",
                "status": status,
                "statusLabel": status_label,
                "message": job.get("message"),
                "createdAt": created_at,
                "startedAt": job.get("startedAt"),
                "completedAt": job.get("completedAt") or (job.get("updatedAt") if status == "completed" else None),
                "finishedAt": job.get("finishedAt") or (job.get("updatedAt") if status in {"completed", "failed", "cancelled", "canceled", "interrupted"} else None),
                "updatedAt": job.get("updatedAt"),
                "epoch": job.get("epoch"),
                "totalEpochs": job.get("totalEpochs") or job.get("epochs"),
                "lastCheckpointUpdatedAt": last_checkpoint_update,
                "elapsedSeconds": job.get("elapsedSeconds"),
                "resources": job.get("resources", {}),
                "resourceSampledAt": job.get("resourceSampledAt"),
                "stale": is_stale_job(job, status),
                "bestMetrics": job.get("bestMetrics", {}),
                "latestMetrics": job.get("metrics", {}),
                "checkpointLineage": {"parentRunId": resumed_parent(metadata, job)[0], "resumedFrom": job.get("checkpointId"), "resumeParentEpoch": job.get("resumeParentEpoch") if job.get("resumeParentEpoch") is not None else resumed_parent(metadata, job)[1]},
                "resumedBy": job.get("resumedBy"),
                "inheritedCheckpoint": inherited_checkpoint_metadata(metadata, job),
                "bestEpoch": job.get("bestEpoch"),
                "selectionMetric": job.get("selectionMetric"),
                "testMetrics": job.get("testMetrics", {}),
                "averageEpochDurationSeconds": job.get("averageEpochDurationSeconds"),
                "recentEpochDurationSeconds": job.get("recentEpochDurationSeconds"),
                "estimatedRemainingSeconds": job.get("estimatedRemainingSeconds"),
                "recentTrainingSpeed": job.get("recentTrainingSpeed", {}),
                "environment": job.get("environment", {}),
                "config": recorded_training_config(metadata, job),
                "datasetFingerprint": job.get("datasetFingerprint"),
                "datasetDiagnostics": job.get("datasetDiagnostics", {}),
                "failureReason": job.get("failureReason"),
                "checkpoints": checkpoints,
            })
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return sorted(runs, key=lambda run: run.get("startedAt") or run.get("createdAt") or "", reverse=True)


def list_checkpoints() -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    jobs_dir = RUNS_DIR / "jobs"
    if jobs_dir.is_dir():
        for metadata in jobs_dir.glob("*/job.json"):
            try:
                job = json.loads(metadata.read_text(encoding="utf-8"))
                job_id = str(job["id"])
                if job.get("resumedBy"):
                    continue
                weight_dir = checkpoint_weight_dir(metadata, job)
                for kind in ("best", "last"):
                    path = weight_dir / f"{kind}.pt"
                    if not path.is_file():
                        continue
                    checkpoints.append({
                        "id": f"{job_id}-{kind}",
                        "name": f"{Path(job.get('baseModel', 'YOLO')).stem.upper()} {kind}.pt · {job_id[:8]}",
                        "filename": path.name,
                        "kind": kind,
                        "source": "training run",
                        "datasetSource": job.get("datasetSource") or (job.get("request") or {}).get("datasetSource") or "container_detector",
                        "jobStatus": job.get("status"),
                        "fineTuneAllowed": job.get("status") in FINAL_TRAINING_STATUSES,
                        "resumeAllowed": kind == "last" and job.get("status") in {"failed", "interrupted", "stopped", "cancelled"},
                        "createdAt": job.get("updatedAt"),
                        "sizeBytes": path.stat().st_size,
                        "lastModifiedAt": checkpoint_timestamp(path),
                        "downloadAllowed": job.get("status") in FINAL_TRAINING_STATUSES,
                        "interim": job.get("status") in ACTIVE_TRAINING_STATUSES,
                        "downloadUrl": f"/api/checkpoints/{job_id}-{kind}/download",
                    })
            except (OSError, ValueError, KeyError):
                continue
    uploads_dir = RUNS_DIR / "checkpoints"
    if uploads_dir.is_dir():
        for metadata in uploads_dir.glob("upload-*.json"):
            try:
                saved = json.loads(metadata.read_text(encoding="utf-8"))
                path = uploads_dir / f"{metadata.stem}.pt"
                if path.is_file():
                    checkpoints.append({
                        "id": metadata.stem,
                        "name": saved.get("name", path.name),
                        "filename": path.name,
                        "kind": "uploaded",
                        "source": "uploaded checkpoint",
                        "fineTuneAllowed": True,
                        "resumeAllowed": False,
                        "createdAt": saved.get("createdAt"),
                        "sizeBytes": path.stat().st_size,
                        "lastModifiedAt": checkpoint_timestamp(path),
                        "downloadAllowed": True,
                        "interim": False,
                        "downloadUrl": f"/api/checkpoints/{metadata.stem}/download",
                    })
            except (OSError, ValueError):
                continue
    return sorted(checkpoints, key=lambda item: item.get("createdAt") or "", reverse=True)


def resolve_checkpoint(checkpoint_id: str) -> tuple[Path, dict[str, Any]]:
    upload_match = re.fullmatch(r"upload-[a-f0-9]{32}", checkpoint_id)
    if upload_match:
        directory = RUNS_DIR / "checkpoints"
        path = directory / f"{checkpoint_id}.pt"
        metadata_path = directory / f"{checkpoint_id}.json"
        if not path.is_file() or not metadata_path.is_file():
            raise HTTPException(404, "Uploaded checkpoint not found")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return path, {"id": checkpoint_id, **metadata, "kind": "uploaded", "resumeAllowed": False}
    match = re.fullmatch(r"([a-f0-9]{32})-(best|last)", checkpoint_id)
    if not match:
        raise HTTPException(400, "Invalid checkpoint id")
    job_id, kind = match.groups()
    job = read_job(job_id)
    successor_id = job.get("resumedBy")
    if successor_id:
        try:
            successor = read_job(str(successor_id))
        except HTTPException:
            successor = {}
        if successor.get("status") in ACTIVE_TRAINING_STATUSES:
            raise HTTPException(409, "This checkpoint is being updated by a resumed training run. Use its current run checkpoint after training stops.")
    if job.get("status") not in FINAL_TRAINING_STATUSES:
        raise HTTPException(409, "This checkpoint is not finalized yet and may still be changing. Download it after the training run stops.")
    path = checkpoint_weight_dir(job_path(job_id), job) / f"{kind}.pt"
    if not path.is_file():
        raise HTTPException(404, "Training checkpoint not found")
    descriptor = {"id": checkpoint_id, "jobId": job_id, "kind": kind, "jobStatus": job.get("status"), "resumeAllowed": kind == "last" and job.get("status") in {"failed", "interrupted", "stopped", "cancelled"} and not job.get("resumedBy")}
    return path, descriptor


def export_job_path(export_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", export_id):
        raise HTTPException(400, "Invalid export id")
    return RUNS_DIR / "exports" / f"{export_id}.json"


def save_export(job: dict[str, Any]) -> None:
    path = export_job_path(job["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def selected_model(model_id: str | None = None, dataset_source: str = "container_detector") -> tuple[str, Any]:
    global _loaded_model
    models = [m for m in checkpoint_models() if m.get("datasetSource", "container_detector") == dataset_source]
    match = next((model for model in models if model["id"] == model_id), None) if model_id else (models[0] if models else None)
    if not match:
        raise HTTPException(409 if not models else 404, ("Train a casting-hole detector before running inference" if dataset_source == "casting_holes" else "Train a container detector before running inference") if not models else "Trained model not found")
    path = match["path"]
    with _model_lock:
        if _loaded_model and _loaded_model[0] == path:
            return path, _loaded_model[1]
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise HTTPException(503, "Ultralytics is unavailable in this runtime") from error
        model = YOLO(path)
        _loaded_model = (path, model)
        return path, model


class TrainRequest(BaseModel):
    epochs: int = Field(default=50, ge=1, le=300)
    imageSize: int = Field(default=640, ge=320, le=1280)
    batch: int = Field(default=4, ge=1, le=32)
    baseModel: str = Field(default="yolo26n.pt", pattern=r"^yolo26[nsmlx]\.pt$")
    initMode: str = Field(default="base", pattern=r"^(base|finetune|resume)$")
    checkpointId: str | None = None
    datasetSource: str = Field(default="container_detector", pattern=r"^(container_detector|casting_holes)$")


def label_for(image: Path, image_root: Path, labels_root: Path) -> Path:
    relative = image.relative_to(image_root)
    label = (labels_root / relative).with_suffix(".txt")
    if not label.is_file():
        label = labels_root / f"{image.stem}.txt"
    return label


def validated_boxes(path: Path) -> list[tuple[float, float, float, float]]:
    if not path.is_file():
        return []
    boxes: list[tuple[float, float, float, float]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            class_id, cx, cy, width, height = map(float, fields)
        except ValueError:
            raise ValueError(f"Invalid YOLO label at {path.name}:{number}") from None
        if len(fields) != 5 or class_id != 0 or not class_id.is_integer():
            raise ValueError(f"Expected class 0 and five YOLO values at {path.name}:{number}")
        if width <= 0 or height <= 0 or min(cx - width / 2, cy - height / 2) < 0 or max(cx + width / 2, cy + height / 2) > 1:
            raise ValueError(f"Out-of-range YOLO box at {path.name}:{number}")
        boxes.append((cx, cy, width, height))
    return boxes


def dataset_catalog_path() -> Path:
    return RUNS_DIR / "datasets" / "catalog.sqlite3"


def open_dataset_catalog(read_only: bool = False) -> sqlite3.Connection:
    path=dataset_catalog_path()
    if read_only:
        connection=sqlite3.connect(f"file:{path}?mode=ro",uri=True,timeout=30)
        connection.row_factory=sqlite3.Row
        return connection
    return dataset_catalog.connect(path)


def migrate_legacy_training_records(db: sqlite3.Connection) -> None:
    """Keep old JSON jobs addressable while marking unrecoverable manifests explicitly."""
    candidates = [("container_detector", RUNS_DIR / "jobs"), ("container_count_classifier", classifier_root() / "jobs")]
    for model_key, root in candidates:
        if not root.is_dir():
            continue
        for job_file in root.glob("*/job.json"):
            try:
                job = json.loads(job_file.read_text(encoding="utf-8"))
                run_id = str(job.get("id") or job_file.parent.name)
                if str(job.get("status", "")).lower() in ACTIVE_TRAINING_STATUSES:
                    continue
                if db.execute("SELECT 1 FROM training_runs WHERE run_id=?", (run_id,)).fetchone():
                    continue
                counts = job.get("splitCounts") or {}
                train = int(counts.get("train", 0) or 0)
                validation = int(counts.get("validation", counts.get("val", 0)) or 0)
                test = int(counts.get("test", 0) or 0)
                version_id = f"legacy-{run_id}"
                dataset_id = f"legacy:{job.get('datasetId') or 'unknown'}:{run_id}"
                manifest = {"legacy": True, "reconciled": False,
                            "reason": "This run predates immutable dataset manifests; its exact source images or annotation revisions cannot be proven from the saved job record.",
                            "oldDatasetPath": job.get("datasetPath"), "splitCounts": counts}
                db.execute("INSERT OR IGNORE INTO dataset_versions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (version_id, dataset_id, 1, model_key, job.get("createdAt") or utc_now(), "{}", train, validation, test, 0,
                     0, json.dumps(manifest, sort_keys=True)))
                db.execute("INSERT OR IGNORE INTO training_runs(run_id,model_key,dataset_version_id,legacy_dataset_id,started_at) VALUES (?,?,?,?,?)",
                    (run_id, model_key, version_id, str(job.get("datasetId") or "legacy"), job.get("startedAt")))
            except (OSError, ValueError, TypeError):
                continue
    db.execute("INSERT OR IGNORE INTO schema_migrations VALUES (2,?)", (utc_now(),))
    db.commit()


def sync_dataset_catalog() -> dict[str, Any]:
    with _catalog_lock:
        return _sync_dataset_catalog()


def _sync_dataset_catalog() -> dict[str, Any]:
    """Index the read-only original dataset; binaries remain in their source location."""
    image_root = DATASET_ROOT / "images"
    if not image_root.is_dir():
        image_root = DATASET_ROOT
    labels_root = DATASET_ROOT / "labels"
    now = utc_now()
    db = dataset_catalog.connect(dataset_catalog_path())
    path_ids: dict[str, str] = {}
    image_hashes: dict[str, str] = {}
    try:
        db.execute("UPDATE image_sources SET active=0 WHERE dataset_source='server dataset'")
        for image in sorted((p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED), key=lambda p: p.as_posix().casefold()):
            relative = image.relative_to(DATASET_ROOT).as_posix()
            stat = image.stat()
            known = db.execute("SELECT s.image_id,s.mtime_ns,i.byte_size FROM image_sources s JOIN source_images i USING(image_id) WHERE s.source_path=?", (relative,)).fetchone()
            if known and known["mtime_ns"] == stat.st_mtime_ns and known["byte_size"] == stat.st_size:
                image_id = known["image_id"]
                db.execute("UPDATE image_sources SET active=1 WHERE source_path=?",(relative,))
            else:
                digest = dataset_catalog.hash_file(image)
                image_id = dataset_catalog.stable_image_id(digest)
                width=height=None; embedded={}
                try:
                    from PIL import Image
                    with Image.open(image) as source_image:
                        width,height=source_image.size
                        embedded={str(key):str(value)[:1000] for key,value in source_image.info.items() if isinstance(value,(str,int,float))}
                except Exception:
                    pass
                metadata={"originalFilename":image.name,"sourcePath":relative,"fileMtimeNs":stat.st_mtime_ns,"embeddedMetadata":embedded}
                db.execute("INSERT OR IGNORE INTO source_images(image_id,sha256,byte_size,width,height,format,first_imported_at,metadata_json) VALUES (?,?,?,?,?,?,?,?)",
                           (image_id, digest, stat.st_size, width, height, image.suffix.lower().lstrip("."), now, json.dumps(metadata)))
                group_key = image.stem.removesuffix("_cropped").casefold()
                db.execute("INSERT INTO image_sources(image_id,source_path,original_filename,dataset_source,imported_at,mtime_ns,group_key,active) VALUES (?,?,?,?,?,?,?,1) "
                           "ON CONFLICT(source_path) DO UPDATE SET image_id=excluded.image_id,original_filename=excluded.original_filename,mtime_ns=excluded.mtime_ns,group_key=excluded.group_key,active=1",
                           (image_id, relative, image.name, "server dataset", now, stat.st_mtime_ns, group_key))
            label = label_for(image, image_root, labels_root)
            try:
                payload = validated_boxes(label) if label.is_file() else []
                annotation = json.dumps({"boxes": payload, "labelPath": label.relative_to(DATASET_ROOT).as_posix() if label.is_file() else None}, sort_keys=True)
                previous = db.execute("SELECT version,payload_json FROM annotations WHERE image_id=? AND task='detection' AND model_key='container_detector' ORDER BY version DESC LIMIT 1", (image_id,)).fetchone()
                if not previous or previous["payload_json"] != annotation:
                    version = (previous["version"] if previous else 0) + 1
                    db.execute("INSERT INTO annotations(image_id,task,model_key,version,payload_json,source_path,created_at) VALUES (?,?,?,?,?,?,?)",
                               (image_id, "detection", "container_detector", version, annotation, label.relative_to(DATASET_ROOT).as_posix() if label.is_file() else None, now))
            except (OSError, UnicodeError, ValueError):
                pass
            path_ids[relative] = image_id
            image_hashes[relative] = db.execute("SELECT sha256 FROM source_images WHERE image_id=?", (image_id,)).fetchone()[0]
        group_keys = [row[0] for row in db.execute("SELECT DISTINCT group_key FROM image_sources WHERE active=1")]
        dataset_catalog.assign_group_splits(db, group_keys, now)
        db.commit()
        image_count = db.execute("SELECT COUNT(DISTINCT image_id) FROM image_sources WHERE active=1").fetchone()[0]
        duplicate_count = db.execute("SELECT COUNT(*) FROM image_sources s WHERE active=1 AND EXISTS (SELECT 1 FROM image_sources other WHERE other.image_id=s.image_id AND other.active=1 AND other.source_id<s.source_id)").fetchone()[0]
        return {"pathIds": path_ids, "hashes": image_hashes, "uniqueOriginalImages": image_count, "duplicatePaths": duplicate_count}
    finally:
        db.close()


def validate_label(path: Path) -> int:
    return len(validated_boxes(path))


def advance_batch_count(current: int, total: int) -> int:
    return min(max(0, current) + 1, max(0, total))


def histogram(values: list[float], edges: list[float]) -> list[dict[str, Any]]:
    result = []
    for index, lower in enumerate(edges[:-1]):
        upper = edges[index + 1]
        result.append({"range": f"{lower:g}–{upper:g}", "count": sum(lower <= value < upper for value in values)})
    result.append({"range": f"≥{edges[-1]:g}", "count": sum(value >= edges[-1] for value in values)})
    return result


def update_epoch_timing(job: dict[str, Any], points: list[dict[str, Any]], total_epochs: int) -> None:
    durations = [float(point["epochDurationSeconds"]) for point in points if isinstance(point.get("epochDurationSeconds"), (int, float)) and point["epochDurationSeconds"] > 0]
    if not durations:
        return
    recent = durations[-5:]
    average = sum(durations) / len(durations)
    recent_average = sum(recent) / len(recent)
    remaining_epochs = max(0, total_epochs - int(job.get("epoch") or 0))
    job["averageEpochDurationSeconds"] = average
    job["recentEpochDurationSeconds"] = recent_average
    job["estimatedRemainingSeconds"] = round(remaining_epochs * recent_average)
    job["recentTrainingSpeed"] = {"completedEpochsPerHour": round(3600 / recent_average, 3), "windowEpochs": len(recent), "imagesPerSecond": None, "note": "Epoch wall time includes validation; image throughput is not estimated."}


def build_training_dataset(job_id: str) -> tuple[Path, dict[str, int], str, int, dict[str, Any]]:
    image_root, labels_root = DATASET_ROOT / "images", DATASET_ROOT / "labels"
    if not image_root.is_dir() or not labels_root.is_dir():
        raise ValueError("Expected a YOLO dataset with images/ and labels/ folders.")
    images = sorted((path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED), key=lambda path: path.relative_to(image_root).as_posix().casefold())
    if len(images) < 3:
        raise ValueError("At least three supported images are required to create train/validation/test splits.")
    catalog = sync_dataset_catalog()
    groups: dict[str, list[Path]] = {}
    boxes, skipped, fingerprint = 0, 0, hashlib.sha256()
    content_annotations: dict[str, list[tuple[float, float, float, float]]] = {}
    excluded_samples: list[dict[str, str]] = []
    label_summary: dict[str, Any] = {"sourceImageCount": len(images), "missingLabelFiles": 0, "invalidLabelFiles": 0, "emptyLabelFiles": 0, "validLabelFiles": 0, "widths": [], "heights": [], "areas": [], "boxesPerImage": [], "invalidSamples": []}
    for image in images:
        label = label_for(image, image_root, labels_root)
        if not label.is_file():
            skipped += 1
            label_summary["missingLabelFiles"] += 1
            excluded_samples.append({"imageId": catalog["pathIds"].get(image.relative_to(DATASET_ROOT).as_posix(), ""), "path": image.relative_to(DATASET_ROOT).as_posix(), "reason": "missing detection label"})
            continue
        try:
            image_box_rows = validated_boxes(label)
        except (OSError, ValueError) as error:
            skipped += 1
            label_summary["invalidLabelFiles"] += 1
            if len(label_summary["invalidSamples"]) < 30:
                label_summary["invalidSamples"].append({"image": image.relative_to(image_root).as_posix(), "reason": str(error)[:240]})
            excluded_samples.append({"imageId": catalog["pathIds"].get(image.relative_to(DATASET_ROOT).as_posix(), ""), "path": image.relative_to(DATASET_ROOT).as_posix(), "reason": f"invalid detection label: {str(error)[:180]}"})
            continue
        image_boxes = len(image_box_rows)
        relative_path = image.relative_to(DATASET_ROOT).as_posix()
        image_id = catalog["pathIds"].get(relative_path)
        if image_id in content_annotations:
            if content_annotations[image_id] != image_box_rows:
                raise ValueError(f"Exact duplicate image has conflicting labels and needs manual reconciliation: {image.name}")
            skipped += 1
            excluded_samples.append({"imageId": image_id, "path": relative_path, "reason": "exact duplicate source image; represented by its canonical image ID"})
            continue
        if image_id:
            content_annotations[image_id] = image_box_rows
        label_summary["validLabelFiles"] += 1
        if image_boxes == 0:
            label_summary["emptyLabelFiles"] += 1
        label_summary["boxesPerImage"].append({"image": image.relative_to(image_root).as_posix(), "count": image_boxes})
        for _, _, width, height in image_box_rows:
            label_summary["widths"].append(width)
            label_summary["heights"].append(height)
            label_summary["areas"].append(width * height)
        boxes += image_boxes
        fingerprint.update(f"{image.relative_to(image_root).as_posix()}\0{catalog['hashes'].get(relative_path,'')}\n".encode())
        if label.is_file():
            fingerprint.update(label.read_bytes())
        groups.setdefault(image.stem.removesuffix("_cropped").casefold(), []).append(image)
    if boxes == 0:
        raise ValueError("No valid container bounding boxes were found in the YOLO labels.")
    group_names = sorted(groups)
    if len(group_names) < 3:
        raise ValueError("At least three distinct source-image groups are required for leakage-safe splits.")
    db = dataset_catalog.connect(dataset_catalog_path())
    try:
        assignment = {row["group_key"]: row["split"] for row in db.execute("SELECT group_key,split FROM split_assignments")}
        group_splits = {name: assignment.get(name, "train") for name in group_names}
        splits = {"train": {name for name, split in group_splits.items() if split == "train"},
                  "val": {name for name, split in group_splits.items() if split == "validation"},
                  "test": {name for name, split in group_splits.items() if split == "test"}}
        cross_split_groups = set()
    finally:
        db.close()
    dataset_dir = RUNS_DIR / "jobs" / job_id / "dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    counts = {"train": 0, "val": 0, "test": 0}
    for split, split_groups in splits.items():
        image_out, label_out = dataset_dir / "images" / split, dataset_dir / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        names_seen = set()
        for group in split_groups:
            for image in groups[group]:
                if image.name.casefold() in names_seen:
                    raise ValueError(f"Repeated image filename in dataset split: {image.name}")
                names_seen.add(image.name.casefold())
                label = label_for(image, image_root, labels_root)
                (image_out / image.name).symlink_to(image.resolve())
                destination_label = label_out / f"{image.stem}.txt"
                shutil.copyfile(label, destination_label) if label.is_file() else destination_label.write_text("", encoding="utf-8")
                counts[split] += 1
    yaml_path = dataset_dir / "data.yaml"
    # Ultralytics resolves a relative `path: .` against its configured global
    # datasets_dir (often /app/datasets), not against this YAML's directory.
    # An explicit mounted snapshot path keeps training independent of that setting.
    yaml_path.write_text(f"path: {json.dumps(str(dataset_dir.resolve()))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: container\n", encoding="utf-8")
    per_image = sorted((item["count"] for item in label_summary["boxesPerImage"]))
    median_boxes = per_image[len(per_image) // 2] if per_image else 0
    deviations = sorted(abs(count - median_boxes) for count in per_image)
    mad = deviations[len(deviations) // 2] if deviations else 0
    many_threshold = max(10, median_boxes + 3 * max(1, mad))
    few_threshold = max(0, median_boxes - 3 * max(1, mad))
    normalized_widths, normalized_heights, normalized_areas = label_summary.pop("widths"), label_summary.pop("heights"), label_summary.pop("areas")
    label_summary["boxCount"] = len(normalized_areas)
    label_summary["boxesPerImage"] = {"minimum": min(per_image, default=0), "median": median_boxes, "maximum": max(per_image, default=0), "imagesWithNoBoxes": sum(count == 0 for count in per_image), "imagesWithOneBox": sum(count == 1 for count in per_image), "unusuallyFewThreshold": few_threshold, "unusuallyFewImages": [item["image"] for item in label_summary["boxesPerImage"] if item["count"] <= few_threshold][:30], "unusuallyManyThreshold": many_threshold, "unusuallyManyImages": [item["image"] for item in label_summary["boxesPerImage"] if item["count"] >= many_threshold][:30]}
    label_summary["normalizedBoxWidthDistribution"] = histogram(normalized_widths, [0, 0.05, 0.1, 0.25, 0.5, 1])
    label_summary["normalizedBoxHeightDistribution"] = histogram(normalized_heights, [0, 0.05, 0.1, 0.25, 0.5, 1])
    label_summary["normalizedBoxAreaDistribution"] = {"small_lt_1_percent": sum(value < 0.01 for value in normalized_areas), "medium_1_to_10_percent": sum(0.01 <= value < 0.1 for value in normalized_areas), "large_gte_10_percent": sum(value >= 0.1 for value in normalized_areas), "definition": "Bounding-box area as a fraction of image area; small <1%, medium 1–10%, large ≥10%."}
    label_summary["duplicateImageCheck"] = {"performed": True, "uniqueOriginalImages": catalog["uniqueOriginalImages"], "duplicatePaths": catalog["duplicatePaths"]}
    label_summary["splitLeakageCheck"] = {"grouping": "Persistent source-image filename groups; terminal _cropped suffix removed.", "groupCount": len(group_names), "groupsInMultipleSplits": len(cross_split_groups), "passed": not cross_split_groups}
    label_summary["fingerprintAlgorithm"] = "SHA-256 of source image bytes and annotation contents."
    dataset_id = hashlib.sha256(f"container_detector\0{DATASET_ROOT}".encode()).hexdigest()[:32]
    version_db = dataset_catalog.connect(dataset_catalog_path())
    members = []
    seen_ids = set()
    for group, images_in_group in groups.items():
        for image in images_in_group:
            relative = image.relative_to(DATASET_ROOT).as_posix()
            image_id = catalog["pathIds"].get(relative)
            if not image_id or image_id in seen_ids:
                continue
            seen_ids.add(image_id)
            ann = version_db.execute("SELECT MAX(version) FROM annotations WHERE image_id=? AND task='detection' AND model_key='container_detector'", (image_id,)).fetchone()[0]
            split = "validation" if group_splits[group] == "validation" else group_splits[group]
            members.append({"image_id": image_id, "split": split, "annotation_version": ann})
    dataset_version_id = dataset_catalog.next_dataset_version(version_db, dataset_id, "container_detector", utc_now(), members,
        {"container_detector": max((row["annotation_version"] or 0 for row in members), default=0)}, excluded=excluded_samples, derived_count=0)
    version_db.executemany("INSERT OR IGNORE INTO derived_samples(sample_id,dataset_version_id,image_id,model_key,sample_type,label_json,preprocessing_version,created_at) VALUES (?,?,?,?,?,?,?,?)",
        [(hashlib.sha256(f"{dataset_version_id}:{row['image_id']}".encode()).hexdigest(),dataset_version_id,row["image_id"],"container_detector","original-image-input",json.dumps({"annotationVersion":row.get("annotation_version")}),"yolo-original-image-v1",utc_now()) for row in members])
    version_db.execute("INSERT OR REPLACE INTO training_runs(run_id,model_key,dataset_version_id,started_at) VALUES (?,?,?,?)", (job_id, "container_detector", dataset_version_id, utc_now()))
    version_db.commit(); version_db.close()
    diagnostics = {"labelValidation": label_summary, "splitStrategy": "Persistent 80/10/10 grouped assignment created once for the shared source dataset.", "splitCounts": counts, "fingerprintAlgorithm": label_summary["fingerprintAlgorithm"], "datasetVersionId": dataset_version_id}
    return yaml_path, counts, fingerprint.hexdigest(), skipped, diagnostics


def _validate_yolo_text(text: str, label_ref: str) -> list[tuple[float, float, float, float]]:
    """Parse and validate single-class (0) normalized YOLO bounding boxes."""
    boxes: list[tuple[float, float, float, float]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        try:
            class_id, cx, cy, width, height = map(float, fields)
        except ValueError:
            raise ValueError(f"Invalid YOLO label at {label_ref}:{number}") from None
        if len(fields) != 5 or class_id != 0 or not class_id.is_integer():
            raise ValueError(f"Expected class 0 and five YOLO values at {label_ref}:{number}")
        if width <= 0 or height <= 0 or min(cx - width / 2, cy - height / 2) < 0 or max(cx + width / 2, cy + height / 2) > 1:
            raise ValueError(f"Out-of-range YOLO box at {label_ref}:{number}")
        boxes.append((cx, cy, width, height))
    return boxes


def build_casting_hole_dataset(job_id: str) -> tuple[Path, dict[str, int], str, int, dict[str, Any]]:
    """Build a leakage-safe YOLO dataset from container-labeler casting_hole_annotations."""
    if not LABELER_DB.is_file():
        raise ValueError(f"Labeler database not found at {LABELER_DB}")
    conn = sqlite3.connect(f"file:{LABELER_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT path, source_path, cached_crop_path, yolo_labels FROM casting_hole_annotations WHERE status='annotated' ORDER BY path"
        ).fetchall()
    finally:
        conn.close()
    if len(rows) < 3:
        raise ValueError("At least three annotated casting-hole crops are required.")
    groups: dict[str, list[tuple[Path, str, str]]] = {}
    skipped, box_count = 0, 0
    fingerprint = hashlib.sha256()
    for row in rows:
        crop_rel = (row["cached_crop_path"] or "").strip()
        yolo_text = (row["yolo_labels"] or "").strip()
        if not crop_rel or not yolo_text:
            skipped += 1
            continue
        crop_path = (DATASET_ROOT / crop_rel).resolve()
        if not crop_path.is_file() or not crop_path.is_relative_to(DATASET_ROOT.resolve()):
            skipped += 1
            continue
        try:
            boxes = _validate_yolo_text(yolo_text, row["path"])
        except ValueError:
            skipped += 1
            continue
        if not boxes:
            skipped += 1
            continue
        box_count += len(boxes)
        group_key = Path(row["source_path"] or row["path"]).stem.removesuffix("_cropped").casefold()
        groups.setdefault(group_key, []).append((crop_path, yolo_text, row["path"]))
        fingerprint.update(f"{crop_rel}\0{yolo_text}\n".encode())
    if box_count == 0 or len(groups) < 3:
        raise ValueError("At least three valid source-image groups with casting-hole boxes are required.")
    splits: dict[str, list[tuple[Path, str, str]]] = {"train": [], "val": [], "test": []}
    for group_key in sorted(groups):
        bucket = int(hashlib.sha256(group_key.encode()).hexdigest()[:8], 16) % 10
        split = "train" if bucket < 8 else ("val" if bucket == 8 else "test")
        splits[split].extend(groups[group_key])
    dataset_dir = RUNS_DIR / "jobs" / job_id / "dataset"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    counts = {"train": 0, "val": 0, "test": 0}
    for split, items in splits.items():
        image_out, label_out = dataset_dir / "images" / split, dataset_dir / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for idx, (crop_path, yolo_text, _) in enumerate(items):
            stem = f"{crop_path.stem}_{idx}"
            (image_out / f"{stem}{crop_path.suffix}").symlink_to(crop_path)
            (label_out / f"{stem}.txt").write_text(yolo_text + "\n", encoding="utf-8")
            counts[split] += 1
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {json.dumps(str(dataset_dir.resolve()))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: casting_hole\n",
        encoding="utf-8",
    )
    diagnostics = {
        "datasetSource": "casting_holes",
        "splitStrategy": "Deterministic 80/10/10 grouped by source_path.",
        "splitCounts": counts,
        "boxCount": box_count,
        "skipped": skipped,
    }
    return yaml_path, counts, fingerprint.hexdigest(), skipped, diagnostics


def training_stop_requested(job_id: str) -> bool:
    if job_queue.cancel_requested(RUNS_DIR / "jobs" / job_id):
        return True
    try:
        return bool(json.loads(job_path(job_id).read_text(encoding="utf-8")).get("stopRequested"))
    except (OSError, ValueError, HTTPException):
        return False


def train_worker(job_id: str, request: TrainRequest) -> None:
    global _active_training
    job = read_job(job_id)
    try:
        job.update(status="preparing", startedAt=job.get("startedAt") or utc_now(), updatedAt=utc_now(), message="Validating labels and creating leakage-aware dataset splits.")
        save_job(job)
        if request.datasetSource == "casting_holes":
            yaml_path, split_counts, fingerprint, skipped_images, dataset_diagnostics = build_casting_hole_dataset(job_id)
        else:
            yaml_path, split_counts, fingerprint, skipped_images, dataset_diagnostics = build_training_dataset(job_id)
        job.update(status="running", updatedAt=utc_now(), message=f"Training on CPU. {skipped_images} images with missing or invalid labels were excluded. The first run downloads YOLO26n weights.", splitCounts=split_counts, skippedImages=skipped_images, datasetFingerprint=fingerprint, datasetVersionId=dataset_diagnostics.get("datasetVersionId"), datasetDiagnostics=dataset_diagnostics)
        save_job(job)
        from ultralytics import YOLO
        checkpoint_path = None
        checkpoint = None
        if request.initMode in {"finetune", "resume"}:
            if not request.checkpointId:
                raise ValueError("Choose a saved or uploaded .pt checkpoint first.")
            checkpoint_path, checkpoint = resolve_checkpoint(request.checkpointId)
            if request.initMode == "resume" and not checkpoint.get("resumeAllowed"):
                raise ValueError("Resume is available only for a saved last.pt checkpoint from an interrupted or failed training run. Choose Fine-tune to start a new run from other .pt files.")
            if request.initMode == "resume":
                job["resumedFromJobId"] = checkpoint.get("jobId")
                source_stat = checkpoint_path.stat()
                job["inheritedCheckpoint"] = {"runId": checkpoint.get("jobId"), "path": str(checkpoint_path), "sizeBytes": source_stat.st_size, "modifiedAt": datetime.fromtimestamp(source_stat.st_mtime, timezone.utc).isoformat(), "capturedAt": utc_now()}
        job_dir = RUNS_DIR / "jobs" / job_id
        if request.initMode == "resume" and checkpoint_path:
            from .yolo_resume import prepare_resume_checkpoint
            resumed_checkpoint = job_dir / "results" / "container" / "weights" / "last.pt"
            prepare_resume_checkpoint(checkpoint_path, resumed_checkpoint, job_dir / "results", "container")
            job["checkpointOutputDir"] = str(resumed_checkpoint.parent)
            initial_model = str(resumed_checkpoint)
        else:
            initial_model = str(checkpoint_path) if checkpoint_path else request.baseModel
        model = YOLO(initial_model)
        try:
            import torch
            import ultralytics
            environment = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "ultralytics": ultralytics.__version__, "application": app.version}
        except Exception:
            environment = {"python": platform.python_version(), "platform": platform.platform(), "application": app.version}
        if request.initMode == "base":
            start_message = f"Starting from {request.baseModel}; its weights download if they are not cached."
        elif request.initMode == "resume":
            start_message = f"Resuming optimizer and epoch state from {checkpoint_path.name}."
        else:
            start_message = f"Fine-tuning weights from {checkpoint_path.name} as a new training run."
        job["message"] = start_message
        job["initialModel"] = str(checkpoint_path) if checkpoint_path else request.baseModel
        job["parentRunId"] = checkpoint.get("jobId") if request.initMode == "resume" and checkpoint else None
        if request.initMode == "resume" and checkpoint:
            parent_csv = results_csv_path(RUNS_DIR / "jobs" / str(checkpoint.get("jobId")) / "job.json", json.loads((RUNS_DIR / "jobs" / str(checkpoint.get("jobId")) / "job.json").read_text(encoding="utf-8")))
            parent_rows = csv_epoch_metrics(parent_csv)
            parent_job = json.loads((RUNS_DIR / "jobs" / str(checkpoint.get("jobId")) / "job.json").read_text(encoding="utf-8"))
            job["resumeParentEpoch"] = int(parent_job.get("epoch") or max((int(row["epoch"]) for row in parent_rows), default=int(checkpoint.get("epoch") or 0)))
        job["environment"] = environment
        job["config"] = {"mode": request.initMode, "initialModel": str(checkpoint_path) if checkpoint_path else request.baseModel, "datasetYaml": str(yaml_path), "epochs": request.epochs, "imageSize": request.imageSize, "batch": request.batch, "device": "cpu", "workers": 0, "cache": False, "amp": False, "patience": 20, "seed": 42, "deterministic": True, "optimizer": "Ultralytics default (actual runtime optimizer is recorded after trainer initialization)"}
        job.setdefault("events", [])
        append_job_event(job, "info", start_message, "training")
        save_job(job)

        last_progress_save = 0.0
        last_resource_sample = 0.0
        train_batches_seen = 0
        previous_cpu_sample: dict[str, Any] | None = None

        def update_phase(phase: str, current: int, total: int, epoch: int | None = None, force: bool = False) -> None:
            nonlocal last_progress_save, last_resource_sample, previous_cpu_sample
            now = time.monotonic()
            phases = job.setdefault("phaseProgress", {})
            entry = phases.setdefault(phase, {"epoch": epoch or 0, "current": 0, "total": max(0, total), "status": "waiting"})
            if epoch is not None and entry.get("epoch") != epoch:
                entry.update(epoch=epoch, current=0, total=max(0, total), status="running")
            entry.update(current=max(0, current), total=max(0, total))
            if total > 0 and current >= total:
                entry["status"] = "completed"
            elif entry.get("status") == "waiting":
                entry["status"] = "running"
            job["currentStage"] = phase
            job["updatedAt"] = utc_now()
            if force or now - last_resource_sample >= 10.0:
                resources, previous_cpu_sample = resource_sample(previous_cpu_sample)
                if resources:
                    job["resources"] = resources
                    job["resourceSampledAt"] = utc_now()
                last_resource_sample = now
            if job.get("startedAt"):
                try:
                    start = datetime.fromisoformat(str(job["startedAt"]).replace("Z", "+00:00"))
                    job["elapsedSeconds"] = max(0, int((datetime.now(timezone.utc) - start).total_seconds()))
                except ValueError:
                    pass
            if force or now - last_progress_save >= 3.0 or (total > 0 and current >= total):
                save_job(job)
                last_progress_save = now

        def on_train_epoch_start(trainer: Any) -> None:
            nonlocal train_batches_seen
            train_batches_seen = 0
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            job.update(epoch=epoch, currentStage="train")
            phases = job.setdefault("phaseProgress", {})
            phases["train"] = {"epoch": epoch, "current": 0, "total": len(getattr(trainer, "train_loader", [])), "status": "running"}
            phases["validation"] = {"epoch": epoch, "current": 0, "total": 0, "status": "waiting"}
            phases.setdefault("test", {"epoch": 0, "current": 0, "total": 0, "status": "waiting"})
            update_phase("train", 0, len(getattr(trainer, "train_loader", [])), epoch, force=True)

        def on_train_batch_end(trainer: Any) -> None:
            nonlocal train_batches_seen
            total = len(getattr(trainer, "train_loader", []))
            train_batches_seen = advance_batch_count(train_batches_seen, total)
            update_phase("train", train_batches_seen, total, int(getattr(trainer, "epoch", 0)) + 1)

        def on_train_epoch_end(trainer: Any) -> None:
            total = len(getattr(trainer, "train_loader", []))
            update_phase("train", total, total, int(getattr(trainer, "epoch", 0)) + 1, force=True)
            if training_stop_requested(job_id):
                trainer.stop = True

        def on_val_start(validator: Any) -> None:
            phase = "test" if job.get("evaluatingTest") else "validation"
            update_phase(phase, 0, len(getattr(validator, "dataloader", [])), int(job.get("epoch", 0)), force=True)

        def on_val_batch_end(validator: Any) -> None:
            phase = "test" if job.get("evaluatingTest") else "validation"
            total = len(getattr(validator, "dataloader", []))
            update_phase(phase, int(getattr(validator, "batch_i", -1)) + 1, total, int(job.get("epoch", 0)))

        def on_val_end(validator: Any) -> None:
            phase = "test" if job.get("evaluatingTest") else "validation"
            total = len(getattr(validator, "dataloader", []))
            update_phase(phase, total, total, int(job.get("epoch", 0)), force=True)

        def on_epoch_end(trainer: Any) -> None:
            metrics = json_safe({str(k): float(v) for k, v in getattr(trainer, "metrics", {}).items() if isinstance(v, (int, float))})
            metadata = job_path(job_id)
            epoch_rows = csv_epoch_metrics(results_csv_path(metadata, job))
            if epoch_rows:
                job["epochMetrics"] = merge_epoch_metrics(job, epoch_rows, utc_now())
                job["metricsUpdatedAt"] = utc_now()
            trainer_args = getattr(trainer, "args", None)
            if trainer_args is not None:
                job["config"] = json_safe(vars(trainer_args))
            current_epoch = int(getattr(trainer, "epoch", -1)) + 1
            job.update(epoch=current_epoch, totalEpochs=request.epochs, metrics=metrics, updatedAt=utc_now())
            update_epoch_timing(job, job.get("epochMetrics", []), request.epochs)
            append_job_event(job, "info", f"Epoch {current_epoch} completed.", "training")
            fitness = getattr(trainer, "fitness", None)
            best_fitness = getattr(trainer, "best_fitness", None)
            if fitness is not None:
                job["latestFitness"] = json_safe(fitness)
            if best_fitness is not None:
                job["bestFitness"] = json_safe(best_fitness)
                if fitness is not None and math.isclose(float(fitness), float(best_fitness), rel_tol=1e-9, abs_tol=1e-12):
                    job["bestMetrics"] = (job.get("epochMetrics") or [{}])[-1].get("metrics", metrics).copy()
                    job["bestEpoch"] = current_epoch
                    job["selectionMetric"] = {"name": "Ultralytics trainer.best_fitness / trainer.fitness", "value": json_safe(best_fitness), "detail": "Recorded from the installed trainer; the UI does not infer this criterion from a displayed metric name."}
            if current_epoch == 1 and trainer_args is not None:
                job["optimizer"] = str(getattr(getattr(trainer, "optimizer", None), "__class__", type(None)).__name__)
            save_job(job)

        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_train_batch_end", on_train_batch_end)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_val_start", on_val_start)
        model.add_callback("on_val_batch_end", on_val_batch_end)
        model.add_callback("on_val_end", on_val_end)
        model.add_callback("on_fit_epoch_end", on_epoch_end)
        if training_stop_requested(job_id):
            stopped_at = utc_now()
            job.update(status="stopped", updatedAt=stopped_at, finishedAt=stopped_at,
                       message="Training stopped before the next epoch. The latest saved checkpoint is available to resume.")
            append_job_event(job, "info", "Training stop completed before the next epoch.", "training")
            save_job(job)
            return
        model.train(data=str(yaml_path), epochs=request.epochs, imgsz=request.imageSize, batch=request.batch, device="cpu", workers=0, cache=False, amp=False, project=str(job_dir / "results"), name="container", exist_ok=True, plots=True, save=True, patience=20, resume=request.initMode == "resume")
        weight_dir = Path(job.get("checkpointOutputDir")) if job.get("checkpointOutputDir") else job_dir / "results" / "container" / "weights"
        best = weight_dir / "best.pt"
        if training_stop_requested(job_id):
            last = weight_dir / "last.pt"
            checkpoint = best if best.is_file() else last if last.is_file() else None
            stopped_at = utc_now()
            job.update(status="stopped", updatedAt=stopped_at, finishedAt=stopped_at,
                       message="Training stopped at the epoch boundary. The latest saved checkpoint is available to resume.",
                       checkpoint=str(checkpoint) if checkpoint else None)
            append_job_event(job, "info", "Training stopped at the requested epoch boundary; final test evaluation was skipped.", "training")
            save_job(job)
            return
        if not best.is_file():
            raise RuntimeError("Training ended without producing weights/best.pt")
        job.update(status="evaluating", evaluatingTest=True, updatedAt=utc_now(), message="Training finished. Evaluating the held-out test split once.", checkpoint=str(best))
        save_job(job)
        test_model = YOLO(str(best))
        test_model.add_callback("on_val_start", on_val_start)
        test_model.add_callback("on_val_batch_end", on_val_batch_end)
        test_model.add_callback("on_val_end", on_val_end)
        test_result = test_model.val(data=str(yaml_path), split="test", imgsz=request.imageSize, batch=request.batch, device="cpu", workers=0, plots=True, verbose=False)
        test_metrics = json_safe({str(key): float(value) for key, value in getattr(test_result, "results_dict", {}).items() if isinstance(value, (int, float))})
        completed_at = utc_now()
        metadata = job_path(job_id)
        epoch_rows = csv_epoch_metrics(results_csv_path(metadata, job))
        if epoch_rows:
            job["epochMetrics"] = merge_epoch_metrics(job, epoch_rows, completed_at)
        job.update(status="completed", evaluatingTest=False, currentStage="complete", updatedAt=completed_at, completedAt=completed_at, finishedAt=completed_at, elapsedSeconds=job.get("elapsedSeconds") or int((datetime.now(timezone.utc) - datetime.fromisoformat(job["startedAt"])).total_seconds()), message="Training and final test evaluation completed. The best checkpoint is ready for detection.", checkpoint=str(best), epoch=request.epochs, testMetrics=test_metrics)
        append_job_event(job, "info", "Training and final test evaluation completed.", "training")
        save_job(job)
    except Exception as error:
        finished_at = utc_now()
        elapsed = job.get("elapsedSeconds")
        if job.get("startedAt"):
            try:
                elapsed = int((datetime.now(timezone.utc) - datetime.fromisoformat(str(job["startedAt"]).replace("Z", "+00:00"))).total_seconds())
            except ValueError:
                pass
        job.update(status="failed", updatedAt=finished_at, finishedAt=finished_at, elapsedSeconds=elapsed, message=str(error)[:1200], failureReason=traceback.format_exc()[-8000:])
        append_job_event(job, "error", str(error), "training")
        save_job(job)
    finally:
        with _lock:
            _active_training = None


@app.get("/api/health")
def health():
    return {"status": "ok", "project": "AI-VIsion-demo", "readOnly": OBSERVABILITY_ONLY}


@app.get("/api/worker/status")
def worker_status():
    claims = {}
    for pipeline in ("yolo", "classifier"):
        path = job_queue.claim_path(RUNS_DIR, pipeline)
        try:
            claims[pipeline] = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            claims[pipeline] = None
        except (OSError, ValueError) as error:
            claims[pipeline] = {"error": str(error)}
    workers = []
    registry_dir = RUNS_DIR / "worker" / "registry"
    if registry_dir.is_dir():
        for path in registry_dir.glob("*.json"):
            try:
                worker = json.loads(path.read_text(encoding="utf-8"))
                heartbeat = datetime.fromisoformat(str(worker.get("heartbeatAt", "")).replace("Z", "+00:00"))
                age = max(0.0, (datetime.now(timezone.utc) - heartbeat).total_seconds())
                workers.append({**worker, "heartbeatAgeSeconds": round(age, 1), "healthy": age < 60})
            except (OSError, ValueError, TypeError):
                continue
    return {"status": "ok", "workers": workers, "workerClaims": claims}


@app.get("/api/dataset")
def dataset_info():
    images = list_images()
    labels = list((DATASET_ROOT / "labels").glob("**/*.txt")) if (DATASET_ROOT / "labels").is_dir() else []
    return {"root": str(DATASET_ROOT), "imageCount": len(images), "labelCount": len(labels), "readOnly": True, "classes": ["container"], "imageSizeBytes": sum(path.stat().st_size for path in images)}


@app.get("/api/datasets/summary")
def datasets_summary():
    db_path = dataset_catalog_path()
    if not OBSERVABILITY_ONLY:
        catalog = sync_dataset_catalog()
    elif db_path.is_file():
        read_db=open_dataset_catalog(read_only=True)
        try: catalog={"uniqueOriginalImages":0,"duplicatePaths":0,"pathIds":{row["source_path"]:row["image_id"] for row in read_db.execute("SELECT source_path,image_id FROM image_sources")}}
        finally: read_db.close()
    else:
        return {"catalogAvailable": False, "message": "Dataset catalog will be created by the main training service."}
    db = open_dataset_catalog(read_only=OBSERVABILITY_ONLY)
    try:
        migrate_legacy_training_records(db)
        unique_images = db.execute("SELECT COUNT(DISTINCT image_id) FROM image_sources WHERE active=1").fetchone()[0]
        duplicate_paths = db.execute("SELECT COUNT(*) FROM image_sources s WHERE active=1 AND EXISTS (SELECT 1 FROM image_sources other WHERE other.image_id=s.image_id AND other.active=1 AND other.source_id<s.source_id)").fetchone()[0]
        split_by_group = {row["group_key"]: row["split"] for row in db.execute("SELECT group_key,split FROM split_assignments")}
        detector = {"trainingSamples": 0, "train": 0, "validation": 0, "test": 0, "excluded": 0, "derivedSamples": 0}
        classifier = {"trainingSamples": 0, "train": 0, "validation": 0, "test": 0, "excluded": 0, "derivedSamples": 0}
        verified_counts = server_container_count_labels()
        seen_ids=set()
        image_root=DATASET_ROOT/"images"
        if not image_root.is_dir(): image_root=DATASET_ROOT
        labels_root=DATASET_ROOT/"labels"
        for image in (p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED):
            relative=image.relative_to(DATASET_ROOT).as_posix(); image_id=(catalog.get("pathIds") or {}).get(relative)
            if not image_id or image_id in seen_ids: continue
            seen_ids.add(image_id); group=image.stem.removesuffix("_cropped").casefold(); split=split_by_group.get(group,"train")
            label=label_for(image,image_root,labels_root)
            try: box_count=len(validated_boxes(label)) if label.is_file() else None
            except (OSError,UnicodeError,ValueError): box_count=None
            if box_count is None: detector["excluded"]+=1
            else: detector["trainingSamples"]+=1; detector[split]+=1
            count_label=verified_counts["labels"].get(relative)
            if count_label in {1,2}: classifier["trainingSamples"]+=1; classifier[split]+=1
            else: classifier["excluded"]+=1
        split_rows = {row["split"]: row["n"] for row in db.execute("SELECT split,COUNT(*) n FROM split_assignments GROUP BY split")}
        latest_versions = {}
        for row in db.execute("SELECT model_key,dataset_id,version,train_count,validation_count,test_count,excluded_count,derived_count,created_at,dataset_version_id FROM dataset_versions ORDER BY version DESC"):
            latest_versions.setdefault(row["model_key"], dict(row))
        legacy_crop_count=0
        for row in classifier_dataset_records():
            if row.get("sourceType") != "server": legacy_crop_count+=int(row.get("imageCount",0) or 0)
        classifier["derivedSamples"]=0
        return {"catalogAvailable": True, "sharedOriginalImages": unique_images, "duplicateImagePaths": duplicate_paths,
                "persistentGroups": sum(split_rows.values()), "persistentSplits": split_rows,
                "models": {"containerDetector": detector, "containerCountClassifier": classifier},
                "modelDatasets": latest_versions,
                "legacyClassifierDatasets": sum(1 for row in classifier_dataset_records() if row.get("sourceType") != "server"),
                "manualReconciliation": sum(1 for row in classifier_dataset_records() if row.get("sourceType") != "server"),
                "legacyImportedSamples": legacy_crop_count,
                "datasetDatabase": "runs/datasets/catalog.sqlite3"}
    finally:
        db.close()


@app.get("/api/dataset/images")
def dataset_images(page: int = 1, pageSize: int = 30):
    page, page_size = max(1, page), max(1, min(100, pageSize))
    images = list_images()
    return {"page": page, "pageSize": page_size, "total": len(images), "items": [path.relative_to(DATASET_ROOT).as_posix() for path in images[(page - 1) * page_size:page * page_size]]}


@app.get("/api/dataset/file")
def dataset_file(path: str):
    return FileResponse(safe_dataset_path(path))


def dataset_explorer_page(page: int, page_size: int) -> tuple[int, int]:
    return max(1, page), max(1, min(100, page_size))


@app.get("/api/dataset/explorer/detection")
def dataset_explorer_detection(page: int = 1, pageSize: int = 50, search: str = "", annotation: str = "all", nameDirection: str = "all"):
    page, page_size = dataset_explorer_page(page, pageSize)
    if annotation not in {"all", "annotated", "unannotated"}:
        raise HTTPException(400, "Invalid annotation filter")
    if nameDirection not in {"all", "left", "right", "top"}:
        raise HTTPException(400, "Invalid filename direction filter")
    image_root = DATASET_ROOT / "images"
    if not image_root.is_dir():
        image_root = DATASET_ROOT
    labels_root = DATASET_ROOT / "labels"
    records = []
    total_boxes = annotated_images = total_images = 0
    images = sorted((p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED), key=lambda p: p.relative_to(image_root).as_posix().casefold())
    query = search.casefold().strip()
    for image in images:
        total_images += 1
        relative = image.relative_to(DATASET_ROOT).as_posix()
        label = label_for(image, image_root, labels_root)
        error = None
        try:
            boxes = validated_boxes(label)
        except (OSError, UnicodeError, ValueError) as exc:
            boxes, error = [], str(exc)
        count = len(boxes)
        total_boxes += count
        annotated_images += count > 0
        status = "invalid" if error else "annotated" if count else "unannotated"
        if query and query not in image.name.casefold():
            continue
        if nameDirection != "all" and nameDirection not in image.name.casefold():
            continue
        if annotation != "all" and status != annotation:
            continue
        records.append({"path": relative, "filename": image.name, "annotationStatus": status, "boxCount": count, "boxes": boxes, "annotationError": error})
    total = len(records)
    start = (page - 1) * page_size
    return {"page": page, "pageSize": page_size, "total": total, "stats": {"totalImages": total_images, "annotatedImages": annotated_images, "unannotatedImages": max(0, total_images - annotated_images), "totalBoxes": total_boxes}, "items": records[start:start + page_size]}


def server_container_count_labels(required: bool = False) -> dict[str, Any]:
    """Read image-level count labels from Container-labeler's verified dataset."""
    if not LABELER_DB.is_file():
        if required:
            raise HTTPException(400, "Container-labeler database is unavailable; mount the verified count labels to train the server classifier.")
        return {"labels": {}, "conflicts": set(), "available": False}
    try:
        connection = sqlite3.connect(f"file:{LABELER_DB}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        rows = connection.execute("""SELECT a.source_path,c.container_count
            FROM text_sessions t
            JOIN text_annotations a ON a.session_id=t.id
            JOIN container_counts c ON c.session_id=a.session_id AND c.path=a.path
            WHERE t.root_name='container_images_verified' AND a.active=1
              AND a.source_path IS NOT NULL AND c.container_count IN (1,2)""").fetchall()
        direct_rows = []
        if connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_container_counts'").fetchone():
            direct_rows = connection.execute("SELECT source_path,container_count FROM source_container_counts").fetchall()
        connection.close()
    except sqlite3.Error as error:
        if required:
            raise HTTPException(400, f"Could not read verified container-count labels: {error}") from error
        return {"labels": {}, "conflicts": set(), "available": False}
    values: dict[str, set[int]] = {}
    for row in rows:
        source_path = Path(str(row["source_path"]).replace("\\", "/"))
        if source_path.is_absolute() or ".." in source_path.parts:
            continue
        relative = source_path.as_posix()
        values.setdefault(relative, set()).add(int(row["container_count"]))
    conflicts = {path for path, labels in values.items() if len(labels) > 1}
    labels = {path: next(iter(found)) for path, found in values.items() if len(found) == 1}
    for row in direct_rows:
        source_path = Path(str(row["source_path"]).replace("\\", "/"))
        if source_path.is_absolute() or ".." in source_path.parts or int(row["container_count"]) not in {1, 2}:
            continue
        relative = source_path.as_posix()
        labels[relative] = int(row["container_count"])
        conflicts.discard(relative)
    return {"labels": labels, "conflicts": conflicts, "available": True}


def prepare_server_classifier_dataset() -> dict[str, Any]:
    image_root, labels_root = DATASET_ROOT / "images", DATASET_ROOT / "labels"
    if not image_root.is_dir() or not labels_root.is_dir():
        raise HTTPException(400, "The server dataset does not contain images/ and labels/ folders.")
    catalog = sync_dataset_catalog()
    images = sorted((path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED), key=lambda path: path.relative_to(image_root).as_posix().casefold())
    verified = server_container_count_labels(required=True)
    verified_labels = verified["labels"]
    content_fingerprint = hashlib.sha256()
    eligible: list[tuple[Path, int]] = []
    conflicted = 0
    for image in images:
        relative = image.relative_to(DATASET_ROOT).as_posix()
        count_label = verified_labels.get(relative)
        content_fingerprint.update(relative.encode())
        content_fingerprint.update(catalog["hashes"].get(relative, "").encode())
        if relative in verified["conflicts"]:
            conflicted += 1
            content_fingerprint.update(b"conflict")
            continue
        if count_label not in {1, 2}:
            content_fingerprint.update(b"unlabeled")
            continue
        content_fingerprint.update(str(count_label).encode())
        eligible.append((image, count_label))
    source_fingerprint = content_fingerprint.hexdigest()
    dataset_id = hashlib.sha256(f"{DATASET_ROOT}\0server-verified-count-v3\0{source_fingerprint}".encode()).hexdigest()[:32]
    target = classifier_root() / "datasets" / dataset_id
    metadata_path = target / "import.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("sourceType") == "server" and all((target / name).is_dir() for name in ("1_container", "2_containers")):
                return metadata
        except (OSError, ValueError, TypeError):
            pass
    if target.exists():
        raise ValueError(f"Classifier dataset directory already exists but is incomplete; preserving it for manual inspection: {target}")
    groups: dict[str, str] = {}
    counts = {"1_container": 0, "2_containers": 0}
    unique_labels: dict[str, int] = {}
    for image, box_count in eligible:
        relative = image.relative_to(image_root).as_posix()
        source_id=catalog["pathIds"].get(image.relative_to(DATASET_ROOT).as_posix())
        if source_id in unique_labels:
            if unique_labels[source_id] != box_count:
                raise HTTPException(409,f"Exact duplicate images have conflicting container-count labels; reconcile before training: {image.name}")
            continue
        if source_id: unique_labels[source_id]=box_count
        label_name = "1_container" if box_count == 1 else "2_containers"
        filename = f"{hashlib.sha256(relative.encode()).hexdigest()[:16]}{image.suffix.lower()}"
        output = target / label_name / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        output.symlink_to(image.resolve())
        groups[filename] = image.stem.removesuffix("_cropped").casefold()
        counts[label_name] += 1
    total = sum(counts.values())
    if not total:
        raise HTTPException(400, "The server dataset has no valid images labeled with exactly one or two containers.")
    metadata = {"datasetId": dataset_id, "path": str(target), "imageCount": total, "classCounts": counts, "invalid": [], "source": "Container-labeler verified image counts", "sourceType": "server", "sourceFingerprint": source_fingerprint, "labelSource": "container_images_verified", "unlabeledCount": max(0, len(images) - sum(counts.values()) - conflicted), "conflictedCount": conflicted}
    (target / "groups.json").write_text(json.dumps(groups), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def classifier_dataset_records() -> list[dict[str, Any]]:
    root = classifier_root() / "datasets"
    records = []
    if not root.is_dir():
        return records
    for metadata in root.glob("*/import.json"):
        try:
            info = json.loads(metadata.read_text(encoding="utf-8"))
            dataset_id = metadata.parent.name
            if not re.fullmatch(r"[a-f0-9]{32}", dataset_id):
                continue
            classes = {"1_container": 0, "2_containers": 0, "unclassified": 0}
            for folder, key in (("1_container", "1_container"), ("2_containers", "2_containers"), ("unclassified", "unclassified"), ("Unclassified", "unclassified")):
                directory = metadata.parent / folder
                if directory.is_dir():
                    classes[key] += sum(1 for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)
            known = classes["1_container"] + classes["2_containers"]
            total = known + classes["unclassified"]
            records.append({"id": dataset_id, "name": info.get("sessionName") or info.get("source") or f"Classifier dataset · {dataset_id[:8]}", "source": info.get("source", "Imported classifier dataset"), "sourceType": info.get("sourceType"), "labelSource": info.get("labelSource"), "imageCount": total, "classCounts": classes, "createdAt": metadata.stat().st_mtime})
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return sorted(records, key=lambda row: (row["createdAt"], row["id"]), reverse=True)


@app.get("/api/dataset/explorer/classification")
def dataset_explorer_classification(datasetId: str | None = None, page: int = 1, pageSize: int = 50, search: str = "", label: str = "all", nameDirection: str = "all"):
    page, page_size = dataset_explorer_page(page, pageSize)
    if label not in {"all", "1", "2", "unclassified"}:
        raise HTTPException(400, "Invalid classification filter")
    if nameDirection not in {"all", "left", "right", "top"}:
        raise HTTPException(400, "Invalid filename direction filter")
    verified = server_container_count_labels()
    image_root = DATASET_ROOT / "images"
    if not image_root.is_dir():
        image_root = DATASET_ROOT
    counts = {
        "1_container": sum(1 for value in verified["labels"].values() if value == 1),
        "2_containers": sum(1 for value in verified["labels"].values() if value == 2),
        "unclassified": 0,
    }
    items: list[dict[str, Any]] = []
    query = search.casefold().strip()
    for relative, count in verified["labels"].items():
        if count not in {1, 2}:
            continue
        image = (DATASET_ROOT / relative).resolve()
        if not image.is_relative_to(DATASET_ROOT) or not image.is_file() or image.suffix.lower() not in SUPPORTED:
            continue
        class_label = str(count)
        if label not in {"all", class_label} or (query and query not in image.name.casefold()):
            continue
        if nameDirection != "all" and nameDirection not in image.name.casefold():
            continue
        items.append({"path": relative, "filename": image.name, "label": class_label, "labelText": f"{count} container" if count == 1 else "2 containers"})
    items.sort(key=lambda row: row["filename"].casefold())
    total_images = counts["1_container"] + counts["2_containers"]
    selected = {"id": "server-verified", "name": "Verified server image counts", "source": "container_images_verified", "imageCount": total_images, "classCounts": counts}
    return {"datasets": [selected], "dataset": selected, "page": page, "pageSize": page_size, "total": len(items), "stats": {"totalImages": total_images, "oneContainer": counts["1_container"], "twoContainers": counts["2_containers"], "unclassified": 0, "onePercent": round(counts["1_container"] * 100 / total_images, 1) if total_images else 0, "twoPercent": round(counts["2_containers"] * 100 / total_images, 1) if total_images else 0}, "items": items[(page - 1) * page_size:page * page_size]}


@app.get("/api/dataset/explorer/classification/file")
def dataset_explorer_classification_file(datasetId: str, path: str):
    if datasetId == "server-verified":
        return FileResponse(safe_dataset_path(path))
    if not re.fullmatch(r"[a-f0-9]{32}", datasetId):
        raise HTTPException(404, "Classifier dataset not found")
    root = (classifier_root() / "datasets" / datasetId).resolve()
    candidate = (root / path).resolve()
    if not candidate.is_file() or not (candidate.is_relative_to(root) or candidate.is_relative_to(DATASET_ROOT)) or candidate.suffix.lower() not in SUPPORTED:
        raise HTTPException(404, "Classifier image not found")
    return FileResponse(candidate)


@app.get("/api/models")
def models():
    references={}
    if not OBSERVABILITY_ONLY:
        catalog_db=dataset_catalog.connect(dataset_catalog_path())
        try:
            migrate_legacy_training_records(catalog_db)
            references={row["run_id"]:row["dataset_version_id"] for row in catalog_db.execute("SELECT run_id,dataset_version_id FROM training_runs")}
        finally: catalog_db.close()
    detector_runs=training_runs(); classifier_rows=classifier_runs()
    for row in detector_runs+classifier_rows:
        version=references.get(row.get("id"))
        if version:
            row["datasetVersionId"]=version
            row["legacyDatasetVersion"]=str(version).startswith("legacy-")
    return {"models": checkpoint_models(), "checkpoints": list_checkpoints(), "trainingRuns": detector_runs, "classifiers": classifier_models(), "classifierRuns": classifier_rows}


@app.get("/api/datasets/versions")
def list_dataset_versions(modelKey: str | None = None):
    if not dataset_catalog_path().is_file(): return {"versions": []}
    db=open_dataset_catalog(read_only=OBSERVABILITY_ONLY)
    try:
        if modelKey:
            rows=db.execute("SELECT * FROM dataset_versions WHERE model_key=? ORDER BY created_at DESC",(modelKey,)).fetchall()
        else:
            rows=db.execute("SELECT * FROM dataset_versions ORDER BY created_at DESC").fetchall()
        versions=[]
        for row in rows:
            item=dict(row); item["annotationVersions"]=json.loads(item.pop("annotation_versions_json")); item["manifest"]=json.loads(item.pop("manifest_json")); versions.append(item)
        return {"versions":versions}
    finally: db.close()


def classifier_root() -> Path:
    return RUNS_DIR / "classifiers"


def classifier_runs() -> list[dict[str, Any]]:
    result = []
    for meta in (classifier_root() / "jobs").glob("*/job.json"):
        try:
            job = json.loads(meta.read_text(encoding="utf-8"))
            for key in ("best.pt", "last.pt"):
                job[key.replace(".pt", "Path")] = str(meta.parent / key) if (meta.parent / key).is_file() else None
            job["resumePath"] = str(meta.parent / "resume_state.pt") if (meta.parent / "resume_state.pt").is_file() else None
            result.append(job)
        except (OSError, ValueError):
            continue
    return sorted(result, key=lambda row: row.get("createdAt", ""), reverse=True)


def classifier_models() -> list[dict[str, Any]]:
    return [{"id": row["id"], "name": f"{row.get('architecture','mobilenet_v3_small')} · {row['id'][:8]}", "createdAt": row.get("updatedAt"), "accuracy": row.get("bestAccuracy")} for row in classifier_runs() if row.get("status") == "completed" and row.get("bestPath") and Path(row["bestPath"]).is_file()]


def save_classifier_job(job: dict[str, Any]) -> None:
    directory = classifier_root() / "jobs" / job["id"]
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "job.json"
    with _lock:
        if target.is_file():
            try:
                if json.loads(target.read_text(encoding="utf-8")).get("stopRequested"):
                    job["stopRequested"] = True
            except (OSError, ValueError):
                pass
        job_queue.atomic_json(target, job)


def classifier_stop_requested(run_id: str) -> bool:
    path = classifier_root() / "jobs" / run_id / "job.json"
    if job_queue.cancel_requested(path.parent):
        return True
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("stopRequested"))
    except (OSError, ValueError, TypeError):
        return False


CLASSIFIER_CLASSES = ("1_container", "2_containers")
CLASSIFIER_TARGETS = {"70:30": (0.7, 0.3), "60:40": (0.6, 0.4), "50:50": (0.5, 0.5)}
CLASSIFIER_YOLO_EXPERIMENT = "container-count-yolo-crops-0cb261aa-v1"
CLASSIFIER_YOLO_SYNC_PREFIX = "container-count-yolo-crops-sync-"


def yolo_classifier_dataset_root(dataset_id: str | None = None) -> Path:
    if not dataset_id or dataset_id == CLASSIFIER_YOLO_EXPERIMENT:
        return ROOT.parent / "experiments" / CLASSIFIER_YOLO_EXPERIMENT
    if not re.fullmatch(rf"{re.escape(CLASSIFIER_YOLO_SYNC_PREFIX)}[a-f0-9]{{16}}", dataset_id):
        raise HTTPException(400, "Invalid synchronized YOLO crop dataset ID.")
    return classifier_root() / "yolo_crop_datasets" / dataset_id


def _source_dataset_relative_path(value: str) -> str | None:
    normalized = str(value or "").replace("\\", "/")
    path = Path(normalized)
    parts = path.parts
    if path.is_absolute():
        if len(parts) < 3 or parts[1] != "dataset":
            return None
        parts = parts[2:]
    elif parts and parts[0] == "dataset":
        parts = parts[1:]
    relative = Path(*parts)
    if not parts or relative.is_absolute() or ".." in relative.parts:
        return None
    return relative.as_posix()


def synchronize_yolo_crop_labels() -> dict[str, Any]:
    """Create an immutable crop dataset version using current verified image labels."""
    source_info = production_matched_classifier_dataset(validate_image_contents=False)
    source_root = Path(source_info["datasetRoot"])
    verified = server_container_count_labels(required=True)
    manifest_path = Path(source_info["manifestPath"])
    manifest_bytes = manifest_path.read_bytes()
    label_by_source: dict[str, int] = {}
    for source_path, count in verified["labels"].items():
        relative = _source_dataset_relative_path(source_path)
        if relative and count in {1, 2}:
            label_by_source[relative] = count

    rows: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    label_hash = hashlib.sha256()
    excluded_unlabeled = excluded_conflicts = 0
    split_counts = {split: {name: 0 for name in CLASSIFIER_CLASSES} for split in ("train", "val", "test")}
    for line_number, line in enumerate(manifest_bytes.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        relative_source = _source_dataset_relative_path(row.get("source_path", ""))
        if not relative_source:
            raise HTTPException(400, f"YOLO crop manifest row {line_number} has an unsupported source path.")
        if relative_source in seen_sources:
            raise HTTPException(409, f"YOLO crop manifest contains multiple crops for source image {relative_source}; refusing ambiguous relabeling.")
        seen_sources.add(relative_source)
        source_image = (DATASET_ROOT / relative_source).resolve()
        if not source_image.is_relative_to(DATASET_ROOT) or not source_image.is_file():
            raise HTTPException(409, f"YOLO crop source image is missing from the verified dataset: {relative_source}.")
        if relative_source in verified["conflicts"]:
            excluded_conflicts += 1
            label_hash.update(f"{relative_source}\0conflict\n".encode())
            continue
        count = label_by_source.get(relative_source)
        if count not in {1, 2}:
            excluded_unlabeled += 1
            label_hash.update(f"{relative_source}\0unlabeled\n".encode())
            continue
        split = "val" if row.get("split") == "validation" else row.get("split")
        if split not in split_counts:
            raise HTTPException(400, f"YOLO crop manifest row {line_number} has an invalid split.")
        label = f"{count}_container" if count == 1 else "2_containers"
        old_crop = Path(str(row.get("crop_path", "")))
        if old_crop.is_absolute() or ".." in old_crop.parts:
            raise HTTPException(400, f"YOLO crop manifest row {line_number} has an unsafe crop path.")
        crop_source = (source_root / old_crop).resolve()
        if not crop_source.is_relative_to(source_root.resolve()) or not crop_source.is_file():
            raise HTTPException(409, f"YOLO crop image is missing: {old_crop.as_posix()}.")
        row["original_label"] = row.get("label")
        row["_source_crop_path"] = old_crop.as_posix()
        row["label"] = label
        row["label_source"] = "container_images_verified"
        row["source_label"] = count
        row["crop_path"] = f"{split}/{label}/{Path(row['sample_id']).name}"
        rows.append(row)
        split_counts[split][label] += 1
        label_hash.update(f"{relative_source}\0{count}\n".encode())

    if not rows:
        raise HTTPException(409, "No YOLO crops match current verified Container-labeler labels.")
    for split in split_counts:
        if any(split_counts[split][name] == 0 for name in CLASSIFIER_CLASSES):
            raise HTTPException(409, f"Current verified labels leave a class empty in the {split} split; refusing to publish an unusable dataset.")

    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    source_fingerprint = label_hash.hexdigest()
    version_hash = hashlib.sha256(f"{manifest_sha}\0{source_fingerprint}".encode()).hexdigest()[:16]
    dataset_id = f"{CLASSIFIER_YOLO_SYNC_PREFIX}{version_hash}"
    destination = yolo_classifier_dataset_root(dataset_id)
    if (destination / "metadata.json").is_file():
        try:
            return production_matched_classifier_dataset(validate_image_contents=False, dataset_id=dataset_id)
        except HTTPException:
            raise

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{dataset_id}-", dir=destination.parent))
    try:
        for split in split_counts:
            for label in CLASSIFIER_CLASSES:
                (staging / split / label).mkdir(parents=True, exist_ok=True)
        output_rows = []
        for row in rows:
            label = row["label"]
            relative_source = _source_dataset_relative_path(row["source_path"])
            source_crop = source_root / row.pop("_source_crop_path")
            if not source_crop.is_file():
                raise HTTPException(409, f"Original crop is missing for {relative_source}.")
            target = staging / row["crop_path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source_crop.resolve())
            output_rows.append(row)
        (staging / "manifest.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows), encoding="utf-8")
        output_counts = {"train": split_counts["train"], "validation": split_counts["val"], "test": split_counts["test"]}
        metadata = {
            "experiment": CLASSIFIER_YOLO_EXPERIMENT, "dataset_id": dataset_id, "status": "complete",
            "created_at": utc_now(), "label_source": "container_images_verified",
            "source_experiment": CLASSIFIER_YOLO_EXPERIMENT, "source_manifest_sha256": manifest_sha,
            "source_label_fingerprint": source_fingerprint, "source_counts": {
                "1_container": sum(split_counts[s]["1_container"] for s in split_counts),
                "2_containers": sum(split_counts[s]["2_containers"] for s in split_counts)},
            "dataset_size": len(output_rows), "output_counts": output_counts,
            "excluded_unlabeled_sources": excluded_unlabeled,
            "excluded_conflicted_sources": excluded_conflicts,
            "split_assignments_preserved": True, "crop_files_linked_to": str(source_root),
        }
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        try:
            os.rename(staging, destination)
        except FileExistsError:
            shutil.rmtree(staging)
        return production_matched_classifier_dataset(validate_image_contents=False, dataset_id=dataset_id)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def atomic_torch_save(payload: Any, destination: Path) -> None:
    from .classifier_checkpoints import save_model_checkpoint
    save_model_checkpoint(payload, destination)


def production_matched_classifier_dataset(validate_image_contents: bool = True, dataset_id: str | None = None) -> dict[str, Any]:
    """Validate and describe the immutable, pre-generated YOLO crop dataset."""
    root = yolo_classifier_dataset_root(dataset_id)
    metadata_path = root / "metadata.json"
    manifest_path = root / "manifest.jsonl"
    if not metadata_path.is_file() or not manifest_path.is_file():
        raise HTTPException(400, f"Production-matched YOLO dataset metadata or manifest is missing under {root}.")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise HTTPException(400, f"Could not read YOLO dataset metadata: {error}") from error
    if metadata.get("experiment") != CLASSIFIER_YOLO_EXPERIMENT or metadata.get("status") != "complete":
        raise HTTPException(400, "The production-matched YOLO dataset metadata does not identify a completed expected experiment.")
    try:
        import json as json_module
        from PIL import Image
        linked_crop_root = None
        if metadata.get("crop_files_linked_to"):
            linked_crop_root = Path(metadata["crop_files_linked_to"]).resolve()
            expected_crop_root = (ROOT.parent / "experiments" / CLASSIFIER_YOLO_EXPERIMENT).resolve()
            if linked_crop_root != expected_crop_root:
                raise ValueError("synchronized dataset links do not target the immutable source crop dataset")
        observed: dict[str, dict[str, int]] = {split: {name: 0 for name in CLASSIFIER_CLASSES} for split in ("train", "val", "test")}
        manifest_rows: dict[str, dict[str, int]] = {split: {name: 0 for name in CLASSIFIER_CLASSES} for split in ("train", "val", "test")}
        manifest_paths: dict[str, dict[str, set[str]]] = {split: {name: set() for name in CLASSIFIER_CLASSES} for split in ("train", "val", "test")}
        sample_ids: set[str] = set()
        source_group_splits: dict[str, str] = {}
        with manifest_path.open(encoding="utf-8") as manifest_file:
            for line_number, line in enumerate(manifest_file, start=1):
                if not line.strip():
                    continue
                row = json_module.loads(line)
                raw_split, label = row.get("split"), row.get("label")
                split = "val" if raw_split == "validation" else raw_split
                if split not in manifest_rows or label not in CLASSIFIER_CLASSES:
                    raise ValueError(f"manifest line {line_number} has an invalid split or class")
                sample_id = row.get("sample_id")
                crop_path = str(row.get("crop_path", "")).replace("\\", "/")
                if not sample_id or sample_id in sample_ids:
                    raise ValueError(f"manifest line {line_number} has a missing or duplicate sample_id")
                sample_ids.add(sample_id)
                relative_crop = Path(crop_path)
                if relative_crop.is_absolute() or ".." in relative_crop.parts or relative_crop.parts[:2] != (split, label):
                    raise ValueError(f"manifest line {line_number} has an unsafe or mismatched crop_path")
                manifest_paths[split][label].add(relative_crop.as_posix())
                if row.get("status") != "cropped":
                    raise ValueError(f"manifest line {line_number} is not a successfully generated crop")
                source_group = row.get("source_group")
                if source_group:
                    previous_split = source_group_splits.setdefault(str(source_group), split)
                    if previous_split != split:
                        raise ValueError(f"source group {source_group!r} crosses {previous_split}/{split} splits")
                manifest_rows[split][label] += 1
        for split in ("train", "val", "test"):
            split_root = root / split
            if not split_root.is_dir():
                raise ValueError(f"missing {split} directory: {split_root}")
            folder_classes = sorted(path.name for path in split_root.iterdir() if path.is_dir())
            if folder_classes != sorted(CLASSIFIER_CLASSES):
                raise ValueError(f"{split} must contain exactly the class folders {', '.join(CLASSIFIER_CLASSES)}")
            for label in CLASSIFIER_CLASSES:
                folder = split_root / label
                files = sorted(path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED)
                if not files:
                    raise ValueError(f"{split}/{label} contains no supported images")
                actual_paths = {path.relative_to(root).as_posix() for path in files}
                if actual_paths != manifest_paths[split][label]:
                    raise ValueError(f"{split}/{label} image paths do not exactly match manifest")
                for image_path in files:
                    resolved_image = image_path.resolve()
                    if not resolved_image.is_relative_to(root.resolve()) and not (linked_crop_root and resolved_image.is_relative_to(linked_crop_root)):
                        raise ValueError(f"image path escapes the dataset root: {image_path}")
                    if validate_image_contents:
                        with Image.open(image_path) as image:
                            image.verify()
                observed[split][label] = len(files)
        if observed != manifest_rows:
            raise ValueError(f"image counts do not match manifest (folders={observed}, manifest={manifest_rows})")
        expected = metadata.get("output_counts", {})
        if expected.get("val") is None and expected.get("validation") is None:
            raise ValueError("dataset metadata is missing validation output counts")
        for split, metadata_split in (("train", "train"), ("val", "val" if expected.get("val") is not None else "validation"), ("test", "test")):
            if observed[split] != expected.get(metadata_split):
                raise ValueError(f"{split} folder counts do not match dataset metadata")
        class_totals = {name: sum(observed[split][name] for split in ("train", "val", "test")) for name in CLASSIFIER_CLASSES}
        return {"datasetRoot": str(root), "experimentId": metadata.get("dataset_id", CLASSIFIER_YOLO_EXPERIMENT),
                "manifestPath": str(manifest_path), "metadataPath": str(metadata_path),
                "splitCounts": observed, "classToIndex": {name: index for index, name in enumerate(CLASSIFIER_CLASSES)},
                "classCounts": class_totals, "imageCount": sum(class_totals.values()), "metadata": metadata}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(400, f"Invalid production-matched YOLO dataset: {error}") from error


def load_yolo_classifier_splits(root: Path, image_folder: Any, train_transform: Any, eval_transform: Any) -> dict[str, Any]:
    """Load the manifest-preserved split directories without re-splitting."""
    split_roots = {"train": root / "train", "validation": root / "val", "test": root / "test"}
    datasets_by_split = {
        "train": image_folder(split_roots["train"], transform=train_transform),
        "validation": image_folder(split_roots["validation"], transform=eval_transform),
        "test": image_folder(split_roots["test"], transform=eval_transform),
    }
    expected = list(CLASSIFIER_CLASSES)
    for split, dataset in datasets_by_split.items():
        if dataset.classes != expected:
            raise ValueError(f"{split} class mapping differs from {expected}")
    return {"datasets": datasets_by_split, "paths": split_roots, "classToIndex": {name: index for index, name in enumerate(expected)}}


def classifier_distribution(labels: list[int], indices: list[int] | None = None) -> dict[str, Any]:
    selected = labels if indices is None else [labels[index] for index in indices]
    counts = [selected.count(class_index) for class_index in range(2)]
    total = len(selected)
    percentages = [round(count * 100 / total, 2) if total else 0.0 for count in counts]
    low, high = sorted(counts)
    return {
        "counts": dict(zip(CLASSIFIER_CLASSES, counts)),
        "percentages": dict(zip(CLASSIFIER_CLASSES, percentages)),
        "total": total,
        "imbalanceRatio": round(high / low, 3) if low else (None if high else 1.0),
    }


def classifier_sampling_weights(labels: list[int], indices: list[int], target: tuple[float, float]) -> list[float]:
    counts = [sum(labels[index] == class_index for index in indices) for class_index in range(2)]
    if not indices or min(counts) == 0:
        raise ValueError("Both classes must be present in the training split for balanced sampling.")
    weights_by_class = [target[class_index] / counts[class_index] for class_index in range(2)]
    return [weights_by_class[labels[index]] for index in indices]


def classifier_metrics(actual: list[int], predicted: list[int]) -> dict[str, Any]:
    matrix = [[sum(a == row and p == col for a, p in zip(actual, predicted)) for col in range(2)] for row in range(2)]
    per_class = {}
    for class_index, name in enumerate(CLASSIFIER_CLASSES):
        tp = matrix[class_index][class_index]
        fp = sum(matrix[row][class_index] for row in range(2) if row != class_index)
        fn = sum(matrix[class_index][col] for col in range(2) if col != class_index)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(matrix[class_index])}
    minority = per_class["2_containers"]
    return {
        "accuracy": sum(a == p for a, p in zip(actual, predicted)) / len(actual) if actual else 0.0,
        "perClass": per_class,
        "precision": minority["precision"], "recall": minority["recall"], "f1": minority["f1"],
        "confusionMatrix": matrix,
    }


def classifier_checkpoint_state(payload: Any, architecture: str, classes: list[str]) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("architecture") != architecture:
        raise ValueError("The selected classifier checkpoint does not match this architecture.")
    if payload.get("classes") != classes or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("The selected classifier checkpoint has incompatible classes or weights.")
    return payload["state_dict"]


def classifier_worker(run_id: str) -> None:
    global _active_classifier
    from .classifier_checkpoints import build_resume_state, load_resume_state, restore_rng_states
    root = classifier_root() / "jobs" / run_id
    job = json.loads((root / "job.json").read_text(encoding="utf-8"))
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
        from torchvision import datasets, models, transforms
        job.update(status="running", message="Preparing image classification data.", startedAt=utc_now()); save_classifier_job(job)
        if job.get("datasetSource") == "yolo_crops":
            yolo_info = production_matched_classifier_dataset(validate_image_contents=True, dataset_id=job.get("datasetId"))
            data_root = Path(yolo_info["datasetRoot"])
            if Path(job["datasetPath"]).resolve() != data_root.resolve():
                raise ValueError("Saved YOLO dataset path does not match the validated production-matched dataset root.")
            if job["imageSize"] != 224:
                raise ValueError("The production-matched YOLO experiment requires 224x224 classifier input.")
            train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
            eval_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
            loaded = load_yolo_classifier_splits(data_root, datasets.ImageFolder, train_transform, eval_transform)
            split_roots = loaded["paths"]
            train_dataset, val_dataset, test_dataset = (loaded["datasets"][name] for name in ("train", "validation", "test"))
            dataset = train_dataset
            train_set, val_set, test_set = train_dataset, val_dataset, test_dataset
            labels = list(train_dataset.targets)
            counts = yolo_info["classCounts"]
            n_train, n_val, n_test = len(train_dataset), len(val_dataset), len(test_dataset)
            if min(n_train, n_val, n_test) < 1:
                raise ValueError("Production-matched YOLO dataset must have non-empty train, validation, and test splits.")
            split_distributions = {
                "train": classifier_distribution(list(train_dataset.targets)),
                "validation": classifier_distribution(list(val_dataset.targets)),
                "test": classifier_distribution(list(test_dataset.targets)),
            }
            job.update(datasetPath=str(data_root), datasetPaths={"train": str(split_roots["train"]), "validation": str(split_roots["validation"]), "test": str(split_roots["test"])}, datasetExperimentId=yolo_info["experimentId"], manifestPath=yolo_info["manifestPath"], metadataPath=yolo_info["metadataPath"], datasetSize=yolo_info["imageCount"], classCounts=counts, splitCounts={"train": n_train, "validation": n_val, "test": n_test}, splitDistributions=split_distributions, classToIndex=yolo_info["classToIndex"], inputSize=224, transformConfig={"color": "PIL ImageFolder RGB", "resize": [224, 224], "toTensor": True, "normalization": {"mean": [.485, .456, .406], "std": [.229, .224, .225]}, "augmentation": "none"}, optimizer={"name": "AdamW", "learningRate": job["learningRate"]}, lossConfig={"name": "CrossEntropyLoss", "balancingStrategy": job.get("balancingStrategy", "none"), "targetDistribution": job.get("targetDistribution", "70:30")}, trainingSchedule={"epochs": job["epochs"], "batchSize": job["batchSize"], "device": job["device"]}, splitStrategy="fixed production-matched manifest splits", message="Preparing classifier training.")
            save_classifier_job(job)
        else:
            if job.get("datasetSource") == "server":
                server_info = prepare_server_classifier_dataset()
                data_root = Path(server_info["path"])
                job.update(datasetId=server_info["datasetId"], datasetPath=str(data_root),
                           datasetSize=server_info["imageCount"], classCounts=server_info["classCounts"])
                save_classifier_job(job)
            else:
                data_root = Path(job["datasetPath"])
            transform = transforms.Compose([transforms.Resize((job["imageSize"], job["imageSize"])), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
            dataset = datasets.ImageFolder(data_root, transform=transform)
            if dataset.classes != ["1_container", "2_containers"] or len(dataset) < 10:
                raise ValueError("Dataset must contain at least 10 readable images across 1_container/ and 2_containers/.")
            counts = {name: sum(1 for _, label in dataset.samples if dataset.classes[label] == name) for name in dataset.classes}
            group_file=data_root/"groups.json"
            groups=json.loads(group_file.read_text()) if group_file.is_file() else {}
            crop_meta_path=data_root/"cropMetadata.json"
            crop_metadata=json.loads(crop_meta_path.read_text()) if crop_meta_path.is_file() else {}
            grouped: dict[str,list[int]]={}
            item_groups=[]; item_source_ids=[]; item_types=[]; item_crop_metadata=[]
            catalog_sync = sync_dataset_catalog()
            source_db=dataset_catalog.connect(dataset_catalog_path())
            source_lookup={row["source_path"]:dict(row) for row in source_db.execute("SELECT source_path,image_id,group_key FROM image_sources")}
            source_db.close()
            for index,(path,_) in enumerate(dataset.samples):
                crop_info=crop_metadata.get(Path(path).name,{})
                source_group = str(crop_info.get("parentSourcePath") or groups.get(Path(path).name,Path(path).name))
                normalized = Path(source_group).stem.removesuffix("_cropped").casefold()
                lookup_key=source_group.replace("\\","/")
                if Path(lookup_key).is_absolute():
                    try: lookup_key=Path(lookup_key).resolve().relative_to(DATASET_ROOT).as_posix()
                    except ValueError: pass
                source_record=source_lookup.get(lookup_key)
                if not source_record and job.get("datasetSource")=="server":
                    try: source_record=source_lookup.get(Path(path).resolve().relative_to(DATASET_ROOT).as_posix())
                    except ValueError: source_record=None
                group_key = source_record["group_key"] if source_record else (normalized if job.get("datasetSource") == "server" else f"legacy:{job.get('datasetId')}:{normalized}")
                item_groups.append(group_key)
                item_source_ids.append(source_record["image_id"] if source_record else None)
                item_types.append("original-image-input" if source_record and Path(path).resolve()==(DATASET_ROOT/source_record["source_path"]).resolve() else "derived-crop-input")
                item_crop_metadata.append(crop_info)
                grouped.setdefault(group_key,[]).append(index)
            db = dataset_catalog.connect(dataset_catalog_path())
            try:
                dataset_catalog.assign_group_splits(db, grouped.keys(), utc_now())
                assignments={row["group_key"]:row["split"] for row in db.execute("SELECT group_key,split FROM split_assignments")}
                item_splits=[assignments[key] for key in item_groups]
                train_indices=[i for i,split in enumerate(item_splits) if split=="train"]
                val_indices=[i for i,split in enumerate(item_splits) if split=="validation"]
                test_indices=[i for i,split in enumerate(item_splits) if split=="test"]
                manifest_path=data_root/"splitAssignments.json"
                manifest_path.write_text(json.dumps({Path(path).name:split for (path,_),split in zip(dataset.samples,item_splits)},indent=2),encoding="utf-8")
            finally:
                db.close()
            n_train,n_val,n_test=len(train_indices),len(val_indices),len(test_indices)
            if min(n_train,n_val,n_test) < 1: raise ValueError("Need enough images for train/validation/test splits.")
            labels = list(dataset.targets)
            split_distributions = {
                "train": classifier_distribution(labels, train_indices),
                "validation": classifier_distribution(labels, val_indices),
                "test": classifier_distribution(labels, test_indices),
            }
            if job.get("balancingStrategy") != "none" and min(split_distributions["train"]["counts"].values()) == 0:
                raise ValueError("Class balancing requires both classes in the training split.")
            train_set,val_set,test_set=Subset(dataset,train_indices),Subset(dataset,val_indices),Subset(dataset,test_indices)
            db = dataset_catalog.connect(dataset_catalog_path())
            members=[]; member_by_id={}; annotations={}; reconciliation=[]; derived_rows=[]
            for item_index,((sample_path,label_index),split) in enumerate(zip(dataset.samples,item_splits)):
                source_image_id=item_source_ids[item_index]
                if source_image_id:
                    label_name=dataset.classes[label_index]
                    payload=json.dumps({"count":1 if label_name=="1_container" else 2},sort_keys=True)
                    previous=db.execute("SELECT version,payload_json FROM annotations WHERE image_id=? AND task='classification' AND model_key='container_count_classifier' ORDER BY version DESC LIMIT 1",(source_image_id,)).fetchone()
                    if not previous or previous["payload_json"]!=payload:
                        annotation_version=(previous["version"] if previous else 0)+1
                        db.execute("INSERT INTO annotations(image_id,task,model_key,version,payload_json,created_at) VALUES (?,?,?,?,?,?)",(source_image_id,"classification","container_count_classifier",annotation_version,payload,utc_now()))
                    else: annotation_version=previous["version"]
                    if source_image_id in member_by_id and member_by_id[source_image_id]["split"]!=split:
                        raise ValueError("Samples linked to the same source image were assigned to different splits.")
                    member_by_id[source_image_id]={"image_id":source_image_id,"split":split,"annotation_version":annotation_version}
                    annotations["container_count_classifier"]=max(annotations.get("container_count_classifier",0),annotation_version)
                    sample_id=hashlib.sha256(f"{run_id}:{Path(sample_path).name}:{label_index}:container_count:v1".encode()).hexdigest()
                    preprocessing=f"mobilenet-imagefolder-v1-size-{job['imageSize']}-imagenet-normalize"
                    derived_rows.append((sample_id,source_image_id,"container_count_classifier",item_types[item_index],payload,json.dumps(item_crop_metadata[item_index].get("cropCoordinates")) if item_crop_metadata[item_index].get("cropCoordinates") is not None else None,preprocessing,utc_now()))
                else:
                    reconciliation.append({"sample":Path(sample_path).name,"path":str(sample_path),"label":1 if dataset.classes[label_index]=="1_container" else 2,"split":split,"reason":"Legacy imported sample has no confidently identified original image."})
            members=list(member_by_id.values())
            version_id=dataset_catalog.next_dataset_version(db,str(job.get("datasetId")),"container_count_classifier",utc_now(),members,annotations,
                derived_count=sum(1 for row in derived_rows if row[3]=="derived-crop-input"),split_counts_override={"train":n_train,"validation":n_val,"test":n_test},manual_reconciliation=reconciliation)
            db.executemany("INSERT OR IGNORE INTO derived_samples(sample_id,dataset_version_id,image_id,model_key,sample_type,label_json,crop_json,preprocessing_version,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                [(sample_id,version_id,image_id,model_key,sample_type,label,crop,preprocessing,created) for sample_id,image_id,model_key,sample_type,label,crop,preprocessing,created in derived_rows])
            db.executemany("INSERT INTO unreconciled_samples(sample_id,dataset_version_id,legacy_dataset_id,sample_path,label_json,split,reason) VALUES (?,?,?,?,?,?,?)",
                [(hashlib.sha256(f"{version_id}:{row['path']}".encode()).hexdigest(),version_id,str(job.get("datasetId")),row["path"],json.dumps({"count":row["label"]}),row["split"],row["reason"]) for row in reconciliation])
            db.execute("INSERT OR REPLACE INTO training_runs(run_id,model_key,dataset_version_id,legacy_dataset_id,started_at) VALUES (?,?,?,?,?)",(run_id,"container_count_classifier",version_id,str(job.get("datasetId")) if reconciliation else None,utc_now()))
            db.commit(); db.close()
            job.update(datasetVersionId=version_id,datasetSize=len(dataset), classCounts=counts, splitCounts={"train":n_train,"validation":n_val,"test":n_test}, splitDistributions=split_distributions, uniqueTrainingImages=n_train, splitStrategy="Persistent grouped source-image split; legacy crops retain their imported source groups", manualReconciliationCount=len(reconciliation), message="Preparing classifier training."); save_classifier_job(job)
        arch=job["architecture"]
        net = models.mobilenet_v3_small(weights=None) if arch == "mobilenet_v3_small" else models.mobilenet_v3_large(weights=None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 2)
        checkpoint = job.get("initialCheckpoint")
        if checkpoint:
            source = next((row for row in classifier_runs() if row.get("id") == checkpoint.get("runId") and row.get("status") == "completed"), None)
            checkpoint_path = source.get("bestPath" if checkpoint.get("kind") == "best" else "lastPath") if source else None
            if not checkpoint_path or not Path(checkpoint_path).is_file():
                raise ValueError("The selected classifier checkpoint is no longer available.")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            net.load_state_dict(classifier_checkpoint_state(payload, arch, dataset.classes))
        device=torch.device("cuda" if job["device"]=="cuda" and torch.cuda.is_available() else "cpu"); net.to(device)
        strategy=job.get("balancingStrategy","none")
        target=CLASSIFIER_TARGETS[job.get("targetDistribution","70:30")]
        if strategy == "weighted_sampling":
            effective_counts={CLASSIFIER_CLASSES[0]:round(n_train*target[0]),CLASSIFIER_CLASSES[1]:n_train-round(n_train*target[0])}
            effective_train_distribution={"counts":effective_counts,"percentages":{name:round(count*100/n_train,2) for name,count in effective_counts.items()},"total":n_train}
        else:
            effective_train_distribution=split_distributions["train"]
        job.update(effectiveTrainDistribution=effective_train_distribution)
        save_classifier_job(job)
        if strategy == "class_weighted_loss":
            train_counts=[split_distributions["train"]["counts"][name] for name in CLASSIFIER_CLASSES]
            observed=[count/n_train for count in train_counts]
            loss_weights=torch.tensor([target[i]/observed[i] for i in range(2)],dtype=torch.float32)
            loss_weights=loss_weights/loss_weights.mean()
            criterion=nn.CrossEntropyLoss(weight=loss_weights.to(device))
        else:
            criterion=nn.CrossEntropyLoss()
        optimizer=torch.optim.AdamW(net.parameters(),lr=job["learningRate"])
        start_epoch = 0
        best = 0.0
        history = []
        resume_run_id = job.get("resumeFromRunId")
        if resume_run_id:
            resume_path = classifier_root() / "jobs" / str(resume_run_id) / "resume_state.pt"
            if not resume_path.is_file():
                raise ValueError("The selected classifier run has no resumable training state.")
            resume_state = load_resume_state(resume_path)
            if resume_state.get("architecture") != arch:
                raise ValueError("The classifier resume checkpoint is incompatible with this run.")
            if resume_state.get("classes") != dataset.classes or resume_state.get("trainingConfig", {}).get("datasetId") != job.get("datasetId"):
                raise ValueError("Classifier resume dataset or class mapping differs from the source run.")
            net.load_state_dict(resume_state["modelState"])
            optimizer.load_state_dict(resume_state["optimizerState"])
            for optimizer_state in optimizer.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(device)
            start_epoch = int(resume_state["epoch"])
            best = float(resume_state.get("bestMetric", 0.0))
            history = list(resume_state.get("history", []))
            restore_rng_states(resume_state.get("rngStates", {}))
            job.update(resumedFromRunId=resume_run_id, resumeStartEpoch=start_epoch,
                       epoch=start_epoch, history=history, bestAccuracy=best)
            save_classifier_job(job)
            last_payload = {"state_dict": resume_state["modelState"], "architecture": arch,
                            "classes": dataset.classes, "imageSize": job["imageSize"]}
            atomic_torch_save(last_payload, root / "last.pt")
            source_best = classifier_root() / "jobs" / str(resume_run_id) / "best.pt"
            if source_best.is_file():
                shutil.copy2(source_best, root / "best.pt")
            else:
                atomic_torch_save(last_payload, root / "best.pt")
        if strategy == "weighted_sampling":
            weights=classifier_sampling_weights(labels,train_indices,target)
            train_sampler=WeightedRandomSampler(weights,num_samples=n_train,replacement=True)
            train_loader=DataLoader(train_set,batch_size=job["batchSize"],sampler=train_sampler,num_workers=0)
        else:
            train_loader=DataLoader(train_set,batch_size=job["batchSize"],shuffle=True,num_workers=0)
        evaluation_loader=lambda subset: DataLoader(subset,batch_size=job["batchSize"],shuffle=False,num_workers=0)
        train_batches=len(train_loader)

        def stop_at_batch_boundary(epoch_index: int, completed_batches: int) -> None:
            model_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            atomic_torch_save({"state_dict": model_state, "architecture": arch,
                               "classes": dataset.classes, "imageSize": job["imageSize"]}, root / "last.pt")
            resume_payload = build_resume_state(
                model=net, optimizer=optimizer, scheduler=None, epoch=epoch_index,
                best_metric=best, history=history, architecture=arch,
                classes=list(dataset.classes), image_size=job["imageSize"],
                training_config={key: job.get(key) for key in (
                    "datasetId", "datasetSource", "epochs", "batchSize", "imageSize",
                    "learningRate", "architecture", "device", "balancingStrategy",
                    "targetDistribution", "datasetVersionId")},
            )
            atomic_torch_save(resume_payload, root / "resume_state.pt")
            job.update(status="stopped", epoch=epoch_index, currentStage="train",
                       batchProgress={"current": completed_batches, "total": train_batches},
                       finishedAt=utc_now(), updatedAt=utc_now(),
                       message="Training stopped at a batch boundary; resumable checkpoint saved.")
            save_classifier_job(job)

        for epoch in range(start_epoch, job["epochs"]):
            if classifier_stop_requested(run_id):
                stop_at_batch_boundary(epoch, 0)
                return
            net.train(); total=correct=0; loss_sum=0.0
            for batch_index,(x,y) in enumerate(train_loader,start=1):
                x,y=x.to(device),y.to(device); optimizer.zero_grad(); logits=net(x); loss=criterion(logits,y); loss.backward(); optimizer.step(); loss_sum+=loss.item()*y.size(0); correct+=(logits.argmax(1)==y).sum().item(); total+=y.size(0)
                if classifier_stop_requested(run_id):
                    stop_at_batch_boundary(epoch, batch_index)
                    return
                if batch_index % 25 == 0 or batch_index == train_batches:
                    job.update(epoch=epoch,status="running",currentStage="train",batchProgress={"current":batch_index,"total":train_batches},updatedAt=utc_now(),message=f"Epoch {epoch+1}/{job['epochs']} · batch {batch_index}/{train_batches}")
                    save_classifier_job(job)
            net.eval(); vloss=0.; vy=[]; vp=[]
            with torch.no_grad():
                for x,y in evaluation_loader(val_set):
                    logits=net(x.to(device)); vloss+=criterion(logits,y.to(device)).item()*y.size(0); vy.extend(y.tolist()); vp.extend(logits.argmax(1).cpu().tolist())
                    if classifier_stop_requested(run_id):
                        stop_at_batch_boundary(epoch, train_batches)
                        return
            item={"epoch":epoch+1,"trainLoss":loss_sum/total,"validationLoss":vloss/len(vy),**classifier_metrics(vy,vp)}; history.append(item)
            model_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            atomic_torch_save({"state_dict":model_state,"architecture":arch,"classes":dataset.classes,"imageSize":job["imageSize"]},root/"last.pt")
            if item["accuracy"]>=best:
                best=item["accuracy"]
                atomic_torch_save({"state_dict":model_state,"architecture":arch,"classes":dataset.classes,"imageSize":job["imageSize"]},root/"best.pt")
            resume_payload = build_resume_state(
                model=net, optimizer=optimizer, scheduler=None, epoch=epoch + 1,
                best_metric=best, history=history, architecture=arch,
                classes=list(dataset.classes), image_size=job["imageSize"],
                training_config={key: job.get(key) for key in ("datasetId", "datasetSource", "epochs", "batchSize", "imageSize", "learningRate", "architecture", "device", "balancingStrategy", "targetDistribution", "datasetVersionId")},
            )
            atomic_torch_save(resume_payload, root/"resume_state.pt")
            job.update(epoch=epoch+1,status="running",metrics=item,history=history,bestAccuracy=best,updatedAt=utc_now(),message=f"Epoch {epoch+1}/{job['epochs']}"); save_classifier_job(job)
            if classifier_stop_requested(run_id):
                job.update(status="stopped",finishedAt=utc_now(),message="Training stopped at an epoch boundary; resumable checkpoint is available.")
                save_classifier_job(job)
                return
        net.eval(); predictions=[]; actual=[]
        with torch.no_grad():
            for x,y in evaluation_loader(test_set): predictions.extend(net(x.to(device)).argmax(1).cpu().tolist()); actual.extend(y.tolist())
        job.update(status="completed",testMetrics={**classifier_metrics(actual,predictions),"distribution":split_distributions["test"]},updatedAt=utc_now(),message="Training, validation, and held-out testing completed."); save_classifier_job(job)
    except Exception as error:
        job.update(status="failed",message=str(error)[:1000],updatedAt=utc_now()); save_classifier_job(job)
    finally:
        _active_classifier=None


@app.post("/api/classifiers/import")
async def import_classifier_dataset(file: UploadFile = File(...)):
    if OBSERVABILITY_ONLY: raise HTTPException(403,"This observability instance is read-only.")
    raw=await file.read(1024*1024*1024+1)
    if not raw or len(raw)>1024*1024*1024: raise HTTPException(413,"Dataset archive is empty or exceeds 1 GB")
    try:
        archive=zipfile.ZipFile(__import__('io').BytesIO(raw)); members=archive.infolist()
        if any(Path(m.filename).is_absolute() or ".." in Path(m.filename).parts for m in members): raise ValueError("Unsafe archive path")
        manifest=next((m.filename for m in members if Path(m.filename).name=="labels.csv"),None)
        if not manifest: raise ValueError("ZIP must contain labels.csv exported by Container-labeler")
        import csv, io
        rows=list(csv.DictReader(io.StringIO(archive.read(manifest).decode("utf-8-sig"))))
        if not rows or not {"image_path","container_count"}.issubset(rows[0]): raise ValueError("labels.csv needs image_path and container_count columns")
        dsid=uuid.uuid4().hex; target=classifier_root()/"datasets"/dsid; target.mkdir(parents=True,exist_ok=True)
        seen=set(); invalid=[]; valid=0; groups={}; crop_metadata={}
        from PIL import Image
        for row in rows:
            image=row.get("image_path",""); label=row.get("container_count","")
            if image in seen: invalid.append({"image":image,"error":"duplicate manifest entry"}); continue
            seen.add(image)
            if label not in {"1","2"}: invalid.append({"image":image,"error":f"invalid label {label!r}"}); continue
            try:
                info=next(m for m in members if m.filename==image)
                payload=archive.read(info)
                with Image.open(__import__('io').BytesIO(payload)) as im: im.verify()
                basename=f"{hashlib.sha256(image.encode()).hexdigest()[:12]}-{Path(image).name}"
                out=target/("1_container" if label=="1" else "2_containers")/basename; out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(payload); groups[basename]=row.get("source_image_path") or image
                parent_path=row.get("source_image_path")
                coordinates=row.get("crop_coordinates") or row.get("crop_box")
                try: coordinates=json.loads(coordinates) if isinstance(coordinates,str) else coordinates
                except ValueError: coordinates=None
                crop_metadata[basename]={"parentSourcePath":parent_path,"cropCoordinates":coordinates}
                valid+=1
            except Exception as error: invalid.append({"image":image,"error":str(error)[:180]})
        counts={name:len(list((target/name).glob("*"))) for name in ("1_container","2_containers")}
        if not valid: raise ValueError("No readable labeled images found")
        meta={"datasetId":dsid,"path":str(target),"imageCount":valid,"classCounts":counts,"invalid":invalid}
        (target/"import.json").write_text(json.dumps({**meta,"invalidCount":len(invalid)},indent=2),encoding="utf-8"); (target/"groups.json").write_text(json.dumps(groups),encoding="utf-8"); (target/"cropMetadata.json").write_text(json.dumps(crop_metadata,indent=2),encoding="utf-8")
        return meta
    except HTTPException: raise
    except Exception as error: raise HTTPException(400,f"Invalid classification dataset: {error}") from error


@app.post("/api/classifiers/import-labeler")
def import_labeler_dataset(request: dict[str, Any] | None = None):
    if OBSERVABILITY_ONLY: raise HTTPException(403,"This observability instance is read-only.")
    request=request or {}
    if not LABELER_DB.is_file(): raise HTTPException(404,"Container-labeler database is not mounted")
    try:
        connection=sqlite3.connect(f"file:{LABELER_DB}?mode=ro",uri=True); connection.row_factory=sqlite3.Row
        query="""SELECT t.id,t.root_name,COUNT(c.container_count) labeled,
          SUM(c.container_count=1) one_count,SUM(c.container_count=2) two_count
          FROM text_sessions t JOIN container_counts c ON c.session_id=t.id
          JOIN text_annotations a ON a.session_id=t.id AND a.path=c.path
          WHERE a.active=1 AND a.cached_crop_path IS NOT NULL AND c.container_count IN (1,2)
          GROUP BY t.id ORDER BY labeled DESC"""
        sessions=connection.execute(query).fetchall()
        selected=request.get("sessionId")
        session=next((row for row in sessions if row["id"]==selected),None) if selected else (sessions[0] if sessions else None)
        if not session: raise ValueError("No labeled container count crops were found")
        rows=connection.execute("""SELECT a.path,a.source_path,a.cached_crop_path,a.container_box,a.box_index,c.container_count
          FROM text_annotations a JOIN container_counts c ON c.session_id=a.session_id AND c.path=a.path
          WHERE a.session_id=? AND a.active=1 AND a.cached_crop_path IS NOT NULL AND c.container_count IN (1,2)
          ORDER BY a.path""",(session["id"],)).fetchall()
        connection.close()
        dsid=uuid.uuid4().hex; target=classifier_root()/"datasets"/dsid; groups={}; crop_metadata={}; invalid=[]; counts={"1_container":0,"2_containers":0}
        for row in rows:
            source=(DATASET_ROOT/row["cached_crop_path"]).resolve()
            if not source.is_relative_to(DATASET_ROOT) or not source.is_file():
                invalid.append({"image":row["cached_crop_path"],"error":"missing or unsafe crop path"}); continue
            label="1_container" if row["container_count"]==1 else "2_containers"
            filename=f"{hashlib.sha256(row['path'].encode()).hexdigest()[:12]}-{source.name}"
            output=target/label/filename; output.parent.mkdir(parents=True,exist_ok=True); output.symlink_to(source)
            groups[filename]=row["source_path"] or row["path"]; counts[label]+=1
            try: crop_coordinates=json.loads(row["container_box"]) if row["container_box"] else None
            except (TypeError,ValueError): crop_coordinates=None
            crop_metadata[filename]={"parentSourcePath":row["source_path"],"sourceAnnotationPath":row["path"],"boxIndex":row["box_index"],"cropCoordinates":crop_coordinates}
        total=sum(counts.values())
        if not total: raise ValueError("No readable container crops found for the selected labeling session")
        meta={"datasetId":dsid,"path":str(target),"imageCount":total,"classCounts":counts,"invalid":invalid,"source":"Container-labeler","sessionId":session["id"],"sessionName":session["root_name"]}
        (target/"groups.json").write_text(json.dumps(groups),encoding="utf-8")
        (target/"cropMetadata.json").write_text(json.dumps(crop_metadata,indent=2),encoding="utf-8")
        (target/"import.json").write_text(json.dumps({**meta,"invalidCount":len(invalid)},indent=2),encoding="utf-8")
        return meta
    except HTTPException: raise
    except Exception as error: raise HTTPException(400,f"Could not import Container-labeler labels: {error}") from error


@app.get("/api/classifiers/dataset-summary")
def classifier_dataset_summary(source: str = "server", datasetId: str | None = None):
    if source == "server":
        labels = server_container_count_labels(required=True)["labels"].values()
        counts = {"1_container": sum(value == 1 for value in labels), "2_containers": sum(value == 2 for value in labels)}
        distribution = classifier_distribution([0] * counts["1_container"] + [1] * counts["2_containers"])
    elif source == "yolo_crops":
        info = synchronize_yolo_crop_labels() if not OBSERVABILITY_ONLY else production_matched_classifier_dataset(validate_image_contents=False)
        split_counts = info["splitCounts"]
        class_counts = {name: sum(split_counts[split][name] for split in ("train", "val", "test")) for name in CLASSIFIER_CLASSES}
        distribution = classifier_distribution([0] * class_counts["1_container"] + [1] * class_counts["2_containers"])
        return {**distribution, "warning": bool(distribution["imbalanceRatio"] and distribution["imbalanceRatio"] >= 2.0),
                "datasetName": "Production-matched YOLO union crops", "datasetType": "Production-matched YOLO union crops",
                "datasetRoot": info["datasetRoot"], "totalImages": info["imageCount"], "splitCounts": split_counts,
                "classCounts": class_counts, "datasetId": info["experimentId"],
                "labelSource": info.get("metadata", {}).get("label_source", "original frozen YOLO crop labels"),
                "strategyOptions": ["none", "weighted_sampling", "class_weighted_loss"],
                "targetOptions": list(CLASSIFIER_TARGETS)}
    elif source == "imported":
        if not datasetId or not re.fullmatch(r"[a-f0-9]{32}", datasetId):
            raise HTTPException(400, "Import a labeled dataset before requesting its summary.")
        root = (classifier_root() / "datasets" / datasetId).resolve()
        if not (root / "import.json").is_file():
            raise HTTPException(404, "Imported classifier dataset not found.")
        counts = {name: sum(1 for path in (root / name).rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED) for name in CLASSIFIER_CLASSES}
    else:
        raise HTTPException(400, "Choose the server, production-matched YOLO crop, or imported dataset.")
    if source == "imported":
        distribution = classifier_distribution([0] * counts["1_container"] + [1] * counts["2_containers"])
    ratio = distribution["imbalanceRatio"]
    return {**distribution, "warning": bool(ratio and ratio >= 2.0), "strategyOptions": ["none", "weighted_sampling", "class_weighted_loss"], "targetOptions": list(CLASSIFIER_TARGETS)}


@app.post("/api/classifiers/train")
def start_classifier_training(request: dict[str, Any]):
    if OBSERVABILITY_ONLY:
        raise HTTPException(403, "This observability instance is read-only.")
    resume_run_id = request.get("resumeRunId")
    if resume_run_id:
        if not re.fullmatch(r"[a-f0-9]{32}", str(resume_run_id)):
            raise HTTPException(400, "Choose a valid classifier run to resume.")
        source_path = classifier_root() / "jobs" / str(resume_run_id) / "job.json"
        if not source_path.is_file():
            raise HTTPException(404, "Classifier run not found.")
        source_job = json.loads(source_path.read_text(encoding="utf-8"))
        if source_job.get("status") not in {"stopped", "cancelled", "failed", "interrupted"}:
            raise HTTPException(409, "Only a stopped, failed, or interrupted classifier run can be resumed.")
        resume_state_path = source_path.parent / "resume_state.pt"
        if not resume_state_path.is_file():
            raise HTTPException(409, "This run has no versioned resumable checkpoint. Start a new fine-tune run from best.pt or last.pt.")
        request = {
            "datasetSource": source_job.get("datasetSource", "imported"),
            "datasetId": source_job.get("datasetId"), "epochs": source_job.get("epochs", 30),
            "batchSize": source_job.get("batchSize", 16), "imageSize": source_job.get("imageSize", 224),
            "learningRate": source_job.get("learningRate", .001), "architecture": source_job.get("architecture", "mobilenet_v3_small"),
            "device": source_job.get("device", "cpu"), "balancingStrategy": source_job.get("balancingStrategy", "none"),
            "targetDistribution": source_job.get("targetDistribution", "70:30"), "resumeRunId": str(resume_run_id),
        }
        if source_job.get("epoch", 0) >= source_job.get("epochs", 0):
            raise HTTPException(409, "The saved classifier checkpoint has no remaining epochs to resume.")
    dataset_source = request.get("datasetSource", "imported")
    if dataset_source == "server":
        dataset_id = "server-verified-counts"
        dataset_path = str(classifier_root() / "datasets" / "server-pending")
        dataset_size = None
        class_counts = None
    elif dataset_source == "yolo_crops":
        if int(request.get("imageSize", 224)) != 224:
            raise HTTPException(400, "The production-matched YOLO crop experiment uses 224×224 input. Set image size to 224.")
        yolo_info = synchronize_yolo_crop_labels()
        dataset_id = yolo_info["experimentId"]
        dataset_path = yolo_info["datasetRoot"]
        dataset_size = yolo_info["imageCount"]
        class_counts = {name: sum(yolo_info["splitCounts"][split][name] for split in ("train", "val", "test")) for name in CLASSIFIER_CLASSES}
    elif dataset_source == "imported":
        dataset_id = str(request.get("datasetId", ""))
        if not re.fullmatch(r"[a-f0-9]{32}", dataset_id):
            raise HTTPException(400, "Import a labeled ZIP dataset first")
        meta = classifier_root() / "datasets" / dataset_id / "import.json"
        if not meta.is_file():
            raise HTTPException(404, "Imported dataset not found")
        cfg = json.loads(meta.read_text(encoding="utf-8"))
        dataset_path = str(cfg.get("datasetRoot", cfg.get("path", meta.parent)))
        dataset_size = cfg.get("imageCount")
        class_counts = cfg.get("classCounts")
    else:
        raise HTTPException(400, "Choose the server, production-matched YOLO crops, or an imported classifier dataset.")

    try:
        epochs = max(1, min(200, int(request.get("epochs", 30))))
        batch = max(1, min(64, int(request.get("batchSize", 16))))
        size = max(96, min(512, int(request.get("imageSize", 224))))
        learning_rate = float(request.get("learningRate", .001))
    except Exception as error:
        raise HTTPException(400, "Invalid training configuration") from error
    if not .000001 <= learning_rate <= .1:
        raise HTTPException(400, "Learning rate must be between 0.000001 and 0.1")
    architecture = request.get("architecture", "mobilenet_v3_small")
    if not isinstance(architecture, str) or architecture not in {"mobilenet_v3_small", "mobilenet_v3_large"}:
        raise HTTPException(400, "Unsupported architecture")
    balancing_strategy = request.get("balancingStrategy", "none")
    target_distribution = request.get("targetDistribution", "70:30")
    if balancing_strategy not in {"none", "weighted_sampling", "class_weighted_loss"}:
        raise HTTPException(400, "Unsupported class balancing strategy")
    if target_distribution not in CLASSIFIER_TARGETS:
        raise HTTPException(400, "Unsupported target class distribution")

    initial_checkpoint = None
    checkpoint_run_id = request.get("checkpointRunId")
    checkpoint_kind = request.get("checkpointKind", "best")
    if checkpoint_run_id:
        if not re.fullmatch(r"[a-f0-9]{32}", str(checkpoint_run_id)) or checkpoint_kind not in {"best", "last"}:
            raise HTTPException(400, "Choose a valid saved classifier checkpoint")
        source = next((row for row in classifier_runs() if row.get("id") == checkpoint_run_id and row.get("status") == "completed"), None)
        if not source:
            raise HTTPException(404, "Completed classifier run not found")
        checkpoint_path = source.get("bestPath" if checkpoint_kind == "best" else "lastPath")
        if not checkpoint_path or not Path(checkpoint_path).is_file():
            raise HTTPException(404, "Classifier checkpoint not found")
        if source.get("architecture") != architecture:
            raise HTTPException(400, "Choose the same architecture as the selected checkpoint")
        initial_checkpoint = {"runId": checkpoint_run_id, "kind": checkpoint_kind}

    run_id = uuid.uuid4().hex
    job = {
        "id": run_id, "modelType": "classifier", "status": "queued", "createdAt": utc_now(),
        "updatedAt": utc_now(), "datasetPath": dataset_path, "datasetId": dataset_id,
        "datasetSize": dataset_size, "classCounts": class_counts, "epochs": epochs,
        "batchSize": batch, "imageSize": size, "learningRate": learning_rate,
        "architecture": architecture, "device": request.get("device", "cpu"), "epoch": 0,
        "datasetSource": dataset_source, "balancingStrategy": balancing_strategy,
        "targetDistribution": target_distribution, "initialCheckpoint": initial_checkpoint,
        "request": dict(request), "message": "Training queued for the independent worker.",
    }
    if resume_run_id:
        job["resumeFromRunId"] = str(resume_run_id)
    if initial_checkpoint:
        job["checkpointSource"] = {"runId": initial_checkpoint["runId"], "kind": initial_checkpoint["kind"]}
    try:
        job_queue.create_job(RUNS_DIR, "classifier", job)
    except FileExistsError as error:
        raise HTTPException(409, "Generated job ID already exists; retry submission.") from error
    return job


@app.get("/api/classifiers/runs/{run_id}")
def get_classifier_run(run_id: str):
    path=classifier_root()/"jobs"/run_id/"job.json"
    if not re.fullmatch(r"[a-f0-9]{32}",run_id) or not path.is_file(): raise HTTPException(404,"Classifier run not found")
    job = json.loads(path.read_text(encoding="utf-8"))
    if job_queue.cancel_requested(path.parent): job["stopRequested"] = True
    return job


@app.post("/api/classifiers/runs/{run_id}/stop")
def stop_classifier_run(run_id: str):
    job=get_classifier_run(run_id)
    if job["status"] not in ACTIVE_TRAINING_STATUSES: raise HTTPException(409,"Classifier run is not active")
    job_queue.request_cancel(classifier_root() / "jobs" / run_id)
    job["stopRequested"] = True
    job["message"] = "Stop requested; training will stop after the current batch."
    return {"ok":True,"stopRequested":True,"job":job}


@app.post("/api/classifiers/runs/{run_id}/resume")
def resume_classifier_run(run_id: str):
    return start_classifier_training({"resumeRunId": run_id})


@app.get("/api/classifiers/runs/{run_id}/download")
def download_classifier(run_id: str, kind: str="best"):
    job=get_classifier_run(run_id)
    if job.get("status")!="completed" or kind not in {"best","last"}: raise HTTPException(404,"Completed classifier checkpoint not found")
    path=classifier_root()/"jobs"/run_id/f"{kind}.pt"
    if not path.is_file(): raise HTTPException(404,"Checkpoint file is missing")
    return FileResponse(path,filename=f"{run_id}-{kind}.pt",media_type="application/octet-stream")


@app.post("/api/checkpoints/upload")
async def upload_checkpoint(file: UploadFile = File(...)):
    if OBSERVABILITY_ONLY:
        raise HTTPException(403, "This observability instance is read-only.")
    filename = Path(file.filename or "checkpoint.pt").name
    if Path(filename).suffix.lower() != ".pt":
        raise HTTPException(400, "Upload a PyTorch .pt checkpoint")
    contents = await file.read(250 * 1024 * 1024 + 1)
    if not contents or len(contents) > 250 * 1024 * 1024:
        raise HTTPException(413, "Checkpoint is empty or larger than 250 MB")
    checkpoint_id = f"upload-{uuid.uuid4().hex}"
    directory = RUNS_DIR / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{checkpoint_id}.pt"
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(contents)
    os.replace(temporary, path)
    metadata = {"name": filename, "originalName": filename, "size": len(contents), "createdAt": utc_now()}
    metadata_path = directory / f"{checkpoint_id}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"checkpoint": {"id": checkpoint_id, "name": filename, "filename": path.name, "kind": "uploaded", "source": "uploaded checkpoint", "fineTuneAllowed": True, "resumeAllowed": False, "createdAt": metadata["createdAt"], "downloadUrl": f"/api/checkpoints/{checkpoint_id}/download"}}


@app.get("/api/checkpoints/{checkpoint_id}/download")
def download_checkpoint(checkpoint_id: str):
    path, descriptor = resolve_checkpoint(checkpoint_id)
    filename = descriptor.get("originalName") or (f"{descriptor['kind']}.pt" if descriptor["kind"] in {"best", "last"} else path.name)
    return FileResponse(path, filename=Path(filename).name, media_type="application/octet-stream")


@app.get("/api/inference/status")
def inference_status():
    path = inference_model_path()
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import onnxruntime  # noqa: F401
        from sliding_inference_final import detect  # noqa: F401
        runtime_available = True
    except ImportError:
        runtime_available = False
    return {"available": path.is_file() and runtime_available, "model": path.name if path.is_file() else None,
            "runtime": "Multi-scale sliding-window · ONNX Runtime CPU"}


@app.post("/api/inference")
async def run_onnx_inference(file: UploadFile = File(...), confidence: float = 0.25,
                            scales: str = "1280,512", overlap: float = 0.4,
                            ios: float = 0.5, verticality: float = 0.15,
                            max_det: int = 10, use_slicing: bool = True):
    started = time.perf_counter()
    suffix = Path(file.filename or "image.jpg").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(400, "Upload a JPG, PNG, WebP, BMP or TIFF image")
    if not 0.01 <= confidence <= 0.99:
        raise HTTPException(400, "Confidence must be between 0.01 and 0.99")
    try:
        scale_values = tuple(int(value.strip()) for value in scales.split(",") if value.strip())
    except ValueError as error:
        raise HTTPException(400, "Scales must be comma-separated pixel sizes") from error
    if not scale_values or len(scale_values) > 8 or any(not 256 <= value <= 1280 for value in scale_values):
        raise HTTPException(400, "Provide 1–8 scales between 256 and 1280 pixels")
    if not 0 <= overlap <= 0.8:
        raise HTTPException(400, "Window overlap must be between 0 and 0.8")
    if not 0 <= ios <= 1 or not 0 <= verticality <= 1 or not 1 <= max_det <= 100:
        raise HTTPException(400, "IOS and verticality must be between 0 and 1; max_det must be 1–100")
    contents = await file.read(20 * 1024 * 1024 + 1)
    if not contents or len(contents) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image is empty or larger than 20 MB")
    try:
        import io
        import numpy as np
        from PIL import Image, ImageOps
        with Image.open(io.BytesIO(contents)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            rgb = np.asarray(image)
        height, width = rgb.shape[:2]
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(400, "Uploaded file is not a readable image") from error

    model_path, session = load_inference_session()
    inference_started = time.perf_counter()
    try:
        import cv2
        import numpy as np
        from sliding_inference_final import decode, detect as sliding_detect, preprocess

        # The script's preprocessing and coordinate mapping use OpenCV BGR.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if use_slicing:
            detections, _ = sliding_detect(
                session, bgr, scales=scale_values, overlap=overlap, conf=confidence,
                ios=ios, vert_min=verticality, max_det=max_det,
            )
        else:
            height, width = bgr.shape[:2]
            tensor, scale, _ = preprocess(bgr)
            detections = decode(session.run(None, {session.get_inputs()[0].name: tensor})[0],
                                scale, width, height, confidence)
            detections = detections[np.argsort(-detections[:, 4])][:max_det] if len(detections) else detections
    except Exception as error:
        raise HTTPException(500, f"Sliding-window inference failed: {str(error)[:400]}") from error
    runtime_ms = (time.perf_counter() - inference_started) * 1000
    boxes = [{"x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]),
              "confidence": float(box[4]), "className": "container"} for box in detections]
    return {"width": width, "height": height, "boxes": boxes, "count": len(boxes),
            "containerDetected": bool(boxes), "model": model_path.name,
            "runtime": "Multi-scale sliding-window · ONNX Runtime CPU", "slicing": {"enabled": use_slicing,
                "scales": list(scale_values), "overlap": overlap, "ios": ios,
                "verticality": verticality, "maxDet": max_det}, "inferenceElapsedMs": round(runtime_ms, 1),
            "totalElapsedMs": round((time.perf_counter() - started) * 1000, 1)}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), modelId: str | None = None, classifierId: str | None = None, castingModelId: str | None = None, confidence: float = 0.25, ocrEnabled: bool = False, ocrWholeImage: bool = False, ocrLanguage: str = "en", ocrModel: str = "PP-OCRv4", ocrConfidence: float = 0.25, ocrMode: str = "roi", roiPositions: str = "0,50,100", roiWidth: float = 15, roiHeight: float = 75, roiPadding: float = 2, cropPadding: float = 0.04, ocrRotations: str = "0,90,180,270", ocrMaxAttempts: int = 8, ocrFullFallback: bool = True, ocrContrast: bool = False, ocrGrayscale: bool = False, ocrSharpen: bool = False, ocrUpscale: float = 1, ocrGroundTruth: str = ""):
    inference_started = time.perf_counter()
    suffix = Path(file.filename or "image.jpg").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(400, "Upload a JPG, PNG, WebP, BMP or TIFF image")
    skip_container = modelId == "none"
    if skip_container:
        classifierId = None
        ocrEnabled = False
        modelId = None
    if not 0.01 <= confidence <= 0.99:
        raise HTTPException(400, "Confidence must be between 0.01 and 0.99")
    if not 0.01 <= ocrConfidence <= 0.99:
        raise HTTPException(400, "OCR confidence must be between 0.01 and 0.99")
    if ocrLanguage != "en":
        raise HTTPException(400, "OCR language must be en")
    if ocrModel not in {"PP-OCRv3", "PP-OCRv4"}:
        raise HTTPException(400, "Unsupported PaddleOCR model selection")
    if ocrMode not in {"full", "container", "roi"}:
        raise HTTPException(400, "Unsupported OCR processing mode")
    try:
        roi_values=tuple(float(v.strip())/100 for v in roiPositions.split(",") if v.strip())
        rotation_values=tuple(int(v) for v in ocrRotations.split(",") if v.strip())
    except ValueError as error:
        raise HTTPException(400, "ROI percentages and rotations must be comma-separated numbers") from error
    if len(roi_values)!=3 or any(not 0 <= v <= 1 for v in roi_values) or not rotation_values or any(v not in {0,90,180,270} for v in rotation_values):
        raise HTTPException(400, "Provide exactly three ROI center positions from 0 to 100 and valid rotations")
    if not 5 <= roiWidth <= 75 or not 10 <= roiHeight <= 100 or not 0 <= roiPadding <= 20 or not 0 <= cropPadding <= 0.25 or not 3 <= ocrMaxAttempts <= 32 or not 1 <= ocrUpscale <= 3:
        raise HTTPException(400, "Invalid OCR padding, attempt limit, or upscale factor")
    contents = await file.read(20 * 1024 * 1024 + 1)
    if not contents or len(contents) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image is empty or larger than 20 MB")
    try:
        from PIL import Image, ImageOps
        import io
        with Image.open(io.BytesIO(contents)) as image:
            # Work in the same orientation browsers use when displaying photos.
            # Keep this normalized payload for detection and every later crop.
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            width, height = normalized.size
        normalized_buffer = io.BytesIO()
        normalized.save(normalized_buffer, format="PNG")
        inference_contents = normalized_buffer.getvalue()
    except Exception as error:
        raise HTTPException(400, "Uploaded file is not a readable image") from error
    def run_detector(dataset_source: str, model_id: str | None, class_name: str):
        path, detector = selected_model(model_id, dataset_source)
        with tempfile.NamedTemporaryFile(suffix=".png") as temporary:
            temporary.write(inference_contents)
            temporary.flush()
            try:
                results = detector.predict(source=temporary.name, conf=confidence, device="cpu", verbose=False)
            except Exception as error:
                raise HTTPException(500, f"Detection failed: {str(error)[:500]}") from error
        found = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                found.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": float(box.conf[0]), "className": class_name})
        return path, found

    checkpoint, boxes = (None, []) if skip_container else run_detector("container_detector", modelId, "container")
    response = {"width": width, "height": height, "boxes": boxes, "model": Path(checkpoint).name if checkpoint else None, "count": len(boxes), "containerDetected": bool(boxes), "containerSkipped": skip_container}
    if castingModelId:
        casting_path, casting_boxes = run_detector("casting_holes", None if castingModelId == "latest" else castingModelId, "casting_hole")
        response["castingHoles"] = {"boxes": casting_boxes, "count": len(casting_boxes), "model": Path(casting_path).name, "detected": bool(casting_boxes)}
    # OCR is opt-in per request and runs after the existing detector. Keep failures
    # local to the OCR result so detector and classifier output remain usable.
    if classifierId:
        try:
            if not boxes:
                raise ValueError("No container area was detected; count classification was skipped")
            descriptor=next((row for row in classifier_runs() if row.get("id")==classifierId and row.get("status")=="completed"),None)
            if not descriptor: raise ValueError("Selected classifier is not a completed run")
            import torch
            from PIL import Image
            from torchvision import models, transforms
            saved=torch.load(descriptor["bestPath"],map_location="cpu",weights_only=True)
            arch=saved["architecture"]; classifier=models.mobilenet_v3_small(weights=None) if arch=="mobilenet_v3_small" else models.mobilenet_v3_large(weights=None)
            classifier.classifier[-1]=torch.nn.Linear(classifier.classifier[-1].in_features,2); classifier.load_state_dict(saved["state_dict"]); classifier.eval()
            with Image.open(io.BytesIO(inference_contents)) as source:
                source=source.convert("RGB")
                if boxes:
                    # Use the full extent of all detected containers as the
                    # classifier input. Passing these generator expressions
                    # directly to int() raises TypeError at runtime.
                    x1=max(0,min(int(min(b["x1"] for b in boxes)),source.width-1)); y1=max(0,min(int(min(b["y1"] for b in boxes)),source.height-1)); x2=max(x1+1,min(int(max(b["x2"] for b in boxes)),source.width)); y2=max(y1+1,min(int(max(b["y2"] for b in boxes)),source.height)); source=source.crop((x1,y1,x2,y2))
                prep=transforms.Compose([transforms.Resize((saved["imageSize"],saved["imageSize"])),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]); tensor=prep(source).unsqueeze(0)
            with torch.no_grad(): probs=torch.softmax(classifier(tensor),dim=1)[0].tolist()
            idx=max(range(2),key=lambda i:probs[i]); response["classification"]={"count":idx+1,"confidence":probs[idx],"probabilities":{"1":probs[0],"2":probs[1]},"model":descriptor.get("name") or f"{descriptor.get('architecture','classifier')} · {descriptor.get('id',classifierId)[:8]}"}
        except Exception as error:
            response["classificationError"]=str(error)[:300]
    if ocrEnabled:
        from io import BytesIO
        from PIL import Image
        from .container_number_validator import candidates as number_candidates
        from .ocr_service import recognize, recognize_container

        ocr_started = time.perf_counter()
        container_results = []
        try:
            with Image.open(BytesIO(inference_contents)) as source_image:
                source_image = source_image.convert("RGB")
                whole_image = source_image.copy() if ocrWholeImage or ocrMode == "full" else None
                for index, box in enumerate(boxes if ocrMode != "full" else []):
                    x1 = max(0, min(math.floor(box["x1"]), source_image.width - 1))
                    y1 = max(0, min(math.floor(box["y1"]), source_image.height - 1))
                    x2 = max(x1 + 1, min(math.ceil(box["x2"]), source_image.width))
                    y2 = max(y1 + 1, min(math.ceil(box["y2"]), source_image.height))
                    crop = source_image.crop((x1, y1, x2, y2))
                    if ocrMode == "roi":
                        outcome = recognize_container(source_image, box, language=ocrLanguage, model_name=ocrModel,
                            confidence=ocrConfidence, padding=0, roi_positions=roi_values,
                            roi_width=roiWidth/100, roi_height=roiHeight/100, roi_padding=roiPadding/100,
                            rotations=rotation_values, max_attempts=ocrMaxAttempts,
                            full_fallback=ocrFullFallback, preprocessing={"contrast":ocrContrast,"grayscale":ocrGrayscale,"sharpen":ocrSharpen,"upscale":ocrUpscale})
                        container_results.append({"containerIndex":index+1,"containerBox":box,**outcome})
                        continue
                    result = recognize(crop, language=ocrLanguage, model_name=ocrModel)
                    raw_text = " ".join(region["text"] for region in result["regions"])
                    regions = [region for region in result["regions"] if region["confidence"] >= ocrConfidence]
                    accepted_text = " ".join(region["text"] for region in regions)
                    container_results.append({"containerIndex": index + 1, "containerBox": box,
                        "cropRect": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                        "regions": regions, "rawText": raw_text, "recognizedText": accepted_text,
                        "candidates": number_candidates(accepted_text), "elapsedMs": result["elapsedMs"],
                        "error": result.get("error")})
            response["ocr"] = {"enabled": True, "mode":ocrMode,"language": ocrLanguage, "model": ocrModel,
                "confidenceThreshold": ocrConfidence, "containers": container_results,
                "elapsedMs": round((time.perf_counter() - ocr_started) * 1000, 1)}
            if ocrWholeImage or ocrMode == "full":
                whole = recognize(whole_image, language=ocrLanguage, model_name=ocrModel)
                whole_raw_text = " ".join(region["text"] for region in whole["regions"])
                whole_regions = [region for region in whole["regions"] if region["confidence"] >= ocrConfidence]
                whole_text = " ".join(region["text"] for region in whole_regions)
                response["ocr"]["wholeImage"] = {"regions": whole_regions, "rawText": whole_raw_text, "recognizedText": whole_text,
                    "candidates": number_candidates(whole_text), "elapsedMs": whole["elapsedMs"],
                    "error": whole.get("error")}
                response["ocr"]["elapsedMs"] = round((time.perf_counter() - ocr_started) * 1000, 1)
            # Side-by-side count metrics; exact match is computed only when the
            # operator supplies a manually verified ground-truth number.
            roi_regions=sum(len(item.get("regions",[])) for item in container_results)
            roi_numbers=sum(len(item.get("candidates",[])) for item in container_results)
            response["ocr"]["comparison"]={"relativeRoi":{"containersProcessed":len(container_results),"textRegionsDetected":roi_regions,"numbersRecognized":roi_numbers,"failedRecognitions":sum(not item.get("candidates") for item in container_results),"averageProcessingMs":round(sum(item.get("elapsedMs",0) for item in container_results)/max(1,len(container_results)),1)},"existing":{"containersProcessed":len(boxes) if ocrWholeImage or ocrMode == "full" else 0,"textRegionsDetected":len(response["ocr"].get("wholeImage",{}).get("regions",[])),"numbersRecognized":len(response["ocr"].get("wholeImage",{}).get("candidates",[])),"failedRecognitions":int(not bool(response["ocr"].get("wholeImage",{}).get("candidates"))) if ocrWholeImage or ocrMode == "full" else 0,"averageProcessingMs":response["ocr"].get("wholeImage",{}).get("elapsedMs")},"groundTruth":ocrGroundTruth.strip().upper() or None}
            truth=ocrGroundTruth.strip().upper()
            if truth:
                response["ocr"]["comparison"]["relativeRoi"]["exactMatch"]=any(c.get("candidate")==truth for item in container_results for c in item.get("candidates",[]))
                response["ocr"]["comparison"]["existing"]["exactMatch"]=any(c.get("candidate")==truth for c in response["ocr"].get("wholeImage",{}).get("candidates",[])) if ocrWholeImage or ocrMode == "full" else None
        except Exception as error:
            response["ocr"] = {"enabled": True, "error": str(error)[:500],
                "containers": container_results, "elapsedMs": round((time.perf_counter() - ocr_started) * 1000, 1)}
    response["inferenceElapsedMs"] = round((time.perf_counter() - inference_started) * 1000, 1)
    return response


@app.post("/api/training")
def start_training(request: TrainRequest):
    if OBSERVABILITY_ONLY:
        raise HTTPException(403, "This observability instance is read-only.")
    if request.initMode == "base" and request.checkpointId:
        raise HTTPException(400, "Select Fine-tune or Resume to use a checkpoint")
    if request.initMode in {"finetune", "resume"}:
        if not request.checkpointId:
            raise HTTPException(400, "Choose a saved or uploaded .pt checkpoint first")
        _, checkpoint = resolve_checkpoint(request.checkpointId)
        if request.initMode == "resume" and not checkpoint.get("resumeAllowed"):
            raise HTTPException(400, "This checkpoint cannot resume training. Choose Fine-tune to start a new run from its weights.")
    job_id = uuid.uuid4().hex
    request_data = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    job = {"id": job_id, "modelType": "yolo", "status": "queued", "createdAt": utc_now(),
           "updatedAt": utc_now(), "epoch": 0, "totalEpochs": request.epochs,
           "epochs": request.epochs, "imageSize": request.imageSize, "batch": request.batch,
           "baseModel": request.baseModel, "initMode": request.initMode,
           "datasetSource": request.datasetSource,
           "checkpointId": request.checkpointId, "request": request_data,
           "message": "Training queued for the independent worker."}
    try:
        job_queue.create_job(RUNS_DIR, "yolo", job)
    except FileExistsError as error:
        raise HTTPException(409, "Generated job ID already exists; retry submission.") from error
    return job


class ExportRequest(BaseModel):
    imageSize: int = Field(default=640, ge=320, le=1280)
    dynamic: bool = False
    opset: int = Field(default=17, ge=12, le=23)


def export_worker(export_id: str, request: ExportRequest) -> None:
    global _active_export
    path = export_job_path(export_id)
    job = json.loads(path.read_text(encoding="utf-8"))
    try:
        job.update(status="running", message="Loading checkpoint and preparing ONNX export.", updatedAt=utc_now())
        save_export(job)
        checkpoint, _ = resolve_checkpoint(job["checkpointId"])
        output_dir = RUNS_DIR / "exports" / export_id
        output_dir.mkdir(parents=True, exist_ok=True)
        staged_checkpoint = output_dir / "checkpoint.pt"
        shutil.copy2(checkpoint, staged_checkpoint)
        from ultralytics import YOLO
        import onnx

        model = YOLO(str(staged_checkpoint))
        job.update(message=f"Exporting ONNX at {request.imageSize}px, opset {request.opset}.", updatedAt=utc_now())
        save_export(job)
        output = model.export(format="onnx", imgsz=request.imageSize, dynamic=request.dynamic, opset=request.opset, simplify=False, device="cpu")
        exported = Path(output).resolve()
        if not exported.is_file() or exported.stat().st_size == 0:
            raise RuntimeError("Exporter did not produce a nonempty ONNX file")
        onnx.checker.check_model(onnx.load(str(exported)))
        final_path = output_dir / "model.onnx"
        if exported != final_path:
            shutil.move(str(exported), final_path)
        job.update(status="completed", message="ONNX graph exported and validated.", updatedAt=utc_now(), filename=final_path.name, size=final_path.stat().st_size)
        save_export(job)
    except Exception as error:
        job.update(status="failed", message=str(error)[:1200], updatedAt=utc_now())
        save_export(job)
    finally:
        with _export_lock:
            _active_export = None


@app.post("/api/checkpoints/{checkpoint_id}/export/onnx")
def start_onnx_export(checkpoint_id: str, request: ExportRequest):
    global _active_export
    if OBSERVABILITY_ONLY:
        raise HTTPException(403, "This observability instance is read-only.")
    resolve_checkpoint(checkpoint_id)
    try:
        import onnx  # noqa: F401
    except ImportError as error:
        raise HTTPException(503, "ONNX export support is not installed. Rebuild AI-VIsion-demo and try again.") from error
    with _export_lock:
        if _active_export:
            raise HTTPException(409, "An ONNX export is already running")
        export_id = uuid.uuid4().hex
        job = {"id": export_id, "checkpointId": checkpoint_id, "status": "queued", "createdAt": utc_now(), "updatedAt": utc_now(), "imageSize": request.imageSize, "dynamic": request.dynamic, "opset": request.opset, "message": "Export queued."}
        save_export(job)
        _active_export = export_id
        threading.Thread(target=export_worker, args=(export_id, request), daemon=True).start()
    return job


@app.get("/api/exports")
def list_exports():
    directory = RUNS_DIR / "exports"
    jobs = []
    if directory.is_dir():
        for path in sorted(directory.glob("[a-f0-9]" * 32 + ".json"), key=lambda item: item.stat().st_mtime, reverse=True):
            try:
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
    return {"exports": jobs}


@app.get("/api/exports/{export_id}")
def get_export(export_id: str):
    path = export_job_path(export_id)
    if not path.is_file():
        raise HTTPException(404, "Export not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/api/exports/{export_id}/download")
def download_export(export_id: str, format: str = "onnx"):
    if format not in {"onnx", "pt"}:
        raise HTTPException(400, "Format must be onnx or pt")
    job_file = export_job_path(export_id)
    if not job_file.is_file():
        raise HTTPException(404, "Export not found")
    job = json.loads(job_file.read_text(encoding="utf-8"))
    if format == "pt":
        source, _ = resolve_checkpoint(job["checkpointId"])
        return FileResponse(source, filename=source.name, media_type="application/octet-stream")
    path = RUNS_DIR / "exports" / export_id / "model.onnx"
    if job.get("status") != "completed" or not path.is_file() or not path.resolve().is_relative_to((RUNS_DIR / "exports" / export_id).resolve()):
        raise HTTPException(404, "Completed ONNX export not found")
    return FileResponse(path, filename=path.name, media_type="application/octet-stream")


@app.get("/api/training/latest")
def latest_training():
    jobs_dir = RUNS_DIR / "jobs"
    files = sorted(jobs_dir.glob("*/job.json"), key=lambda path: path.stat().st_mtime, reverse=True) if jobs_dir.exists() else []
    return json.loads(files[0].read_text(encoding="utf-8")) if files else {"status": "none"}


@app.get("/api/training/{job_id}")
def training_status(job_id: str):
    return read_job(job_id)


@app.post("/api/training/{job_id}/stop")
def stop_training(job_id: str):
    job = read_job(job_id)
    if job.get("status") not in {"queued", "preparing", "running"}:
        raise HTTPException(409, "Training can only be stopped before final evaluation")
    job_queue.request_cancel(job_path(job_id).parent)
    job["stopRequested"] = True
    job["message"] = "Stop requested. The independent worker will stop at its next safe checkpoint boundary."
    return {"ok": True, "stopRequested": True, "job": job}


@app.get("/api/training/{job_id}/metrics")
def training_metrics(job_id: str):
    return metric_history(job_id)


@app.get("/api/training/{job_id}/logs")
def training_logs(job_id: str):
    job = read_job(job_id)
    metadata = job_path(job_id)
    log_files = [metadata.parent / "training.log", results_csv_path(metadata, job).parent / "results.csv"]
    content = []
    for path in log_files:
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            content.append({"name": path.name, "path": str(path.relative_to(RUNS_DIR)), "tail": lines[-300:]})
        except OSError:
            continue
    return {"runId": job_id, "events": job.get("events", []), "failureReason": job.get("failureReason"), "files": content}


@app.on_event("startup")
def recover_jobs():
    if OBSERVABILITY_ONLY:
        return
    jobs_dir = RUNS_DIR / "jobs"
    # Job execution belongs to the independent worker. Web startup must never
    # rewrite worker-owned active statuses or infer that its restart interrupted them.
    classifier_jobs = classifier_root() / "jobs"
    backfill_job_metrics()
    exports_dir = RUNS_DIR / "exports"
    if exports_dir.is_dir():
        for metadata in exports_dir.glob("[a-f0-9]" * 32 + ".json"):
            try:
                export = json.loads(metadata.read_text(encoding="utf-8"))
                if export.get("status") in {"queued", "running"}:
                    export.update(status="failed", updatedAt=utc_now(), message="Server restarted during ONNX export. Start the export again.")
                    save_export(export)
            except (OSError, ValueError):
                continue


@app.get("/")
def home():
    return FileResponse(ROOT / "static" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
