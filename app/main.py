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

ROOT = Path(__file__).parent
DATASET_ROOT = Path(os.getenv("DATASET_ROOT", "/dataset")).resolve()
RUNS_DIR = Path(os.getenv("RUNS_DIR", "/runs")).resolve()
SUPPORTED = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
_lock = threading.RLock()
_active_training: str | None = None
_model_lock = threading.Lock()
_loaded_model: tuple[str, Any] | None = None
_export_lock = threading.Lock()
_active_export: str | None = None
_active_classifier: str | None = None
OBSERVABILITY_ONLY = os.getenv("OBSERVABILITY_ONLY", "0").lower() in {"1", "true", "yes"}
LABELER_DB = Path(os.getenv("LABELER_DB", "/labeler-data/container-labeler.sqlite3"))
app = FastAPI(title="AI Vision Demo", version="1.0.0")


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
    return json.loads(path.read_text(encoding="utf-8"))


def save_job(job: dict[str, Any]) -> None:
    path = job_path(job["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with _lock:
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
                models.append({"id": job["id"], "name": f"Container {Path(job.get('baseModel', 'yolo26n.pt')).stem.upper()} · {job['id'][:8]}", "path": str(best), "createdAt": job.get("updatedAt"), "epochs": job.get("epochs"), "metrics": job.get("metrics", {})})
        except (OSError, ValueError, KeyError):
            continue
    return sorted(models, key=lambda model: model.get("createdAt") or "", reverse=True)


ACTIVE_TRAINING_STATUSES = {"queued", "preparing", "running", "evaluating"}
FINAL_TRAINING_STATUSES = {"completed", "failed", "cancelled", "canceled", "interrupted"}


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
                "CANCELLED" if status in {"cancelled", "canceled", "interrupted"} else
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
                        "jobStatus": job.get("status"),
                        "fineTuneAllowed": job.get("status") in FINAL_TRAINING_STATUSES,
                        "resumeAllowed": kind == "last" and job.get("status") in {"failed", "interrupted"},
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
    descriptor = {"id": checkpoint_id, "jobId": job_id, "kind": kind, "jobStatus": job.get("status"), "resumeAllowed": kind == "last" and job.get("status") in {"failed", "interrupted"} and not job.get("resumedBy")}
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


def selected_model(model_id: str | None = None) -> tuple[str, Any]:
    global _loaded_model
    models = checkpoint_models()
    match = next((model for model in models if model["id"] == model_id), None) if model_id else (models[0] if models else None)
    if not match:
        raise HTTPException(409 if not models else 404, "Train a container detector before running inference" if not models else "Trained model not found")
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
    groups: dict[str, list[Path]] = {}
    boxes, skipped, fingerprint = 0, 0, hashlib.sha256()
    label_summary: dict[str, Any] = {"sourceImageCount": len(images), "missingLabelFiles": 0, "invalidLabelFiles": 0, "emptyLabelFiles": 0, "validLabelFiles": 0, "widths": [], "heights": [], "areas": [], "boxesPerImage": [], "invalidSamples": []}
    for image in images:
        label = label_for(image, image_root, labels_root)
        if not label.is_file():
            skipped += 1
            label_summary["missingLabelFiles"] += 1
            continue
        try:
            image_box_rows = validated_boxes(label)
        except (OSError, ValueError) as error:
            skipped += 1
            label_summary["invalidLabelFiles"] += 1
            if len(label_summary["invalidSamples"]) < 30:
                label_summary["invalidSamples"].append({"image": image.relative_to(image_root).as_posix(), "reason": str(error)[:240]})
            continue
        image_boxes = len(image_box_rows)
        label_summary["validLabelFiles"] += 1
        if image_boxes == 0:
            label_summary["emptyLabelFiles"] += 1
        label_summary["boxesPerImage"].append({"image": image.relative_to(image_root).as_posix(), "count": image_boxes})
        for _, _, width, height in image_box_rows:
            label_summary["widths"].append(width)
            label_summary["heights"].append(height)
            label_summary["areas"].append(width * height)
        boxes += image_boxes
        stat = image.stat()
        fingerprint.update(f"{image.relative_to(image_root).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        if label.is_file():
            fingerprint.update(label.read_bytes())
        groups.setdefault(image.stem.removesuffix("_cropped").casefold(), []).append(image)
    if boxes == 0:
        raise ValueError("No valid container bounding boxes were found in the YOLO labels.")
    group_names = sorted(groups)
    random.Random(42).shuffle(group_names)
    if len(group_names) < 3:
        raise ValueError("At least three distinct source-image groups are required for leakage-safe splits.")
    test_count, val_count = max(1, round(len(group_names) * 0.1)), max(1, round(len(group_names) * 0.1))
    if test_count + val_count >= len(group_names):
        test_count, val_count = 0, 1
    splits = {"test": set(group_names[:test_count]), "val": set(group_names[test_count:test_count + val_count])}
    splits["train"] = set(group_names) - splits["test"] - splits["val"]
    cross_split_groups = (splits["train"] & splits["val"]) | (splits["train"] & splits["test"]) | (splits["val"] & splits["test"])
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
    label_summary["duplicateImageCheck"] = {"performed": False, "reason": "Exact image hashing was skipped to avoid an additional full read of the large source dataset during training preparation."}
    label_summary["splitLeakageCheck"] = {"grouping": "Source image filename stem; terminal _cropped suffix removed.", "seed": 42, "groupCount": len(group_names), "groupsInMultipleSplits": len(cross_split_groups), "passed": not cross_split_groups}
    label_summary["fingerprintAlgorithm"] = "SHA-256 of relative image path, image byte size, image mtime, and label contents. Image contents are not read for the fingerprint."
    diagnostics = {"labelValidation": label_summary, "splitStrategy": "Deterministic 80/10/10 split by source-image filename group, keeping _cropped derivatives together.", "splitSeed": 42, "splitCounts": counts, "fingerprintAlgorithm": label_summary["fingerprintAlgorithm"]}
    return yaml_path, counts, fingerprint.hexdigest(), skipped, diagnostics


def train_worker(job_id: str, request: TrainRequest) -> None:
    global _active_training
    job = read_job(job_id)
    try:
        job.update(status="preparing", startedAt=job.get("startedAt") or utc_now(), updatedAt=utc_now(), message="Validating labels and creating leakage-aware dataset splits.")
        save_job(job)
        yaml_path, split_counts, fingerprint, skipped_images, dataset_diagnostics = build_training_dataset(job_id)
        job.update(status="running", updatedAt=utc_now(), message=f"Training on CPU. {skipped_images} images with missing or invalid labels were excluded. The first run downloads YOLO26n weights.", splitCounts=split_counts, skippedImages=skipped_images, datasetFingerprint=fingerprint, datasetDiagnostics=dataset_diagnostics)
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
                job["checkpointOutputDir"] = str(checkpoint_path.parent)
                job["resumedFromJobId"] = checkpoint.get("jobId")
                source_stat = checkpoint_path.stat()
                job["inheritedCheckpoint"] = {"runId": checkpoint.get("jobId"), "path": str(checkpoint_path), "sizeBytes": source_stat.st_size, "modifiedAt": datetime.fromtimestamp(source_stat.st_mtime, timezone.utc).isoformat(), "capturedAt": utc_now()}
        initial_model = str(checkpoint_path) if checkpoint_path else request.baseModel
        model = YOLO(initial_model)
        try:
            import torch
            import ultralytics
            environment = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__, "ultralytics": ultralytics.__version__, "application": app.version}
        except Exception:
            environment = {"python": platform.python_version(), "platform": platform.platform(), "application": app.version}
        job_dir = RUNS_DIR / "jobs" / job_id
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
        model.train(data=str(yaml_path), epochs=request.epochs, imgsz=request.imageSize, batch=request.batch, device="cpu", workers=0, cache=False, amp=False, project=str(job_dir / "results"), name="container", exist_ok=True, plots=True, save=True, patience=20, resume=request.initMode == "resume")
        weight_dir = Path(job.get("checkpointOutputDir")) if job.get("checkpointOutputDir") else job_dir / "results" / "container" / "weights"
        best = weight_dir / "best.pt"
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


@app.get("/api/dataset")
def dataset_info():
    images = list_images()
    labels = list((DATASET_ROOT / "labels").glob("**/*.txt")) if (DATASET_ROOT / "labels").is_dir() else []
    return {"root": str(DATASET_ROOT), "imageCount": len(images), "labelCount": len(labels), "readOnly": True, "classes": ["container"], "imageSizeBytes": sum(path.stat().st_size for path in images)}


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
def dataset_explorer_detection(page: int = 1, pageSize: int = 50, search: str = "", annotation: str = "all"):
    page, page_size = dataset_explorer_page(page, pageSize)
    if annotation not in {"all", "annotated", "unannotated"}:
        raise HTTPException(400, "Invalid annotation filter")
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
        if annotation != "all" and status != annotation:
            continue
        records.append({"path": relative, "filename": image.name, "annotationStatus": status, "boxCount": count, "boxes": boxes, "annotationError": error})
    total = len(records)
    start = (page - 1) * page_size
    return {"page": page, "pageSize": page_size, "total": total, "stats": {"totalImages": total_images, "annotatedImages": annotated_images, "unannotatedImages": max(0, total_images - annotated_images), "totalBoxes": total_boxes}, "items": records[start:start + page_size]}


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
            records.append({"id": dataset_id, "name": info.get("sessionName") or info.get("source") or f"Classifier dataset · {dataset_id[:8]}", "source": info.get("source", "Imported classifier dataset"), "imageCount": total, "classCounts": classes, "createdAt": metadata.stat().st_mtime})
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return sorted(records, key=lambda row: (row["createdAt"], row["id"]), reverse=True)


@app.get("/api/dataset/explorer/classification")
def dataset_explorer_classification(datasetId: str | None = None, page: int = 1, pageSize: int = 50, search: str = "", label: str = "all"):
    page, page_size = dataset_explorer_page(page, pageSize)
    if label not in {"all", "1", "2", "unclassified"}:
        raise HTTPException(400, "Invalid classification filter")
    datasets = classifier_dataset_records()
    selected = next((row for row in datasets if row["id"] == datasetId), None) if datasetId else (datasets[0] if datasets else None)
    if datasetId and not selected:
        raise HTTPException(404, "Classifier dataset not found")
    if not selected:
        return {"datasets": datasets, "dataset": None, "page": page, "pageSize": page_size, "total": 0, "stats": {"totalImages": 0, "oneContainer": 0, "twoContainers": 0, "unclassified": 0, "onePercent": 0, "twoPercent": 0}, "items": []}
    root = classifier_root() / "datasets" / selected["id"]
    folder_specs = (("1_container", "1"), ("2_containers", "2"), ("unclassified", "unclassified"), ("Unclassified", "unclassified"))
    items = []
    query = search.casefold().strip()
    for folder, class_label in folder_specs:
        directory = root / folder
        if not directory.is_dir():
            continue
        for image in directory.rglob("*"):
            if not image.is_file() or image.suffix.lower() not in SUPPORTED or (query and query not in image.name.casefold()):
                continue
            if label != "all" and class_label != label:
                continue
            rel = image.relative_to(root).as_posix()
            resolved = image.resolve()
            if not resolved.is_file() or not (resolved.is_relative_to(root.resolve()) or resolved.is_relative_to(DATASET_ROOT)):
                continue
            items.append({"path": rel, "filename": image.name, "label": class_label, "labelText": {"1": "1 container", "2": "2 containers", "unclassified": "Unclassified"}[class_label]})
    items.sort(key=lambda row: row["filename"].casefold())
    total_images = selected["imageCount"]
    counts = selected["classCounts"]
    return {"datasets": datasets, "dataset": selected, "page": page, "pageSize": page_size, "total": len(items), "stats": {"totalImages": total_images, "oneContainer": counts["1_container"], "twoContainers": counts["2_containers"], "unclassified": counts["unclassified"], "onePercent": round(counts["1_container"] * 100 / total_images, 1) if total_images else 0, "twoPercent": round(counts["2_containers"] * 100 / total_images, 1) if total_images else 0}, "items": items[(page - 1) * page_size:page * page_size]}


@app.get("/api/dataset/explorer/classification/file")
def dataset_explorer_classification_file(datasetId: str, path: str):
    if not re.fullmatch(r"[a-f0-9]{32}", datasetId):
        raise HTTPException(404, "Classifier dataset not found")
    root = (classifier_root() / "datasets" / datasetId).resolve()
    candidate = (root / path).resolve()
    if not candidate.is_file() or not (candidate.is_relative_to(root) or candidate.is_relative_to(DATASET_ROOT)) or candidate.suffix.lower() not in SUPPORTED:
        raise HTTPException(404, "Classifier image not found")
    return FileResponse(candidate)


@app.get("/api/models")
def models():
    return {"models": checkpoint_models(), "checkpoints": list_checkpoints(), "trainingRuns": training_runs(), "classifiers": classifier_models(), "classifierRuns": classifier_runs()}


def classifier_root() -> Path:
    return RUNS_DIR / "classifiers"


def classifier_runs() -> list[dict[str, Any]]:
    result = []
    for meta in (classifier_root() / "jobs").glob("*/job.json"):
        try:
            job = json.loads(meta.read_text(encoding="utf-8"))
            for key in ("best.pt", "last.pt"):
                job[key.replace(".pt", "Path")] = str(meta.parent / key) if (meta.parent / key).is_file() else None
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
    temporary = target.with_suffix(".tmp")
    with _lock:
        if target.is_file():
            try:
                if json.loads(target.read_text(encoding="utf-8")).get("stopRequested"):
                    job["stopRequested"] = True
            except (OSError, ValueError):
                pass
        temporary.write_text(json.dumps(job, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, target)


def classifier_stop_requested(run_id: str) -> bool:
    path = classifier_root() / "jobs" / run_id / "job.json"
    try:
        return bool(json.loads(path.read_text(encoding="utf-8")).get("stopRequested"))
    except (OSError, ValueError, TypeError):
        return False


def classifier_worker(run_id: str) -> None:
    global _active_classifier
    root = classifier_root() / "jobs" / run_id
    job = json.loads((root / "job.json").read_text(encoding="utf-8"))
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Subset
        from torchvision import datasets, models, transforms
        job.update(status="running", message="Preparing image classification data.", startedAt=utc_now()); save_classifier_job(job)
        data_root = Path(job["datasetPath"])
        transform = transforms.Compose([transforms.Resize((job["imageSize"], job["imageSize"])), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        dataset = datasets.ImageFolder(data_root, transform=transform)
        if dataset.classes != ["1_container", "2_containers"] or len(dataset) < 10:
            raise ValueError("Dataset must contain at least 10 readable images across 1_container/ and 2_containers/.")
        counts = {name: sum(1 for _, label in dataset.samples if dataset.classes[label] == name) for name in dataset.classes}
        group_file=data_root/"groups.json"
        groups=json.loads(group_file.read_text()) if group_file.is_file() else {}
        grouped: dict[str,list[int]]={}
        for index,(path,_) in enumerate(dataset.samples): grouped.setdefault(groups.get(Path(path).name,Path(path).name),[]).append(index)
        keys=sorted(grouped); random.Random(42).shuffle(keys)
        train_cut=max(1,int(len(keys)*.8)); val_cut=max(train_cut+1,int(len(keys)*.9))
        train_indices=[i for group in keys[:train_cut] for i in grouped[group]]; val_indices=[i for group in keys[train_cut:val_cut] for i in grouped[group]]; test_indices=[i for group in keys[val_cut:] for i in grouped[group]]
        n_train,n_val,n_test=len(train_indices),len(val_indices),len(test_indices)
        if min(n_train,n_val,n_test) < 1: raise ValueError("Need enough images for train/validation/test splits.")
        train_set,val_set,test_set=Subset(dataset,train_indices),Subset(dataset,val_indices),Subset(dataset,test_indices)
        job.update(datasetSize=len(dataset), classCounts=counts, splitCounts={"train":n_train,"validation":n_val,"test":n_test}, splitStrategy="Deterministic 80/10/10 by source image/capture group", message="Preparing classifier training."); save_classifier_job(job)
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
            if not isinstance(payload, dict) or payload.get("architecture") != arch:
                raise ValueError("The selected classifier checkpoint does not match this architecture.")
            if payload.get("classes") != dataset.classes or not isinstance(payload.get("state_dict"), dict):
                raise ValueError("The selected classifier checkpoint has incompatible classes or weights.")
            net.load_state_dict(payload["state_dict"])
        device=torch.device("cuda" if job["device"]=="cuda" and torch.cuda.is_available() else "cpu"); net.to(device)
        criterion=nn.CrossEntropyLoss(); optimizer=torch.optim.AdamW(net.parameters(),lr=job["learningRate"])
        loader=lambda subset,shuffle: DataLoader(subset,batch_size=job["batchSize"],shuffle=shuffle,num_workers=0)
        train_batches=len(loader(train_set,False))
        best=0.0; history=[]
        for epoch in range(job["epochs"]):
            if classifier_stop_requested(run_id):
                job.update(status="stopped",finishedAt=utc_now(),message="Training stopped by user."); save_classifier_job(job); return
            net.train(); total=correct=0; loss_sum=0.0
            for batch_index,(x,y) in enumerate(loader(train_set,True),start=1):
                if classifier_stop_requested(run_id):
                    job.update(status="stopped",finishedAt=utc_now(),epoch=epoch,message="Training stopped by user during the current epoch."); save_classifier_job(job); return
                x,y=x.to(device),y.to(device); optimizer.zero_grad(); logits=net(x); loss=criterion(logits,y); loss.backward(); optimizer.step(); loss_sum+=loss.item()*y.size(0); correct+=(logits.argmax(1)==y).sum().item(); total+=y.size(0)
                if batch_index % 25 == 0 or batch_index == train_batches:
                    job.update(epoch=epoch,status="running",currentStage="train",batchProgress={"current":batch_index,"total":train_batches},updatedAt=utc_now(),message=f"Epoch {epoch+1}/{job['epochs']} · batch {batch_index}/{train_batches}")
                    save_classifier_job(job)
            net.eval(); vloss=0.; vy=[]; vp=[]
            with torch.no_grad():
                for x,y in loader(val_set,False):
                    logits=net(x.to(device)); vloss+=criterion(logits,y.to(device)).item()*y.size(0); vy.extend(y.tolist()); vp.extend(logits.argmax(1).cpu().tolist())
            tp=sum(a==b==1 for a,b in zip(vy,vp)); fp=sum(a==1 and b==0 for a,b in zip(vy,vp)); fn=sum(a==0 and b==1 for a,b in zip(vy,vp)); precision=tp/max(1,tp+fp); recall=tp/max(1,tp+fn); f1=2*precision*recall/max(1e-12,precision+recall); acc=sum(a==b for a,b in zip(vy,vp))/len(vy)
            item={"epoch":epoch+1,"trainLoss":loss_sum/total,"validationLoss":vloss/len(vy),"accuracy":acc,"precision":precision,"recall":recall,"f1":f1,"confusionMatrix":[[sum(a==0 and b==0 for a,b in zip(vy,vp)),sum(a==0 and b==1 for a,b in zip(vy,vp))],[sum(a==1 and b==0 for a,b in zip(vy,vp)),sum(a==1 and b==1 for a,b in zip(vy,vp))]]}; history.append(item)
            torch.save({"state_dict":net.cpu().state_dict(),"architecture":arch,"classes":dataset.classes,"imageSize":job["imageSize"]},root/"last.pt"); net.to(device)
            if acc>=best: best=acc; torch.save({"state_dict":net.cpu().state_dict(),"architecture":arch,"classes":dataset.classes,"imageSize":job["imageSize"]},root/"best.pt"); net.to(device)
            job.update(epoch=epoch+1,status="running",metrics=item,history=history,bestAccuracy=best,updatedAt=utc_now(),message=f"Epoch {epoch+1}/{job['epochs']}"); save_classifier_job(job)
        net.eval(); predictions=[]; actual=[]
        with torch.no_grad():
            for x,y in loader(test_set,False): predictions.extend(net(x.to(device)).argmax(1).cpu().tolist()); actual.extend(y.tolist())
        job.update(status="completed",testMetrics={"accuracy":sum(a==b for a,b in zip(actual,predictions))/len(actual)},updatedAt=utc_now(),message="Training, validation, and held-out testing completed."); save_classifier_job(job)
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
        seen=set(); invalid=[]; valid=0; groups={}
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
                out=target/("1_container" if label=="1" else "2_containers")/basename; out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(payload); groups[basename]=row.get("source_image_path") or image; valid+=1
            except Exception as error: invalid.append({"image":image,"error":str(error)[:180]})
        counts={name:len(list((target/name).glob("*"))) for name in ("1_container","2_containers")}
        if not valid: raise ValueError("No readable labeled images found")
        meta={"datasetId":dsid,"path":str(target),"imageCount":valid,"classCounts":counts,"invalid":invalid}
        (target/"import.json").write_text(json.dumps({**meta,"invalidCount":len(invalid)},indent=2),encoding="utf-8"); (target/"groups.json").write_text(json.dumps(groups),encoding="utf-8")
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
        rows=connection.execute("""SELECT a.path,a.source_path,a.cached_crop_path,c.container_count
          FROM text_annotations a JOIN container_counts c ON c.session_id=a.session_id AND c.path=a.path
          WHERE a.session_id=? AND a.active=1 AND a.cached_crop_path IS NOT NULL AND c.container_count IN (1,2)
          ORDER BY a.path""",(session["id"],)).fetchall()
        connection.close()
        dsid=uuid.uuid4().hex; target=classifier_root()/"datasets"/dsid; groups={}; invalid=[]; counts={"1_container":0,"2_containers":0}
        for row in rows:
            source=(DATASET_ROOT/row["cached_crop_path"]).resolve()
            if not source.is_relative_to(DATASET_ROOT) or not source.is_file():
                invalid.append({"image":row["cached_crop_path"],"error":"missing or unsafe crop path"}); continue
            label="1_container" if row["container_count"]==1 else "2_containers"
            filename=f"{hashlib.sha256(row['path'].encode()).hexdigest()[:12]}-{source.name}"
            output=target/label/filename; output.parent.mkdir(parents=True,exist_ok=True); output.symlink_to(source)
            groups[filename]=row["source_path"] or row["path"]; counts[label]+=1
        total=sum(counts.values())
        if not total: raise ValueError("No readable container crops found for the selected labeling session")
        meta={"datasetId":dsid,"path":str(target),"imageCount":total,"classCounts":counts,"invalid":invalid,"source":"Container-labeler","sessionId":session["id"],"sessionName":session["root_name"]}
        (target/"groups.json").write_text(json.dumps(groups),encoding="utf-8")
        (target/"import.json").write_text(json.dumps({**meta,"invalidCount":len(invalid)},indent=2),encoding="utf-8")
        return meta
    except HTTPException: raise
    except Exception as error: raise HTTPException(400,f"Could not import Container-labeler labels: {error}") from error


@app.post("/api/classifiers/train")
def start_classifier_training(request: dict[str,Any]):
    global _active_classifier
    if OBSERVABILITY_ONLY: raise HTTPException(403,"This observability instance is read-only.")
    with _lock:
        if _active_classifier: raise HTTPException(409,"A classifier training run is already active")
        dataset_id=str(request.get("datasetId",""))
        if not re.fullmatch(r"[a-f0-9]{32}",dataset_id): raise HTTPException(400,"Import a labeled ZIP dataset first")
        meta=classifier_root()/"datasets"/dataset_id/"import.json"
        if not meta.is_file(): raise HTTPException(404,"Imported dataset not found")
        try: cfg=json.loads(meta.read_text()); epochs=max(1,min(200,int(request.get("epochs",30)))); batch=max(1,min(64,int(request.get("batchSize",16)))); size=max(96,min(512,int(request.get("imageSize",224)))); lr=float(request.get("learningRate",.001))
        except Exception as e: raise HTTPException(400,"Invalid training configuration") from e
        if not .000001<=lr<=.1: raise HTTPException(400,"Learning rate must be between 0.000001 and 0.1")
        architecture=request.get("architecture","mobilenet_v3_small")
        if not isinstance(architecture,str) or architecture not in {"mobilenet_v3_small","mobilenet_v3_large"}: raise HTTPException(400,"Unsupported architecture")
        initial_checkpoint=None
        checkpoint_run_id=request.get("checkpointRunId")
        checkpoint_kind=request.get("checkpointKind","best")
        if checkpoint_run_id:
            if not re.fullmatch(r"[a-f0-9]{32}",str(checkpoint_run_id)) or not isinstance(checkpoint_kind,str) or checkpoint_kind not in {"best","last"}: raise HTTPException(400,"Choose a valid saved classifier checkpoint")
            source=next((row for row in classifier_runs() if row.get("id")==checkpoint_run_id and row.get("status")=="completed"),None)
            if not source: raise HTTPException(404,"Completed classifier run not found")
            checkpoint_path=source.get("bestPath" if checkpoint_kind=="best" else "lastPath")
            if not checkpoint_path or not Path(checkpoint_path).is_file(): raise HTTPException(404,"Classifier checkpoint not found")
            if source.get("architecture")!=architecture: raise HTTPException(400,"Choose the same architecture as the selected checkpoint")
            initial_checkpoint={"runId":checkpoint_run_id,"kind":checkpoint_kind}
        run_id=uuid.uuid4().hex; job={"id":run_id,"status":"queued","createdAt":utc_now(),"updatedAt":utc_now(),"datasetPath":cfg["path"],"datasetId":dataset_id,"datasetSize":cfg["imageCount"],"classCounts":cfg["classCounts"],"epochs":epochs,"batchSize":batch,"imageSize":size,"learningRate":lr,"architecture":architecture,"device":request.get("device","cpu"),"epoch":0,"initialCheckpoint":initial_checkpoint}
        if initial_checkpoint: job["checkpointSource"]={"runId":initial_checkpoint["runId"],"kind":initial_checkpoint["kind"]}
        save_classifier_job(job); _active_classifier=run_id; threading.Thread(target=classifier_worker,args=(run_id,),daemon=True).start(); return job


@app.get("/api/classifiers/runs/{run_id}")
def get_classifier_run(run_id: str):
    path=classifier_root()/"jobs"/run_id/"job.json"
    if not re.fullmatch(r"[a-f0-9]{32}",run_id) or not path.is_file(): raise HTTPException(404,"Classifier run not found")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/classifiers/runs/{run_id}/stop")
def stop_classifier_run(run_id: str):
    job=get_classifier_run(run_id)
    if job["status"] not in ACTIVE_TRAINING_STATUSES: raise HTTPException(409,"Classifier run is not active")
    job["stopRequested"]=True; save_classifier_job(job); return {"ok":True}


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


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), modelId: str | None = None, classifierId: str | None = None, confidence: float = 0.25):
    suffix = Path(file.filename or "image.jpg").suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(400, "Upload a JPG, PNG, WebP, BMP or TIFF image")
    if not 0.01 <= confidence <= 0.99:
        raise HTTPException(400, "Confidence must be between 0.01 and 0.99")
    contents = await file.read(20 * 1024 * 1024 + 1)
    if not contents or len(contents) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image is empty or larger than 20 MB")
    try:
        from PIL import Image
        import io
        with Image.open(io.BytesIO(contents)) as image:
            width, height = image.size
            image.verify()
    except Exception as error:
        raise HTTPException(400, "Uploaded file is not a readable image") from error
    checkpoint, model = selected_model(modelId)
    with tempfile.NamedTemporaryFile(suffix=suffix) as temporary:
        temporary.write(contents)
        temporary.flush()
        try:
            results = model.predict(source=temporary.name, conf=confidence, device="cpu", verbose=False)
        except Exception as error:
            raise HTTPException(500, f"Detection failed: {str(error)[:500]}") from error
    boxes = []
    if results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "confidence": float(box.conf[0]), "className": "container"})
    response = {"width": width, "height": height, "boxes": boxes, "model": Path(checkpoint).name, "count": len(boxes), "containerDetected": bool(boxes)}
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
            with Image.open(io.BytesIO(contents)) as source:
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
    return response


@app.post("/api/training")
def start_training(request: TrainRequest):
    global _active_training
    if OBSERVABILITY_ONLY:
        raise HTTPException(403, "This observability instance is read-only. Start training from the main app after the current run finishes.")
    with _lock:
        if _active_training:
            raise HTTPException(409, "A training job is already active")
        if request.initMode == "base" and request.checkpointId:
            raise HTTPException(400, "Select Fine-tune or Resume to use a checkpoint")
        if request.initMode in {"finetune", "resume"}:
            if not request.checkpointId:
                raise HTTPException(400, "Choose a saved or uploaded .pt checkpoint first")
            _, checkpoint = resolve_checkpoint(request.checkpointId)
            if request.initMode == "resume" and not checkpoint.get("resumeAllowed"):
                raise HTTPException(400, "This checkpoint cannot resume training. Choose Fine-tune to start a new run from its weights.")
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "status": "queued", "createdAt": utc_now(), "updatedAt": utc_now(), "epoch": 0, "totalEpochs": request.epochs, "epochs": request.epochs, "imageSize": request.imageSize, "batch": request.batch, "baseModel": request.baseModel, "initMode": request.initMode, "checkpointId": request.checkpointId, "message": "Training queued."}
        save_job(job)
        _active_training = job_id
        threading.Thread(target=train_worker, args=(job_id, request), daemon=True).start()
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
    global _active_training
    if OBSERVABILITY_ONLY:
        return
    jobs_dir = RUNS_DIR / "jobs"
    if not jobs_dir.is_dir():
        return
    for metadata in jobs_dir.glob("*/job.json"):
        try:
            job = json.loads(metadata.read_text(encoding="utf-8"))
            if job.get("status") in {"queued", "preparing", "running", "evaluating"}:
                interrupted_at = utc_now()
                job.update(status="interrupted", updatedAt=interrupted_at, finishedAt=interrupted_at, message="Server restarted during training. A saved last checkpoint may be available for a future resume feature.")
                append_job_event(job, "warning", "Server restarted while the run was active; status changed to interrupted.", "recovery")
                save_job(job)
        except (OSError, ValueError):
            continue
    _active_training = None
    classifier_jobs = classifier_root() / "jobs"
    if classifier_jobs.is_dir():
        for metadata in classifier_jobs.glob("*/job.json"):
            try:
                classifier = json.loads(metadata.read_text(encoding="utf-8"))
                if classifier.get("status") in ACTIVE_TRAINING_STATUSES:
                    classifier.update(status="interrupted", updatedAt=utc_now(), message="Classifier worker stopped when the server process restarted.")
                    save_classifier_job(classifier)
            except (OSError, ValueError, KeyError, TypeError):
                continue
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
