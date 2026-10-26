from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import threading
import time
import uuid
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
    temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def checkpoint_models() -> list[dict[str, Any]]:
    models = []
    jobs_dir = RUNS_DIR / "jobs"
    if not jobs_dir.is_dir():
        return models
    for metadata in jobs_dir.glob("*/job.json"):
        try:
            job = json.loads(metadata.read_text(encoding="utf-8"))
            best = metadata.parent / "results" / "container" / "weights" / "best.pt"
            if best.is_file() and job.get("status") == "completed":
                models.append({"id": job["id"], "name": f"Container {Path(job.get('baseModel', 'yolo26n.pt')).stem.upper()} · {job['id'][:8]}", "path": str(best), "createdAt": job.get("updatedAt"), "epochs": job.get("epochs"), "metrics": job.get("metrics", {})})
        except (OSError, ValueError, KeyError):
            continue
    return sorted(models, key=lambda model: model.get("createdAt") or "", reverse=True)


def list_checkpoints() -> list[dict[str, Any]]:
    checkpoints: list[dict[str, Any]] = []
    jobs_dir = RUNS_DIR / "jobs"
    if jobs_dir.is_dir():
        for metadata in jobs_dir.glob("*/job.json"):
            try:
                job = json.loads(metadata.read_text(encoding="utf-8"))
                job_id = str(job["id"])
                weight_dir = metadata.parent / "results" / "container" / "weights"
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
                        "fineTuneAllowed": True,
                        "resumeAllowed": kind == "last" and job.get("status") in {"failed", "interrupted"},
                        "createdAt": job.get("updatedAt"),
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
    path = RUNS_DIR / "jobs" / job_id / "results" / "container" / "weights" / f"{kind}.pt"
    if not path.is_file():
        raise HTTPException(404, "Training checkpoint not found")
    descriptor = {"id": checkpoint_id, "kind": kind, "jobStatus": job.get("status"), "resumeAllowed": kind == "last" and job.get("status") in {"failed", "interrupted"}}
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


def validate_label(path: Path) -> int:
    if not path.is_file():
        return 0
    count = 0
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
        count += 1
    return count


def build_training_dataset(job_id: str) -> tuple[Path, dict[str, int], str, int]:
    image_root, labels_root = DATASET_ROOT / "images", DATASET_ROOT / "labels"
    if not image_root.is_dir() or not labels_root.is_dir():
        raise ValueError("Expected a YOLO dataset with images/ and labels/ folders.")
    images = sorted((path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED), key=lambda path: path.relative_to(image_root).as_posix().casefold())
    if len(images) < 3:
        raise ValueError("At least three supported images are required to create train/validation/test splits.")
    groups: dict[str, list[Path]] = {}
    boxes, skipped, fingerprint = 0, 0, hashlib.sha256()
    for image in images:
        label = label_for(image, image_root, labels_root)
        try:
            image_boxes = validate_label(label)
        except (OSError, ValueError):
            skipped += 1
            continue
        if not label.is_file():
            skipped += 1
            continue
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
    return yaml_path, counts, fingerprint.hexdigest(), skipped


def train_worker(job_id: str, request: TrainRequest) -> None:
    global _active_training
    job = read_job(job_id)
    try:
        job.update(status="preparing", updatedAt=utc_now(), message="Validating labels and creating leakage-aware dataset splits.")
        save_job(job)
        yaml_path, split_counts, fingerprint, skipped_images = build_training_dataset(job_id)
        job.update(status="running", updatedAt=utc_now(), message=f"Training on CPU. {skipped_images} images with missing or invalid labels were excluded. The first run downloads YOLO26n weights.", splitCounts=split_counts, skippedImages=skipped_images, datasetFingerprint=fingerprint)
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
        initial_model = str(checkpoint_path) if checkpoint_path else request.baseModel
        model = YOLO(initial_model)
        job_dir = RUNS_DIR / "jobs" / job_id
        if request.initMode == "base":
            start_message = f"Starting from {request.baseModel}; its weights download if they are not cached."
        elif request.initMode == "resume":
            start_message = f"Resuming optimizer and epoch state from {checkpoint_path.name}."
        else:
            start_message = f"Fine-tuning weights from {checkpoint_path.name} as a new training run."
        job["message"] = start_message
        job["initialModel"] = str(checkpoint_path) if checkpoint_path else request.baseModel
        save_job(job)

        last_progress_save = 0.0

        def update_phase(phase: str, current: int, total: int, epoch: int | None = None, force: bool = False) -> None:
            nonlocal last_progress_save
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
            if force or now - last_progress_save >= 3.0 or (total > 0 and current >= total):
                save_job(job)
                last_progress_save = now

        def on_train_epoch_start(trainer: Any) -> None:
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            job.update(epoch=epoch, currentStage="train")
            phases = job.setdefault("phaseProgress", {})
            phases["train"] = {"epoch": epoch, "current": 0, "total": len(getattr(trainer, "train_loader", [])), "status": "running"}
            phases["validation"] = {"epoch": epoch, "current": 0, "total": 0, "status": "waiting"}
            phases.setdefault("test", {"epoch": 0, "current": 0, "total": 0, "status": "waiting"})
            update_phase("train", 0, len(getattr(trainer, "train_loader", [])), epoch, force=True)

        def on_train_batch_end(trainer: Any) -> None:
            total = len(getattr(trainer, "train_loader", []))
            update_phase("train", int(getattr(trainer, "batch_i", -1)) + 1, total, int(getattr(trainer, "epoch", 0)) + 1)

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
            metrics = {str(k): float(v) for k, v in getattr(trainer, "metrics", {}).items() if isinstance(v, (int, float))}
            job.update(epoch=int(getattr(trainer, "epoch", -1)) + 1, totalEpochs=request.epochs, metrics=metrics, updatedAt=utc_now())
            save_job(job)

        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_train_batch_end", on_train_batch_end)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_val_start", on_val_start)
        model.add_callback("on_val_batch_end", on_val_batch_end)
        model.add_callback("on_val_end", on_val_end)
        model.add_callback("on_fit_epoch_end", on_epoch_end)
        model.train(data=str(yaml_path), epochs=request.epochs, imgsz=request.imageSize, batch=request.batch, device="cpu", workers=0, cache=False, amp=False, project=str(job_dir / "results"), name="container", exist_ok=True, plots=True, save=True, patience=20, resume=request.initMode == "resume")
        best = job_dir / "results" / "container" / "weights" / "best.pt"
        if not best.is_file():
            raise RuntimeError("Training ended without producing weights/best.pt")
        job.update(status="evaluating", evaluatingTest=True, updatedAt=utc_now(), message="Training finished. Evaluating the held-out test split once.", checkpoint=str(best))
        save_job(job)
        test_model = YOLO(str(best))
        test_model.add_callback("on_val_start", on_val_start)
        test_model.add_callback("on_val_batch_end", on_val_batch_end)
        test_model.add_callback("on_val_end", on_val_end)
        test_result = test_model.val(data=str(yaml_path), split="test", imgsz=request.imageSize, batch=request.batch, device="cpu", workers=0, plots=True, verbose=False)
        test_metrics = {str(key): float(value) for key, value in getattr(test_result, "results_dict", {}).items() if isinstance(value, (int, float))}
        job.update(status="completed", evaluatingTest=False, currentStage="complete", updatedAt=utc_now(), message="Training and final test evaluation completed. The best checkpoint is ready for detection.", checkpoint=str(best), epoch=request.epochs, testMetrics=test_metrics)
        save_job(job)
    except Exception as error:
        job.update(status="failed", updatedAt=utc_now(), message=str(error)[:1200])
        save_job(job)
    finally:
        with _lock:
            _active_training = None


@app.get("/api/health")
def health():
    return {"status": "ok", "project": "AI-VIsion-demo"}


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


@app.get("/api/models")
def models():
    return {"models": checkpoint_models(), "checkpoints": list_checkpoints()}


@app.post("/api/checkpoints/upload")
async def upload_checkpoint(file: UploadFile = File(...)):
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
async def detect(file: UploadFile = File(...), modelId: str | None = None, confidence: float = 0.25):
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
    return {"width": width, "height": height, "boxes": boxes, "model": Path(checkpoint).name, "count": len(boxes)}


@app.post("/api/training")
def start_training(request: TrainRequest):
    global _active_training
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


@app.on_event("startup")
def recover_jobs():
    global _active_training
    jobs_dir = RUNS_DIR / "jobs"
    if not jobs_dir.is_dir():
        return
    for metadata in jobs_dir.glob("*/job.json"):
        try:
            job = json.loads(metadata.read_text(encoding="utf-8"))
            if job.get("status") in {"queued", "preparing", "running", "evaluating"}:
                job.update(status="interrupted", updatedAt=utc_now(), message="Server restarted during training. A saved last checkpoint may be available for a future resume feature.")
                save_job(job)
        except (OSError, ValueError):
            continue
    _active_training = None
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
