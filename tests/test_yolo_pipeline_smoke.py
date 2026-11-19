"""One-epoch YOLO smoke test on generated data and a read-only seed checkpoint.

Set YOLO_TEST_CHECKPOINT to a completed local checkpoint. Test outputs are
created under pytest's temporary directory; the checkpoint source is never opened
for writing.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app import main
from app.job_queue import claim_pipeline, create_job, release_claim
from worker import main as worker_main


def test_yolo_container_detection_training_smoke(tmp_path: Path, monkeypatch):
    source_value = os.getenv("YOLO_TEST_CHECKPOINT")
    if not source_value:
        pytest.skip("Mount a completed YOLO checkpoint read-only and set YOLO_TEST_CHECKPOINT")
    source_checkpoint = Path(source_value)
    if not source_checkpoint.is_file():
        pytest.skip("Configured YOLO test checkpoint does not exist")
    source_hash = hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()
    # Canonical checkpoint layout: jobs/<run>/results/container/weights/best.pt
    source_run_id = source_checkpoint.parents[3].name
    source_metadata = source_checkpoint.parents[3] / "job.json"
    source_job = json.loads(source_metadata.read_text())

    runs = tmp_path / "runs"
    seed_run = runs / "jobs" / source_run_id
    seed_dir = seed_run / "results" / "container" / "weights"
    seed_dir.mkdir(parents=True)
    shutil.copy2(source_checkpoint, seed_dir / "best.pt")
    (seed_run / "job.json").write_text(json.dumps({**source_job, "status": "completed"}))

    dataset = tmp_path / "dataset"
    image_root, label_root = dataset / "images", dataset / "labels"
    image_root.mkdir(parents=True)
    label_root.mkdir(parents=True)
    for index in range(10):
        image = Image.new("RGB", (320, 320), (70 + index, 95, 110))
        draw = ImageDraw.Draw(image)
        x = 75 + index * 3
        draw.rectangle((x, 90, x + 120, 230), outline=(230, 230, 230), width=5)
        image.save(image_root / f"sample-{index:02}.jpg", quality=85)
        center_x = (x + 60) / 320
        (label_root / f"sample-{index:02}.txt").write_text(f"0 {center_x:.6f} 0.5 0.375 0.4375\n")

    monkeypatch.setattr(main, "RUNS_DIR", runs)
    monkeypatch.setattr(main, "DATASET_ROOT", dataset)
    monkeypatch.setattr(worker_main, "RUNS_DIR", runs)
    job_id = uuid.uuid4().hex
    request = main.TrainRequest(epochs=1, imageSize=320, batch=2, initMode="finetune",
                                checkpointId=f"{source_run_id}-best")
    create_job(runs, "yolo", {"id": job_id, "modelType": "yolo", "status": "queued",
                               "createdAt": main.utc_now(), "updatedAt": main.utc_now(),
                               "request": request.model_dump(), "epochs": 1})
    owner = claim_pipeline(runs, "yolo", worker_main.WORKER_ID)
    assert owner is not None
    try:
        worker_main.execute_job("yolo", runs / "jobs" / job_id / "job.json", owner)
    finally:
        release_claim(runs, "yolo", worker_main.WORKER_ID)
    result = json.loads((runs / "jobs" / job_id / "job.json").read_text())
    assert result["status"] == "completed", result
    assert result["epoch"] == 1
    assert Path(result["checkpoint"]).is_file()
    assert hashlib.sha256(source_checkpoint.read_bytes()).hexdigest() == source_hash
    assert (seed_dir / "best.pt").is_file()
