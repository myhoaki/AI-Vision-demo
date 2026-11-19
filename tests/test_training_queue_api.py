import json
from pathlib import Path

import pytest

from app import main
from app.yolo_resume import prepare_resume_checkpoint


def test_yolo_submission_only_creates_queued_job(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(main, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(main, "OBSERVABILITY_ONLY", False)
    monkeypatch.setattr(main.threading, "Thread", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("web attempted to start a thread")))
    submitted = main.start_training(main.TrainRequest(epochs=2, imageSize=640, batch=4))
    saved = json.loads((tmp_path / "runs" / "jobs" / submitted["id"] / "job.json").read_text())
    assert submitted["status"] == "queued"
    assert saved["modelType"] == "yolo"
    assert saved["request"]["epochs"] == 2


def test_classifier_submission_only_creates_queued_job(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(main, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(main, "OBSERVABILITY_ONLY", False)
    dataset_id = "a" * 32
    dataset_dir = tmp_path / "runs" / "classifiers" / "datasets" / dataset_id
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "import.json").write_text(json.dumps({"path": str(dataset_dir), "imageCount": 12,
                                                            "classCounts": {"1_container": 6, "2_containers": 6}}))
    monkeypatch.setattr(main.threading, "Thread", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("web attempted to start a thread")))
    submitted = main.start_classifier_training({"datasetSource": "imported", "datasetId": dataset_id,
                                                "architecture": "mobilenet_v3_small", "epochs": 2})
    saved = json.loads((tmp_path / "runs" / "classifiers" / "jobs" / submitted["id"] / "job.json").read_text())
    assert submitted["status"] == "queued"
    assert saved["modelType"] == "classifier"
    assert saved["request"]["epochs"] == 2


def test_web_startup_preserves_worker_owned_active_status(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs"
    job_dir = runs / "jobs" / ("b" * 32)
    job_dir.mkdir(parents=True)
    metadata = job_dir / "job.json"
    original = {"id": "b" * 32, "status": "running", "epoch": 2, "epochMetrics": []}
    metadata.write_text(json.dumps(original))
    monkeypatch.setattr(main, "RUNS_DIR", runs)
    monkeypatch.setattr(main, "OBSERVABILITY_ONLY", False)
    main.recover_jobs()
    assert json.loads(metadata.read_text()) == original


def test_classifier_resume_requires_versioned_checkpoint(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs"
    job_dir = runs / "classifiers" / "jobs" / ("c" * 32)
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps({"id": "c" * 32, "status": "stopped", "epoch": 2,
                                                    "epochs": 5, "architecture": "mobilenet_v3_small"}))
    monkeypatch.setattr(main, "RUNS_DIR", runs)
    monkeypatch.setattr(main, "OBSERVABILITY_ONLY", False)
    with pytest.raises(main.HTTPException) as error:
        main.start_classifier_training({"resumeRunId": "c" * 32})
    assert error.value.status_code == 409


def test_classifier_resume_creates_new_job_without_changing_source(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs"
    run_id = "e" * 32
    source_dir = runs / "classifiers" / "jobs" / run_id
    source_dir.mkdir(parents=True)
    source_job = {"id": run_id, "status": "stopped", "epoch": 1, "epochs": 2,
                  "architecture": "mobilenet_v3_small", "datasetSource": "imported",
                  "datasetId": "f" * 32, "batchSize": 2, "imageSize": 224,
                  "learningRate": 0.001, "device": "cpu", "balancingStrategy": "none",
                  "targetDistribution": "70:30"}
    source = source_dir / "job.json"
    original_bytes = json.dumps(source_job).encode()
    source.write_bytes(original_bytes)
    (source_dir / "resume_state.pt").write_bytes(b"test fixture; route validates existence only")
    dataset_dir = runs / "classifiers" / "datasets" / source_job["datasetId"]
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "import.json").write_text(json.dumps({"path": str(dataset_dir), "imageCount": 4}))
    monkeypatch.setattr(main, "RUNS_DIR", runs)
    monkeypatch.setattr(main, "OBSERVABILITY_ONLY", False)
    resumed = main.start_classifier_training({"resumeRunId": run_id})
    assert resumed["id"] != run_id and resumed["status"] == "queued"
    assert resumed["resumeFromRunId"] == run_id
    assert source.read_bytes() == original_bytes
    assert (runs / "classifiers" / "jobs" / resumed["id"] / "job.json").is_file()


def test_yolo_resume_checkpoint_is_cloned_and_redirected(tmp_path: Path):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    source = tmp_path / "source.pt"
    torch.save({"train_args": {"project": "/old/run/results", "name": "container", "exist_ok": True},
                "ema": SimpleNamespace(args=SimpleNamespace(project="/old/run/results", name="container", exist_ok=True)),
                "model": None}, source)
    original = source.read_bytes()
    destination = tmp_path / "new" / "results" / "container" / "weights" / "last.pt"
    prepare_resume_checkpoint(source, destination, destination.parents[2], "container")
    resumed = torch.load(destination, map_location="cpu", weights_only=False)
    assert source.read_bytes() == original
    assert resumed["train_args"]["project"] == str(destination.parents[2].resolve())
    assert resumed["ema"].args.project == str(destination.parents[2].resolve())
    with pytest.raises(FileExistsError):
        prepare_resume_checkpoint(source, destination, destination.parents[2], "container")
