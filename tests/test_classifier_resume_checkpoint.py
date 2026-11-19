from pathlib import Path
import json
import hashlib

import torch
from PIL import Image

from app import main
from app.classifier_checkpoints import build_resume_state, load_resume_state, restore_rng_states, save_model_checkpoint
from app.job_queue import claim_pipeline, release_claim
from worker import main as worker_main
from app.main import classifier_checkpoint_state


def test_classifier_resume_checkpoint_restores_model_optimizer_and_epoch(tmp_path: Path):
    torch.manual_seed(23)
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    inputs = torch.randn(3, 4)
    model(inputs).sum().backward()
    optimizer.step()
    payload = build_resume_state(
        model=model, optimizer=optimizer, scheduler=None, epoch=4, best_metric=0.75,
        history=[{"epoch": 4, "accuracy": 0.75}], architecture="mobilenet_v3_small",
        classes=["1_container", "2_containers"], image_size=224,
        training_config={"datasetId": "dataset-1", "epochs": 12},
    )
    path = tmp_path / "resume_state.pt"
    save_model_checkpoint(payload, path)
    loaded = load_resume_state(path)
    copy = torch.nn.Linear(4, 2)
    copy.load_state_dict(loaded["modelState"])
    copy_optimizer = torch.optim.AdamW(copy.parameters(), lr=0.001)
    copy_optimizer.load_state_dict(loaded["optimizerState"])
    restore_rng_states(loaded["rngStates"])
    assert loaded["epoch"] == 4
    assert loaded["bestMetric"] == 0.75
    assert loaded["schedulerState"] is None
    assert loaded["trainingConfig"]["datasetId"] == "dataset-1"
    assert path.is_file()


def test_classifier_resume_rejects_unknown_schema(tmp_path: Path):
    path = tmp_path / "bad.pt"
    save_model_checkpoint({"schemaVersion": 100}, path)
    try:
        load_resume_state(path)
    except ValueError as error:
        assert "schema" in str(error).lower()
    else:
        raise AssertionError("unknown checkpoint schema was accepted")


def test_legacy_classifier_weight_checkpoint_remains_compatible():
    weight = torch.nn.Linear(4, 2).state_dict()
    payload = {"architecture": "mobilenet_v3_small", "classes": ["1_container", "2_containers"], "state_dict": weight}
    restored = classifier_checkpoint_state(payload, "mobilenet_v3_small", ["1_container", "2_containers"])
    assert set(restored) == set(weight)


def test_explicit_classifier_resume_trains_in_a_new_run(tmp_path: Path, monkeypatch):
    runs = tmp_path / "runs"
    dataset_id = "f" * 32
    dataset = runs / "classifiers" / "datasets" / dataset_id
    for class_index, name in enumerate(("1_container", "2_containers")):
        class_dir = dataset / name
        class_dir.mkdir(parents=True)
        for index in range(10):
            color = (40 + class_index * 100, 20 + index, 80)
            Image.new("RGB", (20, 20), color).save(class_dir / f"{name}-{index:02}.png")
    (dataset / "import.json").write_text(json.dumps({"path": str(dataset), "imageCount": 20}))
    monkeypatch.setattr(main, "RUNS_DIR", runs)
    monkeypatch.setattr(main, "DATASET_ROOT", tmp_path / "empty-dataset")
    monkeypatch.setattr(worker_main, "RUNS_DIR", runs)

    architecture = "mobilenet_v3_small"
    classes = ["1_container", "2_containers"]
    from torchvision import models
    from torch import nn
    net = models.mobilenet_v3_small(weights=None)
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 2)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001)
    source_id = "a" * 32
    source_dir = runs / "classifiers" / "jobs" / source_id
    source_dir.mkdir(parents=True)
    source_job = {"id": source_id, "modelType": "classifier", "status": "stopped", "epoch": 0,
                  "epochs": 1, "batchSize": 8, "imageSize": 96, "learningRate": 0.001,
                  "architecture": architecture, "device": "cpu", "datasetSource": "imported",
                  "datasetId": dataset_id, "datasetPath": str(dataset), "balancingStrategy": "none",
                  "targetDistribution": "70:30"}
    source_metadata = source_dir / "job.json"
    source_metadata.write_text(json.dumps(source_job))
    state = build_resume_state(model=net, optimizer=optimizer, scheduler=None, epoch=0,
                               best_metric=0.0, history=[], architecture=architecture,
                               classes=classes, image_size=96,
                               training_config={"datasetId": dataset_id})
    source_checkpoint = source_dir / "resume_state.pt"
    save_model_checkpoint(state, source_checkpoint)
    source_hash = hashlib.sha256(source_checkpoint.read_bytes()).hexdigest()

    queued = main.start_classifier_training({"resumeRunId": source_id})
    new_id = queued["id"]
    assert new_id != source_id
    owner = claim_pipeline(runs, "classifier", worker_main.WORKER_ID)
    assert owner is not None
    try:
        worker_main.execute_job("classifier", runs / "classifiers" / "jobs" / new_id / "job.json", owner)
    finally:
        release_claim(runs, "classifier", worker_main.WORKER_ID)
    result = json.loads((runs / "classifiers" / "jobs" / new_id / "job.json").read_text())
    assert result["status"] == "completed", result
    assert result["epoch"] == 1
    assert (runs / "classifiers" / "jobs" / new_id / "resume_state.pt").is_file()
    assert source_metadata.read_text() == json.dumps(source_job)
    assert hashlib.sha256(source_checkpoint.read_bytes()).hexdigest() == source_hash
