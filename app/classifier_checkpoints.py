"""Versioned MobileNetV3 training checkpoints with explicit resume state."""
from __future__ import annotations

import os
import random
import uuid
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def save_model_checkpoint(payload: dict[str, Any], destination: Path) -> None:
    import torch

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        try:
            dir_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    finally:
        temporary.unlink(missing_ok=True)


def build_resume_state(*, model: Any, optimizer: Any, scheduler: Any, epoch: int,
                       best_metric: float, history: list[dict[str, Any]], architecture: str,
                       classes: list[str], image_size: int, training_config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch

    model_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    return {
        "schemaVersion": SCHEMA_VERSION,
        "modelState": model_state,
        "optimizerState": optimizer.state_dict(),
        "schedulerState": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "bestMetric": float(best_metric),
        "history": list(history),
        "architecture": architecture,
        "classes": list(classes),
        "imageSize": int(image_size),
        "trainingConfig": dict(training_config),
        "rngStates": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def load_resume_state(path: Path) -> dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schemaVersion") != SCHEMA_VERSION:
        raise ValueError("Unsupported or invalid classifier resume checkpoint schema.")
    required = {"modelState", "optimizerState", "epoch", "bestMetric", "history", "architecture", "classes", "trainingConfig", "rngStates"}
    if not required.issubset(payload):
        raise ValueError("Classifier resume checkpoint is missing required state.")
    return payload


def restore_rng_states(states: dict[str, Any]) -> None:
    import numpy as np
    import torch

    if states.get("python") is not None:
        random.setstate(states["python"])
    if states.get("numpy") is not None:
        np.random.set_state(states["numpy"])
    if states.get("torch") is not None:
        torch.set_rng_state(states["torch"])
    if torch.cuda.is_available() and states.get("cuda") is not None:
        torch.cuda.set_rng_state_all(states["cuda"])
