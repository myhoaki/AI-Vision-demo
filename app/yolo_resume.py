"""Create a non-destructive Ultralytics resume checkpoint in a new run directory."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any


def _set_arg_container(container: Any, name: str, value: Any) -> None:
    if isinstance(container, dict):
        container[name] = value
    elif container is not None and hasattr(container, name):
        setattr(container, name, value)


def prepare_resume_checkpoint(source: Path, destination: Path, project: Path, name: str) -> Path:
    """Clone a checkpoint and redirect its saved trainer arguments to a new output.

    Ultralytics restores ``project`` and ``name`` from the checkpoint's EMA model
    arguments when ``resume=True``. Updating only call arguments is insufficient
    and can overwrite the source experiment.
    """
    import torch

    source, destination, project = source.resolve(), destination.resolve(), project.resolve()
    if source == destination:
        raise ValueError("A YOLO resume checkpoint must be written to a new run directory.")
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists():
        raise FileExistsError(destination)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Unsupported YOLO checkpoint format.")
    train_args = checkpoint.get("train_args")
    if not isinstance(train_args, dict):
        raise ValueError("YOLO checkpoint is missing saved training arguments.")
    output_dir = project / name
    train_args.update(project=str(project), name=name, exist_ok=True, save_dir=str(output_dir))
    ema = checkpoint.get("ema")
    ema_args = getattr(ema, "args", None)
    if ema_args is None:
        raise ValueError("YOLO checkpoint is missing EMA training arguments required for resume.")
    for key, value in (("project", str(project)), ("name", name), ("exist_ok", True), ("save_dir", str(output_dir))):
        _set_arg_container(ema_args, key, value)
    model = checkpoint.get("model")
    model_args = getattr(model, "args", None)
    if model_args is not None:
        for key, value in (("project", str(project)), ("name", name), ("exist_ok", True), ("save_dir", str(output_dir))):
            _set_arg_container(model_args, key, value)

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        torch.save(checkpoint, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        # Avoid replacing any output if a duplicate destination appeared.
        os.link(temporary, destination)
        dir_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
