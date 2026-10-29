# AI-VIsion-demo

A standalone container detection and training workspace inspired by the PlateLens project structure. It provides a browser UI, FastAPI backend, YOLO26 container training, model-backed image inference, and dataset browsing.

## What it does

- **Detect:** upload a JPG/PNG/WebP/TIFF image or choose one from the server dataset. The latest completed container checkpoint predicts boxes and confidence values on the CPU.
- **Dataset:** browse the existing YOLO dataset in a paged gallery. Images and labels are mounted read-only.
- **Train model:** train a YOLO26n/s/m checkpoint on the container annotations. Epoch, image size, and batch size are configurable. Training runs in a background thread; live train/validation batch progress and per-epoch metrics are saved, then the held-out test split runs once after training. Run details include metric charts, the effective Ultralytics arguments, runtime versions, dataset fingerprint and label diagnostics, elapsed time, resource samples, checkpoint lineage, and bounded events. Progress and checkpoints are saved under `runs/jobs/`.
- **Continue training:** start a new fine-tune run from a saved `best.pt`/`last.pt` or an uploaded trusted YOLO `.pt` checkpoint. Resume an interrupted run from its `last.pt` to restore optimizer and epoch state. Fine-tuning starts a separate run and leaves the source checkpoint intact.
- **Export models:** download saved `.pt` checkpoints and export any saved/uploaded weights to ONNX. ONNX files are validated and saved under `runs/exports/` for download.
- **Leakage-aware split:** source images and `_cropped` derivatives are assigned to the same train/validation/test split. Source images are symlinked into each run snapshot (no second copy of the 21 GB dataset); labels and a `data.yaml` with an explicit absolute snapshot path are written in the run directory so Ultralytics does not resolve paths against its global dataset setting.

The pretrained COCO checkpoint is only a training starting point; it has no dedicated shipping-container class. Detection is enabled after a container-specific run completes. The first training run downloads the chosen YOLO26 starting weights, so the server needs internet access at that time. Training and inference currently use CPU. The image includes OpenCV's small required system libraries for its headless server runtime.

## Run as a separate Docker app

This Compose project uses host port `8768`, separate from the labeler (`8766`), dataset audit (`8767`), and PlateLens (`8000`). It reads the dataset at `../container-labeler/data/server_dataset/container_images_verified` read-only and writes checkpoints only into `./runs/`.

The read-only training observer is available on port `8769` by default. It shares the saved run files read-only and can show persisted metrics by epoch without restarting or changing the active training service. Open `http://<server-ip>:8769` to monitor training safely.

From the project folder on the server:

```bash
cd /home/bovanmy/AI-VIsion-demo
docker compose up -d --build
docker compose ps
```

The service publishes port `8768` on the server interfaces, like PlateLens on port `8000`. From your laptop, visit **http://your-server-address:8768** in Chrome or Edge. The server firewall/network must allow inbound TCP traffic on port `8768`. You can still use an SSH tunnel if direct access is unavailable.

Check logs with `docker compose logs -f ai-vision-demo`. Stop only this app with `docker compose down`. Training outputs are in `runs/jobs/<job-id>/`; avoid deleting a run directory if you want to keep its checkpoint.

## Dataset format

The configured dataset is expected to have `images/` and `labels/` folders, with YOLO `.txt` labels matched by relative image path or filename stem. It uses one class, class `0: container`, with normalized `class_id center_x center_y width height` rows. Empty label files are treated as negative examples. Images with missing or invalid label rows are excluded and counted in the training progress. This project does not edit the source dataset or save uploaded detection images. Uploaded `.pt` checkpoints are kept under `runs/checkpoints/`; upload only trusted PyTorch/Ultralytics checkpoints because loading them may execute serialized code.

## Runtime endpoints

- `GET /api/health`
- `GET /api/dataset` and `GET /api/dataset/images`
- `POST /api/detect` (`multipart/form-data`, field `file`)
- `GET /api/models`
- `POST /api/checkpoints/upload`, `GET /api/checkpoints/{checkpoint_id}/download`
- `POST /api/training`, `GET /api/training/latest`, `GET /api/training/{job_id}`, `GET /api/training/{job_id}/metrics`, `GET /api/training/{job_id}/logs`
- `POST /api/checkpoints/{checkpoint_id}/export/onnx`, `GET /api/exports`, `GET /api/exports/{export_id}/download`

Training and export job state is written atomically to JSON. Epoch history is also recoverable from Ultralytics `results.csv`; on service startup, readable CSV history is backfilled into terminal job records without modifying active jobs. Historical runs show fields that were never captured as unavailable. If the server restarts mid-training, the run is marked interrupted and its `last.pt` is available for resume when it exists. Dataset diagnostics skip exact-image hashing to avoid another full pass over the source images; split leakage checks group by filename stem and remove a terminal `_cropped` suffix.

ONNX export follows Ultralytics' documented `model.export(format="onnx")` flow. Training resume uses `last.pt` with `resume=True`, which restores the optimizer and epoch state; use fine-tune for completed checkpoints or uploaded pretrained `.pt` files. [Ultralytics train/resume docs](https://docs.ultralytics.com/modes/train), [Ultralytics ONNX export docs](https://docs.ultralytics.com/modes/export).
