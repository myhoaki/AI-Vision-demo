# AI-VIsion-demo

A standalone container detection and training workspace inspired by the PlateLens project structure. It provides a browser UI, FastAPI backend, YOLO26 container training, model-backed image inference, and dataset browsing.

## What it does

- **Detect:** upload a JPG/PNG/WebP/TIFF image or choose one from the server dataset. The latest completed container checkpoint predicts boxes and confidence values on the CPU.
- **Inference:** run the supplied `Container-Inference/best.onnx` model with a multi-scale sliding-window detector and ONNX Runtime CPU using scales 1280,512, overlap 0.4, confidence 0.25, IOS 0.5, verticality 0.15, and a 10-detection limit. Overlapping tile hypotheses are merged with IOS; low-verticality boxes are filtered. Sliding windows are enabled by default and can be disabled for a full-image pass. The model is mounted read-only into the app container.
- **Dataset:** browse the existing YOLO dataset in a paged gallery. Images and labels are mounted read-only.
- **Train model:** submit YOLO26n/s/m work to the independent Training Worker. Epoch, image size, and batch size are configurable. Live train/validation progress and per-epoch metrics are saved, then the held-out test split runs once after training. Run details include metric charts, effective Ultralytics arguments, runtime versions, dataset fingerprint and label diagnostics, elapsed time, resource samples, checkpoint lineage, and bounded events. Progress and checkpoints are saved under `runs/jobs/`.
- **Continue training:** start a new fine-tune run from a saved `best.pt`/`last.pt` or an uploaded trusted YOLO `.pt` checkpoint. Explicit YOLO resume from `last.pt` restores Ultralytics optimizer and epoch state in a new run while preserving the source checkpoint.
- **Export models:** download saved `.pt` checkpoints and export any saved/uploaded weights to ONNX. ONNX files are validated and saved under `runs/exports/` for download.
- **Leakage-aware split:** source images and `_cropped` derivatives are assigned to the same train/validation/test split. Source images are symlinked into each run snapshot (no second copy of the 21 GB dataset); labels and a `data.yaml` with an explicit absolute snapshot path are written in the run directory so Ultralytics does not resolve paths against its global dataset setting.

The pretrained COCO checkpoint is only a training starting point; it has no dedicated shipping-container class. Detection is enabled after a container-specific run completes. The first training run downloads the chosen YOLO26 starting weights, so the server needs internet access at that time. Training and inference currently use CPU. The image includes OpenCV's small required system libraries for its headless server runtime.

## Run as a separate Docker app

The Web App and Training Worker use separate Compose projects and share `./runs/`. The Web App serves the UI, inference, dataset browsing, job submission, and monitoring; the worker alone executes YOLO and MobileNetV3 training. Rebuilding the web project never targets or recreates the worker project. The web service uses host port `8768`, separate from the labeler (`8766`), dataset audit (`8767`), and PlateLens (`8000`). Datasets and experiment sources are mounted read-only; job records, logs, checkpoints, and metrics are written under `./runs/`.

The read-only training observer is available on port `8769` by default. It shares the saved run files read-only and can show persisted metrics by epoch without restarting or changing the active training service. Open `http://<server-ip>:8769` to monitor training safely.

From the project folder on the server, start the worker and web app independently:

```bash
cd /home/bovanmy/AI-VIsion-demo
docker compose -p ai-vision-training -f docker-compose.training.yml up -d --build
docker compose -p ai-vision-web -f docker-compose.web.yml up -d --build
docker compose -p ai-vision-training -f docker-compose.training.yml ps
docker compose -p ai-vision-web -f docker-compose.web.yml ps
```

The service publishes port `8768` on the server interfaces, like PlateLens on port `8000`. From your laptop, visit **http://your-server-address:8768** in Chrome or Edge. The server firewall/network must allow inbound TCP traffic on port `8768`. You can still use an SSH tunnel if direct access is unavailable.

Check the app with `docker compose -p ai-vision-web -f docker-compose.web.yml logs -f web` and worker with `docker compose -p ai-vision-training -f docker-compose.training.yml logs -f training-worker`. Do not use broad `docker compose down --remove-orphans` across projects. Job data is in `runs/jobs/<job-id>/` and `runs/classifiers/jobs/<run-id>/`; never delete a run directory if you want to keep its history or checkpoints.

### Independent worker operations

Use only the Web App project for web updates. This leaves the worker project untouched:

```bash
docker compose -p ai-vision-web -f docker-compose.web.yml up -d --build --no-deps web
```

Inspect worker health and pipeline claims:

```bash
docker compose -p ai-vision-training -f docker-compose.training.yml ps
docker compose -p ai-vision-training -f docker-compose.training.yml logs --tail=100 training-worker
docker compose -p ai-vision-training -f docker-compose.training.yml exec -T training-worker python -m worker.control status
curl -fsS http://localhost:8768/api/worker/status
```

Job progress is available in the Training UI and at `/api/training/<job-id>` or `/api/classifiers/runs/<run-id>`. YOLO log/metric files are available from `/api/training/<job-id>/logs` and `/api/training/<job-id>/metrics`; classifier batch status and epoch history are in its run JSON. Graceful per-job cancellation is requested through the existing stop endpoints:

```bash
curl -X POST http://localhost:8768/api/training/<job-id>/stop
curl -X POST http://localhost:8768/api/classifiers/runs/<run-id>/stop
```

Stopping the worker service sends cancellation requests and waits up to 30 minutes for a safe epoch boundary. Avoid stopping it during a long-running job unless cancellation is intended:

```bash
docker compose -p ai-vision-training -f docker-compose.training.yml stop training-worker
```

YOLO resume remains an explicit new job using a resumable `last.pt` checkpoint and `initMode: "resume"`. New MobileNetV3 jobs save `resume_state.pt` at completed epoch boundaries; use **Resume in a new run** in the classifier history, or call:

```bash
curl -X POST http://localhost:8768/api/classifiers/runs/<stopped-run-id>/resume
```

The new classifier run preserves its source run and restores model, AdamW state, epoch, best validation accuracy, history, and random-number-generator states. No scheduler is currently used. Older classifier runs without `resume_state.pt` cannot be exact-resumed; they can still seed a new fine-tune from compatible `.pt` weights.

Claims are exclusive per pipeline and have heartbeats. A worker never reclaims a stale claim automatically. If the worker was forcibly killed, first verify its container is stopped, then inspect the claim and explicitly recover it:

```bash
docker compose -p ai-vision-training -f docker-compose.training.yml exec -T training-worker python -m worker.control status
# After independently confirming the owning worker container is stopped:
docker compose -p ai-vision-training -f docker-compose.training.yml run --rm --no-deps training-worker python -m worker.control release-stale --pipeline yolo --confirm-owner-stopped
```

Use `--pipeline classifier` for a classifier claim. Recovery archives the claim, marks its current job interrupted, and never starts that job again automatically. Explicitly resume it as a new run when supported.

To roll back the web app, restore/rebuild the desired web source or image and operate only on `ai-vision-web`; leave `ai-vision-training` running:

```bash
docker compose -p ai-vision-web -f docker-compose.web.yml up -d --build --no-deps web
```

Never use `docker compose down -v`, volume-prune commands, or commands that combine both project files for routine updates.

## Dataset format

The configured dataset is expected to have `images/` and `labels/` folders, with YOLO `.txt` labels matched by relative image path or filename stem. It uses one class, class `0: container`, with normalized `class_id center_x center_y width height` rows. Empty label files are treated as negative examples. Images with missing or invalid label rows are excluded and counted in the training progress. This project does not edit the source dataset or save uploaded detection images. Uploaded `.pt` checkpoints are kept under `runs/checkpoints/`; upload only trusted PyTorch/Ultralytics checkpoints because loading them may execute serialized code.

## Runtime endpoints

- `GET /api/health`
- `GET /api/dataset` and `GET /api/dataset/images`
- `POST /api/detect` (`multipart/form-data`, field `file`)
- `GET /api/inference/status`, `POST /api/inference` (`multipart/form-data`, field `file`; optional `confidence`, `scales`, `overlap`, `ios`, `verticality`, `max_det`, `use_slicing`)
- `GET /api/models`
- `POST /api/checkpoints/upload`, `GET /api/checkpoints/{checkpoint_id}/download`
- `POST /api/training` (persistent queue submission), `POST /api/training/{job_id}/stop` (cancellation marker), `GET /api/training/latest`, `GET /api/training/{job_id}`, `GET /api/training/{job_id}/metrics`, `GET /api/training/{job_id}/logs`
- `POST /api/classifiers/train` (persistent queue submission), `POST /api/classifiers/runs/{run_id}/stop`, `POST /api/classifiers/runs/{run_id}/resume`, `GET /api/worker/status`
- `POST /api/checkpoints/{checkpoint_id}/export/onnx`, `GET /api/exports`, `GET /api/exports/{export_id}/download`

Training job state is atomically written to JSON in the shared run mount. The web app creates queued records and reads status; the worker owns active job metadata. Pipeline claim files under `runs/worker/claims/` prevent duplicate execution across worker containers. Heartbeat delays never cause automatic claim theft. A confirmed stale claim is recovered explicitly and its job becomes interrupted. Epoch history is also recoverable from Ultralytics `results.csv`; web startup backfills only terminal job records. Historical runs show fields that were never captured as unavailable. Dataset diagnostics skip exact-image hashing to avoid another full pass over the source images; split leakage checks group by filename stem and remove a terminal `_cropped` suffix.

ONNX export follows Ultralytics' documented `model.export(format="onnx")` flow. Training resume uses `last.pt` with `resume=True`, which restores the optimizer and epoch state; use fine-tune for completed checkpoints or uploaded pretrained `.pt` files. [Ultralytics train/resume docs](https://docs.ultralytics.com/modes/train), [Ultralytics ONNX export docs](https://docs.ultralytics.com/modes/export).
# Shared dataset catalog

The application keeps original image files in the existing read-only dataset mount. It does not copy them into the catalog. On the first dataset summary or model-list request, it creates `/runs/datasets/catalog.sqlite3` and indexes the mounted images by SHA-256. Content-identical paths point to one stable image ID, while each original filename and source path remains recorded.

The catalog stores versioned annotations by task and model, persistent train/validation/test assignments by source-image group, dataset-version manifests, per-version image membership, derived sample metadata, and training-run references. Detector and server classifier snapshots use symlinks to original files. The classifier receives original images labeled with image-level counts from the Container-labeler `container_images_verified` session. Detector bounding-box counts are never used as a proxy for container counts. Conflicting image-level labels are excluded for manual review. Existing crop imports retain their crop path, parent path, and coordinates when those fields are available.

Dataset versions are listed at `/api/datasets/versions`; the Dataset tab displays unique originals, duplicate paths, per-model sample and split counts, exclusions, derived samples, and legacy imports needing reconciliation. New training jobs record a `datasetVersionId` in their run JSON and in the catalog.

Migration is additive. Existing datasets, run JSON, checkpoints, labels, and cached crops are retained. Old training records receive a legacy dataset-version reference; when their exact source membership or split assignments were not saved, the manifest marks them unreconciled and classifier samples are listed for manual review. The migration does not infer missing crop-to-source relationships.

## Experimental PaddleOCR in Detect

The Detect endpoint can optionally run PaddleOCR after the existing YOLO26 container detector. The default **Three relative ROIs** mode pads each detected box, crops it from the original resolution, and OCRs exactly three independently configured regions along its longitudinal axis. The initial centers are 15%, 50%, and 85%, with 25% longitudinal width and full transverse height. Horizontal and vertical boxes are supported; each ROI is normalized for vertical orientation before OCR. All three are processed even when one returns a reliable number. A full-container attempt runs only when none yields an ISO 6346 check-digit-valid candidate. The UI displays each ROI boundary, text, confidence, status, and elapsed time, plus aggregated candidates and optional full-image comparison counts. A manually entered ground truth enables exact-match reporting; confidence is not treated as accuracy. OCR initialization failures are reported without discarding detector/classifier results. No crop is written into the dataset.

The Docker image installs CPU `paddlepaddle==3.0.0` and `paddleocr==2.10.0`, using PP-OCRv4 English by default. PaddleOCR initializes lazily once per process/configuration and downloads its pretrained models on the first OCR request. To use local model directories, set `PADDLEOCR_DET_MODEL_DIR` and `PADDLEOCR_REC_MODEL_DIR` in the service environment and mount those directories read-only into the container. Detect settings include processing mode, three ROI centers, longitudinal width, transverse height, ROI and container padding, optional contrast/grayscale/sharpen/upscale, fallback, confidence threshold, and maximum attempts. Results preserve raw OCR strings and list alternate candidates; no characters are repaired or invented. Distinct conflicting candidates remain marked for verification. The optional full-image run is an approximate baseline for side-by-side region/number counts, while ground truth is required to report an exact match.

Rebuild and start the app with:

```bash
docker compose -p ai-vision-web -f docker-compose.web.yml up -d --build --no-deps web
```

CPU OCR can add several seconds per image, especially when whole-image comparison is enabled. Recognition quality depends on source resolution, blur, lighting, and whether the container identification code is visible. ISO 6346 validation only checks candidates OCR actually returned; it does not repair OCR text.

### Compare OCR modes manually

Rebuild the `ai-vision-demo` service, open Detect, upload an image with known container numbers, enable PaddleOCR, and select **Three relative ROIs**. Keep **Compare full image** enabled to populate the side-by-side region and number counts. Enter manually checked ISO 6346 numbers in **Verified ground truth** to see exact-match outcomes. Repeat with **Full container** or **Full image** mode while keeping the image and OCR model fixed. Counts and elapsed times are per submitted image; they are not dataset-level accuracy estimates. The UI does not persist these evaluation runs.
