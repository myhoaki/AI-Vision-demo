"""Read worker claims and explicitly release a verified stale claim."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.job_queue import atomic_json, claim_is_stale, claim_path, exclusive_recovery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices={"status", "release-stale"})
    parser.add_argument("--pipeline", choices={"yolo", "classifier"})
    parser.add_argument("--runs-dir", default="/runs")
    parser.add_argument("--max-age-seconds", type=int, default=180)
    parser.add_argument("--confirm-owner-stopped", action="store_true",
                        help="Required after independently verifying the owning worker container is stopped.")
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)
    pipelines = [args.pipeline] if args.pipeline else ["yolo", "classifier"]
    if args.command == "status":
        registry = runs_dir / "worker" / "registry"
        for path in sorted(registry.glob("*.json")) if registry.is_dir() else []:
            print("worker", json.dumps(json.loads(path.read_text(encoding="utf-8")), sort_keys=True))
        for pipeline in pipelines:
            path = claim_path(runs_dir, pipeline)
            if path.is_file():
                print(pipeline, json.dumps(json.loads(path.read_text(encoding="utf-8")), sort_keys=True))
            else:
                print(pipeline, "idle")
        return
    if not args.pipeline or not args.confirm_owner_stopped:
        parser.error("release-stale requires --pipeline and --confirm-owner-stopped")
    path = claim_path(runs_dir, args.pipeline)
    try:
        recovery_lock = exclusive_recovery(runs_dir, args.pipeline)
    except RuntimeError as error:
        parser.error(str(error))
    with recovery_lock:
        try:
            claim = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            print("No claim exists.")
            return
        except (OSError, ValueError) as error:
            parser.error(f"Claim is unreadable; refusing recovery: {error}")
        stale = claim_is_stale(claim, time.time(), args.max_age_seconds, owner_alive=lambda _claim: False)
        if not stale:
            parser.error("Claim heartbeat is not stale; refusing to release it.")
        archive = path.with_name(f"{path.stem}.stale-{int(time.time())}-{claim.get('workerId', 'unknown')}.json")
        # The lock serializes recovery tools; verify the claim has not been replaced since inspection.
        current = json.loads(path.read_text(encoding="utf-8"))
        if current.get("workerId") != claim.get("workerId") or current.get("heartbeatAt") != claim.get("heartbeatAt"):
            parser.error("Claim changed during recovery; refusing to release it.")
        path.rename(archive)
        job_id = claim.get("jobId")
        if job_id:
            relative = "jobs" if args.pipeline == "yolo" else "classifiers/jobs"
            metadata = runs_dir / relative / str(job_id) / "job.json"
            if metadata.is_file():
                job = json.loads(metadata.read_text(encoding="utf-8"))
                if job.get("status") in {"queued", "preparing", "running", "evaluating"}:
                    now = datetime.now(timezone.utc).isoformat()
                    job.update(status="interrupted", updatedAt=now, finishedAt=now,
                               message="Worker ownership was explicitly recovered after confirming the worker container stopped. Resume only as a new job.")
                    job.setdefault("events", []).append({"timestamp": now, "level": "WARNING", "source": "worker-recovery",
                                                          "message": "Stale worker claim explicitly released; no automatic restart was performed."})
                    atomic_json(metadata, job)
        print(f"Released stale claim by archiving {archive.name}; no job was resumed automatically.")


if __name__ == "__main__":
    main()
