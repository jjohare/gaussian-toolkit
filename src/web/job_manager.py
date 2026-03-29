# SPDX-FileCopyrightText: 2026 LichtFeld Studio Authors
# SPDX-License-Identifier: GPL-3.0-or-later

"""Job state management for the video-to-scene web pipeline.

Jobs are persisted as individual JSON files in /data/jobs/<job_id>.json.
Thread-safe access via a module-level lock.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

JOBS_DIR = Path(os.environ.get("LFS_JOBS_DIR", "/data/jobs"))
INPUT_DIR = Path(os.environ.get("LFS_INPUT_DIR", "/data/input"))
OUTPUT_BASE = Path(os.environ.get("LFS_OUTPUT_DIR", "/data/output"))

MAX_JOB_AGE_SECONDS = 7 * 24 * 3600  # 1 week

_lock = threading.Lock()


class JobState:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    # Stage-specific states are dynamically generated as "stage_<name>"
    @staticmethod
    def stage(name: str) -> str:
        return f"stage_{name}"


@dataclass
class HardwareSnapshot:
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization_pct: float = 0.0
    ram_used_mb: float = 0.0
    ram_total_mb: float = 0.0
    disk_free_gb: float = 0.0
    timestamp: float = 0.0


@dataclass
class StageInfo:
    name: str
    status: str = "pending"  # pending, running, completed, failed, skipped
    started_at: float | None = None
    finished_at: float | None = None
    duration_seconds: float = 0.0
    hardware: dict[str, Any] = field(default_factory=dict)
    preview_path: str | None = None
    error: str | None = None


@dataclass
class Job:
    job_id: str
    filename: str
    input_video_path: str
    output_dir: str
    state: str = JobState.QUEUED
    progress: float = 0.0
    current_stage: str = ""
    stages: dict[str, dict[str, Any]] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    previews: dict[str, str] = field(default_factory=dict)
    created_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    result_archive: str | None = None
    file_size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Job:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


PIPELINE_STAGES = [
    "INGEST",
    "REMOVE_PEOPLE",
    "SELECT_FRAMES",
    "RECONSTRUCT",
    "QUALITY_GATE_1",
    "DECOMPOSE",
    "EXTRACT_OBJECTS",
    "MESH_OBJECTS",
    "TEXTURE_BAKE",
    "QUALITY_GATE_2",
    "INPAINT_BG",
    "RETRAIN_BG",
    "USD_ASSEMBLE",
    "VALIDATE",
]


def _ensure_dirs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _save_job(job: Job) -> None:
    path = _job_path(job.job_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(job.to_dict(), indent=2, default=str))
    tmp.replace(path)


def _load_job(job_id: str) -> Job | None:
    path = _job_path(job_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return Job.from_dict(data)
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        logger.warning("Corrupt job file %s: %s", path, exc)
        return None


def create_job(filename: str, input_video_path: str, file_size_bytes: int = 0) -> Job:
    """Create a new job and persist it."""
    _ensure_dirs()
    job_id = uuid.uuid4().hex[:12]
    output_dir = str(OUTPUT_BASE / job_id)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    stages = {}
    for stage_name in PIPELINE_STAGES:
        stages[stage_name] = asdict(StageInfo(name=stage_name))

    job = Job(
        job_id=job_id,
        filename=filename,
        input_video_path=input_video_path,
        output_dir=output_dir,
        state=JobState.QUEUED,
        stages=stages,
        created_at=time.time(),
        file_size_bytes=file_size_bytes,
    )

    with _lock:
        _save_job(job)

    logger.info("Created job %s for %s", job_id, filename)
    return job


def get_job(job_id: str) -> Job | None:
    """Retrieve a job by ID."""
    with _lock:
        return _load_job(job_id)


def update_job(job_id: str, **kwargs: Any) -> Job | None:
    """Atomically update job fields and persist."""
    with _lock:
        job = _load_job(job_id)
        if job is None:
            return None
        for key, value in kwargs.items():
            if hasattr(job, key):
                setattr(job, key, value)
        _save_job(job)
        return job


def append_log(job_id: str, line: str) -> None:
    """Append a log line to the job. Keeps last 2000 lines on disk."""
    with _lock:
        job = _load_job(job_id)
        if job is None:
            return
        job.logs.append(line)
        if len(job.logs) > 2000:
            job.logs = job.logs[-2000:]
        _save_job(job)


def set_stage_status(
    job_id: str,
    stage_name: str,
    status: str,
    hardware: dict[str, Any] | None = None,
    preview_path: str | None = None,
    error: str | None = None,
) -> None:
    """Update a specific stage's status within a job."""
    with _lock:
        job = _load_job(job_id)
        if job is None:
            return

        stage = job.stages.get(stage_name, {"name": stage_name})

        if status == "running":
            stage["status"] = "running"
            stage["started_at"] = time.time()
            job.current_stage = stage_name
            job.state = JobState.stage(stage_name)
        elif status == "completed":
            stage["status"] = "completed"
            stage["finished_at"] = time.time()
            if stage.get("started_at"):
                stage["duration_seconds"] = stage["finished_at"] - stage["started_at"]
        elif status == "failed":
            stage["status"] = "failed"
            stage["finished_at"] = time.time()
            stage["error"] = error
            if stage.get("started_at"):
                stage["duration_seconds"] = stage["finished_at"] - stage["started_at"]

        if hardware:
            stage["hardware"] = hardware
        if preview_path:
            stage["preview_path"] = preview_path
            job.previews[stage_name] = preview_path

        job.stages[stage_name] = stage
        _save_job(job)


def list_jobs() -> list[dict[str, Any]]:
    """Return summary of all jobs, sorted by creation time descending."""
    _ensure_dirs()
    jobs = []
    with _lock:
        for path in JOBS_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                jobs.append({
                    "job_id": data.get("job_id", path.stem),
                    "filename": data.get("filename", ""),
                    "state": data.get("state", "unknown"),
                    "progress": data.get("progress", 0.0),
                    "current_stage": data.get("current_stage", ""),
                    "created_at": data.get("created_at", 0),
                    "started_at": data.get("started_at"),
                    "finished_at": data.get("finished_at"),
                    "error": data.get("error"),
                    "file_size_bytes": data.get("file_size_bytes", 0),
                })
            except (json.JSONDecodeError, KeyError):
                continue

    jobs.sort(key=lambda j: j.get("created_at", 0), reverse=True)
    return jobs


def delete_job(job_id: str) -> bool:
    """Delete a job's metadata and output directory."""
    with _lock:
        job = _load_job(job_id)
        if job is None:
            return False

        # Remove job file
        path = _job_path(job_id)
        if path.exists():
            path.unlink()

        # Remove output directory
        output = Path(job.output_dir)
        if output.exists():
            shutil.rmtree(output, ignore_errors=True)

        # Remove input video
        input_path = Path(job.input_video_path)
        if input_path.exists():
            input_path.unlink(missing_ok=True)

        logger.info("Deleted job %s", job_id)
        return True


def cleanup_old_jobs(max_age_seconds: int = MAX_JOB_AGE_SECONDS) -> int:
    """Remove completed/failed jobs older than max_age_seconds. Returns count removed."""
    _ensure_dirs()
    now = time.time()
    removed = 0

    with _lock:
        for path in JOBS_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                state = data.get("state", "")
                created = data.get("created_at", 0)
                if state in (JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED):
                    if now - created > max_age_seconds:
                        job_id = data.get("job_id", path.stem)
                        # Remove without lock (we already hold it)
                        path.unlink(missing_ok=True)
                        output_dir = data.get("output_dir", "")
                        if output_dir and Path(output_dir).exists():
                            shutil.rmtree(output_dir, ignore_errors=True)
                        input_path = data.get("input_video_path", "")
                        if input_path and Path(input_path).exists():
                            Path(input_path).unlink(missing_ok=True)
                        removed += 1
                        logger.info("Cleaned up old job %s", job_id)
            except (json.JSONDecodeError, KeyError, OSError):
                continue

    return removed
