import asyncio
import uuid
from datetime import datetime
from threading import Lock
from typing import Any, Dict, List, Optional

_lock = Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_tasks: Dict[str, asyncio.Task] = {}


def _timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"


def create_job(query: str) -> str:
    job_id = str(uuid.uuid4())
    now = _timestamp()
    with _lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "query": query,
            "status": "started",
            "current_agent": None,
            "progress_percentage": 0,
            "logs": [],
            "partial_results": {},
            "final_result": None,
            "error_message": None,
            "created_at": now,
            "updated_at": now,
        }
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job is not None else None


def _update(job_id: str, **values: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.update(values)
        job["updated_at"] = _timestamp()


def append_log(job_id: str, message: str) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job["logs"].append({"timestamp": _timestamp(), "message": message})
        job["updated_at"] = _timestamp()


def update_job(
    job_id: str,
    status: Optional[str] = None,
    current_agent: Optional[str] = None,
    progress_percentage: Optional[int] = None,
    partial_results: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
) -> None:
    values: Dict[str, Any] = {}
    if status is not None:
        values["status"] = status
    if current_agent is not None:
        values["current_agent"] = current_agent
    if progress_percentage is not None:
        values["progress_percentage"] = max(0, min(100, progress_percentage))
    if partial_results is not None:
        values["partial_results"] = partial_results
    if error_message is not None:
        values["error_message"] = error_message
    if values:
        _update(job_id, **values)


def complete_job(job_id: str, final_result: Dict[str, Any]) -> None:
    update_job(job_id, status="completed", progress_percentage=100, partial_results=final_result)
    _update(job_id, final_result=final_result)


def fail_job(job_id: str, message: str) -> None:
    append_log(job_id, f"Job failed: {message}")
    update_job(job_id, status="failed", error_message=message)


def cancel_job(job_id: str) -> bool:
    with _lock:
        task = _tasks.get(job_id)
    if task is None or task.done():
        return False
    task.cancel()
    update_job(job_id, status="cancelled", error_message="Job cancelled by user.")
    append_log(job_id, "Job cancellation requested.")
    return True


def add_task(job_id: str, task: asyncio.Task) -> None:
    with _lock:
        _tasks[job_id] = task


def get_running_jobs() -> List[Dict[str, Any]]:
    with _lock:
        return [dict(job) for job in _jobs.values()]
