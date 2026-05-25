import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import login

from app.core.job_manager import add_task, append_log, cancel_job, complete_job, create_job, fail_job, get_job, update_job
from app.core.pipeline import run_pipeline, run_pipeline_state, serialize_state

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(hf_token)

app = FastAPI(
    title="ResearchPilot API",
    description="Backend API for ResearchPilot multi-agent research assistant.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
SESSIONS_DIR = BASE_DIR / "sessions"


class ResearchRequest(BaseModel):
    query: str


@app.post("/api/research/run")
async def run_research(request: ResearchRequest) -> Dict[str, Any]:
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query is required.")

    job_id = create_job(query)

    async def _run_job() -> None:
        try:
            update_job(job_id, status="running", current_agent="QueryAgent", progress_percentage=0)

            def progress_callback(agent_name: str, event: str, detail: str, progress: int | None, state: Any) -> None:
                append_log(job_id, f"{agent_name}: {detail}")
                update_job(
                    job_id,
                    status="running",
                    current_agent=agent_name,
                    progress_percentage=progress if progress is not None else None,
                    partial_results=serialize_state(state) if state is not None else None,
                )

            state = await asyncio.to_thread(run_pipeline_state, query, progress_callback)
            final_result = serialize_state(state)
            complete_job(job_id, final_result)
        except asyncio.CancelledError:
            fail_job(job_id, "Job cancelled by user.")
        except Exception as exc:
            fail_job(job_id, str(exc))

    task = asyncio.create_task(_run_job())
    add_task(job_id, task)
    return {"job_id": job_id, "status": "started"}


@app.get("/api/research/status/{job_id}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "current_agent": job["current_agent"],
        "progress_percentage": job["progress_percentage"],
        "logs": job["logs"],
        "partial_results": job["partial_results"],
        "error_message": job.get("error_message"),
    }


@app.get("/api/research/result/{job_id}")
async def get_job_result(job_id: str) -> Dict[str, Any]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "completed":
        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "partial_results": job["partial_results"],
            "progress_percentage": job["progress_percentage"],
            "current_agent": job["current_agent"],
            "logs": job["logs"],
        }
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "final_result": job["final_result"],
        "logs": job["logs"],
    }


@app.delete("/api/research/cancel/{job_id}")
async def cancel_research_job(job_id: str) -> Dict[str, Any]:
    if not get_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")
    if not cancel_job(job_id):
        raise HTTPException(status_code=400, detail="Job cannot be cancelled.")
    return {"job_id": job_id, "status": "cancelled"}


@app.get("/api/research/history")
async def get_research_history() -> List[Dict[str, Any]]:
    history = []
    if not SESSIONS_DIR.exists():
        return history

    for session_dir in sorted(SESSIONS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not session_dir.is_dir():
            continue
        draft_file = session_dir / "draft_report.json"
        if not draft_file.exists():
            continue

        try:
            payload = json.loads(draft_file.read_text(encoding="utf-8"))
            history.append(
                {
                    "id": session_dir.name,
                    "created_at": datetime.fromtimestamp(session_dir.stat().st_mtime).isoformat() + "Z",
                    "paper_count": payload.get("papers_used", 0),
                    "title": payload.get("report", "Research session"),
                    "summary": payload.get("report", ""),
                }
            )
        except Exception:
            continue

    return history


@app.get("/api/research/session/{session_id}")
async def get_research_session(session_id: str) -> Dict[str, Any]:
    draft_file = SESSIONS_DIR / session_id / "draft_report.json"
    if not draft_file.exists():
        raise HTTPException(status_code=404, detail="Session not found.")

    payload = json.loads(draft_file.read_text(encoding="utf-8"))
    return {
        "id": session_id,
        "report": payload.get("report", ""),
        "paper_count": payload.get("papers_used", 0),
        "sources": payload.get("sources", []),
    }


if __name__ == "__main__":
    query = input("Enter your research query: ")
    result = run_pipeline(query)

    print("\n=== FINAL OUTPUT ===\n")
    print(result)