# ResearchPilot Frontend

A React + Vite frontend for ResearchPilot that connects to the existing backend API endpoints.

## Install

```bash
cd frontend
npm install
```

## Run locally

```bash
npm run dev
```

The app will use `VITE_API_BASE_URL` from `.env`, which defaults to `http://localhost:8000`.

## Build

```bash
npm run build
```

## Backend integration

The frontend expects the backend to expose these API routes:
-- `POST /api/research/run` - accepts `{ query: string }` and returns `{ job_id, status }` immediately.
-- `GET /api/research/status/{job_id}` - returns live job progress, logs, and partial results.
-- `GET /api/research/result/{job_id}` - returns the final report once the background job completes.
-- `DELETE /api/research/cancel/{job_id}` - cancels a running job if possible.

If running with the backend in `/backend`, start the backend with:

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Then run the frontend in a second terminal.

## Example API responses

### Start a job
```json
{
	"job_id": "8b0f6ae0-1b4b-4f5c-a81c-0e1234567890",
	"status": "started"
}
```

### Job status
```json
{
	"job_id": "8b0f6ae0-1b4b-4f5c-a81c-0e1234567890",
	"status": "running",
	"current_agent": "RetrievalAgent",
	"progress_percentage": 42,
	"logs": [
		{ "timestamp": "2026-05-18T12:34:56Z", "message": "QueryAgent: QueryAgent started" }
	],
	"partial_results": {
		"query": "...",
		"papers": [],
		"generated_answer": ""
	}
}
```

### Final result
```json
{
	"job_id": "8b0f6ae0-1b4b-4f5c-a81c-0e1234567890",
	"status": "completed",
	"final_result": {
		"query": "...",
		"papers": [],
		"generated_answer": "...",
		"references": [],
		"diagnostics": {}
	}
}
```
