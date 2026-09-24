# Multi Agent RAG system for Academic Research Assistance

ResearchPilot is a full-stack, multi-agent research assistant. It accepts a
natural-language research question, discovers relevant open-access papers,
extracts and ranks evidence, and produces a cited research report through a
web interface.

## Features

- Multi-agent research pipeline for query analysis, expansion, retrieval,
	evidence extraction, and report synthesis
- Hybrid paper retrieval using OpenAlex and optional CORE API results
- Open-access PDF download, text extraction, chunking, embeddings, and FAISS
	vector search
- Evidence-grounded reports with references, diagnostics, and progress updates
- Background research jobs that can be monitored or cancelled from the UI
- Research session history stored locally in `backend/sessions`

## Architecture

- `backend/`: FastAPI API and the research pipeline
- `frontend/`: React, TypeScript, and Vite web application
- `backend/data/`: downloaded and indexed research data
- `backend/sessions/`: generated reports and per-session indexes

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer and npm
- Ollama for local language-model generation
- Internet access for OpenAlex paper retrieval and PDF downloads

The backend uses the Ollama model `qwen2.5:7b` by default. Install Ollama from
[ollama.com](https://ollama.com), then download the model:

```bash
ollama pull qwen2.5:7b
```

## Installation

Clone the repository and enter its root directory:

```bash
git clone <repository-url>
cd ResearchPilot
```

### Backend

Create and activate a virtual environment, then install the Python
dependencies:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### Frontend

In a second terminal, install the JavaScript dependencies:

```bash
cd frontend
npm install
```

## Configuration

The application works with OpenAlex without an API key. Optional credentials
and service settings can be placed in `backend/.env`:

```dotenv
# Optional: improves OpenAlex request identification and quotas
OPENALEX_API_KEY=
OPENALEX_MAILTO=you@example.com

# Optional: enables additional CORE paper retrieval
CORE_API_KEY=
CORE_API_URL=https://api.core.ac.uk/v3/search/works

# Optional: Hugging Face authentication for model downloads
HF_TOKEN=

# Optional: use a remote Ollama server instead of the local default
OLLAMA_BASE_URL=http://localhost:11434
```

Do not commit `.env` files or API keys. The frontend uses
`VITE_API_BASE_URL` and defaults to `http://localhost:8000`:

```dotenv
# frontend/.env
VITE_API_BASE_URL=http://localhost:8000
```

## Running Locally

Start Ollama first, if it is not already running:

```bash
ollama serve
```

Start the backend from the `backend` directory in one terminal:

```bash
source .venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Start the frontend from the `frontend` directory in another terminal:

```bash
npm run dev
```

Open the URL printed by Vite, usually `http://localhost:5173`.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| `POST` | `/api/research/run` | Start a research job with `{ "query": "..." }` |
| `GET` | `/api/research/status/{job_id}` | Read progress, logs, and partial results |
| `GET` | `/api/research/result/{job_id}` | Read the completed report |
| `DELETE` | `/api/research/cancel/{job_id}` | Cancel a running job |
| `GET` | `/api/research/history` | List saved research sessions |
| `GET` | `/api/research/session/{session_id}` | Read a saved session |

Interactive API documentation is available at
`http://localhost:8000/docs` while the backend is running.

## Testing and Builds

Run the backend test suite from `backend`:

```bash
python -m pytest
```

Build the frontend for production from `frontend`:

```bash
npm run build
```

Preview the production frontend build with:

```bash
npm run preview
```

## Tech Stack

- Python, FastAPI, and Uvicorn
- LangChain and LangChain Ollama
- Sentence Transformers and FAISS
- OpenAlex and optional CORE API retrieval
- React, TypeScript, Vite, Tailwind CSS, and Axios
