# Agent Guide - Model Benchmark Explorer

This document is a guide for AI Agents (and humans) working on this codebase. It outlines the tech stack, architecture, design philosophy, and standard workflows.

## 🌟 Design Philosophy

1.  **Premium & Modern UI**: We prioritize a "Wow" factor. The UI should use **glassmorphism**, **smooth transitions**, **dark mode by default**, and **curated color palettes** (Slate/Zinc/Blue/Orange). Later on dark mode will be ported to catppuccin mocha. Avoid generic layouts.
    - _Keywords_: Backdrop blur, rounded corners (xl/2xl), translucent panels, subtle borders.
2.  **Fairness & Rigor**: This is a benchmark tool. Fairness is paramount.
    - **Parameter Checks**: Always validate generation params against existing data.
    - **Coverage Checks**: Ensure models have equal data (prompts/image counts) before comparison.
    - **Reproducibility**: Seeds, exact timestamps, and metadata preservation are critical.
3.  **Modular Generation**: While the current implementation has built-in generation, the long-term goal is to move toward **external providers** (A1111, ComfyUI, etc.) via API. Avoid monolithic generation logic.

## 🛠 Tech Stack

### Frontend

- **Framework**: React 19 + Vite 6
- **Language**: TypeScript
- **Styling**: Tailwind CSS (Utility-first, heavily used for layout/design)
- **Linting & Formatting**: **Biome** (configured in `biome.json`)
- **Type Checking**: `tsc` (TypeScript compiler)
- **Icons**: Lucide React
- **Charts**: Recharts (Scatter plots for metrics)
- **State**: React Context + Hooks (minimal global state libraries)

### Backend

- **Framework**: FastAPI (Python 3.10+)
- **Linting & Formatting**: **Ruff** (configured in `backend/pyproject.toml`)
- **Type Checking**: **Ty**
- **ORM**: SQLModel (SQLite)
- **Migrations**: Alembic (configured in `backend/alembic/`)
- **ML Engine**: PyTorch, Diffusers, Safetensors
- **Image Processing**: Pillow, OpenCV
- **Metrics**: LPIPS, CLIP, Aesthetica
- **External Integration (Planned)**: A1111 API, ComfyUI API

## 📂 Architecture

### Directory Structure

```
model-benchmark-explorer/
├── backend/                  # Python Backend
│   ├── app/
│   │   ├── api/              # API Endpoints (Routes)
│   │   │   ├── generation.py # /generate, /analyze, /check-params
│   │   │   ├── models.py     # /models, /models/register
│   │   │   └── ...
│   │   ├── services/         # Business Logic
│   │   │   ├── generation.py # Image generation logic & queue
│   │   │   ├── analysis.py   # Metrics computation & scoring
│   │   │   ├── model_manager.py # DB Sync & Local Scanning
│   │   │   └── prompt_manager.py # Prompt loading & metadata
│   │   ├── lib/              # Low-level ML wrappers (Legacy/Temporary)
│   │   │   ├── inference.py  # SDXL Pipeline wrapper (To be replaced by APIs)
│   │   │   ├── metrics.py    # Torch metrics (CLIP, LPIPS)
│   │   │   └── lpw_utils.py  # Long Prompt Weighting utilities
│   │   └── core/             # Config, DB, State
│   ├── assets/               # Runtime Data (Gitignored)
│   │   ├── models/           # Large model files (.safetensors)
│   │   ├── outputs/          # Generated images (grouped by Model Name)
│   │   ├── database.db       # SQLModel (SQLite) database
│   │   ├── sources.json      # Tracked import paths
│   │   └── model_cache.json  # Fast startup cache
│   └── requirements.txt
├── src/                      # React Frontend
│   ├── components/           # Reusable UI (Button, Modal, Card)
│   ├── pages/                # Main Views (Dashboard, Gallery, Compare)
│   ├── services/api.ts       # Typed API Client (Fetch-based)
│   ├── vite-env.d.ts         # Vite Environment Types
│   └── index.css             # Tailwind Directives & Global Styles
```

### Key Data Flows

1.  **Generation**:
    - User configures params in `Dashboard.tsx`.
    - Request sent to `/api/generate`.
    - Backend `services/generation.py` adds to queue.
    - Frontend polls `/api/status` for progress.
    - Images saved to `backend/assets/outputs/{model_name}/`.

2.  **Analysis**:
    - User clicks "Analyze".
    - `services/analysis.py` computes metrics (CLIP Accuracy, LPIPS Diversity).
    - A `BenchmarkRun` is created in the SQLite DB.
    - `ModelResult` records (linking model hash to metrics) are saved to the DB.
    - Frontend fetches results from the database via `/api/models`.

3.  **Model Identification**:
    - Models are identified by **SHA256 Hash** (Industry Standard).
    - Older BLAKE3 hashes are deprecated.
    - Local cache (`model_cache.json`) speeds up startup.

4.  **Database & Indexing**:
    - **Alembic**: All schema changes MUST be done via Alembic migrations. The `init_db()` function is initialized but doesn't auto-create tables anymore.
    - **Image Indexing**: Generated images have an 8-character UUID in their PNG metadata. The `ImageOutput` table indexes these for speed.
    - **Stable Prompts**: Prompts are stored in a dedicated `Prompt` table. Images should link to `Prompt.id` for analytics consistency.

## 🤖 Agent Workflows

### 1. Adding Features

- **Check `TODO.md`** first.
- **Plan**: Create an `implementation_plan.md` artifact before coding complex features.
- **Task Mode**: Use `task_boundary` to structure your work into granular steps.
- **Code Review**: Use the `review-mcp` tool (e.g., `mcp_review_with_context`) to verify your changes. You can review `staged`, `unstaged`, or `HEAD~1` changes before notifying the user.

### 2. UI Development

- **Never** use raw CSS unless absolutely necessary. Use Tailwind.
- **Components**: Create small, reusable components in `src/components` if a UI element is used twice.
- **Icons**: Import from `lucide-react`.

### 3. Backend Development

- **Type Hinting**: Use strict Python type hints (`def foo(x: int) -> str:`).
- **Async**: FastAPI endpoints should be `async def` if they do I/O.
- **ML Code**: Avoid adding new monolithic ML code to `app/lib/`. New features should prioritize **External APIs** (A1111/ComfyUI) for generation.

### 4. Code Quality

- **Linting (Frontend)**: Run `npm run check` for Biome.
- **Linting (Backend)**: Run `npm run check:backend` for Ruff.
  - Check: `ruff check .`
  - Fix: `ruff check . --fix`
  - Format: `ruff format .`
- **Global Check**: Run `npm run check:all` to verify the entire repository.
- **Type Checking**:
  - **Frontend**: Run `npm run typecheck` for `tsc`.
  - **Backend**: Use `Ty`.
    - Check all: `uvx ty check`
    - Check directory: `uvx ty check backend/services`
    - Check file: `uvx ty check backend/services/analysis.py`
- **Style**: Follow existing styles (Tabs for JS/TS, Spaces for Python).
- **Imports**: Keep imports organized (handled by Biome and Ruff isort).
- **Comments**: Comment complex logic, especially ML pipeline steps.
- **Testing**: Follow the guidelines in [TESTING_STRATEGY.md](file:///d:/Projects/model-benchmark-explorer/TESTING_STRATEGY.md).

## 📝 Common Tasks

- **Adding a dependency**:
  - Frontend: `npm install <package>` -> Update `package.json`.
  - Backend: `pip install <package>` -> Update `backend/requirements.txt`.
- **New API Endpoint**:
  1.  Define Pydantic schema in `backend/app/core/state.py`.
  2.  Create route in `backend/app/api/`.
  3.  Add service logic in `backend/app/services/`.
  4.  Add typed function in `src/services/api.ts`.
- **Database Schema Change**:
  1.  Modify models in `backend/app/core/database.py`.
  2.  Run `alembic revision --autogenerate -m "description"`.
  3.  Review the migration script (ensure `import sqlmodel` is present if needed).
  4.  Apply with `alembic upgrade head`.

## ⚠️ Gotchas

- **File Paths**: Use `pathlib` in backend. Remember `assets` is in `backend/assets/`.
- **Windows**: The user is on Windows. Ensure paths work with backslashes/forward slashes (use `Path` object), but Linux should also be supported.
- **Vite Proxy**: Frontend calls `/api/...`, Vite proxies to `localhost:8000`. Don't hardcode `localhost:8000` in fetch calls, use the relative path or configured base.
- **CORS**: Configured in `backend/app/main.py`.
- **Environment**:
  - **Terminal**: This project is developed in a **PowerShell** (`pwsh`) environment.
  - **Backend Venv**: Always ensure the Python virtual environment (`backend/venv`) is **active** when running backend scripts or the server.
  - **Scripts**: Prefer using the defined `npm` scripts in `package.json` for frontend tasks.
