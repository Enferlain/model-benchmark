# Agent Guide - Model Benchmark Explorer

This document is a guide for AI Agents (and humans) working on this codebase. It outlines the tech stack, architecture, design philosophy, and standard workflows.

## 🌟 Design Philosophy

1.  **Premium & Modern UI**: We prioritize a "Wow" factor. The UI should use **glassmorphism**, **smooth transitions**, **dark mode by default**, and **curated color palettes** (Slate/Zinc/Blue/Orange). Later on dark mode will be ported to catppuccin mocha. Avoid generic layouts.
    - _Keywords_: Backdrop blur, rounded corners (xl/2xl), translucent panels, subtle borders.
2.  **Fairness & Rigor**: This is a benchmark tool. Fairness is paramount.
    - **Parameter Checks**: Always validate generation params against existing data.
    - **Coverage Checks**: Ensure models have equal data (prompts/image counts) before comparison.
    - **Reproducibility**: Seeds, exact timestamps, and metadata preservation are critical.
3.  **Local-First / Privacy**: Everything runs locally. No external API calls for generation unless downloading models (HF/CivitAI).

## 🛠 Tech Stack

### Frontend

- **Framework**: React 19 + Vite 6
- **Language**: TypeScript
- **Styling**: Tailwind CSS (Utility-first, heavily used for layout/design)
- **Icons**: Lucide React
- **Charts**: Recharts (Scatter plots for metrics)
- **State**: React Context + Hooks (minimal global state libraries)

### Backend

- **Framework**: FastAPI (Python 3.10+)
- **ORM**: SQLModel (SQLite)
- **ML Engine**: PyTorch, Diffusers, Safetensors
- **Image Processing**: Pillow, OpenCV
- **Metrics**: LPIPS, CLIP, Aesthetica

## 📂 Architecture

### Directory Structure

```
model-benchmark-explorer/
├── backend/                  # Python Backend
│   ├── app/
│   │   ├── api/              # API Endpoints (Routes)
│   │   │   ├── generation.py # /generate, /analyze
│   │   │   ├── models.py     # /models
│   │   │   └── ...
│   │   ├── services/         # Business Logic (Heavy lifting)
│   │   │   ├── generation.py # Image generation logic & queue
│   │   │   ├── analysis.py   # Metrics computation & scoring
│   │   │   └── ...
│   │   ├── lib/              # Low-level ML wrappers
│   │   │   ├── inference.py  # SDXL Pipeline wrapper
│   │   │   └── metrics.py    # Torch metrics implementation
│   │   └── core/             # Config, DB, State
│   ├── assets/               # Runtime Data (Gitignored)
│   │   ├── models/           # Large model files
│   │   ├── outputs/          # Generated images
│   │   └── database.db       # SQLModel DB
│   └── requirements.txt
├── src/                      # React Frontend
│   ├── components/           # Reusable UI (Button, Modal, Card)
│   ├── pages/                # Main Views
│   │   ├── Dashboard.tsx     # Hero view (Scatter plot)
│   │   ├── Gallery.tsx       # Image Grid/Filtering
│   │   ├── Compare.tsx       # Side-by-Side Slider
│   │   └── ...
│   ├── services/api.ts       # Typed API Client
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
    - `services/analysis.py` computes metrics (CLIP, LPIPS).
    - Results saved to `assets/outputs/.../results.json`.
    - Results synced to SQLite DB for fast querying.

3.  **Model Identification**:
    - Models are identified by **BLAKE3 Hash** (or SHA256 in future).
    - Filenames are secondary.
    - Local cache (`model_cache.json`) speeds up startup.

## 🤖 Agent Workflows

### 1. Adding Features

- **Check `TODO.md`** first.
- **Plan**: Create an `implementation_plan.md` artifact before coding complex features.
- **Task Mode**: Use `task_boundary` to structure your work into granular steps.

### 2. UI Development

- **Never** use raw CSS unless absolutely necessary. Use Tailwind.
- **Components**: Create small, reusable components in `src/components` if a UI element is used twice.
- **Icons**: Import from `lucide-react`.

### 3. Backend Development

- **Type Hinting**: Use strict Python type hints (`def foo(x: int) -> str:`).
- **Async**: FastAPI endpoints should be `async def` if they do I/O.
- **ML Code**: Keep heavy ML code in `app/lib/` or `app/services/`, not directly in API routes.

### 4. Code Quality

- **Linting**: Follow the existing style.
- **Imports**: Keep imports organized.
- **Comments**: Comment complex logic, especially ML pipeline steps.

## 📝 Common Tasks

- **Adding a dependency**:
  - Frontend: `npm install <package>` -> Update `package.json`.
  - Backend: `pip install <package>` -> Update `backend/requirements.txt`.
- **New API Endpoint**:
  1.  Define Pydantic schema in `backend/app/core/state.py`.
  2.  Create route in `backend/app/api/`.
  3.  Add service logic in `backend/app/services/`.
  4.  Add typed function in `src/services/api.ts`.

## ⚠️ Gotchas

- **File Paths**: Use `pathlib` in backend. Remember `assets` is in `backend/assets/`.
- **Windows**: The user is on Windows. Ensure paths work with backslashes/forward slashes (use `Path` object), but Linux should also be supported.
- **Vite Proxy**: Frontend calls `/api/...`, Vite proxies to `localhost:8000`. Don't hardcode `localhost:8000` in fetch calls, use the relative path or configured base.
- **CORS**: Configured in `backend/app/main.py`.
