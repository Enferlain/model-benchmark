# Model Benchmark Explorer WIP

A tool for benchmarking and comparing SDXL models by generating images from test prompts and calculating quality metrics.

<img width="1893" height="962" alt="image" src="https://github.com/user-attachments/assets/84bd7ac2-f10f-4ef0-9b53-784e62af5b76" />

<img width="1354" height="679" alt="image" src="https://github.com/user-attachments/assets/ceb852b9-4a27-466d-8a37-3888ee347ad4" />

<img width="1333" height="371" alt="image" src="https://github.com/user-attachments/assets/1ff081b3-21d8-4fb3-b633-05ffc83b3859" />

## Quick Start

### 1. Install Dependencies

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head # Run migrations

# Frontend (from root)
npm install
```

### 2. Assets Migration (Important!)

**Note:** The `assets` directory has been moved from the project root to `backend/assets`.
If you are upgrading from an older version, please move your existing `assets` folder into `backend/`.

### 3. Add Models

You can add models in two ways:

#### Option A: Manual Placement (Local)

Place your `.safetensors` model files directly in:

```
backend/assets/models/
```

#### Option B: Download via UI (Hugging Face / CivitAI)

The easiest way to add models is via the Dashboard UI:

1.  Navigate to the **Dashboard**.
2.  Locate the **"Add Model"** panel on the left sidebar.
3.  Paste a **Direct Download Link** (e.g., from Hugging Face or CivitAI) into the "MODEL URL" input field.
4.  Click **"Download Model"**.

- Supported Sources: Hugging Face (resolve/main links), CivitAI (model download links).
- **Note:** Only public models are supported. Authentication (API keys) for private or gated models is not currently implemented.
- The system will automatically attempt to parse the model name from the URL, or you can use the API (Option C) for custom naming.
- A progress bar will show the download status.

#### Option C: Download via API (Advanced)

For automated or headless setups, you can trigger downloads via the API:

**Example (cURL):**

```bash
curl -X POST "http://localhost:8000/api/models/download" \
     -H "Content-Type: application/json" \
     -d '{
           "url": "https://huggingface.co/author/repo/resolve/main/model.safetensors",
           "name": "MyModel",
           "source": "HuggingFace"
         }'
```

- **URL**: Direct download link to the model file.
- **Name**: Desired filename (without extension).
- **Source**: Metadata tag (e.g., "HuggingFace", "CivitAI").

You can check the download status via:

```bash
curl "http://localhost:8000/api/models/download/status"
```

### 4. Manage Prompts

The application features a full **Prompt Manager** tab.

- **Interactive Queue**: Toggle prompts ON/OFF. Only enabled prompts are used for benchmarking.
- **CRUD**: Create, Edit, Delete prompts directly in the UI.
- **Reference Images**: Upload reference images for specific prompts (useful for Img2Img or visual comparison).
- **Organization**: Search and index tracking.

_Alternatively, you can still place files manually:_

#### Option A: Paired Image+Prompt (Preferred)

Place images and matching `.txt` files in:

```
backend/assets/image_prompts/
├── example1.png     # Reference image
├── example1.txt     # Prompt for that image
```

#### Option B: Separate Folders (Fallback)

```
backend/assets/images/     # Reference images (optional)
backend/assets/prompts/    # Text files with prompts
```

### 5. Run

```bash
# Terminal 1: Backend
cd backend
venv\Scripts\activate
uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Key Features

- **Dashboard**: Visual scatter plot of model metrics with linked table selection
- **Arena**: Pairwise model comparison with **Bradley-Terry (BT)** ranking system
- **Gallery**: Filter generated images by Prompt, Seed, and Model (with optimized UUID indexing)
- **Compare**: Side-by-side image comparison with slider overlay
- **Prompt Editor**: Full CRUD for stable test prompts with reference images
- **Database**: View and search benchmark run history and arena votes
- **Model Comparison View**: Slider-based visual comparison between models

### Fairness Features

- **Parameter Mismatch Detection**: Warns when generation settings don't match existing images
- **Prompt Coverage Check**: Ensures all models have equal image counts before analysis
- **Equalize Mode**: Automatically generates missing images to match across models

---

## How Image Generation Works

### Directory Structure

```
backend/assets/
├── models/          # Your .safetensors models
├── prompts/         # Text files with prompts
└── outputs/         # Generated images (per model)
    └── ModelName/
        ├── p000_i00_s218.png   # Prompt 0, Image 0, Seed 218
        ├── p000_i01_s219.png   # Prompt 0, Image 1, Seed 219
        └── p001_i00_s218.png   # Prompt 1, Image 0, Seed 218
```

### Filename Convention

`p{prompt_index}_i{image_index}_s{seed}.png`

- **prompt_index**: Which prompt (0-indexed)
- **image_index**: Which image for that prompt (for diversity/LPIPS)
- **seed**: The exact seed used

### Image Metadata

Each generated image contains embedded metadata:

- `model_name`: Which model generated it
- `prompt`: The full prompt text
- `parameters`: JSON with steps, cfg, sampler, seed, dimensions
- `generation_time`: ISO timestamp
- `id`: Unique short identifier

### Seed Logic

- Each prompt uses `base_seed + image_index`
- Different prompts use the **same seed sequence** for fair comparison
- Example with `seed=218, images_per_prompt=2`:
  - Prompt 0: seeds 218, 219
  - Prompt 1: seeds 218, 219 (same!)

### Gap Handling

If an image is missing (e.g., `p001_i01` doesn't exist), the system:

1. Detects the exact missing indices
2. Regenerates only those with the correct seed
3. Won't duplicate existing images

---

## API Endpoints

### Generation & Analysis

| Endpoint                      | Method | Description                                  |
| ----------------------------- | ------ | -------------------------------------------- |
| `/api/generate`               | POST   | Generate images (no metrics)                 |
| `/api/analyze`                | POST   | Compute metrics on existing images           |
| `/api/scan`                   | POST   | Generate + Analyze                           |
| `/api/status`                 | GET    | Current generation progress                  |
| `/api/cancel`                 | POST   | Cancel running generation                    |
| `/api/check-params`           | POST   | Check if params match existing images        |
| `/api/archive/{model}`        | POST   | Archive model's images to timestamped folder |
| `/api/analyze/check-coverage` | POST   | Check prompt coverage across models          |

### Models

| Endpoint                      | Method | Description             |
| ----------------------------- | ------ | ----------------------- |
| `/api/models`                 | GET    | List analyzed models    |
| `/api/models/download`        | POST   | Download model from URL |
| `/api/models/download/status` | GET    | Download progress       |
| `/api/models/{id}`            | DELETE | Remove model            |

### Arena

| Endpoint                 | Method | Description                     |
| ------------------------ | ------ | ------------------------------- |
| `/api/arena/vote`        | POST   | Cast vote and update BT ratings |
| `/api/arena/leaderboard` | GET    | Get BT-ranked model leaderboard |

### Generation Options

```json
{
  "sampler": "euler_a",
  "steps": 28,
  "guidance_scale": 5.0,
  "seed": 218,
  "images_per_prompt": 2,
  "num_prompts": 10,
  "width": 1024,
  "height": 1536,
  "common_only": false,
  "equalize_counts": false
}
```

---

## Metrics

- **CLIP Score**: Prompt adherence (how well image matches text)
- **LPIPS Diversity**: Visual diversity between images of the same prompt
- **VQA Score**: Visual question answering based quality assessment
- **Rating**: Aesthetic rating prediction

---

## Project Structure

```
model-benchmark-explorer/
├── backend/
│   ├── main.py               # Entry point
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py           # FastAPI app setup
│   │   ├── api/              # API route handlers
│   │   │   ├── generation.py # /generate, /analyze, /scan
│   │   │   ├── models.py     # /models endpoints
│   │   │   ├── prompts.py    # /prompts endpoints
│   │   │   └── system.py     # /status, /runs
│   │   ├── services/         # Business logic
│   │   │   ├── generation.py # Image generation
│   │   │   ├── analysis.py   # Metrics computation
│   │   │   ├── model_manager.py
│   │   │   ├── prompt_manager.py
│   │   │   └── downloader.py
│   │   ├── lib/              # Core utilities
│   │   │   ├── inference.py  # SDXL pipeline wrapper
│   │   │   └── metrics.py    # CLIP & LPIPS calculation
│   │   └── core/             # Database, state, config
│   └── sd-scripts/           # Self-contained sd-scripts library
├── src/                      # React Frontend
│   ├── pages/
│   │   ├── Dashboard.tsx     # Main view with scatter plot
│   │   ├── Gallery.tsx       # Image browser
│   │   ├── Compare.tsx       # Side-by-side comparison
│   │   ├── PromptEditor.tsx  # Prompt management
│   │   └── Database.tsx      # Benchmark history
│   ├── components/           # Reusable UI components
│   ├── services/api.ts       # API client
│   └── context/              # React Context (Theme)
└── package.json
```
