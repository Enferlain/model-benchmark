# Model Benchmark Explorer

A tool for benchmarking and comparing SDXL models by generating images from test prompts and calculating quality metrics.

## Quick Start

### 1. Install Dependencies

```bash
# Backend
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Frontend (from root)
npm install
```

### 2. Add Models

Place your `.safetensors` model files in:

```
backend/assets/models/
```

### 3. Add Test Data

You have **two options** for test data:

#### Option A: Paired Image+Prompt (Preferred)

Place images and matching `.txt` files in:

```
backend/assets/image_prompts/
├── example1.png     # Reference image
├── example1.txt     # Prompt for that image
├── example2.jpg
└── example2.txt
```

This links each prompt to a reference image for quality comparison.

#### Option B: Separate Folders (Fallback)

If no paired data exists, the system uses:

```
backend/assets/images/     # Reference images (optional)
backend/assets/prompts/    # Text files with prompts (one prompt per file)
```

Note: These are loaded independently without pairing.

### 4. Run

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

| Endpoint        | Method | Description                        |
| --------------- | ------ | ---------------------------------- |
| `/api/generate` | POST   | Generate images (no metrics)       |
| `/api/analyze`  | POST   | Compute metrics on existing images |
| `/api/scan`     | POST   | Generate + Analyze                 |
| `/api/status`   | GET    | Current generation progress        |
| `/api/cancel`   | POST   | Cancel running generation          |
| `/api/models`   | GET    | List analyzed models               |

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
  "height": 1536
}
```

---

## Metrics

- **CLIP Score**: Prompt adherence (how well image matches text)
- **LPIPS Diversity**: Visual diversity between images of the same prompt

---

## Project Structure

```
model-benchmark-explorer/
├── backend/
│   ├── main.py           # FastAPI server
│   ├── inference.py      # SDXL generation wrapper
│   ├── metrics.py        # CLIP & LPIPS calculation
│   ├── data_loader.py    # Model/prompt discovery
│   ├── sd-scripts/       # Self-contained sd-scripts library
│   └── requirements.txt
├── src/                  # Vite frontend
└── package.json
```
