from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import threading
from pathlib import Path

# Import backend modules
import data_loader
import state
import scanner
import downloader

app = FastAPI()

# Suppress uvicorn access log spam for polling endpoints
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Mount assets directory for serving images
# Ensure the directory exists to avoid errors on startup if it's missing
data_loader.ASSETS_DIR.mkdir(exist_ok=True)
app.mount("/assets", StaticFiles(directory=data_loader.ASSETS_DIR), name="assets")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    print("Starting up... (no auto-generation, use /api/generate or /api/analyze)")
    # Populate models_db on startup
    print("Scanning for local models and existing outputs...")
    scanner.analyze_models_only(state.ScanOptions())

# API Endpoints
@app.get("/api/models")
def get_models():
    return state.models_db

@app.get("/api/models/{model_id}/outputs")
def get_model_outputs(model_id: str):
    output_dir = data_loader.ASSETS_DIR / "outputs" / model_id
    if not output_dir.exists():
        return []

    # Get prompts (cached or reloaded)
    _, prompts = data_loader.load_test_data()

    images = []
    # Filename format: p{prompt_idx:03d}_i{image_idx:02d}_s{seed}.png
    # We sort by filename to keep them in order
    for img_path in sorted(list(output_dir.glob("p*_i*_s*.png"))):
        try:
            name = img_path.stem
            parts = name.split('_')
            # robust parsing
            prompt_idx = -1
            seed = -1

            for part in parts:
                if part.startswith('p') and part[1:].isdigit():
                    prompt_idx = int(part[1:])
                elif part.startswith('s') and part[1:].isdigit():
                    seed = int(part[1:])

            if prompt_idx != -1:
                prompt_text = prompts[prompt_idx] if prompt_idx < len(prompts) else "Unknown prompt"

                # Construct URL: mounted /assets points to backend/assets
                # Output dir is backend/assets/outputs/{model_id}
                # So URL is /assets/outputs/{model_id}/{filename}
                # Use standard forward slashes for URLs
                url = f"/assets/outputs/{model_id}/{img_path.name}"

                images.append({
                    "filename": img_path.name,
                    "url": url,
                    "prompt": prompt_text,
                    "seed": seed,
                    "prompt_idx": prompt_idx
                })
        except Exception as e:
            print(f"Error parsing metadata for {img_path}: {e}")

    return images

@app.post("/api/generate")
def generate_endpoint(options: state.ScanOptions = Body(default=state.ScanOptions())):
    """Generate images only (no metrics calculation)."""
    if state.generation_state["is_running"]:
        return {"status": "error", "message": "Generation already in progress"}
    return scanner.generate_images_only(options)

@app.post("/api/analyze")
def analyze_endpoint(options: state.ScanOptions = Body(default=state.ScanOptions())):
    """Analyze existing images and compute metrics."""
    return scanner.analyze_models_only(options)

@app.post("/api/scan")
def scan_models(options: state.ScanOptions = Body(default=state.ScanOptions())):
    """Generate images AND analyze (legacy endpoint)."""
    if state.generation_state["is_running"]:
        return {"status": "error", "message": "Generation already in progress"}
    scanner.generate_images_only(options)
    return scanner.analyze_models_only(options)

@app.post("/api/cancel")
def cancel_generation():
    """Cancel ongoing generation."""
    if state.generation_state["is_running"]:
        state.generation_state["should_cancel"] = True
        return {"status": "ok", "message": "Cancellation requested"}
    return {"status": "ok", "message": "No generation in progress"}

@app.get("/api/status")
def get_status():
    """Get current generation status."""
    return {
        "is_running": state.generation_state["is_running"],
        "current_model": state.generation_state["current_model"],
        "progress": state.generation_state["progress"]
    }

@app.delete("/api/models/{model_id}")
def delete_model(model_id: str, delete_file: bool = False):
    target_model = next((m for m in state.models_db if m.id == model_id), None)

    if delete_file and target_model and target_model.path:
        try:
            file_path = Path(target_model.path).resolve()
            models_dir_resolved = data_loader.MODELS_DIR.resolve()

            # Security check: Ensure file is within MODELS_DIR
            if not str(file_path).startswith(str(models_dir_resolved)):
                 raise HTTPException(status_code=403, detail="Cannot delete file outside models directory")

            if file_path.exists():
                if file_path.is_dir() or file_path.is_symlink():
                     raise HTTPException(status_code=403, detail="Cannot delete directories or symlinks")

                file_path.unlink()
                print(f"Deleted file: {file_path}")
            else:
                print(f"File not found for deletion: {file_path}")
        except HTTPException:
            raise
        except Exception as e:
            print(f"Error deleting file: {e}")
            raise HTTPException(status_code=500, detail=str(e)) from e

    # Update DB globally
    # Note: reassigning the list reference in state requires modifying the list in place or using the module attribute
    state.models_db[:] = [m for m in state.models_db if m.id != model_id]
    return {"status": "ok"}

@app.post("/api/models/download")
def download_model(request: state.ModelRequest, background_tasks: BackgroundTasks):
    # TOCTOU protection
    with state.download_state_lock:
        if state.download_state["is_downloading"]:
            raise HTTPException(status_code=400, detail="Download already in progress")
        state.download_state["is_downloading"] = True

    background_tasks.add_task(downloader.download_model_task, request.url, request.name, request.source, request.api_token)
    return {"status": "started"}

@app.get("/api/models/download/status")
def get_download_status():
    with state.download_state_lock:
        return state.download_state.copy()
