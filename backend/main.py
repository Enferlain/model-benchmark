from fastapi import FastAPI, HTTPException, Body, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict
import time
import random
import os
import torch
import logging
import requests
import re
import threading
import traceback
from pathlib import Path

# Suppress uvicorn access log spam for polling endpoints
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Import backend modules
from metrics import MetricsCalculator
import data_loader
import inference

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_calc = MetricsCalculator()
metrics_calc.load_clip() # Load on startup since requirements are installed
metrics_calc.load_lpips() # Load LPIPS for diversity calculation

class ModelRequest(BaseModel):
    url: str
    name: Optional[str] = None
    source: Optional[str] = "Unknown"
    api_token: Optional[str] = None

class ModelResult(BaseModel):
    id: str
    name: str
    source: str
    accuracy: float
    diversity: float
    rating: float
    vqa_score: Optional[float] = 0.0
    lpips_loss: Optional[float] = 0.0
    metrics: Dict[str, float] = {}
    url: str
    path: Optional[str] = None

# In-memory database
models_db: List[ModelResult] = []

@app.on_event("startup")
async def startup_event():
    print("Starting up... (no auto-generation, use /api/generate or /api/analyze)")

# Generation state management
generation_state = {
    "is_running": False,
    "should_cancel": False,
    "current_model": None,
    "progress": {"current": 0, "total": 0}
}

# Download state management
download_state = {
    "is_downloading": False,
    "current_file": None,
    "progress": 0,
    "total": 0,
    "status": "idle", # idle, downloading, completed, error
    "error": None
}
download_state_lock = threading.Lock()

def download_model_task(url: str, name: str, source: str, api_token: Optional[str] = None):
    global download_state

    with download_state_lock:
        download_state["current_file"] = name
        download_state["progress"] = 0
        download_state["total"] = 0
        download_state["status"] = "downloading"
        download_state["error"] = None

    try:
        print(f"Starting download: {url}")
        timeout = int(os.environ.get("REQUEST_TIMEOUT", 30))

        # Helper: Transform URL for HuggingFace blob -> resolve
        if "huggingface.co" in url and "/blob/" in url:
            print("Detected HuggingFace /blob/ URL, converting to /resolve/...")
            url = url.replace("/blob/", "/resolve/")

        headers = {
            "User-Agent": "ModelBenchmarkExplorer/1.0"
        }
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        # Helper: Resolve Civitai Model ID to Download URL
        if "civitai.com/models/" in url and "api/download" not in url:
            try:
                # Extract potential ID
                match = re.search(r"civitai\.com/models/(\d+)", url)
                if match:
                    model_id = match.group(1)
                    print(f"Detected Civitai Model ID {model_id}, attempting to resolve download URL via API...")
                    api_url = f"https://civitai.com/api/v1/models/{model_id}"

                    api_resp = requests.get(api_url, headers=headers, timeout=10)
                    if api_resp.ok:
                        data = api_resp.json()
                        if "modelVersions" in data and len(data["modelVersions"]) > 0:
                            # Use the first (latest) version's download URL
                            download_url = data["modelVersions"][0].get("downloadUrl")
                            if download_url:
                                print(f"Resolved to: {download_url}")
                                url = download_url
                            else:
                                print("No downloadUrl found in API response.")
                        else:
                            print("No modelVersions found in API response.")
                    else:
                        print(f"Civitai API lookup failed: {api_resp.status_code}")
            except Exception as e:
                print(f"Error resolving Civitai URL: {e}")

        response = requests.get(url, stream=True, allow_redirects=True, timeout=timeout, headers=headers)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        with download_state_lock:
            download_state["total"] = total_size

        # Determine filename
        filename = None
        if "content-disposition" in response.headers:
             cd = response.headers["content-disposition"]
             fname = re.findall("filename=(.+)", cd)
             if len(fname) > 0:
                 filename = fname[0].strip('"').strip("'")

        if not filename:
            filename = url.split("/")[-1]
            if "?" in filename:
                filename = filename.split("?")[0]

        # Basic sanitization for extension check
        if not filename or (not filename.endswith(".safetensors") and not filename.endswith(".ckpt")):
             filename = f"{name or 'model'}.safetensors"

        # Safe filename generation (path traversal prevention)
        filename = os.path.basename(filename) # Strip directory components
        # Whitelist safe characters only
        filename = "".join([c for c in filename if c.isalnum() or c in "._- "])
        # Ensure it has a valid extension (again, after sanitization)
        if not (filename.endswith(".safetensors") or filename.endswith(".ckpt")):
             filename += ".safetensors"

        save_path = data_loader.MODELS_DIR / filename

        # Verify save path is within MODELS_DIR
        try:
             save_path = save_path.resolve()
             models_dir_resolved = data_loader.MODELS_DIR.resolve()
             if not str(save_path).startswith(str(models_dir_resolved)):
                 raise ValueError(f"Invalid path: {save_path}")
        except Exception as e:
             raise ValueError(f"Path verification failed: {e}") from e

        print(f"Saving to: {save_path}")

        downloaded = 0
        last_update = 0
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Update progress every 1MB to reduce lock contention
                    if downloaded - last_update > 1024 * 1024:
                        with download_state_lock:
                            download_state["progress"] = downloaded
                        last_update = downloaded

        # Final update
        with download_state_lock:
            download_state["progress"] = downloaded
            download_state["status"] = "completed"

        print("Download complete.")

        # Add to models_db
        new_model = {
            "id": save_path.stem,
            "name": name or save_path.stem.replace("-", " ").title(),
            "source": source,
            "url": url,
            "path": str(save_path),
            "accuracy": 0.0,
            "diversity": 0.0,
            "rating": 0.0,
            "metrics": {"accuracy": 0.0, "diversity": 0.0}
        }

        # Check if exists
        exists = False
        for m in models_db:
             if m.id == new_model["id"]:
                 exists = True
                 break
        if not exists:
             models_db.append(ModelResult(**new_model))

    except Exception as e:
        print(f"Download error details: {traceback.format_exc()}")
        with download_state_lock:
            download_state["status"] = "error"
            download_state["error"] = str(e)
        print(f"Download error: {e}")
    finally:
        with download_state_lock:
            download_state["is_downloading"] = False

def check_cancelled():
    """Check if generation should be cancelled. Call this in generation loops."""
    return generation_state["should_cancel"]

class ScanOptions(BaseModel):
    sampler: Literal["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver", "dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"] = "euler_a"
    steps: int = 28
    guidance_scale: float = 5.0
    seed: int = 218
    images_per_prompt: int = 1  # Set > 1 for LPIPS diversity measurement
    num_prompts: int = 10  # Number of prompts to use from test data
    width: int = 1024
    height: int = 1536

def load_local_models(options: ScanOptions = ScanOptions()):
    print(f"Loading local models with options: {options}")
    local_models = data_loader.get_available_models_from_disk()
    print(f"Found {len(local_models)} local models.")
    
    # Get prompts (we only need the text prompts)
    _, prompts = data_loader.load_test_data()
    if not prompts:
        print("No prompts found in assets. Skipping inference.")
        return

    inferencer = None # Lazy load

    for lm in local_models:
        # Check if already exists in DB
        if any(m.id == lm['id'] for m in models_db):
            continue
            
        model_id = lm['id']
        model_path = lm['path'] # data_loader needs to ensure this field exists
        
        # Check output directory
        output_dir = data_loader.ASSETS_DIR / "outputs" / model_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check existing images using new naming scheme: p{prompt_idx:03d}_i{image_idx:02d}_s{seed}.png
        existing_images = list(output_dir.glob("p*_i*_s*.png"))
        
        # Count images per prompt
        prompt_image_counts = {}
        for img_path in existing_images:
            # Parse prompt index from filename
            name = img_path.stem  # e.g., p000_i00_s218
            try:
                prompt_idx = int(name.split('_')[0][1:])  # Extract number after 'p'
                prompt_image_counts[prompt_idx] = prompt_image_counts.get(prompt_idx, 0) + 1
            except:
                pass
        
        # Determine which prompts need more images
        target_prompts = prompts[:options.num_prompts]
        prompts_needing_images = []
        images_needed_per_prompt = []
        
        for i, prompt in enumerate(target_prompts):
            current_count = prompt_image_counts.get(i, 0)
            if current_count < options.images_per_prompt:
                prompts_needing_images.append((i, prompt))
                images_needed_per_prompt.append(options.images_per_prompt - current_count)
        
        if prompts_needing_images:
            print(f"Need to generate images for {len(prompts_needing_images)} prompts for {model_id}")
            
            try:
                if inferencer is None:
                    inferencer = inference.SDXLInferencer()
                
                inferencer.load_model(model_path)
                
                # Detect V-Prediction models
                extra_args = []
                lower_name = model_path.lower()
                if any(x in lower_name for x in ["v-prediction", "v-pred", "v_pred", "_v2"]):
                    print(f"Detected V-Prediction model: {model_id}")
                    extra_args.append("--v_parameterization")

                # Generate images for each prompt that needs them
                for (prompt_idx, prompt), needed_count in zip(prompts_needing_images, images_needed_per_prompt):
                    existing_for_prompt = prompt_image_counts.get(prompt_idx, 0)
                    
                    for img_num in range(needed_count):
                        image_idx = existing_for_prompt + img_num
                        current_seed = options.seed + prompt_idx * 1000 + image_idx  # Unique seed per image
                        
                        gen_iterator = inferencer.generate(
                            prompts=[prompt],
                            negative_prompt="worst quality, low quality, lowres, artist name, signature, bad anatomy",
                            steps=options.steps, 
                            guidance_scale=options.guidance_scale,
                            width=options.width,
                            height=options.height,
                            seed=current_seed,
                            sampler=options.sampler,
                            images_per_prompt=1,  # Generate one at a time for proper naming
                            extra_args=extra_args
                        )
                        
                        for img in gen_iterator:
                            if img:
                                # Naming: p{prompt_idx}_i{image_idx}_s{seed}.png
                                save_path = output_dir / f"p{prompt_idx:03d}_i{image_idx:02d}_s{current_seed}.png"
                                img.save(save_path)
                                print(f"Saved {save_path}")
                
                # Reload from disk
                existing_images = list(output_dir.glob("p*_i*_s*.png"))
            except Exception as e:
                print(f"Failed to run inference on {model_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Now compute metrics on these images (existing or new)
        # Group images by prompt for LPIPS diversity calculation
        from PIL import Image
        grouped_images = {}  # prompt_idx -> [PIL Images]
        flat_images = []
        flat_prompts = []
        
        for img_path in sorted(existing_images):
            try:
                name = img_path.stem  # e.g., p000_i00_s218
                prompt_idx = int(name.split('_')[0][1:])  # Extract number after 'p'
                
                img = Image.open(img_path).convert("RGB")
                flat_images.append(img)
                
                # Map prompt index to prompt text
                if prompt_idx < len(prompts):
                    flat_prompts.append(prompts[prompt_idx])
                else:
                    flat_prompts.append("")
                
                # Group for LPIPS
                if prompt_idx not in grouped_images:
                    grouped_images[prompt_idx] = []
                grouped_images[prompt_idx].append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

        print(f"Loaded {len(flat_images)} images for analysis of {model_id}.")
        print(f"Grouped into {len(grouped_images)} prompt groups for LPIPS.")
        
        # Calculate metrics if we have any images
        if flat_images:
            try:
                # Pass grouped_images for LPIPS diversity
                metrics = metrics_calc.calculate_metrics(flat_images, flat_prompts, grouped_images)
                lm['accuracy'] = round(metrics['clip_score'], 3)
                lm['diversity'] = round(metrics['diversity_score'], 3)
                lm['vqa_score'] = round(random.uniform(0.7, 0.9), 2)  # Still mocked
                lm['lpips_loss'] = round(metrics.get('lpips_diversity', 0.0), 3)  # Real LPIPS
                
                lm['metrics'] = {
                    'accuracy': lm['accuracy'],
                    'diversity': lm['diversity'],
                    'rating': lm['rating'],
                    'vqa_score': lm['vqa_score'],
                    'lpips_loss': lm['lpips_loss']
                }
            except Exception as e:
                print(f"Error calculating metrics for {model_id}: {e}")
                import traceback
                traceback.print_exc()
                lm['accuracy'] = 0.0
                lm['diversity'] = 0.0
                lm['metrics'] = {'accuracy': 0.0, 'diversity': 0.0}
        else:
            print(f"No images found for {model_id}. Using zeros.")
            lm['accuracy'] = 0.0
            lm['diversity'] = 0.0
            lm['metrics'] = {'accuracy': 0.0, 'diversity': 0.0}
        
        models_db.append(ModelResult(**lm))

    # Cleanup to save VRAM
    if inferencer:
        del inferencer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def generate_images_only(options: ScanOptions):
    """Generate images without computing metrics. Supports cancellation."""
    generation_state["is_running"] = True
    generation_state["should_cancel"] = False
    
    try:
        local_models = data_loader.get_available_models_from_disk()
        _, prompts = data_loader.load_test_data()
        
        if not prompts:
            return {"status": "error", "message": "No prompts found"}
        
        inferencer = None
        total_images_needed = 0
        images_generated = 0
        
        # Calculate total work
        for lm in local_models:
            output_dir = data_loader.ASSETS_DIR / "outputs" / lm['id']
            output_dir.mkdir(parents=True, exist_ok=True)
            existing_images = list(output_dir.glob("p*_i*_s*.png"))
            
            prompt_image_counts = {}
            for img_path in existing_images:
                try:
                    name = img_path.stem
                    prompt_idx = int(name.split('_')[0][1:])
                    prompt_image_counts[prompt_idx] = prompt_image_counts.get(prompt_idx, 0) + 1
                except:
                    pass
            
            for i in range(min(options.num_prompts, len(prompts))):
                current_count = prompt_image_counts.get(i, 0)
                if current_count < options.images_per_prompt:
                    total_images_needed += options.images_per_prompt - current_count
        
        generation_state["progress"] = {"current": 0, "total": total_images_needed}
        
        for lm in local_models:
            if check_cancelled():
                break
                
            model_id = lm['id']
            model_path = lm['path']
            generation_state["current_model"] = model_id
            
            output_dir = data_loader.ASSETS_DIR / "outputs" / model_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            existing_images = list(output_dir.glob("p*_i*_s*.png"))
            # Track which specific image indices exist for each prompt (handles gaps)
            prompt_existing_indices = {}  # prompt_idx -> set of image indices
            for img_path in existing_images:
                try:
                    name = img_path.stem  # e.g., p001_i02_s218
                    parts = name.split('_')
                    prompt_idx = int(parts[0][1:])  # Extract number after 'p'
                    image_idx = int(parts[1][1:])   # Extract number after 'i'
                    if prompt_idx not in prompt_existing_indices:
                        prompt_existing_indices[prompt_idx] = set()
                    prompt_existing_indices[prompt_idx].add(image_idx)
                except:
                    pass
            
            target_prompts = prompts[:options.num_prompts]
            prompts_needing_images = []
            missing_indices_per_prompt = []  # List of sets of missing indices
            
            for i, prompt in enumerate(target_prompts):
                existing_indices = prompt_existing_indices.get(i, set())
                needed_indices = set(range(options.images_per_prompt))
                missing_indices = needed_indices - existing_indices
                if missing_indices:
                    prompts_needing_images.append((i, prompt))
                    missing_indices_per_prompt.append(sorted(missing_indices))
            
            if not prompts_needing_images:
                continue
                
            try:
                if inferencer is None:
                    inferencer = inference.SDXLInferencer()
                
                inferencer.load_model(model_path)
                
                extra_args = []
                lower_name = model_path.lower()
                if any(x in lower_name for x in ["v-prediction", "v-pred", "v_pred", "_v2"]):
                    extra_args.append("--v_parameterization")
                
                # Batch ALL prompts into a single generate() call (model loads only once!)
                # Build list of (prompt, image_idx) pairs for all missing images
                generation_queue = []  # [(prompt_idx, prompt_text, image_idx), ...]
                
                for (prompt_idx, prompt), missing_indices in zip(prompts_needing_images, missing_indices_per_prompt):
                    for image_idx in missing_indices:
                        generation_queue.append((prompt_idx, prompt, image_idx))
                
                if not generation_queue or check_cancelled():
                    continue
                
                # Group by prompt for batching (script needs unique prompts)
                # We'll repeat each prompt for how many images it needs
                prompts_to_gen = [item[1] for item in generation_queue]
                # Calculate per-prompt seeds (seed + image_idx for each)
                per_prompt_seeds = [options.seed + item[2] for item in generation_queue]
                
                # Single subprocess call - model loads once!
                # Each line gets its specific seed via --d syntax in the prompt file
                gen_iterator = inferencer.generate(
                    prompts=prompts_to_gen,
                    negative_prompt="worst quality, low quality, lowres, artist name, signature, bad anatomy",
                    steps=options.steps,
                    guidance_scale=options.guidance_scale,
                    width=options.width,
                    height=options.height,
                    seed=options.seed,  # Fallback, per_prompt_seeds takes precedence
                    sampler=options.sampler,
                    images_per_prompt=1,  # Each prompt line = 1 image
                    extra_args=extra_args,
                    per_prompt_seeds=per_prompt_seeds
                )
                
                # Save images with exact indices from queue
                for idx, img in enumerate(gen_iterator):
                    if check_cancelled():
                        break
                    
                    if idx >= len(generation_queue):
                        break
                    
                    if img:
                        prompt_idx, _, image_idx = generation_queue[idx]
                        actual_seed = options.seed + image_idx  # Seed matches the image index
                        save_path = output_dir / f"p{prompt_idx:03d}_i{image_idx:02d}_s{actual_seed}.png"
                        img.save(save_path)
                        images_generated += 1
                        generation_state["progress"]["current"] = images_generated
                        print(f"[{images_generated}/{total_images_needed}] Saved {save_path}")
                                
            except Exception as e:
                print(f"Failed to generate for {model_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Cleanup
        if inferencer:
            del inferencer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        return {
            "status": "cancelled" if check_cancelled() else "complete",
            "images_generated": images_generated
        }
    finally:
        generation_state["is_running"] = False
        generation_state["current_model"] = None

def analyze_models_only(options: ScanOptions):
    """Analyze existing images and compute metrics (no generation)."""
    models_db.clear()
    
    local_models = data_loader.get_available_models_from_disk()
    _, prompts = data_loader.load_test_data()
    
    if not prompts:
        return {"status": "error", "message": "No prompts found"}
    
    for lm in local_models:
        model_id = lm['id']
        output_dir = data_loader.ASSETS_DIR / "outputs" / model_id
        
        existing_images = list(output_dir.glob("p*_i*_s*.png"))
        
        # Group images by prompt
        from PIL import Image
        grouped_images = {}
        flat_images = []
        flat_prompts = []
        
        for img_path in sorted(existing_images):
            try:
                name = img_path.stem
                prompt_idx = int(name.split('_')[0][1:])
                
                img = Image.open(img_path).convert("RGB")
                flat_images.append(img)
                
                if prompt_idx < len(prompts):
                    flat_prompts.append(prompts[prompt_idx])
                else:
                    flat_prompts.append("")
                
                if prompt_idx not in grouped_images:
                    grouped_images[prompt_idx] = []
                grouped_images[prompt_idx].append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        print(f"Analyzing {model_id}: {len(flat_images)} images in {len(grouped_images)} groups")
        
        if flat_images:
            try:
                metrics = metrics_calc.calculate_metrics(flat_images, flat_prompts, grouped_images)
                lm['accuracy'] = round(metrics['clip_score'], 3)
                lm['diversity'] = round(metrics['diversity_score'], 3)
                lm['vqa_score'] = round(random.uniform(0.7, 0.9), 2)
                lm['lpips_loss'] = round(metrics.get('lpips_diversity', 0.0), 3)
                
                lm['metrics'] = {
                    'accuracy': lm['accuracy'],
                    'diversity': lm['diversity'],
                    'rating': lm['rating'],
                    'vqa_score': lm['vqa_score'],
                    'lpips_loss': lm['lpips_loss']
                }
            except Exception as e:
                print(f"Error calculating metrics for {model_id}: {e}")
                lm['accuracy'] = 0.0
                lm['diversity'] = 0.0
                lm['metrics'] = {'accuracy': 0.0, 'diversity': 0.0}
        else:
            lm['accuracy'] = 0.0
            lm['diversity'] = 0.0
            lm['metrics'] = {'accuracy': 0.0, 'diversity': 0.0}
        
        models_db.append(ModelResult(**lm))
    
    return models_db

# API Endpoints
@app.get("/api/models")
def get_models():
    return models_db

@app.post("/api/generate")
def generate_endpoint(options: ScanOptions = Body(default=ScanOptions())):
    """Generate images only (no metrics calculation)."""
    if generation_state["is_running"]:
        return {"status": "error", "message": "Generation already in progress"}
    return generate_images_only(options)

@app.post("/api/analyze")
def analyze_endpoint(options: ScanOptions = Body(default=ScanOptions())):
    """Analyze existing images and compute metrics."""
    return analyze_models_only(options)

@app.post("/api/scan")
def scan_models(options: ScanOptions = Body(default=ScanOptions())):
    """Generate images AND analyze (legacy endpoint)."""
    if generation_state["is_running"]:
        return {"status": "error", "message": "Generation already in progress"}
    generate_images_only(options)
    return analyze_models_only(options)

@app.post("/api/cancel")
def cancel_generation():
    """Cancel ongoing generation."""
    if generation_state["is_running"]:
        generation_state["should_cancel"] = True
        return {"status": "ok", "message": "Cancellation requested"}
    return {"status": "ok", "message": "No generation in progress"}

@app.get("/api/status")
def get_status():
    """Get current generation status."""
    return {
        "is_running": generation_state["is_running"],
        "current_model": generation_state["current_model"],
        "progress": generation_state["progress"]
    }

@app.delete("/api/models/{model_id}")
def delete_model(model_id: str, delete_file: bool = False):
    global models_db

    target_model = next((m for m in models_db if m.id == model_id), None)

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

    models_db = [m for m in models_db if m.id != model_id]
    return {"status": "ok"}

@app.post("/api/models/download")
def download_model(request: ModelRequest, background_tasks: BackgroundTasks):
    # TOCTOU protection
    with download_state_lock:
        if download_state["is_downloading"]:
            raise HTTPException(status_code=400, detail="Download already in progress")
        download_state["is_downloading"] = True

    background_tasks.add_task(download_model_task, request.url, request.name, request.source, request.api_token)
    return {"status": "started"}

@app.get("/api/models/download/status")
def get_download_status():
    with download_state_lock:
        return download_state.copy()
