from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Literal, Dict
import time
import random
import os
import torch

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

# In-memory database
models_db: List[ModelResult] = []

@app.on_event("startup")
async def startup_event():
    print("Starting up...")
    load_local_models()

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

@app.get("/api/models")
def get_models():
    return models_db

@app.post("/api/scan")
def scan_models(options: ScanOptions = Body(default=ScanOptions())):
    models_db.clear()
    load_local_models(options)
    return models_db

@app.post("/api/analyze")
def analyze_model(request: ModelRequest):
    # This might be used for URL models, but for local we rely on scan
    # For now, just return mock if not found
    return {"status": "ok", "message": "Analysis started"}

@app.delete("/api/models/{model_id}")
def delete_model(model_id: str):
    global models_db
    models_db = [m for m in models_db if m.id != model_id]
    return {"status": "ok"}
