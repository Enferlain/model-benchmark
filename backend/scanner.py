import random
import torch
import torch
import data_loader
import inference
from metrics import MetricsCalculator
from state import models_db, generation_state, ScanOptions, ModelResult

# Initialize metrics calculator lazily
metrics_calc = None

def get_metrics_calc():
    global metrics_calc
    if metrics_calc is None:
        metrics_calc = MetricsCalculator()
        metrics_calc.load_clip() # Load on startup since requirements are installed
        metrics_calc.load_lpips() # Load LPIPS for diversity calculation
    return metrics_calc

def check_cancelled():
    """Check if generation should be cancelled. Call this in generation loops."""
    return generation_state["should_cancel"]

def load_local_models(options: ScanOptions = ScanOptions()):
    print(f"Loading local models with options: {options}")
    local_models = data_loader.get_available_models_from_disk()
    print(f"Found {len(local_models)} local models.")
    
    # Get prompts (we only need the text prompts)
    # Get prompts (we only need the text prompts)
    prompts = data_loader.load_prompts_only()
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
                mc = get_metrics_calc()
                metrics = mc.calculate_metrics(flat_images, flat_prompts, grouped_images)
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
        prompts = data_loader.load_prompts_only()
        
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
    prompts = data_loader.load_prompts_only()
    
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
                mc = get_metrics_calc()
                metrics = mc.calculate_metrics(flat_images, flat_prompts, grouped_images)
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

def scan_models_light(options: ScanOptions = ScanOptions()):
    """
    Fast scan of models on disk without loading images or calculating metrics.
    Populates models_db with available models and file counts.
    """
    models_db.clear()
    local_models = data_loader.get_available_models_from_disk()
    
    # Just need prompts for count, not for analysis
    # Use optimized prompt loader
    prompts = data_loader.load_prompts_only()
    
    for lm in local_models:
        model_id = lm['id']
        output_dir = data_loader.ASSETS_DIR / "outputs" / model_id
        
        # Count existing images
        existing_images = list(output_dir.glob("p*_i*_s*.png"))
        
        # Basic stats
        lm['accuracy'] = 0.0
        lm['diversity'] = 0.0
        lm['metrics'] = {'accuracy': 0.0, 'diversity': 0.0}
        
        # We could potentially store image count here if needed, 
        # but the frontend fetches outputs separately anyway.
        
        models_db.append(ModelResult(**lm))
        
    print(f"Fast scan complete. Found {len(models_db)} models.")
    return models_db
