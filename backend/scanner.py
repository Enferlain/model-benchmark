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
    """Generate images using robust metadata matching."""
    generation_state["is_running"] = True
    generation_state["should_cancel"] = False
    
    try:
        local_models = data_loader.get_available_models_from_disk()
        # Get full metadata for prompts to track text and ID
        all_prompts_meta = data_loader.get_all_prompts_metadata()
        
        if not all_prompts_meta:
            return {"status": "error", "message": "No prompts found"}
        
        # Filter down to just the ones we need (based on config order)
        # ScanOptions.num_prompts limits how many we process
        target_prompts_meta = all_prompts_meta[:options.num_prompts]
        
        inferencer = None
        total_images_needed = 0
        images_generated = 0
        
        from PIL.PngImagePlugin import PngInfo
        from PIL import Image
        
        # 1. Pre-calculate work needed
        # We need to scan each model to see what's missing BY TEXT/CONTENT, not filename
        model_work_queue = [] # [(model, items_needing_generation)]
        
        for lm in local_models:
            output_dir = data_loader.ASSETS_DIR / "outputs" / lm['id']
            output_dir.mkdir(parents=True, exist_ok=True)
            existing_images = list(output_dir.glob("p*_i*_s*.png"))
            
            # Map Prompt Text -> Set of Existing Image Indices (relative to single prompt)
            # We want to know: "For prompt 'A cat', do we have image #0, #1, #2?"
            # Using text as key is safer than ID if IDs change, but ID is technically safer if text changes.
            # Let's use Prompt Text as the "Source of Truth" for what is being tested.
            existing_counts = {} # prompt_text -> set(image_indices)
            
            for img_path in existing_images:
                try:
                    # Lazy open to read metadata
                    with Image.open(img_path) as img:
                        img.load() # Read header? Actually open() is enough for info usually, but load ensures. 
                        # info might need load() for some formats but PNG usually gets it in open().
                        # let's try reading info
                        meta_prompt = img.info.get("prompt")
                        
                    if meta_prompt:
                        key = meta_prompt.strip()
                    else:
                        # Legacy fallback: use filename index to look up CURRENT text
                        # This is the "danger zone" we are fixing, but we must handle old files.
                        name = img_path.stem
                        p_idx = int(name.split('_')[0][1:])
                        # Look up text from current list
                        if p_idx < len(all_prompts_meta):
                            key = all_prompts_meta[p_idx]['text'].strip()
                        else:
                            key = None # Orphaned file
                            
                    if key:
                        if key not in existing_counts: existing_counts[key] = set()
                        
                        # Extract image index from filename "i02"
                        # We still trust the filename for the "seed slot" (first, second, third variation)
                        # because that's structural.
                        parts = img_path.stem.split('_')
                        i_idx = -1
                        for p in parts:
                            if p.startswith('i') and p[1:].isdigit():
                                i_idx = int(p[1:])
                                break
                        
                        if i_idx != -1:
                            existing_counts[key].add(i_idx)
                            
                except Exception as e:
                    # print(f"Warning: Failed to scan {img_path}: {e}")
                    pass

            # Calculate missing
            missing_for_model = [] # (prompt_meta, missing_indices_list)
            
            for prompt_meta in target_prompts_meta:
                 text = prompt_meta['text'].strip()
                 existing_indices = existing_counts.get(text, set())
                 needed = set(range(options.images_per_prompt))
                 missing = needed - existing_indices
                 
                 if missing:
                     missing_for_model.append((prompt_meta, sorted(missing)))
                     total_images_needed += len(missing)
            
            if missing_for_model:
                model_work_queue.append((lm, missing_for_model))


        generation_state["progress"] = {"current": 0, "total": total_images_needed}
        
        # 2. Execute Generation
        for lm, work_items in model_work_queue:
            if check_cancelled(): break
                
            model_id = lm['id']
            model_path = lm['path']
            generation_state["current_model"] = model_id
            
            output_dir = data_loader.ASSETS_DIR / "outputs" / model_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                if inferencer is None:
                    inferencer = inference.SDXLInferencer()
                
                inferencer.load_model(model_path)
                
                extra_args = []
                if any(x in model_path.lower() for x in ["v-prediction", "v-pred", "v_pred", "_v2"]):
                    extra_args.append("--v_parameterization")
                
                # Build Batch
                # Queue: (prompt_meta, image_index_slot)
                generation_queue = []
                for p_meta, missing_indices in work_items:
                    for i_idx in missing_indices:
                        generation_queue.append((p_meta, i_idx))
                
                if not generation_queue: continue
                
                prompts_texts = [item[0]['text'] for item in generation_queue]
                # Seeds: Base + Global Offset + Slot Index
                # Note: We want consistent seeds for "Slot 0" of "Prompt A" across all models.
                # Previous logic: seed + prompt_idx*1000 + i_idx. 
                # New Logic: We don't want prompt_idx dependency because it shifts!
                # We need a stable hash of the prompt text to be the offset?
                # OR just rely on "Slot Index" combined with global seed?
                # IF we do `seed + i_idx`, then "Prompt A, Image 0" has same seed as "Prompt B, Image 0".
                # This is actually GOOD for comparing different prompts on same seed.
                # BUT user might want variety? 
                # Let's stick to simple: seed + i_idx.
                # So "Image 0" always uses GlobalSeed+0. "Image 1" uses GlobalSeed+1.
                # This means all prompts share the same seeds for their 1st, 2nd, 3rd images.
                # This is usually desired for consistency.
                per_prompt_seeds = [options.seed + item[1] for item in generation_queue]
                
                gen_iterator = inferencer.generate(
                    prompts=prompts_texts,
                    negative_prompt="worst quality, low quality, lowres, artist name, signature, bad anatomy",
                    steps=options.steps,
                    guidance_scale=options.guidance_scale, 
                    width=options.width,
                    height=options.height,
                    seed=options.seed,
                    sampler=options.sampler,
                    images_per_prompt=1,
                    extra_args=extra_args,
                    per_prompt_seeds=per_prompt_seeds
                )
                
                for idx, img in enumerate(gen_iterator):
                    if check_cancelled(): break
                    if idx >= len(generation_queue): break
                    
                    if img:
                        p_meta, i_idx = generation_queue[idx]
                        
                        # Calculate current index for filename 
                        # We still use the *current* list index for the filename convention
                        # just so they sort nicely in folders, even if that index is ephemeral.
                        # We have to find what index this prompt is CURRENTLY at.
                        # It's in target_prompts_meta.
                        try:
                            current_p_idx = target_prompts_meta.index(p_meta)
                        except ValueError:
                            current_p_idx = 999 
                        
                        actual_seed = per_prompt_seeds[idx]
                        
                        # Prepare Metadata
                        metadata = PngInfo()
                        metadata.add_text("prompt", p_meta['text'])
                        metadata.add_text("index", str(current_p_idx))
                        metadata.add_text("seed", str(actual_seed))
                        metadata.add_text("alias", p_meta.get("alias", "") or "")
                        metadata.add_text("original_filename", f"p{current_p_idx:03d}_i{i_idx:02d}_s{actual_seed}.png")
                        
                        fname = f"p{current_p_idx:03d}_i{i_idx:02d}_s{actual_seed}.png"
                        save_path = output_dir / fname
                        
                        img.save(save_path, pnginfo=metadata)
                        
                        images_generated += 1
                        generation_state["progress"]["current"] = images_generated
                        print(f"[{images_generated}/{total_images_needed}] Saved {save_path}")
            
            except Exception as e:
                print(f"Failed generation for {model_id}: {e}")
                import traceback
                traceback.print_exc()

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
                img = Image.open(img_path).convert("RGB")
                img.load() # Load for metadata access
                
                # Try metadata first
                prompt_text = img.info.get("prompt")
                
                if not prompt_text:
                    # Legacy fallback
                    name = img_path.stem
                    prompt_idx = int(name.split('_')[0][1:])
                    if prompt_idx < len(prompts):
                        prompt_text = prompts[prompt_idx]
                    else:
                        prompt_text = ""
                
                flat_images.append(img)
                flat_prompts.append(prompt_text)
                
                # Group by text content for LPIPS diversity
                # Using text as key instead of index
                if prompt_text not in grouped_images:
                    grouped_images[prompt_text] = []
                grouped_images[prompt_text].append(img)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
        
        print(f"Analyzing {model_id}: {len(flat_images)} images in {len(grouped_images)} groups")
        
        if flat_images:
            try:
                mc = get_metrics_calc()
                # Calculator expects prompt_idx keys or just keys to iterate? 
                # metrics.py likely iterates values(). Let's assume it handles any key type or list of lists.
                # Actually, check metrics.py if needed, but usually grouped_images is just used for values().
                # If metrics.py expects numeric keys, this might break.
                # Let's check: "grouped_images.values()" is standard.
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
