import random
import torch
import shutil
import data_loader
import inference
from metrics import MetricsCalculator
from state import models_db, generation_state, ScanOptions, ModelResult

import hashlib
from typing import Optional
from pathlib import Path
from datetime import datetime
from sqlmodel import select, desc
import database as db
from database import Model, BenchmarkRun, ModelResult as DBModelResult

# Initialize metrics calculator lazily
metrics_calc = None


def get_metrics_calc():
    global metrics_calc
    if metrics_calc is None:
        metrics_calc = MetricsCalculator()
        metrics_calc.load_clip()  # Load on startup since requirements are installed
        metrics_calc.load_lpips()  # Load LPIPS for diversity calculation
    return metrics_calc


def compute_model_hash(file_path: Path) -> str:
    """Computes BLAKE3 hash of the file. Reads in chunks."""
    # Start with a fast partial hash? No, user requested SHA256/BLAKE3 for correctness.
    # BLAKE3 is fast. If not available, use SHA256.
    # requirements.txt has `blake3`.
    try:
        import blake3

        hasher = blake3.blake3()
    except ImportError:
        hasher = hashlib.sha256()

    try:
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error hashing {file_path}: {e}")
        return "error_hashing_" + str(random.randint(0, 999999))


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

    inferencer = None  # Lazy load

    for lm in local_models:
        # Check if already exists in DB
        if any(m.id == lm["id"] for m in models_db):
            continue

        model_id = lm["id"]
        model_path = lm["path"]  # data_loader needs to ensure this field exists

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
                prompt_idx = int(name.split("_")[0][1:])  # Extract number after 'p'
                prompt_image_counts[prompt_idx] = (
                    prompt_image_counts.get(prompt_idx, 0) + 1
                )
            except:
                pass

        # Determine which prompts need more images
        target_prompts = prompts[: options.num_prompts]

        prompts_needing_images = []
        images_needed_per_prompt = []

        for i, prompt in enumerate(target_prompts):
            current_count = prompt_image_counts.get(i, 0)
            if current_count < options.images_per_prompt:
                prompts_needing_images.append((i, prompt))
                images_needed_per_prompt.append(
                    options.images_per_prompt - current_count
                )

        if prompts_needing_images:
            print(
                f"Need to generate images for {len(prompts_needing_images)} prompts for {model_id}"
            )

            try:
                if inferencer is None:
                    inferencer = inference.SDXLInferencer()

                inferencer.load_model(model_path)

                # Detect V-Prediction models
                extra_args = []
                lower_name = model_path.lower()
                if any(
                    x in lower_name for x in ["v-prediction", "v-pred", "v_pred", "_v2"]
                ):
                    print(f"Detected V-Prediction model: {model_id}")
                    extra_args.append("--v_parameterization")

                # Generate images for each prompt that needs them
                for (prompt_idx, prompt), needed_count in zip(
                    prompts_needing_images, images_needed_per_prompt
                ):
                    existing_for_prompt = prompt_image_counts.get(prompt_idx, 0)

                    for img_num in range(needed_count):
                        image_idx = existing_for_prompt + img_num
                        current_seed = (
                            options.seed + prompt_idx * 1000 + image_idx
                        )  # Unique seed per image

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
                            extra_args=extra_args,
                        )

                        for img in gen_iterator:
                            if img:
                                # Naming: p{prompt_idx}_i{image_idx}_s{seed}.png
                                save_path = (
                                    output_dir
                                    / f"p{prompt_idx:03d}_i{image_idx:02d}_s{current_seed}.png"
                                )
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
                prompt_idx = int(name.split("_")[0][1:])  # Extract number after 'p'

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
                metrics = mc.calculate_metrics(
                    flat_images, flat_prompts, grouped_images
                )
                lm["accuracy"] = round(metrics["clip_score"], 3)
                lm["diversity"] = round(metrics["diversity_score"], 3)
                lm["vqa_score"] = round(random.uniform(0.7, 0.9), 2)  # Still mocked
                lm["lpips_loss"] = round(
                    metrics.get("lpips_diversity", 0.0), 3
                )  # Real LPIPS

                lm["metrics"] = {
                    "accuracy": lm["accuracy"],
                    "diversity": lm["diversity"],
                    "rating": lm["rating"],
                    "vqa_score": lm["vqa_score"],
                    "lpips_loss": lm["lpips_loss"],
                }
            except Exception as e:
                print(f"Error calculating metrics for {model_id}: {e}")
                import traceback

                traceback.print_exc()
                lm["accuracy"] = 0.0
                lm["diversity"] = 0.0
                lm["metrics"] = {"accuracy": 0.0, "diversity": 0.0}
        else:
            print(f"No images found for {model_id}. Using zeros.")
            lm["accuracy"] = 0.0
            lm["diversity"] = 0.0
            lm["metrics"] = {"accuracy": 0.0, "diversity": 0.0}

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
        # Get full metadata for prompts to track text and ID
        all_prompts_meta = data_loader.get_all_prompts_metadata()

        if not all_prompts_meta:
            return {"status": "error", "message": "No prompts found"}

        target_prompts_meta = all_prompts_meta[: options.num_prompts]

        inferencer = None
        total_images_needed = 0
        images_generated = 0

        from PIL.PngImagePlugin import PngInfo
        from PIL import Image

        with db.get_session() as session:
            # 1. Pre-calculate work needed
            model_work_queue = []  # [(model_result, items_needing_generation)]

            # Iterate over ACTIVE SESSION models only (Active Context)
            valid_models = []
            for m_res in models_db:
                # We need full Model object or at least enough data.
                # state.ModelResult has path, name, prediction_type.
                # We can use that directly.
                valid_models.append(m_res)

            for m in valid_models:
                # Use Name as ID for folder (User Request)
                output_dir = data_loader.ASSETS_DIR / "outputs" / m.name
                output_dir.mkdir(parents=True, exist_ok=True)
                existing_images = list(output_dir.glob("p*_i*_s*.png"))

                existing_counts = {}  # prompt_text -> set(image_indices)

                for img_path in existing_images:
                    try:
                        with Image.open(img_path) as img:
                            img.load()
                            meta_prompt = img.info.get("prompt")

                        if meta_prompt:
                            key = meta_prompt.strip()
                        else:
                            # Legacy fallback (less reliable without filename match to prompts list)
                            key = None

                        if key:
                            if key not in existing_counts:
                                existing_counts[key] = set()

                            parts = img_path.stem.split("_")
                            i_idx = -1
                            for p in parts:
                                if p.startswith("i") and p[1:].isdigit():
                                    i_idx = int(p[1:])
                                    break

                            if i_idx != -1:
                                existing_counts[key].add(i_idx)

                    except Exception:
                        pass

                # Calculate missing
                missing_for_model = []

                for prompt_meta in target_prompts_meta:
                    text = prompt_meta["text"].strip()
                    existing_indices = existing_counts.get(text, set())
                    needed = set(range(options.images_per_prompt))
                    missing = needed - existing_indices

                    if missing:
                        missing_for_model.append((prompt_meta, sorted(missing)))
                        total_images_needed += len(missing)

                if missing_for_model:
                    model_work_queue.append((m, missing_for_model))

            generation_state["progress"] = {"current": 0, "total": total_images_needed}

            # 2. Execute Generation
            for m, work_items in model_work_queue:
                if check_cancelled():
                    break

                model_id = m.hash  # API uses hash as ID
                model_path = m.path
                generation_state["current_model"] = m.name

                output_dir = data_loader.ASSETS_DIR / "outputs" / m.name
                output_dir.mkdir(parents=True, exist_ok=True)

                try:
                    if inferencer is None:
                        inferencer = inference.SDXLInferencer()

                    # Verify path exists
                    if not Path(model_path).exists():
                        print(f"Skipping {m.name}: File not found at {model_path}")
                        continue

                    inferencer.load_model(model_path)

                    extra_args = []
                    # Use DB prediction type if available!
                    if m.prediction_type == "v_prediction":
                        extra_args.append("--v_parameterization")
                    elif any(
                        x in model_path.lower()
                        for x in ["v-prediction", "v-pred", "v_pred", "_v2"]
                    ):
                        # Fallback heuristic
                        extra_args.append("--v_parameterization")

                    # Build Batch
                    generation_queue = []
                    for p_meta, missing_indices in work_items:
                        for i_idx in missing_indices:
                            generation_queue.append((p_meta, i_idx))

                    if not generation_queue:
                        continue

                    prompts_texts = [item[0]["text"] for item in generation_queue]
                    per_prompt_seeds = [
                        options.seed + item[1] for item in generation_queue
                    ]

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
                        per_prompt_seeds=per_prompt_seeds,
                    )

                    for idx, img in enumerate(gen_iterator):
                        if check_cancelled():
                            break
                        if idx >= len(generation_queue):
                            break

                        if img:
                            p_meta, i_idx = generation_queue[idx]

                            try:
                                current_p_idx = target_prompts_meta.index(p_meta)
                            except ValueError:
                                current_p_idx = 999

                            actual_seed = per_prompt_seeds[idx]

                            # Prepare Metadata
                            metadata = PngInfo()
                            metadata.add_text("prompt", p_meta["text"])
                            metadata.add_text("index", str(current_p_idx))
                            metadata.add_text("seed", str(actual_seed))
                            metadata.add_text("alias", p_meta.get("alias", "") or "")
                            metadata.add_text(
                                "original_filename",
                                f"p{current_p_idx:03d}_i{i_idx:02d}_s{actual_seed}.png",
                            )

                            fname = (
                                f"p{current_p_idx:03d}_i{i_idx:02d}_s{actual_seed}.png"
                            )
                            save_path = output_dir / fname

                            img.save(save_path, pnginfo=metadata)

                            images_generated += 1
                            generation_state["progress"]["current"] = images_generated
                            print(
                                f"[{images_generated}/{total_images_needed}] Saved {save_path}"
                            )

                except Exception as e:
                    print(f"Failed generation for {m.name}: {e}")
                    import traceback

                    traceback.print_exc()

        return {
            "status": "cancelled" if check_cancelled() else "complete",
            "images_generated": images_generated,
        }
    finally:
        generation_state["is_running"] = False
        generation_state["current_model"] = None


def analyze_models_only(options: ScanOptions):
    """Analyze existing images and compute metrics (no generation). Persists to DB."""

    prompts = data_loader.load_prompts_only()
    if not prompts:
        return {"status": "error", "message": "No prompts found"}

    with db.get_session() as session:
        # Create Benchmark Run
        run = BenchmarkRun(
            timestamp=datetime.utcnow(),
            parameters=options.dict(),
            prompts=prompts,
            prompt_set_id="default",
        )
        session.add(run)
        session.commit()
        session.refresh(run)
        print(f"Started Benchmark Run ID: {run.id}")

        session.refresh(run)
        print(f"Started Benchmark Run ID: {run.id}")

        # models = session.exec(select(Model)).all()
        # Analyze only ACTIVE SESSION models
        valid_models = models_db

        for m in valid_models:
            output_dir = data_loader.ASSETS_DIR / "outputs" / m.name
            existing_images = list(output_dir.glob("p*_i*_s*.png"))

            # Group images by prompt
            from PIL import Image

            grouped_images = {}
            flat_images = []
            flat_prompts = []

            for img_path in sorted(existing_images):
                try:
                    img = Image.open(img_path).convert("RGB")
                    img.load()

                    prompt_text = img.info.get("prompt")

                    # If prompt missing in metadata, fallback to index if we trust it,
                    # but strictly we should probably limit analysis to "known" prompts?
                    # Let's keep existing fallback logic for robustness.
                    if not prompt_text:
                        name = img_path.stem
                        try:
                            prompt_idx = int(name.split("_")[0][1:])
                            if prompt_idx < len(prompts):
                                prompt_text = prompts[prompt_idx]
                        except:
                            pass

                    if not prompt_text:
                        prompt_text = ""

                    flat_images.append(img)
                    flat_prompts.append(prompt_text)

                    if prompt_text not in grouped_images:
                        grouped_images[prompt_text] = []
                    grouped_images[prompt_text].append(img)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

            print(f"Analyzing {m.name}: {len(flat_images)} images")

            metrics = {
                "accuracy": 0.0,
                "diversity": 0.0,
                "rating": 0.0,
                "vqa_score": 0.0,
                "lpips_loss": 0.0,
            }

            if flat_images:
                try:
                    mc = get_metrics_calc()
                    raw_metrics = mc.calculate_metrics(
                        flat_images, flat_prompts, grouped_images
                    )

                    metrics["accuracy"] = round(raw_metrics["clip_score"], 3)
                    metrics["diversity"] = round(raw_metrics["diversity_score"], 3)
                    metrics["vqa_score"] = round(random.uniform(0.7, 0.9), 2)
                    metrics["lpips_loss"] = round(
                        raw_metrics.get("lpips_diversity", 0.0), 3
                    )
                except Exception as e:
                    print(f"Error metrics {m.name}: {e}")

            # Save Result to DB
            result = DBModelResult(
                run_id=run.id,
                model_hash=m.hash,
                metrics=metrics,
                image_count=len(flat_images),
            )
            session.add(result)

        session.commit()
        print("Analysis complete. Results saved.")

    # Sync to refresh in-memory state for API
    return sync_models_with_db()


def sync_models_with_db():
    """
    Scans disk for models, hashes them, updates DB, and then refreshes the in-memory state.
    """
    print("Syncing models with database...")
    local_models = data_loader.get_available_models_from_disk()

    with db.get_session() as session:
        # 1. Process found files
        found_hashes = set()

        for lm in local_models:
            path = Path(lm["path"])
            if not path.exists():
                continue

            # Efficient check: Do we know this path?
            # Ideally we check by hash, but hashing is slow.
            # Optimization: Check if path exists in DB. If so, verify size/mtime?
            # For now, let's assume if path matches, it's the same model, UNLESS we want to be very strict.
            # User wants foresight -> Hashing.
            # We can cache the hash in a separate sidecar or just rely on the DB.
            # Let's compute hash. Warning: Large files take time.
            # To avoid re-hashing every restart, we should check `meta` for mtime + size.

            stat = path.stat()
            file_mtime = stat.st_mtime
            file_size = stat.st_size

            # Try to find by path first to skip hashing if unchanged
            existing_model = session.exec(
                select(Model).where(Model.path == str(path))
            ).first()

            calculated_hash = None

            if existing_model:
                saved_meta = existing_model.meta or {}
                if (
                    saved_meta.get("mtime") == file_mtime
                    and saved_meta.get("size") == file_size
                ):
                    # Trusted match
                    calculated_hash = existing_model.hash
                    found_hashes.add(calculated_hash)
                    # Update active status if we had one?
                    continue

            # If we are here, it's new, moved, or changed.
            if not calculated_hash:
                print(f"Hashing {path.name}...")
                calculated_hash = compute_model_hash(path)

            found_hashes.add(calculated_hash)

            # --- Migration Logic: Ensure Output Folder follows Name ---
            # User prefers readable folder names (model name) over Hashes.
            # We use lm['name'] (filename stem) as the folder name.

            target_folder_name = lm["name"]
            target_dir = data_loader.ASSETS_DIR / "outputs" / target_folder_name

            # Check for Hash Folder (if we created one recently)
            hash_dir = data_loader.ASSETS_DIR / "outputs" / calculated_hash

            if hash_dir.exists() and hash_dir != target_dir:
                if not target_dir.exists():
                    print(
                        f"Renaming hash folder {hash_dir.name} to {target_folder_name}..."
                    )
                    try:
                        hash_dir.rename(target_dir)
                    except Exception as e:
                        print(f"Failed to rename hash folder: {e}")
                else:
                    print(
                        f"Target folder {target_folder_name} exists. Keeping existing."
                    )

            # Also check for Legacy Filename-ID folder (if we missed it or reverting)
            # The 'lm["id"]' is usually the filename stem, which matches our 'target_folder_name'
            # So the above target check covers it.
            # -----------------------------------------------------

            # Upsert
            model_record = session.get(Model, calculated_hash)
            if not model_record:
                # New Model
                print(f"New model detected: {lm['name']} ({calculated_hash})")

                # Detect type
                m_type = "sd1.5"
                pred_type = "epsilon"

                # Simple heuristic mapping from filename for now,
                # but we can expand this to read the safetensors header later.
                lower_name = path.name.lower()
                if "xl" in lower_name:
                    m_type = "sdxl"
                if "v-pred" in lower_name or "vpred" in lower_name:
                    pred_type = "v_prediction"

                model_record = Model(
                    hash=calculated_hash,
                    name=lm["name"],  # Use filename derived name
                    filename=path.name,
                    path=str(path),
                    type=m_type,
                    prediction_type=pred_type,
                    meta={"mtime": file_mtime, "size": file_size},
                )
                session.add(model_record)
            else:
                # Update path/meta if changed
                if model_record.path != str(path):
                    print(f"Model moved: {model_record.name} to {path}")
                    model_record.path = str(path)
                    model_record.filename = path.name

                # Update meta just in case
                model_record.meta = {"mtime": file_mtime, "size": file_size}
                session.add(model_record)

        session.commit()

        # 2. Refresh In-Memory State for API
        # We fetch all models that currently exist on disk (checked by existence in found_hashes?
        # or just all valid paths).
        # Let's show all valid models in DB.

        # We need to populate the `ModelResult` list for the frontend.
        # This includes fetching the LATEST benchmark result for each model.

        models_db.clear()

        all_db_models = session.exec(select(Model)).all()
        for db_m in all_db_models:
            # Verify existence on disk (optional, but good for "Clean" list)
            if not Path(db_m.path).exists():
                continue

            # Get latest result
            # We want the run with the most recent timestamp?
            # Or just ANY result?
            # Let's get the latest run result.
            latest_res = session.exec(
                select(DBModelResult)
                .join(BenchmarkRun)
                .where(DBModelResult.model_hash == db_m.hash)
                .order_by(desc(BenchmarkRun.timestamp))
            ).first()

            # Use Name for URL/Folder
            folder_name = db_m.name

            api_m = ModelResult(
                id=db_m.hash,  # Using Hash as ID for API now
                hash=db_m.hash,
                name=db_m.name,
                source="Local",
                url=f"/assets/outputs/{folder_name}" if latest_res else "",
                path=db_m.path,
                prediction_type=db_m.prediction_type,
                model_type=db_m.type,
            )

            if latest_res:
                api_m.accuracy = latest_res.metrics.get("accuracy", 0.0)
                api_m.diversity = latest_res.metrics.get("diversity", 0.0)
                api_m.metrics = latest_res.metrics
                api_m.image_count = latest_res.image_count

            models_db.append(api_m)

    print(f"DB Sync complete. Loaded {len(models_db)} models.")
    return models_db


def scan_models_light(options: ScanOptions = ScanOptions()):
    """
    Wrapper for sync_models_with_db to match signature.
    """
    return sync_models_with_db()
