from typing import Optional
from pathlib import Path
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from ..core import database as db
from ..core.state import models_db, generation_state, ScanOptions
from ..lib import inference
from . import prompt_manager as data_loader


def check_cancelled():
    """Check if generation should be cancelled. Call this in generation loops."""
    return generation_state["should_cancel"]


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

        # 1. Pre-calculate work needed
        model_work_queue = []  # [(model_result, items_needing_generation)]

        # Iterate over ACTIVE SESSION models only (Active Context)
        valid_models = []
        for m_res in models_db:
            valid_models.append(m_res)

        # If equalize_counts, first scan all models to find max image count per prompt
        prompt_max_counts: dict[str, int] = {}
        if options.equalize_counts:
            for m in valid_models:
                output_dir = data_loader.ASSETS_DIR / "outputs" / m.name
                if not output_dir.exists():
                    continue
                images = list(output_dir.glob("p*_i*_s*.png"))
                for img_path in images:
                    try:
                        with Image.open(img_path) as img:
                            prompt = img.info.get("prompt")
                            if prompt:
                                prompt = prompt.strip()
                                prompt_max_counts[prompt] = prompt_max_counts.get(
                                    prompt, 0
                                )
                                # Count this image (we'll track per-model counts below)
                    except Exception:
                        pass
                # Now count per prompt for this model
                model_counts: dict[str, int] = {}
                for img_path in images:
                    try:
                        with Image.open(img_path) as img:
                            prompt = img.info.get("prompt")
                            if prompt:
                                prompt = prompt.strip()
                                model_counts[prompt] = model_counts.get(prompt, 0) + 1
                    except Exception:
                        pass
                # Update max counts
                for p, c in model_counts.items():
                    prompt_max_counts[p] = max(prompt_max_counts.get(p, 0), c)
            print(f"Equalize mode: max counts = {prompt_max_counts}")

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

                except Exception as e:
                    print(f"Warning: Could not parse existing image {img_path}: {e}")

            # Calculate missing
            missing_for_model = []

            for prompt_meta in target_prompts_meta:
                text = prompt_meta["text"].strip()
                existing_indices = existing_counts.get(text, set())

                # Use max count from other models if equalize mode, otherwise use setting
                if options.equalize_counts and text in prompt_max_counts:
                    target_count = prompt_max_counts[text]
                else:
                    target_count = options.images_per_prompt

                needed = set(range(target_count))
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

                        # Generate unique ID for this image
                        import uuid
                        import json
                        from datetime import datetime

                        image_id = str(uuid.uuid4())[:8]  # Short UUID
                        generation_time = datetime.utcnow().isoformat()

                        # Build parameters dict
                        params = {
                            "steps": options.steps,
                            "cfg": options.guidance_scale,
                            "sampler": options.sampler,
                            "seed": actual_seed,
                            "width": options.width,
                            "height": options.height,
                        }

                        # Prepare Metadata
                        metadata = PngInfo()
                        metadata.add_text("model_name", m.name)
                        metadata.add_text("prompt", p_meta["text"])
                        metadata.add_text("parameters", json.dumps(params))
                        metadata.add_text("prompt_set", "")  # Future: named prompt sets
                        metadata.add_text("id", image_id)
                        metadata.add_text("index", str(i_idx))  # Image variant index
                        metadata.add_text("seed", str(actual_seed))
                        metadata.add_text("alias", p_meta.get("alias", "") or "")
                        metadata.add_text("generation_time", generation_time)
                        metadata.add_text(
                            "original_filename",
                            f"p{current_p_idx:03d}_i{i_idx:02d}_s{actual_seed}.png",
                        )

                        fname = f"p{current_p_idx:03d}_i{i_idx:02d}_s{actual_seed}.png"
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
