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
