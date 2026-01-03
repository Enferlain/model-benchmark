from fastapi import APIRouter, Body
from pathlib import Path
from datetime import datetime
from PIL import Image
import json
import shutil

from ..core import state
from ..services import generation, analysis
from ..services import prompt_manager as data_loader

router = APIRouter()


@router.post("/generate")
def generate_endpoint(options: state.ScanOptions = Body(default=state.ScanOptions())):
    """Generate images only (no metrics calculation)."""
    if state.generation_state["is_running"]:
        return {"status": "error", "message": "Generation already in progress"}
    return generation.generate_images_only(options)


@router.post("/analyze")
def analyze_endpoint(options: state.ScanOptions = Body(default=state.ScanOptions())):
    """Analyze existing images and compute metrics."""
    return analysis.analyze_models_only(options)


@router.post("/scan")
def scan_models(options: state.ScanOptions = Body(default=state.ScanOptions())):
    """Generate images AND analyze (legacy endpoint)."""
    if state.generation_state["is_running"]:
        return {"status": "error", "message": "Generation already in progress"}

    gen_result = generation.generate_images_only(options)

    # Check for failure or cancellation
    if gen_result.get("status") in ["error", "cancelled"]:
        return gen_result

    return analysis.analyze_models_only(options)


@router.post("/cancel")
def cancel_generation():
    """Cancel ongoing generation."""
    if state.generation_state["is_running"]:
        state.generation_state["should_cancel"] = True
        return {"status": "ok", "message": "Cancellation requested"}
    return {"status": "ok", "message": "No generation in progress"}


@router.post("/check-params")
def check_params(options: state.ScanOptions = Body(default=state.ScanOptions())):
    """Check if current settings match existing image metadata."""
    current_params = {
        "steps": options.steps,
        "cfg": options.guidance_scale,
        "sampler": options.sampler,
        "width": options.width,
        "height": options.height,
    }

    mismatched_models = []
    existing_params = None

    for m in state.models_db:
        output_dir = data_loader.ASSETS_DIR / "outputs" / m.name
        if not output_dir.exists():
            continue

        # Find first image with metadata
        images = list(output_dir.glob("p*_i*_s*.png"))
        if not images:
            continue

        try:
            with Image.open(images[0]) as img:
                params_json = img.info.get("parameters")
                if params_json:
                    img_params = json.loads(params_json)

                    # Compare (ignore seed as it varies per image)
                    if existing_params is None:
                        existing_params = {
                            "steps": img_params.get("steps"),
                            "cfg": img_params.get("cfg"),
                            "sampler": img_params.get("sampler"),
                            "width": img_params.get("width"),
                            "height": img_params.get("height"),
                        }

                    # Check for mismatch
                    mismatch = (
                        current_params["steps"] != img_params.get("steps")
                        or current_params["cfg"] != img_params.get("cfg")
                        or current_params["sampler"] != img_params.get("sampler")
                        or current_params["width"] != img_params.get("width")
                        or current_params["height"] != img_params.get("height")
                    )

                    if mismatch:
                        mismatched_models.append(
                            {
                                "name": m.name,
                                "existing_params": {
                                    "steps": img_params.get("steps"),
                                    "cfg": img_params.get("cfg"),
                                    "sampler": img_params.get("sampler"),
                                    "width": img_params.get("width"),
                                    "height": img_params.get("height"),
                                },
                            }
                        )
        except Exception as e:
            print(f"Error reading params from {images[0]}: {e}")

    return {
        "matches": len(mismatched_models) == 0,
        "existing_params": existing_params,
        "mismatched_models": mismatched_models,
        "current_params": current_params,
    }


@router.post("/archive/{model_name}")
def archive_model(model_name: str):
    """Archive existing images for a model by moving to timestamped folder."""
    output_dir = data_loader.ASSETS_DIR / "outputs" / model_name

    if not output_dir.exists():
        return {"status": "ok", "message": "No images to archive"}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = (
        data_loader.ASSETS_DIR / "outputs" / f"{model_name}_archived_{timestamp}"
    )

    try:
        shutil.move(str(output_dir), str(archive_dir))
        return {"status": "ok", "message": f"Archived to {archive_dir.name}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/analyze/check-coverage")
def check_coverage():
    """Check prompt coverage across all active models."""
    model_coverage: dict[str, set[str]] = {}

    for m in state.models_db:
        output_dir = data_loader.ASSETS_DIR / "outputs" / m.name
        if not output_dir.exists():
            model_coverage[m.name] = set()
            continue

        images = list(output_dir.glob("p*_i*_s*.png"))
        prompts_found = set()

        for img_path in images:
            try:
                with Image.open(img_path) as img:
                    prompt = img.info.get("prompt")
                    if prompt:
                        prompts_found.add(prompt)
            except Exception:
                pass

        model_coverage[m.name] = prompts_found

    # Find common prompts (intersection of all sets)
    all_prompt_sets = list(model_coverage.values())
    if not all_prompt_sets or all(len(s) == 0 for s in all_prompt_sets):
        common_prompts: set[str] = set()
    else:
        non_empty = [s for s in all_prompt_sets if len(s) > 0]
        common_prompts = non_empty[0].copy() if non_empty else set()
        for s in non_empty[1:]:
            common_prompts &= s

    # Check if all models have exactly the same coverage
    all_match = all(s == common_prompts and len(s) > 0 for s in all_prompt_sets)

    # Build response with coverage details
    coverage_details = []
    for name, prompts in model_coverage.items():
        missing = list(common_prompts - prompts) if common_prompts else []
        extra = list(prompts - common_prompts) if common_prompts else list(prompts)
        coverage_details.append(
            {
                "name": name,
                "count": len(prompts),
                "missing_count": len(missing),
                "extra_count": len(extra),
            }
        )

    return {
        "all_match": all_match,
        "common_count": len(common_prompts),
        "model_coverage": coverage_details,
    }
