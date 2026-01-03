from fastapi import APIRouter, Body
from ..core import state
from ..services import generation, analysis

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
