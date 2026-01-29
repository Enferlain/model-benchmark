import threading
from typing import Literal

from pydantic import BaseModel


class ModelRequest(BaseModel):
    url: str
    name: str | None = None
    source: str | None = "Unknown"
    api_token: str | None = None


# Forward compatibility for API responses
class ModelResult(BaseModel):
    id: str  # Kept for frontend compatibility (hash or filename-based)
    hash: str | None = None
    name: str
    source: str = "Local"
    accuracy: float = 0.0
    diversity: float = 0.0
    rating: float = 0.0
    vqa_score: float = 0.0
    lpips_loss: float = 0.0
    bt_score: float = 1000.0
    metrics: dict[str, float] = {}
    url: str = ""
    path: str | None = None
    image_count: int = 0

    # Metadata
    prediction_type: str | None = None
    model_type: str | None = None
    ztsnr: bool = False
    is_missing: bool = False


class ScanOptions(BaseModel):
    sampler: Literal[
        "ddim",
        "pndm",
        "lms",
        "euler",
        "euler_a",
        "heun",
        "dpm_2",
        "dpm_2_a",
        "dpmsolver",
        "dpmsolver++",
        "dpmsingle",
        "k_lms",
        "k_euler",
        "k_euler_a",
        "k_dpm_2",
        "k_dpm_2_a",
    ] = "euler_a"
    steps: int = 28
    guidance_scale: float = 5.0
    seed: int = 218
    images_per_prompt: int = 1  # Set > 1 for LPIPS diversity measurement
    num_prompts: int = 10  # Number of prompts to use from test data
    width: int = 1024
    height: int = 1536
    common_only: bool = False  # If true, only analyze prompts common to all models
    equalize_counts: bool = False  # If true, generate to match max image count per prompt across models
    selected_model_ids: list[str] | None = None  # If set, only process these models


# In-memory List Cache (Populated from DB)
# We will treat this as a read-only cache for GET requests to avoid hitting DB on every poll if needed,
# but for now let's keep it sync.
models_db: list[ModelResult] = []

# Generation state management
generation_state = {
    "is_running": False,
    "should_cancel": False,
    "current_model": None,
    "progress": {"current": 0, "total": 0},
}

# Download state management
download_state = {
    "is_downloading": False,
    "current_file": None,
    "progress": 0,
    "total": 0,
    "status": "idle",  # idle, downloading, completed, error
    "error": None,
}
download_state_lock = threading.Lock()
