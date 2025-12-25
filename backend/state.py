from pydantic import BaseModel
from typing import List, Optional, Literal, Dict
import threading

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

class ScanOptions(BaseModel):
    sampler: Literal["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver", "dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"] = "euler_a"
    steps: int = 28
    guidance_scale: float = 5.0
    seed: int = 218
    images_per_prompt: int = 1  # Set > 1 for LPIPS diversity measurement
    num_prompts: int = 10  # Number of prompts to use from test data
    width: int = 1024
    height: int = 1536

# In-memory database
models_db: List[ModelResult] = []

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
