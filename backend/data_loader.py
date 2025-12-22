import os
from pathlib import Path
from PIL import Image
from typing import List, Tuple

ASSETS_DIR = Path(__file__).parent / "assets"
MODELS_DIR = ASSETS_DIR / "models"
IMAGES_DIR = ASSETS_DIR / "images"
PROMPTS_DIR = ASSETS_DIR / "prompts"
PAIRED_DIR = ASSETS_DIR / "image_prompts"

def get_available_models_from_disk():
    models = []
    if not MODELS_DIR.exists():
        return []
        
    for file in MODELS_DIR.glob("*.safetensors"):
        models.append({
            "id": file.stem,
            "name": file.stem.replace("-", " ").title(),
            "source": "Local",
            "url": str(file.name),
            "path": str(file),
            "accuracy": 0.0, # Computed later
            "diversity": 0.0,
            "rating": 0.0,
        })
    return models

def load_test_data() -> Tuple[List[Image.Image], List[str]]:
    images = []
    prompts = []
    
    # Prefer paired directory
    if PAIRED_DIR.exists():
        # Get all images
        image_files = sorted(
            list(PAIRED_DIR.glob("*.png")) + 
            list(PAIRED_DIR.glob("*.jpg")) + 
            list(PAIRED_DIR.glob("*.jpeg"))
        )
        
        print(f"Found {len(image_files)} images in {PAIRED_DIR}")
        
        for img_path in image_files:
            # Try to find matching text file
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                try:
                    img = Image.open(img_path).convert("RGB")
                    with open(txt_path, "r", encoding="utf-8") as f:
                        prompt = f.read().strip()
                    
                    images.append(img)
                    prompts.append(prompt)
                except Exception as e:
                    print(f"Error loading pair {img_path}: {e}")
    
    # Fallback to separate folders if no pairs found
    if not images and IMAGES_DIR.exists():
        print("Fallback to separate folders")
        for img_path in IMAGES_DIR.glob("*.png"):
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

        if PROMPTS_DIR.exists():
            for txt_path in PROMPTS_DIR.glob("*.txt"):
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        prompts.append(f.read().strip())
                except Exception as e:
                    print(f"Error loading prompt {txt_path}: {e}")
                
    return images, prompts
