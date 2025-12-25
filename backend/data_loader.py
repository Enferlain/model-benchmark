import os
import sys
from pathlib import Path
from PIL import Image
from typing import List, Tuple

ASSETS_DIR = Path(__file__).parent / "assets"
MODELS_DIR = ASSETS_DIR / "models"
IMAGES_DIR = ASSETS_DIR / "images"
PROMPTS_DIR = ASSETS_DIR / "prompts"
PAIRED_DIR = ASSETS_DIR / "image_prompts"

# Migration check
OLD_ASSETS_DIR = Path(__file__).parent.parent / "assets"
if OLD_ASSETS_DIR.exists() and not ASSETS_DIR.exists():
    print("WARNING: It looks like you have assets in the old location:", file=sys.stderr)
    print(f"  Old: {OLD_ASSETS_DIR}", file=sys.stderr)
    print(f"  New: {ASSETS_DIR}", file=sys.stderr)
    print("Please move your 'assets' folder into the 'backend' directory.", file=sys.stderr)

import json
import time
try:
    import blake3
except ImportError:
    print("blake3 not found, falling back to sha256 (slower)")
    import hashlib
    blake3 = None

CACHE_FILE = ASSETS_DIR / "model_cache.json"

def calculate_hash(filepath: Path) -> str:
    """Calculate hash of file content. Uses blake3 if available, else sha256."""
    if blake3:
        hasher = blake3.blake3()
    else:
        hasher = hashlib.sha256()
        
    with open(filepath, 'rb') as f:
        while chunk := f.read(1048576):  # 1MB chunks
            hasher.update(chunk)
    return hasher.hexdigest()

def get_model_cache():
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_model_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)

def get_available_models_from_disk():
    models = []
    if not MODELS_DIR.exists():
        return []
    
    cache = get_model_cache()
    cache_dirty = False
    
    # Process all safetensors files
    for file in MODELS_DIR.glob("*.safetensors"):
        filename = file.name
        mtime = file.stat().st_mtime
        size = file.stat().st_size
        
        # Check cache
        cached_entry = cache.get(filename)
        model_hash = None
        
        if cached_entry:
            # Verify mtime and size match
            if cached_entry.get('mtime') == mtime and cached_entry.get('size') == size:
                model_hash = cached_entry.get('hash')
        
        # If no valid hash found, calculate it
        if not model_hash:
            print(f"Hashing {filename}...")
            model_hash = calculate_hash(file)
            cache[filename] = {
                'mtime': mtime,
                'size': size,
                'hash': model_hash
            }
            cache_dirty = True
            
        models.append({
            "id": file.stem, # ID matches filename (readable, used for folders)
            "hash": model_hash, # Hash used for stable identity/coloring
            "filename": filename,
            "name": file.stem.replace("-", " ").title(),
            "source": "Local",
            "url": str(file.name),
            "path": str(file),
            "accuracy": 0.0, # Computed later
            "diversity": 0.0,
            "rating": 0.0,
        })
        
    if cache_dirty:
        save_model_cache(cache)
        
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

def load_prompts_only() -> List[str]:
    """Load only prompts without loading images for faster API response."""
    prompts = []
    
    # Prefer paired directory
    if PAIRED_DIR.exists():
        # Get all images to find matching text files (mimic load_test_data logic)
        image_files = sorted(
            list(PAIRED_DIR.glob("*.png")) + 
            list(PAIRED_DIR.glob("*.jpg")) + 
            list(PAIRED_DIR.glob("*.jpeg"))
        )
        
        for img_path in image_files:
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        prompts.append(f.read().strip())
                except Exception:
                    pass
    
    # Fallback to separate folders
    if not prompts and PROMPTS_DIR.exists():
        # Check if we would have loaded images from IMAGES_DIR? 
        # load_test_data only checks PROMPTS_DIR if IMAGES_DIR has images.
        # But for just prompts, we probably just want the prompts regardless of images?
        # Let's stick to the existing logic: if pairs found, use pairs. If not, check prompts dir.
        for txt_path in PROMPTS_DIR.glob("*.txt"):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    prompts.append(f.read().strip())
            except Exception:
                pass
                
    return prompts
