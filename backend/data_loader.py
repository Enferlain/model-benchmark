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


    if not deleted:
        raise FileNotFoundError(f"No files found to delete for {filename}")
        
    return True



CONFIG_PATH = ASSETS_DIR / "prompts_config.json"

# Config Structure:
# {
#   "version": 2,
#   "order": ["file1.png", "file2.txt", ...],
#   "states": { "file1.png": true, "file2.txt": false }
# }

def load_prompt_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, 'r') as f:
                data = json.load(f)
                # Migration from v1 (simple dict) to v2
                if "version" not in data:
                    return {"version": 2, "order": [], "states": data}
                return data
        except:
            return {"version": 2, "order": [], "states": {}}
    return {"version": 2, "order": [], "states": {}}

def save_prompt_config(config: dict):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=2)

def is_prompt_enabled(filename: str, config: dict = None) -> bool:
    if config is None:
        config = load_prompt_config()
    return config.get("states", {}).get(filename, True)

def toggle_prompt_active(filename: str, enabled: bool):
    config = load_prompt_config()
    if "states" not in config: config["states"] = {}
    config["states"][filename] = enabled
    save_prompt_config(config)
    return True

def save_prompt_order(filenames: List[str]):
    config = load_prompt_config()
    config["order"] = filenames
    save_prompt_config(config)
    return True

# Override get_all_prompts_metadata to include enabled status and respect order
def get_all_prompts_metadata():
    """Get rich metadata for all prompts for the Prompt Manager."""
    prompts_data = []
    config = load_prompt_config()
    
    # helper to clean IDs
    def clean_id(path):
        return path.stem

    # 1. Scan Paired Directory (Images + Text)
    if PAIRED_DIR.exists():
        image_files = sorted(
            list(PAIRED_DIR.glob("*.png")) + 
            list(PAIRED_DIR.glob("*.jpg")) + 
            list(PAIRED_DIR.glob("*.jpeg"))
        )
        
        for img_path in image_files:
            txt_path = img_path.with_suffix(".txt")
            text_content = ""
            if txt_path.exists():
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        text_content = f.read().strip()
                except:
                    pass
            
            prompts_data.append({
                "id": clean_id(img_path),
                "filename": img_path.name,
                "text": text_content,
                "image": f"/assets/image_prompts/{img_path.name}",
                "type": "paired",
                "enabled": is_prompt_enabled(img_path.name, config)
            })

    # 2. Scan Text-only Directory
    if PROMPTS_DIR.exists():
        for txt_path in PROMPTS_DIR.glob("*.txt"):
            pid = clean_id(txt_path)
            if not any(p['id'] == pid for p in prompts_data):
                 try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    prompts_data.append({
                        "id": pid,
                        "filename": txt_path.name,
                        "text": content,
                        "image": None, 
                        "type": "text_only",
                        "enabled": is_prompt_enabled(txt_path.name, config)
                    })
                 except:
                     pass
                     
    # Sort
    # 1. If 'order' exists in config, use it to prioritize
    order_list = config.get("order", [])
    
    def sort_key(p):
        try:
            return order_list.index(p['filename'])
        except ValueError:
            # If not in order list, put at the end, sorted by ID
            return 999999
            
    # First sort by ID (fallback stability)
    prompts_data.sort(key=lambda x: x['id'])
    
    # Then sort by order list if present
    if order_list:
        prompts_data.sort(key=sort_key)
        
    return prompts_data


# Update filtering in load functions
def load_test_data() -> Tuple[List[Image.Image], List[str]]:
    images = []
    prompts = []
    config = load_prompt_config()
    order_list = config.get("order", [])
    
    # helper
    def sort_key(filename):
        try:
            return order_list.index(filename)
        except ValueError:
            return 999999

    # Collect all candidates first
    candidates = []

    # Prefer paired directory
    if PAIRED_DIR.exists():
        image_files = sorted(
            list(PAIRED_DIR.glob("*.png")) + 
            list(PAIRED_DIR.glob("*.jpg")) + 
            list(PAIRED_DIR.glob("*.jpeg"))
        )
        for img_path in image_files:
            if not is_prompt_enabled(img_path.name, config):
                continue
            candidates.append((img_path, "paired"))
            
    # Sort candidates by order
    # (For now only paired supported here as per original logic, though we can expand later)
    candidates.sort(key=lambda x: sort_key(x[0].name))
    
    for img_path, _ in candidates:
        txt_path = img_path.with_suffix(".txt")
        if txt_path.exists():
            try:
                img = Image.open(img_path).convert("RGB")
                with open(txt_path, "r", encoding="utf-8") as f:
                    prompt = f.read().strip()
                
                images.append(img)
                prompts.append(prompt)
            except Exception as e:
                pass
    
    return images, prompts

def load_prompts_only() -> List[str]:
    """Load only active prompts."""
    # This one is tricky because it mixes two sources.
    # We should gather all items, sort them, then extract content.
    
    prompts_map = {} # filename -> text
    config = load_prompt_config()
    order_list = config.get("order", [])
    
    # 1. Paired
    if PAIRED_DIR.exists():
        image_files = list(PAIRED_DIR.glob("*.png")) + list(PAIRED_DIR.glob("*.jpg")) + list(PAIRED_DIR.glob("*.jpeg"))
        for img_path in image_files:
            if not is_prompt_enabled(img_path.name, config): continue
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        prompts_map[img_path.name] = f.read().strip()
                except: pass
                
    # 2. Text Only
    if PROMPTS_DIR.exists():
         for txt_path in PROMPTS_DIR.glob("*.txt"):
             if not is_prompt_enabled(txt_path.name, config): continue
             # Avoid duplicates if paired already handled it? (Assuming filenames unique)
             if txt_path.name not in prompts_map:
                 try:
                    with open(txt_path, "r", encoding="utf-8") as f:
                        prompts_map[txt_path.name] = f.read().strip()
                 except: pass

    # Sort
    filenames = list(prompts_map.keys())
    
    def sort_key(fname):
        try:
            return order_list.index(fname)
        except ValueError:
            return 999999 + hash(fname) # Deterministic fallback
            
    filenames.sort(key=sort_key)
    
    return [prompts_map[f] for f in filenames]



def save_new_prompt(text: str, image_bytes: bytes = None, filename_hint: str = None):
    """Create a new prompt. If image provided, save to paired dir. Else text dir."""
    import time
    import re
    
    # Generate ID/Filename
    if filename_hint:
        base_name = Path(filename_hint).stem
    else:
        # Create slug from text
        # Take first 30 chars, remove non-alphanumeric, replace spaces with underscores
        slug = re.sub(r'[^a-zA-Z0-9\s]', '', text[:30]).strip().replace(' ', '_').lower()
        if not slug:
            slug = "untitled"
        base_name = f"{slug}_{int(time.time())}"
        
    # Ensure unique
    counter = 0
    while True:
        suffix = f"_{counter}" if counter > 0 else ""
        candidate = base_name + suffix
        # Check both dirs
        if not (PAIRED_DIR / f"{candidate}.png").exists() and \
           not (PROMPTS_DIR / f"{candidate}.txt").exists():
            base_name = candidate
            break
        counter += 1
        
    target_dir = PAIRED_DIR if image_bytes else PROMPTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    
    final_filename = f"{base_name}.png" if image_bytes else f"{base_name}.txt"
    
    # Save Image
    if image_bytes:
        img_path = target_dir / final_filename
        with open(img_path, "wb") as f:
            f.write(image_bytes)
    else:
        # For text only, filename is .txt
        final_filename = f"{base_name}.txt"
            
    # Save Text
    txt_path = target_dir / f"{base_name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
        
    # Update Order (Prepend)
    config = load_prompt_config()
    current_order = config.get("order", [])
    # If using ID tracking, we need the filename
    if final_filename not in current_order:
        config["order"] = [final_filename] + current_order
        save_prompt_config(config)
        
    return base_name

def update_prompt_text(filename: str, new_text: str):
    """Update text content of an existing prompt."""
    # Find file
    # Check PAIRED first
    txt_path = PAIRED_DIR / Path(filename).with_suffix(".txt").name
    if not txt_path.exists():
        txt_path = PROMPTS_DIR / Path(filename).with_suffix(".txt").name
        
    if not txt_path.exists():
        raise FileNotFoundError(f"Prompt text file not found for {filename}")
        
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(new_text)
        
    return True

def delete_prompt(filename: str):
    """Delete prompt files (image and text)."""
    base_name = Path(filename).stem
    
    deleted = False
    
    # Try Paired
    for ext in ['.png', '.jpg', '.jpeg', '.txt']:
        p = PAIRED_DIR / f"{base_name}{ext}"
        if p.exists():
            p.unlink()
            deleted = True
            
    # Try Text Only
    p = PROMPTS_DIR / f"{base_name}.txt"
    if p.exists():
        p.unlink()
        deleted = True
        
    if not deleted:
        raise FileNotFoundError(f"No files found to delete for {filename}")
        
    return True
