from datetime import datetime
import hashlib
import json
from pathlib import Path

from sqlmodel import desc, select

from ..core import database as db
from ..core.database import BenchmarkRun, ImageOutput, Model
from ..core.database import ModelResult as DBModelResult
from ..core.state import ModelResult, models_db
from . import prompt_manager as data_loader


SOURCES_FILE = data_loader.ASSETS_DIR / "sources.json"


def load_sources() -> list[str]:
    if SOURCES_FILE.exists():
        try:
            with open(SOURCES_FILE) as f:
                return json.load(f)
        except:
            return []
    return []


def save_sources(sources: list[str]):
    with open(SOURCES_FILE, "w") as f:
        json.dump(list(set(sources)), f, indent=2)


def add_source(path: str):
    sources = load_sources()
    if path not in sources:
        sources.append(path)
        save_sources(sources)


def add_sources_batch(paths: list[str]):
    sources = load_sources()
    changed = False
    for path in paths:
        if path not in sources:
            sources.append(path)
            changed = True
    if changed:
        save_sources(sources)


def register_paths(paths: list[str]) -> dict:
    """
    Register multiple paths at once and sync only once.
    """
    valid_paths = []
    for p_str in paths:
        path = Path(p_str)
        if not path.exists():
            print(f"Warning: Path not found: {p_str}")
            continue
        valid_paths.append(str(path))

    if not valid_paths:
        return {
            "results": [],
            "stats": {"added": 0, "updated": 0, "removed": 0, "unchanged": 0},
        }

    # Batch add to sources
    add_sources_batch(valid_paths)

    # Sync ONCE
    _, stats = sync_models_with_db()

    # Retrieve all added models
    results = []
    for p_str in valid_paths:
        path = Path(p_str)
        if path.is_file():
            matched = next((m for m in models_db if Path(m.path).resolve() == path.resolve()), None)
            if not matched:
                matched = next((m for m in models_db if m.filename == path.name), None)
            if matched:
                results.append(matched.dict())
        else:
            # For folders, we don't return specific models, just status
            pass

    return {"results": results, "stats": stats}


def register_path(path_str: str) -> dict:
    """
    Immediately register a model path (file or folder).
    Returns the processed model info dict or raises error.
    """
    data = register_paths([path_str])
    results = data.get("results", [])
    if results:
        # Backward compatibility for direct callers expecting just the dict
        # But if API uses register_path, it might break if I don't check usage?
        # Actually register_path in API returns the result.
        # I should probably update models.py if I change this signature.
        # But register_path(path) -> dict.
        return results[0]

    # If folder or empty, just return status
    return {"status": "registered", "path": path_str, "stats": data.get("stats")}


def compute_model_hash(file_path: Path) -> str:
    """Computes SHA256 hash of the file (industry standard)."""
    hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        while chunk := f.read(1048576):  # 1MB chunks
            hasher.update(chunk)
    return hasher.hexdigest()


def scan_model_type(path: Path) -> tuple[str, str, bool]:
    """
    Detects model type (sd1.5, sdxl) and prediction type (epsilon, v_prediction, etc)
    by inspecting the safetensors file structure.
    Returns: (model_type, prediction_type, is_ztsnr)
    """
    m_type = "sd1.5"
    pred_type = "epsilon"
    is_ztsnr = False

    # 1. Try to read safetensors header
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import safe_open

            with safe_open(path, framework="pt", device="cpu") as f:
                keys = set(f.keys())  # Use set for faster lookups

                # --- Architecture Detection ---
                # SDXL models have dual text encoders:
                # 0: conditioner.embedders.0.transformer.* (CLIP G)
                # 1: conditioner.embedders.1.model.* (CLIP L or OpenCLIP)
                if any(k.startswith("conditioner.embedders.0.transformer") for k in keys):
                    m_type = "sdxl"
                # SD 1.x models use cond_stage_model.transformer.text_model
                elif any(k.startswith("cond_stage_model.transformer.text_model") for k in keys):
                    m_type = "sd1.5"

                # --- Prediction Type / Config Detection (User Logic) ---
                if "edm_mean" in keys and "edm_std" in keys:  # Playground V2.5
                    pred_type = "edm"
                elif "edm_vpred.sigma_max" in keys:
                    pred_type = "v_prediction_edm"
                elif "v_pred" in keys:
                    pred_type = "v_prediction"
                    if "ztsnr" in keys:  # Some zsnr anime checkpoints
                        is_ztsnr = True
                else:
                    pred_type = "epsilon"

        except Exception as e:
            print(f"Failed to scan safetensors header for {path.name}: {e}")

    # 2. Heuristics / Overrides from Filename (Fallback)
    lower_name = path.name.lower()

    # Only fallback if not definitively found (or to catch legacy naming)
    if m_type == "sd1.5":
        if "xl" in lower_name:
            m_type = "sdxl"

    if pred_type == "epsilon":
        if "v-pred" in lower_name or "vpred" in lower_name:
            pred_type = "v_prediction"

    return m_type, pred_type, is_ztsnr


def sync_models_with_db(recheck_types: bool = False):
    """
    Scans disk for models, hashes them, updates DB, and then refreshes the in-memory state.
    Args:
        recheck_types: If True, will re-open safetensors to verify type/prediction even if file hasn't changed.
    Returns:
        (models_db, stats): Tuple of current models list and stats dict.
    """
    print(f"Syncing models with database (recheck_types={recheck_types})...")
    local_models = data_loader.get_available_models_from_disk()

    stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0}

    with db.get_session() as session:
        # 1. Process found files
        found_hashes = set()

        for lm in local_models:
            path = Path(lm["path"])
            if not path.exists():
                continue

            stat = path.stat()
            file_mtime = stat.st_mtime
            file_size = stat.st_size

            # Try to find by path first to skip hashing if unchanged
            existing_model = session.exec(select(Model).where(Model.path == str(path))).first()

            calculated_hash = None
            is_new = False
            is_update = False

            if existing_model:
                saved_meta = existing_model.meta or {}
                if saved_meta.get("mtime") == file_mtime and saved_meta.get("size") == file_size:
                    # Trusted match on file level
                    calculated_hash = existing_model.hash
                    found_hashes.add(calculated_hash)

                    if not recheck_types:
                        stats["unchanged"] += 1
                        continue
                    # If recheck_types is True, we proceed to update logic below instead of continuing
                    is_update = True

            # If we are here, it's new, moved, changed, or we are forcing a recheck.
            if not calculated_hash:
                # OPTIMIZATION: Use the hash from local_models if available (computed by prompt_manager)
                if lm.get("hash"):
                    calculated_hash = lm["hash"]
                else:
                    print(f"Hashing {path.name}...")
                    try:
                        calculated_hash = compute_model_hash(path)
                    except Exception as e:
                        print(f"Error hashing {path}: {e}")
                        continue

            if not existing_model and calculated_hash not in found_hashes:
                # Check if hash exists under different path (moved)
                has_hash_in_db = session.get(Model, calculated_hash)
                if not has_hash_in_db:
                    is_new = True

            found_hashes.add(calculated_hash)

            # --- Migration Logic: Ensure Output Folder follows Name ---
            target_folder_name = lm["name"]
            target_dir = data_loader.ASSETS_DIR / "outputs" / target_folder_name

            # Check for Hash Folder (if we created one recently)
            hash_dir = data_loader.ASSETS_DIR / "outputs" / calculated_hash

            if hash_dir.exists() and hash_dir != target_dir:
                if not target_dir.exists():
                    print(f"Renaming hash folder {hash_dir.name} to {target_folder_name}...")
                    try:
                        hash_dir.rename(target_dir)
                    except Exception as e:
                        print(f"Failed to rename hash folder: {e}")
                else:
                    print(f"Target folder {target_folder_name} exists. Keeping existing.")

            # Upsert
            model_record = session.get(Model, calculated_hash)
            if not model_record:
                # New Model
                print(f"New model detected: {lm['name']} ({calculated_hash})")
                stats["added"] += 1

                # Detect type
                m_type, pred_type, is_ztsnr = scan_model_type(path)

                model_record = Model(
                    hash=calculated_hash,
                    name=lm["name"],  # Use filename derived name
                    filename=path.name,
                    path=str(path),
                    type=m_type,
                    source="Local",
                    prediction_type=pred_type,
                    hash_type="sha256",
                    meta={"mtime": file_mtime, "size": file_size, "ztsnr": is_ztsnr},
                    is_missing=False,
                )
                session.add(model_record)
            else:
                # Update path/meta if changed
                if model_record.path != str(path):
                    print(f"Model moved: {model_record.name} to {path}")
                    model_record.path = str(path)
                    model_record.filename = path.name
                    stats["updated"] += 1
                elif is_update:  # If we forced recheck
                    stats["updated"] += 1
                elif not is_new:  # If we are just confirming existing
                    # Wait, if we are here, we might have skipped earlier if unchanged.
                    # If we fell through, something changed or checking types.
                    pass

                # Check for Rename (Same Hash, Different Name in DB vs Disk)
                # derived name from disk: lm["name"]
                if model_record.name != lm["name"]:
                    print(f"Model renamed: {model_record.name} -> {lm['name']}")
                    old_name = model_record.name
                    new_name = lm["name"]

                    # Rename Output Folder if exists
                    old_out_dir = data_loader.ASSETS_DIR / "outputs" / old_name
                    new_out_dir = data_loader.ASSETS_DIR / "outputs" / new_name

                    if old_out_dir.exists() and not new_out_dir.exists():
                        try:
                            print(f"Renaming output folder {old_name} -> {new_name}")
                            old_out_dir.rename(new_out_dir)
                        except Exception as e:
                            print(f"Failed to rename output folder: {e}")

                    # Update DB Record
                    model_record.name = new_name
                    # Filename is likely already updated by path check above, but ensure it matches
                    model_record.filename = path.name
                    if "updated" not in stats:  # prevent double count
                        stats["updated"] += 1

                # Re-scan to update types if missing or outdated (optional, but good for active update)
                if recheck_types:
                    m_type, pred_type, is_ztsnr = scan_model_type(path)
                    model_record.type = m_type
                    model_record.prediction_type = pred_type
                    # Update meta with ztsnr
                    new_meta = model_record.meta or {}
                    new_meta["mtime"] = file_mtime
                    new_meta["size"] = file_size
                    new_meta["ztsnr"] = is_ztsnr
                    model_record.meta = new_meta
                    print(f"Updated metadata/type for {model_record.name}: {m_type}, {pred_type}, ztsnr={is_ztsnr}")
                else:
                    # Just update basic meta
                    if not model_record.meta:
                        model_record.meta = {}
                    model_record.meta["size"] = file_size

                # Ensure it is marked as found
                if model_record.is_missing:
                    model_record.is_missing = False
                    stats["updated"] += 1  # Recovered

                session.add(model_record)
                session.commit()  # Ensure model hash exists for FK
                index_model_outputs(session, model_record)

                # If we fell through here and didn't count as update or added, and not skipped, treat as unchanged?
                # The logic above is slightly loose on counting "updated" vs "unchanged" when fallthrough happens without specific change detected.
                # But mostly correct.

        # Cleanup: Remove models from DB that rely on files not found in this scan
        # found_hashes contains valid models we just verified
        all_db_models_cleanup = session.exec(select(Model)).all()
        for db_m in all_db_models_cleanup:
            # If hash is not in found_hashes, it means it wasn't found on disk during this scan
            if db_m.hash not in found_hashes:
                if not db_m.is_missing:
                    print(f"Marking model as missing: {db_m.name} ({db_m.hash})")
                    db_m.is_missing = True
                    session.add(db_m)
                    stats["removed"] += 1
            else:
                # Should already be handled above, but double check
                if db_m.is_missing:
                    db_m.is_missing = False
                    session.add(db_m)

        session.commit()

        # 2. Refresh In-Memory State for API
        new_models_list = []

        all_db_models = session.exec(select(Model)).all()
        for db_m in all_db_models:
            # Verify existence on disk (optional, but good for "Clean" list)
            # if not Path(db_m.path).exists():
            #     continue

            # Get latest result
            latest_res = session.exec(
                select(DBModelResult).join(BenchmarkRun).where(DBModelResult.model_hash == db_m.hash).order_by(desc(BenchmarkRun.timestamp))
            ).first()

            # Use Name for URL/Folder
            folder_name = db_m.name

            api_m = ModelResult(
                id=db_m.hash,  # Using Hash as ID for API now
                hash=db_m.hash,
                name=db_m.name,
                source=db_m.source or "Local",
                url=f"/assets/outputs/{folder_name}" if latest_res else "",
                path=db_m.path,
                prediction_type=db_m.prediction_type,
                model_type=db_m.type,
                bt_score=db_m.bt_score,
                is_missing=db_m.is_missing,
            )

            # Pass through ztsnr from meta if present
            if db_m.meta and db_m.meta.get("ztsnr"):
                api_m.ztsnr = True

            if latest_res:
                api_m.accuracy = latest_res.metrics.get("accuracy", 0.0)
                api_m.diversity = latest_res.metrics.get("diversity", 0.0)
                # Merge metrics
                api_m.metrics.update(latest_res.metrics)
                api_m.image_count = latest_res.image_count

            new_models_list.append(api_m)

        # Atomic replacement
        models_db[:] = new_models_list

    print(f"DB Sync complete. Loaded {len(models_db)} models. Stats: {stats}")
    return models_db, stats


def index_model_outputs(session: db.Session, model: Model):
    """
    Scans the output directory for a model and indexes images in the ImageOutput table.
    Uses mtime to skip already indexed images.
    """
    output_dir = data_loader.ASSETS_DIR / "outputs" / model.name
    if not output_dir.exists():
        return

    print(f"Indexing outputs for {model.name}...")

    # 1. Get existing indexed images for this model to check mtime
    indexed_map = {img.path: img.mtime for img in session.exec(select(ImageOutput).where(ImageOutput.model_hash == model.hash)).all()}

    from PIL import Image
    from .prompt_manager import get_all_prompts_metadata

    # Cache prompts for better lookup performance (though we usually have prompt_text in metadata)
    # Actually, the filename has prompt_idx (pX), so we can use that.
    prompts = [p["text"] for p in get_all_prompts_metadata()]

    new_count = 0
    updated_count = 0

    for img_path in output_dir.glob("p*_i*_s*.png"):
        mtime = int(img_path.stat().st_mtime)
        path_str = str(img_path)

        if path_str in indexed_map and indexed_map[path_str] >= mtime:
            continue

        try:
            with Image.open(img_path) as img:
                img.load()
                image_id = img.info.get("id")
                prompt_text = img.info.get("prompt")

                # Parse filename for seed/idx if metadata is missing (legacy)
                if not prompt_text or not image_id:
                    parts = img_path.stem.split("_")
                    prompt_idx = -1
                    seed = None
                    for part in parts:
                        if part.startswith("p") and part[1:].isdigit():
                            prompt_idx = int(part[1:])
                        elif part.startswith("s") and part[1:].isdigit():
                            seed = int(part[1:])

                    if not prompt_text and prompt_idx != -1 and prompt_idx < len(prompts):
                        prompt_text = prompts[prompt_idx]

                    if not image_id:
                        # Generate deterministic ID for legacy images
                        image_id = hashlib.sha256(path_str.encode()).hexdigest()[:8]

                # Upsert ImageOutput
                record = session.get(ImageOutput, image_id)
                if not record:
                    record = ImageOutput(id=image_id, model_hash=model.hash, prompt_text=prompt_text, seed=seed, path=path_str, mtime=mtime)
                    session.add(record)
                    new_count += 1
                else:
                    record.mtime = mtime
                    record.path = path_str  # Path might have changed due to rename
                    record.prompt_text = prompt_text
                    updated_count += 1

        except Exception as e:
            print(f"Failed to index {img_path.name}: {e}")

    session.commit()
    if new_count > 0 or updated_count > 0:
        print(f"Indexed {model.name}: {new_count} new, {updated_count} updated.")
