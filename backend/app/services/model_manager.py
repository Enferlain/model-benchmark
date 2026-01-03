import hashlib
from pathlib import Path
from sqlmodel import select, desc
from typing import Optional
from ..core import database as db
from ..core.database import Model, BenchmarkRun, ModelResult as DBModelResult
from ..core.state import models_db, ModelResult
from . import prompt_manager as data_loader


def compute_model_hash(file_path: Path) -> str:
    """Computes BLAKE3 hash of the file. Reads in chunks."""
    try:
        import blake3

        hasher = blake3.blake3()
    except ImportError:
        hasher = hashlib.sha256()

    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
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
                if any(
                    k.startswith("conditioner.embedders.0.transformer") for k in keys
                ):
                    m_type = "sdxl"
                # SD 1.x models use cond_stage_model.transformer.text_model
                elif any(
                    k.startswith("cond_stage_model.transformer.text_model")
                    for k in keys
                ):
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
    """
    print(f"Syncing models with database (recheck_types={recheck_types})...")
    local_models = data_loader.get_available_models_from_disk()

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
            existing_model = session.exec(
                select(Model).where(Model.path == str(path))
            ).first()

            calculated_hash = None

            if existing_model:
                saved_meta = existing_model.meta or {}
                if (
                    saved_meta.get("mtime") == file_mtime
                    and saved_meta.get("size") == file_size
                ):
                    # Trusted match on file level
                    calculated_hash = existing_model.hash
                    found_hashes.add(calculated_hash)

                    if not recheck_types:
                        continue
                    # If recheck_types is True, we proceed to update logic below instead of continuing

            # If we are here, it's new, moved, changed, or we are forcing a recheck.
            if not calculated_hash:
                print(f"Hashing {path.name}...")
                try:
                    calculated_hash = compute_model_hash(path)
                except Exception as e:
                    print(f"Error hashing {path}: {e}")
                    continue

            found_hashes.add(calculated_hash)

            # --- Migration Logic: Ensure Output Folder follows Name ---
            target_folder_name = lm["name"]
            target_dir = data_loader.ASSETS_DIR / "outputs" / target_folder_name

            # Check for Hash Folder (if we created one recently)
            hash_dir = data_loader.ASSETS_DIR / "outputs" / calculated_hash

            if hash_dir.exists() and hash_dir != target_dir:
                if not target_dir.exists():
                    print(
                        f"Renaming hash folder {hash_dir.name} to {target_folder_name}..."
                    )
                    try:
                        hash_dir.rename(target_dir)
                    except Exception as e:
                        print(f"Failed to rename hash folder: {e}")
                else:
                    print(
                        f"Target folder {target_folder_name} exists. Keeping existing."
                    )

            # Upsert
            model_record = session.get(Model, calculated_hash)
            if not model_record:
                # New Model
                print(f"New model detected: {lm['name']} ({calculated_hash})")

                # Detect type
                m_type, pred_type, is_ztsnr = scan_model_type(path)

                model_record = Model(
                    hash=calculated_hash,
                    name=lm["name"],  # Use filename derived name
                    filename=path.name,
                    path=str(path),
                    type=m_type,
                    prediction_type=pred_type,
                    meta={"mtime": file_mtime, "size": file_size, "ztsnr": is_ztsnr},
                )
                session.add(model_record)
            else:
                # Update path/meta if changed
                if model_record.path != str(path):
                    print(f"Model moved: {model_record.name} to {path}")
                    model_record.path = str(path)
                    model_record.filename = path.name

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
                    print(
                        f"Updated metadata/type for {model_record.name}: {m_type}, {pred_type}, ztsnr={is_ztsnr}"
                    )
                else:
                    # Just update basic meta
                    if not model_record.meta:
                        model_record.meta = {}
                    model_record.meta["mtime"] = file_mtime
                    model_record.meta["size"] = file_size

                session.add(model_record)

        # Cleanup: Remove models from DB that rely on files not found in this scan
        # found_hashes contains valid models we just verified
        all_db_models_cleanup = session.exec(select(Model)).all()
        for db_m in all_db_models_cleanup:
            # If hash is not in found_hashes, it means it wasn't found on disk during this scan
            if db_m.hash not in found_hashes:
                print(f"Removing missing model from DB: {db_m.name} ({db_m.hash})")
                session.delete(db_m)

        session.commit()

        # 2. Refresh In-Memory State for API
        new_models_list = []

        all_db_models = session.exec(select(Model)).all()
        for db_m in all_db_models:
            # Verify existence on disk (optional, but good for "Clean" list)
            if not Path(db_m.path).exists():
                continue

            # Get latest result
            latest_res = session.exec(
                select(DBModelResult)
                .join(BenchmarkRun)
                .where(DBModelResult.model_hash == db_m.hash)
                .order_by(desc(BenchmarkRun.timestamp))
            ).first()

            # Use Name for URL/Folder
            folder_name = db_m.name

            api_m = ModelResult(
                id=db_m.hash,  # Using Hash as ID for API now
                hash=db_m.hash,
                name=db_m.name,
                source="Local",
                url=f"/assets/outputs/{folder_name}" if latest_res else "",
                path=db_m.path,
                prediction_type=db_m.prediction_type,
                model_type=db_m.type,
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

    print(f"DB Sync complete. Loaded {len(models_db)} models.")
    return models_db
