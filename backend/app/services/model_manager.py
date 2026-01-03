import hashlib
import random
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


def sync_models_with_db():
    """
    Scans disk for models, hashes them, updates DB, and then refreshes the in-memory state.
    """
    print("Syncing models with database...")
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
                    # Trusted match
                    calculated_hash = existing_model.hash
                    found_hashes.add(calculated_hash)
                    continue

            # If we are here, it's new, moved, or changed.
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
                m_type = "sd1.5"
                pred_type = "epsilon"

                # Simple heuristic mapping from filename
                lower_name = path.name.lower()
                if "xl" in lower_name:
                    m_type = "sdxl"
                if "v-pred" in lower_name or "vpred" in lower_name:
                    pred_type = "v_prediction"

                model_record = Model(
                    hash=calculated_hash,
                    name=lm["name"],  # Use filename derived name
                    filename=path.name,
                    path=str(path),
                    type=m_type,
                    prediction_type=pred_type,
                    meta={"mtime": file_mtime, "size": file_size},
                )
                session.add(model_record)
            else:
                # Update path/meta if changed
                if model_record.path != str(path):
                    print(f"Model moved: {model_record.name} to {path}")
                    model_record.path = str(path)
                    model_record.filename = path.name

                # Update meta just in case
                model_record.meta = {"mtime": file_mtime, "size": file_size}
                session.add(model_record)

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

            if latest_res:
                api_m.accuracy = latest_res.metrics.get("accuracy", 0.0)
                api_m.diversity = latest_res.metrics.get("diversity", 0.0)
                api_m.metrics = latest_res.metrics
                api_m.image_count = latest_res.image_count

            new_models_list.append(api_m)

        # Atomic replacement
        models_db[:] = new_models_list

    print(f"DB Sync complete. Loaded {len(models_db)} models.")
    return models_db
