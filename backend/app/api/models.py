from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from ..core import database as db
from ..core import state
from ..services import downloader, model_manager
from ..services import prompt_manager as data_loader


router = APIRouter()


@router.get("/models")
def get_models():
    return state.models_db


from pydantic import BaseModel


class RegisterModelRequest(BaseModel):
    path: str


class RegisterBatchRequest(BaseModel):
    paths: list[str]


@router.post("/models/register-batch")
def register_batch(request: RegisterBatchRequest):
    """
    Registers multiple paths and syncs ONCE.
    """
    try:
        results = model_manager.register_paths(request.paths)
        return results
    except Exception as e:
        print(f"Batch Registration Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/register")
def register_model_path(request: RegisterModelRequest):
    """
    Registers a local path (file or folder) immediately.
    Logs it to sources.json and adds to DB.
    """
    try:
        result = model_manager.register_path(request.path)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Registration Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/scan")
def scan_models():
    """Triggers a re-scan of the models directory."""
    try:
        updated_list = model_manager.sync_models_with_db(recheck_types=True)
        return {"status": "ok", "count": len(updated_list)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/{model_id}/outputs")
def get_model_outputs(model_id: str):
    # Retrieve model name from state to determine correct output folder
    # model_id is the hash, but folders are stored by Name
    model_entry = next((m for m in state.models_db if m.id == model_id), None)
    if not model_entry:
        return []

    output_dir = data_loader.ASSETS_DIR / "outputs" / model_entry.name
    if not output_dir.exists():
        # Fallback: check if directory exists by ID (legacy or just in case)
        if (data_loader.ASSETS_DIR / "outputs" / model_id).exists():
            output_dir = data_loader.ASSETS_DIR / "outputs" / model_id
        else:
            return []

    # Get prompts (cached or reloaded)
    prompts = data_loader.load_prompts_only()

    images = []
    # Filename format: p{prompt_idx:03d}_i{image_idx:02d}_s{seed}.png
    # We sort by filename to keep them in order
    for img_path in sorted(list(output_dir.glob("p*_i*_s*.png"))):
        try:
            name = img_path.stem
            parts = name.split("_")
            # robust parsing
            prompt_idx = -1
            seed = -1

            for part in parts:
                if part.startswith("p") and part[1:].isdigit():
                    prompt_idx = int(part[1:])
                elif part.startswith("s") and part[1:].isdigit():
                    seed = int(part[1:])

            if prompt_idx != -1:
                # Default fallback
                prompt_text = (
                    prompts[prompt_idx]
                    if prompt_idx < len(prompts)
                    else "Unknown prompt"
                )

                # Metadata check
                try:
                    from PIL import Image

                    with Image.open(img_path) as img:
                        img.load()
                        meta_prompt = img.info.get("prompt")
                        if meta_prompt:
                            prompt_text = meta_prompt
                except Exception:
                    pass

                # Construct URL: mounted /assets points to backend/assets
                # Output dir is backend/assets/outputs/{model_folder_name} (usually Name)
                # So URL is /assets/outputs/{output_dir.name}/{filename}
                # Use standard forward slashes for URLs
                url = f"/assets/outputs/{output_dir.name}/{img_path.name}"

                # Get mtime for cache busting
                mtime = int(img_path.stat().st_mtime)

                images.append(
                    {
                        "filename": img_path.name,
                        "url": url,
                        "prompt": prompt_text,
                        "seed": seed,
                        "prompt_idx": prompt_idx,
                        "mtime": mtime,
                    }
                )
        except Exception as e:
            print(f"Error parsing metadata for {img_path}: {e}")

    return images


@router.delete("/models/{model_id}")
def delete_model(model_id: str, delete_file: bool = False):
    # Find in DB first (model_id is hash now)
    target_hash = model_id

    with db.get_session() as session:
        # We still need to look up path from DB to be safe/correct
        db_model = session.get(db.Model, target_hash)

        if not db_model:
            # If not in DB, maybe it was just in memory?
            # If explicit delete requested, we should probably just return OK or 404
            pass
        else:
            # Delete File if requested
            if delete_file:
                path_str = db_model.path
                try:
                    file_path = Path(path_str).resolve()
                    models_dir_resolved = data_loader.MODELS_DIR.resolve()

                    # Security check: Only allow deleting files in models dir
                    # But wait, what if it's an external import?
                    # External imports should allow deletion if user says so?
                    # The original code had a security check:
                    # try: file_path.relative_to(models_dir_resolved)
                    # This prevents deleting system files if path was spoofed or external.
                    # If the user imported "C:\Windows\System32\kernel32.dll" (as a model??), we shouldn't delete it.
                    # So we KEEP the security check.

                    is_safe = False
                    try:
                        file_path.relative_to(models_dir_resolved)
                        is_safe = True
                    except ValueError:
                        pass

                    # Also allow deleting from known outputs? No, models are input.

                    if is_safe and file_path.exists():
                        if file_path.is_dir() or file_path.is_symlink():
                            # raising error might abort DB delete.
                            # We should probably still delete DB entry if file fails?
                            # Or fail fully. Let's fail fully for safety.
                            raise HTTPException(
                                status_code=403,
                                detail="Cannot delete directories or symlinks",
                            )

                        file_path.unlink()
                        print(f"Deleted file: {file_path}")
                    elif not is_safe:
                        print(
                            f"Skipping file deletion for external/unsafe path: {file_path}"
                        )
                        # If user ASKED to delete file but we can't, do we error?
                        # Probably yes, to warn them.
                        raise HTTPException(
                            status_code=403,
                            detail="Cannot delete file outside models directory (External Import)",
                        )

                except HTTPException:
                    raise
                except Exception as e:
                    print(f"Error deleting file: {e}")
                    raise HTTPException(status_code=500, detail=str(e)) from e

            # ALWAYS Delete from DB
            session.delete(db_model)
            session.commit()

            # Remove from 'sources.json' if it was there?
            # 'sources.json' tracks imported paths. If we delete the model, we should probably remove it from sources
            # to prevent it from re-appearing on next scan (if file still exists or if it was a folder).
            # If it was a file import, removing from DB/File is good.
            # But if it's in sources.json, scan might re-add it?
            # If delete_file=True, scan won't find it.
            # If delete_file=False (ghost cleanup), scan WILL find it again if it exists!
            # So checking sources is tricky.
            # However, for "ghosts", the file doesn't exist, so scan won't find it.
            # So just DB delete is sufficient for ghosts.

    # Remove from Session List
    state.models_db[:] = [m for m in state.models_db if m.id != model_id]

    return {"status": "ok"}


@router.post("/models/download")
def download_model(request: state.ModelRequest, background_tasks: BackgroundTasks):
    # TOCTOU protection
    with state.download_state_lock:
        if state.download_state["is_downloading"]:
            raise HTTPException(status_code=400, detail="Download already in progress")
        state.download_state["is_downloading"] = True

    background_tasks.add_task(
        downloader.download_model_task,
        request.url,
        request.name,
        request.source,
        request.api_token,
    )
    return {"status": "started"}


@router.get("/models/download/status")
def get_download_status():
    with state.download_state_lock:
        return state.download_state.copy()
