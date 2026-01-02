from fastapi import APIRouter, HTTPException, BackgroundTasks
from pathlib import Path
from ..core import state, database as db
from ..services import downloader
from ..services import prompt_manager as data_loader

router = APIRouter()

@router.get("/models")
def get_models():
    return state.models_db

@router.get("/models/{model_id}/outputs")
def get_model_outputs(model_id: str):
    output_dir = data_loader.ASSETS_DIR / "outputs" / model_id
    if not output_dir.exists():
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
                # Output dir is backend/assets/outputs/{model_id}
                # So URL is /assets/outputs/{model_id}/{filename}
                # Use standard forward slashes for URLs
                url = f"/assets/outputs/{model_id}/{img_path.name}"

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

    # Update Global State (Active Session)
    if not any(m.id == model_id for m in state.models_db):
        # If not in session, we might still want to delete the file if requested?
        # But the ID is the hash.
        pass

    # Delete File if requested
    if delete_file:
        with db.get_session() as session:
            # We still need to look up path from DB to be safe/correct
            db_model = session.get(db.Model, target_hash)
            if db_model:
                path_str = db_model.path
                try:
                    file_path = Path(path_str).resolve()
                    models_dir_resolved = data_loader.MODELS_DIR.resolve()

                    # Security check
                    if not str(file_path).startswith(str(models_dir_resolved)):
                        raise HTTPException(
                            status_code=403,
                            detail="Cannot delete file outside models directory",
                        )

                    if file_path.exists():
                        if file_path.is_dir() or file_path.is_symlink():
                            raise HTTPException(
                                status_code=403,
                                detail="Cannot delete directories or symlinks",
                            )

                        file_path.unlink()
                        print(f"Deleted file: {file_path}")
                except HTTPException:
                    raise
                except Exception as e:
                    print(f"Error deleting file: {e}")
                    raise HTTPException(status_code=500, detail=str(e)) from e

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
