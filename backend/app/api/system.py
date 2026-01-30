from fastapi import APIRouter, Body, HTTPException, Response
from sqlmodel import desc, select

from ..core import database as db
from ..core import state
from ..services import image_service, notes_manager


router = APIRouter()


@router.get("/thumbnails/{rel_path:path}")
def get_thumbnail(rel_path: str, w: int = 150, h: int = 150, q: int = 80):
    """
    Generate or retrieve a cached thumbnail.

    Security: Validates that rel_path stays within ASSETS_DIR.
    Caching: Returns immutable cache headers for browser "forever cache".
    """
    # Clamp dimensions to reasonable limits
    width = max(16, min(w, 512))
    height = max(16, min(h, 512))
    quality = max(10, min(q, 100))

    try:
        thumb_bytes, media_type = image_service.get_thumbnail(rel_path, width=width, height=height, quality=quality)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Image not found") from None
    except ValueError as e:
        # Path traversal attempt or invalid path
        raise HTTPException(status_code=400, detail=str(e)) from None

    return Response(
        content=thumb_bytes,
        media_type=media_type,
        headers={
            "Cache-Control": "public, max-age=31536000, immutable",
        },
    )


@router.get("/status")
def get_status():
    """Get current generation status."""
    return {
        "is_running": state.generation_state["is_running"],
        "current_model": state.generation_state["current_model"],
        "progress": state.generation_state["progress"],
    }


@router.get("/runs")
def get_runs():
    """Get list of past benchmark runs with their model results."""
    with db.get_session() as session:
        runs = session.exec(select(db.BenchmarkRun).order_by(desc(db.BenchmarkRun.timestamp))).all()

        result = []
        for run in runs:
            # Get results for this run
            run_results = session.exec(select(db.ModelResult).where(db.ModelResult.run_id == run.id)).all()

            # Build model results with names
            models_data = []
            for res in run_results:
                model = session.get(db.Model, res.model_hash)
                models_data.append(
                    {
                        "model_hash": res.model_hash,
                        "model_name": model.name if model else "Unknown",
                        "metrics": res.metrics,
                        "image_count": res.image_count,
                    }
                )

            result.append(
                {
                    "id": run.id,
                    "timestamp": run.timestamp.isoformat(),
                    "parameters": run.parameters,
                    "prompts": run.prompts,
                    "prompt_set_id": run.prompt_set_id,
                    "results": models_data,
                }
            )

        return result


@router.get("/notes/{note_id}")
def get_note(note_id: str):
    return notes_manager.get_note(note_id)


@router.post("/notes/{note_id}")
def update_note(note_id: str, payload: dict = Body(...)):
    return notes_manager.update_note(note_id, payload)


@router.post("/system/browse")
def browse_system(payload: dict = Body(...)):
    """
    Opens a native file dialog on the server machine.
    Payload: { type: 'file' | 'folder', initial_dir: str }
    """
    import queue
    import tkinter as tk
    from tkinter import filedialog

    target_type = payload.get("type", "folder")

    # We must run tkinter in main thread or handle loop correctly.
    # Since FastAPI is threaded, we are already in a thread.
    # We create a temporary root, hide it, ask, destroy.

    result_queue = queue.Queue()

    def open_dialog():
        try:
            root = tk.Tk()
            root.withdraw()  # Hide main window
            root.attributes("-topmost", True)  # Bring to front

            paths = []
            if target_type == "file":
                # Use askopenfilenames (plural) to return a tuple of paths
                result = filedialog.askopenfilenames(
                    title="Select Model Files",
                    filetypes=[("Safetensors", "*.safetensors"), ("All Files", "*.*")],
                )
                # result is a tuple of strings
                if result:
                    paths = list(result)
            else:
                path = filedialog.askdirectory(title="Select Model Directory")
                if path:
                    paths = [path]

            root.destroy()
            result_queue.put(paths)
        except Exception as e:
            result_queue.put([])
            print(f"Dialog error: {e}")

    # Run in a separate thread to avoid blocking the loop too hard (though waiting is necessary)
    # Actually just running it here is fine as it blocks this request.
    try:
        open_dialog()
        paths = result_queue.get()
        return {"paths": paths}
    except Exception as e:
        return {"paths": [], "error": str(e)}
