from fastapi import APIRouter, Body
from ..services import notes_manager
from ..core import state, database as db
from sqlmodel import select, desc

router = APIRouter()

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
        runs = session.exec(
            select(db.BenchmarkRun).order_by(desc(db.BenchmarkRun.timestamp))
        ).all()
        
        result = []
        for run in runs:
            # Get results for this run
            run_results = session.exec(
                select(db.ModelResult).where(db.ModelResult.run_id == run.id)
            ).all()
            
            # Build model results with names
            models_data = []
            for res in run_results:
                model = session.get(db.Model, res.model_hash)
                models_data.append({
                    "model_hash": res.model_hash,
                    "model_name": model.name if model else "Unknown",
                    "metrics": res.metrics,
                    "image_count": res.image_count
                })
            
            result.append({
                "id": run.id,
                "timestamp": run.timestamp.isoformat(),
                "parameters": run.parameters,
                "prompts": run.prompts,
                "prompt_set_id": run.prompt_set_id,
                "results": models_data
            })
        
        return result

@router.get("/notes/{note_id}")
def get_note(note_id: str):
    return notes_manager.get_note(note_id)

@router.post("/notes/{note_id}")
def update_note(note_id: str, payload: dict = Body(...)):
    return notes_manager.update_note(note_id, payload)
