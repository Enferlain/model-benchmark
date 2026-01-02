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
    """Get list of past benchmark runs."""
    with db.get_session() as session:
        runs = session.exec(
            select(db.BenchmarkRun).order_by(desc(db.BenchmarkRun.timestamp))
        ).all()
        return runs

@router.get("/notes/{note_id}")
def get_note(note_id: str):
    return notes_manager.get_note(note_id)

@router.post("/notes/{note_id}")
def update_note(note_id: str, payload: dict = Body(...)):
    return notes_manager.update_note(note_id, payload)
