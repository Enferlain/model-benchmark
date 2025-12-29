import json
from pathlib import Path
from typing import Dict, Any

from data_loader import ASSETS_DIR

NOTES_FILE = ASSETS_DIR / "notes.json"

def load_notes() -> Dict[str, Any]:
    if NOTES_FILE.exists():
        try:
            with open(NOTES_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_notes(notes: Dict[str, Any]):
    with open(NOTES_FILE, 'w') as f:
        json.dump(notes, f, indent=2)

def get_note(note_id: str) -> str:
    notes = load_notes()
    return notes.get(note_id, "")

def set_note(note_id: str, content: str):
    notes = load_notes()
    notes[note_id] = content
    save_notes(notes)
