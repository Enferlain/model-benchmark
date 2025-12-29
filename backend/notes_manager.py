import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Constants
ASSETS_DIR = Path("assets")
NOTES_FILE = ASSETS_DIR / "notes.json"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _load_notes_db() -> Dict[str, Any]:
    """Loads the notes database from the JSON file."""
    if not NOTES_FILE.exists():
        return {}

    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        logger.error(f"Failed to decode {NOTES_FILE}. Returning empty db.")
        return {}
    except Exception as e:
        logger.error(f"Error loading notes: {e}")
        return {}

def _save_notes_db(db: Dict[str, Any]) -> None:
    """Saves the notes database to the JSON file."""
    # Ensure directory exists
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with open(NOTES_FILE, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving notes: {e}")

def get_note(note_id: str) -> Dict[str, Any]:
    """Retrieves a note by ID."""
    db = _load_notes_db()
    return db.get(note_id, {})

def update_note(note_id: str, content: Dict[str, Any]) -> Dict[str, Any]:
    """Updates or creates a note for the given ID. Merges with existing data."""
    db = _load_notes_db()

    current_note = db.get(note_id, {})
    # Update fields (simple merge)
    current_note.update(content)

    # Update timestamp if not provided?
    # For now, just save what is given.

    db[note_id] = current_note
    _save_notes_db(db)
    return current_note

def delete_note(note_id: str) -> bool:
    """Deletes a note by ID."""
    db = _load_notes_db()
    if note_id in db:
        del db[note_id]
        _save_notes_db(db)
        return True
    return False
