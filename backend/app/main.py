from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import threading

# Import backend modules
from .services import prompt_manager as data_loader
from .services import model_manager
from .core import state, database as db
from .api import models, prompts, generation, system

app = FastAPI(lifespan=lifespan)

# Suppress uvicorn access log spam for polling endpoints
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

# Mount assets directory for serving images
# Ensure the directory exists to avoid errors on startup if it's missing
data_loader.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
db.init_db()
app.mount("/assets", StaticFiles(directory=data_loader.ASSETS_DIR), name="assets")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup/Shutdown Events
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from contextlib import asynccontextmanager

class ModelFileHandler(FileSystemEventHandler):
    def __init__(self):
        self._timer = None
        self.debounce_seconds = 2.0

    def on_any_event(self, event):
        if event.is_directory:
            return
        self._trigger_scan()

    def _trigger_scan(self):
        if self._timer:
            self._timer.cancel()
        self._timer = threading.Timer(self.debounce_seconds, self._run_scan)
        self._timer.start()

    def _run_scan(self):
        print("Detected file changes in models directory. Rescanning...")
        try:
            model_manager.sync_models_with_db()
            print("Auto-scan complete.")
        except Exception as e:
            print(f"Auto-scan failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up... (no auto-generation, use /api/generate or /api/analyze)")

    # 1. Initial Scan
    print("Scanning for local models (fast mode)...")
    model_manager.sync_models_with_db()

    # 2. Start Watcher
    # Ensure models dir exists
    data_loader.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    event_handler = ModelFileHandler()
    observer = Observer()
    observer.schedule(event_handler, str(data_loader.MODELS_DIR), recursive=False)
    observer.start()
    print(f"Started watching {data_loader.MODELS_DIR} for changes.")

    yield

    # Shutdown
    if observer:
        observer.stop()
        observer.join()

# Register Routers
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(prompts.router, prefix="/api", tags=["prompts"])
app.include_router(generation.router, prefix="/api", tags=["generation"])
app.include_router(system.router, prefix="/api", tags=["system"])
