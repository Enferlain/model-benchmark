from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import JSON, Column
from sqlmodel import Field, Relationship, Session, SQLModel, create_engine


# Database setup
DATABASE_FILE = "assets/database.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"


# Models
class Model(SQLModel, table=True):
    hash: str = Field(primary_key=True, index=True, description="SHA256 hash of the model file")
    name: str = Field(index=True)
    filename: str
    path: str
    type: str = Field(default="unknown", description="e.g. sd15, sdxl")
    source: str = Field(default="Local", description="Civitai, HuggingFace, or Local")
    prediction_type: str = Field(default="epsilon", description="epsilon or v_prediction")
    hash_type: str = Field(default="sha256", description="Algorithm (sha256, blake3)")
    compatibility: dict = Field(default={}, sa_column=Column(JSON))
    meta: dict = Field(default={}, sa_column=Column(JSON))
    is_hidden: bool = Field(default=False)
    is_missing: bool = Field(default=False, description="True if model file is not found on disk")
    bt_score: float = Field(default=1000.0, description="Bradley-Terry strength score")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    results: list["ModelResult"] = Relationship(back_populates="model", sa_relationship_kwargs={"cascade": "all, delete-orphan"})


class BenchmarkRun(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    parameters: dict = Field(default={}, sa_column=Column(JSON))
    prompts: list[str] = Field(default=[], sa_column=Column(JSON))
    prompt_set_id: str | None = Field(default=None, index=True)
    # Future: link to Prompt entities
    # prompt_ids: list[str] = Field(default=[], sa_column=Column(JSON))

    results: list["ModelResult"] = Relationship(back_populates="run")


class ModelResult(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="benchmarkrun.id")
    model_hash: str = Field(foreign_key="model.hash")

    metrics: dict = Field(default={}, sa_column=Column(JSON))
    image_count: int = Field(default=0)

    run: BenchmarkRun = Relationship(back_populates="results")
    model: Model = Relationship(back_populates="results")


class Prompt(SQLModel, table=True):
    id: str = Field(primary_key=True, description="Stable UUID for the prompt")
    text: str = Field(index=True)
    filename: str | None = Field(default=None, index=True)
    category: str | None = Field(default=None, index=True)
    tags: list[str] = Field(default=[], sa_column=Column(JSON))
    meta: dict = Field(default={}, sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ArenaVote(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    model_a_hash: str = Field(foreign_key="model.hash", index=True)
    model_b_hash: str = Field(foreign_key="model.hash", index=True)
    winner_hash: str | None = Field(default=None, index=True)
    vote_type: str = Field(description="model_a, model_b, tie, both_bad")

    prompt_id: str | None = Field(default=None, foreign_key="prompt.id", index=True)
    prompt_text: str | None = Field(default=None, description="Fallback if no prompt_id")
    seed: int | None = Field(default=None)
    parameters: dict = Field(default={}, sa_column=Column(JSON))

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ImageOutput(SQLModel, table=True):
    id: str = Field(primary_key=True, description="8-char UUID from PNG metadata")
    model_hash: str = Field(foreign_key="model.hash", index=True)
    prompt_id: str | None = Field(default=None, foreign_key="prompt.id", index=True)
    prompt_text: str | None = Field(default=None, description="Fallback text")
    seed: int | None = Field(default=None)
    path: str = Field(index=True)
    mtime: int = Field(default=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# Engine
engine = create_engine(DATABASE_URL)


def init_db():
    # Ensure assets dir exists
    Path("assets").mkdir(exist_ok=True)
    # SQLModel.metadata.create_all(engine)
    # Legacy migration logic removed in favor of Alembic
    pass


def get_session():
    return Session(engine)
