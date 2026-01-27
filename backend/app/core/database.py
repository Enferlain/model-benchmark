from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlmodel import Field, SQLModel, create_engine, Session, Relationship
from sqlalchemy import Column, JSON
from pathlib import Path

# Database setup
DATABASE_FILE = "assets/database.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"


# Models
class Model(SQLModel, table=True):
    hash: str = Field(
        primary_key=True, index=True, description="BLAKE3 hash of the model file"
    )
    name: str = Field(index=True)
    filename: str
    path: str
    type: str = Field(default="unknown", description="e.g. sd15, sdxl")
    source: str = Field(default="Local", description="Civitai, HuggingFace, or Local")
    prediction_type: str = Field(
        default="epsilon", description="epsilon or v_prediction"
    )
    compatibility: Dict = Field(default={}, sa_column=Column(JSON))
    meta: Dict = Field(default={}, sa_column=Column(JSON))
    is_hidden: bool = Field(default=False)
    is_missing: bool = Field(
        default=False, description="True if model file is not found on disk"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    results: List["ModelResult"] = Relationship(
        back_populates="model", sa_relationship_kwargs={"cascade": "all, delete-orphan"}
    )


class BenchmarkRun(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    parameters: Dict = Field(default={}, sa_column=Column(JSON))
    prompts: List[str] = Field(default=[], sa_column=Column(JSON))
    prompt_set_id: Optional[str] = Field(default=None, index=True)

    results: List["ModelResult"] = Relationship(back_populates="run")


class ModelResult(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: int = Field(foreign_key="benchmarkrun.id")
    model_hash: str = Field(foreign_key="model.hash")

    metrics: Dict = Field(default={}, sa_column=Column(JSON))
    image_count: int = Field(default=0)

    run: BenchmarkRun = Relationship(back_populates="results")
    model: Model = Relationship(back_populates="results")


# Engine
engine = create_engine(DATABASE_URL)


def init_db():
    # Ensure assets dir exists
    Path("assets").mkdir(exist_ok=True)
    SQLModel.metadata.create_all(engine)

    # Ad-hoc migration for is_missing column
    try:
        from sqlalchemy import inspect, text

        inspector = inspect(engine)
        columns = [c["name"] for c in inspector.get_columns("model")]
        if "is_missing" not in columns:
            print("Migrating DB: Adding is_missing column to model table...")
            with engine.connect() as conn:
                conn.execute(
                    text("ALTER TABLE model ADD COLUMN is_missing BOOLEAN DEFAULT 0")
                )
                conn.commit()

        if "source" not in columns:
            print("Migrating DB: Adding source column to model table...")
            with engine.connect() as conn:
                conn.execute(
                    text("ALTER TABLE model ADD COLUMN source VARCHAR DEFAULT 'Local'")
                )
                conn.commit()
    except Exception as e:
        print(f"Migration warning: {e}")


def get_session():
    return Session(engine)
