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
    prediction_type: str = Field(
        default="epsilon", description="epsilon or v_prediction"
    )
    compatibility: Dict = Field(default={}, sa_column=Column(JSON))
    meta: Dict = Field(default={}, sa_column=Column(JSON))
    is_hidden: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    results: List["ModelResult"] = Relationship(back_populates="model")


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


def get_session():
    return Session(engine)
