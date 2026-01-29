from fastapi import APIRouter, HTTPException
from sqlmodel import desc, select
from typing import Optional
from pydantic import BaseModel
from ..core import database as db
from ..services import bt_service

router = APIRouter()


class VoteRequest(BaseModel):
    model_a_hash: str
    model_b_hash: str
    winner_hash: Optional[str] = None
    vote_type: str  # model_a, model_b, tie, both_bad
    prompt_id: Optional[str] = None
    prompt_text: Optional[str] = None
    seed: Optional[int] = None
    parameters: dict = {}


@router.post("/arena/vote")
def arena_vote(vote: VoteRequest):
    with db.get_session() as session:
        # 1. Create Vote Record
        new_vote = db.ArenaVote(
            model_a_hash=vote.model_a_hash,
            model_b_hash=vote.model_b_hash,
            winner_hash=vote.winner_hash,
            vote_type=vote.vote_type,
            prompt_id=vote.prompt_id,
            prompt_text=vote.prompt_text,
            seed=vote.seed,
            parameters=vote.parameters,
        )
        session.add(new_vote)
        session.commit()

        # 2. Trigger BT Rating Update
        bt_service.update_all_bt_ratings(session)

    return {"status": "ok"}


@router.get("/arena/leaderboard")
def get_leaderboard():
    with db.get_session() as session:
        models = session.exec(select(db.Model).order_by(desc(db.Model.bt_score))).all()

        return [{"hash": m.hash, "name": m.name, "bt_score": m.bt_score, "is_missing": m.is_missing} for m in models]
