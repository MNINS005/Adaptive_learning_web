from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.configuration.db_connection import get_db
from src.entity.db_models import User, Attempt, KnowledgeState
from src.logger import logger

router = APIRouter()


@router.get("/{user_id}/stats")
def get_stats(user_id: str, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(404, "User not found")

        total   = db.query(Attempt).filter(
            Attempt.user_id == user_id).count()
        correct = db.query(Attempt).filter(
            Attempt.user_id    == user_id,
            Attempt.is_correct == True
        ).count()

        states = db.query(KnowledgeState).filter(
            KnowledgeState.user_id == user_id
        ).all()

        return {
            "user_id"         : user_id,
            "username"        : user.username,
            "total_attempts"  : total,
            "correct_attempts": correct,
            "accuracy"        : round(correct / total, 4) if total else 0.0,
            "topic_skills"    : {
                ks.topic: round(ks.skill_score, 4) for ks in states
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/{user_id}/knowledge-state")
def get_knowledge_state(user_id: str, db: Session = Depends(get_db)):
    try:
        states = db.query(KnowledgeState).filter(
            KnowledgeState.user_id == user_id
        ).all()

        return {
            "user_id": user_id,
            "topics" : [
                {
                    "topic"      : ks.topic,
                    "skill_score": round(ks.skill_score, 4),
                    "updated_at" : str(ks.updated_at)
                }
                for ks in states
            ]
        }
    except Exception as e:
        raise HTTPException(500, str(e))