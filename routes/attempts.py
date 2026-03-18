from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.configuration.db_connection import get_db
from src.entity.db_models import Attempt, Question, KnowledgeState
from src.entity.schemas import AttemptCreate, AttemptOut
from src.data_access.attempt_data import AttemptData
from src.utils.main_utils import compute_reward
from src.logger import logger

router = APIRouter()


@router.post("/", response_model=AttemptOut)
def log_attempt(payload: AttemptCreate, db: Session = Depends(get_db)):
    try:
        ad      = AttemptData(db)
        attempt = ad.save_attempt(
            user_id     = payload.user_id,
            question_id = payload.question_id,
            is_correct  = payload.is_correct,
            time_taken  = payload.time_taken
        )

        # update knowledge state
        question = db.query(Question).filter(
            Question.id == payload.question_id
        ).first()

        if question:
            ks = db.query(KnowledgeState).filter(
                KnowledgeState.user_id == str(payload.user_id),
                KnowledgeState.topic   == question.topic
            ).first()

            current_skill = ks.skill_score if ks else 0.3
            new_skill     = min(1.0, current_skill + 0.05) \
                if payload.is_correct \
                else max(0.0, current_skill - 0.02)

            ad.update_knowledge_state(
                str(payload.user_id),
                question.topic,
                new_skill
            )

            reward = compute_reward(
                payload.is_correct,
                question.difficulty,
                current_skill
            )
            logger.info(
                f"Attempt: user={payload.user_id} "
                f"correct={payload.is_correct} "
                f"reward={reward:.3f}"
            )

        return attempt

    except Exception as e:
        db.rollback()
        raise HTTPException(500, str(e))