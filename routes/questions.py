import os
import json
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.configuration.db_connection import get_db
from src.entity.db_models import Question, Attempt, KnowledgeState
from src.utils.main_utils import (
    load_dkt_model, load_rl_agent,
    encode_user_sequence, get_knowledge_state,
    filter_candidate_questions, load_json
)
from src.constants import DKT_MODEL_PATH, RL_POLICY_PATH, ARTIFACT_DIR
from src.logger import logger

router = APIRouter()

TOPIC_MAP = {
    "arrays": 0, "linked_lists": 1, "trees": 2,
    "graphs": 3, "dynamic_programming": 4,
    "sorting": 5, "searching": 6,
    "recursion": 7, "general_cs": 8,
}

# load models once at startup
try:
    dkt_model      = load_dkt_model(DKT_MODEL_PATH)
    rl_agent       = load_rl_agent(RL_POLICY_PATH)
    question_index = load_json(
        os.path.join(ARTIFACT_DIR, "transformed", "question_index.json")
    )
    num_questions  = len(question_index)
    logger.info(f"Models loaded — {num_questions} questions indexed")
except Exception as e:
    dkt_model = rl_agent = None
    question_index = {}
    num_questions  = 0
    logger.warning(f"Models not loaded: {e}")


@router.get("/next/{user_id}")
def get_next_question(user_id: str, db: Session = Depends(get_db)):
    try:
        # fallback if models not loaded
        if not dkt_model or not rl_agent:
            q = db.query(Question).first()
            if not q:
                raise HTTPException(404, "No questions found")
            return {
                "question_id": str(q.id), "content": q.content,
                "topic": q.topic, "difficulty": q.difficulty,
                "source": "random_fallback"
            }

        # get user attempts
        attempts = (
            db.query(Attempt)
            .filter(Attempt.user_id == user_id)
            .order_by(Attempt.attempted_at.asc())
            .all()
        )

        # encode sequence
        attempt_dicts = [
            {"question_id": str(a.question_id),
             "is_correct": int(a.is_correct)}
            for a in attempts
        ]
        user_seq   = encode_user_sequence(
            attempt_dicts, question_index, num_questions
        )

        # get knowledge state from DKT
        state      = get_knowledge_state(dkt_model, user_seq)
        user_skill = float(np.mean(state))

        # find weakest topic
        ks = (
            db.query(KnowledgeState)
            .filter(KnowledgeState.user_id == user_id)
            .order_by(KnowledgeState.skill_score.asc())
            .first()
        )
        target_topic = ks.topic if ks else None

        # get candidates
        all_q      = db.query(Question).all()
        candidates = filter_candidate_questions(
            all_q, user_skill, target_topic
        )
        if not candidates:
            candidates = all_q[:10]

        # RL agent picks best
        scores = [
            rl_agent.forward(
                state,
                q.difficulty,
                TOPIC_MAP.get(q.topic, 0)
            )
            for q in candidates
        ]
        best_q = candidates[int(np.argmax(scores))]

        logger.info(
            f"Next Q for {user_id}: "
            f"topic={best_q.topic} diff={best_q.difficulty}"
        )
        return {
            "question_id" : str(best_q.id),
            "content"     : best_q.content,
            "topic"       : best_q.topic,
            "difficulty"  : best_q.difficulty,
            "user_skill"  : round(user_skill, 4),
            "source"      : "rl_agent"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Next question error: {e}")
        raise HTTPException(500, str(e))


@router.get("/all")
def get_questions(
    topic: str = None,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    try:
        query = db.query(Question)
        if topic:
            query = query.filter(Question.topic == topic)
        return [
            {
                "question_id": str(q.id), "content": q.content,
                "topic": q.topic, "difficulty": q.difficulty
            }
            for q in query.limit(limit).all()
        ]
    except Exception as e:
        raise HTTPException(500, str(e))