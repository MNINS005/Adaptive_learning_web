from sqlalchemy.orm import Session
from src.entity.db_models import Attempt, KnowledgeState
from src.logger import logger
from src.exception import CustomException, CustomException
import sys


class AttemptData:

    def __init__(self, db: Session):
        self.db = db

    def get_user_attempts(self, user_id: str) -> list:
        try:
            attempts = (
                self.db.query(Attempt)
                .filter(Attempt.user_id == user_id)
                .order_by(Attempt.attempted_at.asc())
                .all()
            )
            logger.info(f"Fetched {len(attempts)} attempts for user {user_id}")
            return attempts
        except Exception as e:
            raise CustomException(e, sys)

    def get_recent_attempts(self, user_id: str, limit: int = 50) -> list:
        try:
            attempts = (
                self.db.query(Attempt)
                .filter(Attempt.user_id == user_id)
                .order_by(Attempt.attempted_at.desc())
                .limit(limit)
                .all()
            )
            logger.info(f"Fetched {len(attempts)} recent attempts for user {user_id}")
            return attempts
        except Exception as e:
            raise CustomException(e, sys)

    def save_attempt(self, user_id: str, question_id: str,
                     is_correct: bool, time_taken: int = None) -> Attempt:
        try:
            attempt = Attempt(
                user_id     = user_id,
                question_id = question_id,
                is_correct  = is_correct,
                time_taken  = time_taken
            )
            self.db.add(attempt)
            self.db.commit()
            self.db.refresh(attempt)
            logger.info(f"Attempt saved for user {user_id}")
            return attempt
        except Exception as e:
            self.db.rollback()
            raise CustomException(e, sys)

    def get_all_attempts_for_training(self) -> list:
        try:
            attempts = self.db.query(Attempt).all()
            logger.info(f"Fetched {len(attempts)} total attempts for training")
            return attempts
        except Exception as e:
            raise CustomException(e, sys)

    def update_knowledge_state(self, user_id: str,
                                topic: str, skill_score: float):
        try:
            ks = (
                self.db.query(KnowledgeState)
                .filter(
                    KnowledgeState.user_id == user_id,
                    KnowledgeState.topic   == topic
                )
                .first()
            )
            if ks:
                ks.skill_score = skill_score
            else:
                ks = KnowledgeState(
                    user_id     = user_id,
                    topic       = topic,
                    skill_score = skill_score
                )
                self.db.add(ks)
            self.db.commit()
            logger.info(f"Knowledge state updated: user={user_id} topic={topic} skill={skill_score}")
        except Exception as e:
            self.db.rollback()
            raise CustomException(e, sys)

    def get_knowledge_state(self, user_id: str) -> list:
        try:
            states = (
                self.db.query(KnowledgeState)
                .filter(KnowledgeState.user_id == user_id)
                .all()
            )
            return states
        except Exception as e:
            raise CustomException(e, sys)