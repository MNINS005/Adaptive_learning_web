from sqlalchemy.orm import Session
from src.entity.db_models import Question
from src.logger import logger
from src.exception import CustomException
import sys


class QuestionData:

    def __init__(self, db: Session):
        self.db = db

    def get_question_by_id(self, question_id: str) -> Question:
        try:
            question = (
                self.db.query(Question)
                .filter(Question.id == question_id)
                .first()
            )
            return question
        except Exception as e:
            raise AppException(e, sys)

    def get_questions_by_topic(self, topic: str) -> list:
        try:
            questions = (
                self.db.query(Question)
                .filter(Question.topic == topic)
                .all()
            )
            logger.info(f"Fetched {len(questions)} questions for topic {topic}")
            return questions
        except Exception as e:
            raise AppException(e, sys)

    def get_questions_by_difficulty_range(self, min_diff: float,
                                           max_diff: float) -> list:
        try:
            questions = (
                self.db.query(Question)
                .filter(
                    Question.difficulty >= min_diff,
                    Question.difficulty <= max_diff
                )
                .all()
            )
            logger.info(f"Fetched {len(questions)} questions in difficulty {min_diff}-{max_diff}")
            return questions
        except Exception as e:
            raise AppException(e, sys)

    def get_candidate_questions(self, topic: str,
                                 user_skill: float,
                                 zpd_delta: float = 0.15) -> list:
        try:
            target     = user_skill + zpd_delta
            min_diff   = max(0.0, target - 0.2)
            max_diff   = min(1.0, target + 0.2)
            questions  = (
                self.db.query(Question)
                .filter(
                    Question.topic      == topic,
                    Question.difficulty >= min_diff,
                    Question.difficulty <= max_diff
                )
                .all()
            )
            logger.info(f"Found {len(questions)} candidate questions for skill {user_skill}")
            return questions
        except Exception as e:
            raise AppException(e, sys)

    def save_question(self, content: str, topic: str,
                      difficulty: float, source: str) -> Question:
        try:
            question = Question(
                content    = content,
                topic      = topic,
                difficulty = difficulty,
                source     = source
            )
            self.db.add(question)
            self.db.commit()
            self.db.refresh(question)
            logger.info(f"Question saved: topic={topic} difficulty={difficulty}")
            return question
        except Exception as e:
            self.db.rollback()
            raise AppException(e, sys)

    def get_all_questions(self) -> list:
        try:
            questions = self.db.query(Question).all()
            logger.info(f"Fetched all {len(questions)} questions")
            return questions
        except Exception as e:
            raise AppException(e, sys)