import os
import sys
import requests
import pandas as pd

from src.logger import logger
from src.exception import  CustomException
from src.configuration.db_connection import SessionLocal
from src.data_access.question_data import QuestionData


class DataGathering:

    def __init__(self):
        self.db = SessionLocal()
        self.qd = QuestionData(self.db)

    # ── topic mapper ───────────────────────────────────────────────────
    @staticmethod
    def map_topic(raw_tags: str) -> str:
        tag = raw_tags.lower()
        if "linked list"         in tag: return "linked_lists"
        if "dynamic programming" in tag: return "dynamic_programming"
        if "binary search"       in tag: return "searching"
        if "recursion"           in tag: return "recursion"
        if "sort"                in tag: return "sorting"
        if "tree"                in tag: return "trees"
        if "graph"               in tag: return "graphs"
        if "array"               in tag: return "arrays"
        if "stack"               in tag: return "arrays"
        if "queue"               in tag: return "arrays"
        if "hash"                in tag: return "arrays"
        if "string"              in tag: return "arrays"
        return "arrays"

    # ── difficulty mapper ──────────────────────────────────────────────
    @staticmethod
    def map_difficulty(raw: str) -> float:
        mapping = {
            "easy":   0.2,
            "medium": 0.5,
            "hard":   0.8,
        }
        return mapping.get(str(raw).strip().lower(), 0.5)

    # ── Source 1: Kaggle CSV ───────────────────────────────────────────
    def load_from_kaggle_csv(self, csv_path: str) -> int:
        try:
            logger.info(f"Loading from Kaggle CSV: {csv_path}")

            df = pd.read_csv(csv_path)
            logger.info(f"CSV shape: {df.shape}")
            logger.info(f"CSV columns: {df.columns.tolist()}")

            # clean column names — strip spaces, lowercase
            df.columns = df.columns.str.strip().str.lower()

            # try both common column name patterns
            content_col    = next(
                (c for c in df.columns if c in ["title", "question_title", "name", "problem"]),
                None
            )
            difficulty_col = next(
                (c for c in df.columns if "difficulty" in c),
                None
            )
            topic_col      = next(
                (c for c in df.columns if c in ["topic_tags", "tags", "topics", "category"]),
                None
            )

            if not content_col:
                raise AppException("No title/content column found in CSV", sys)

            logger.info(f"Using columns → content: {content_col} | difficulty: {difficulty_col} | topic: {topic_col}")

            saved = 0
            for _, row in df.iterrows():
                content = str(row.get(content_col, "")).strip()
                if not content or content == "nan":
                    continue

                difficulty = self.map_difficulty(
                    row.get(difficulty_col, "medium") if difficulty_col else "medium"
                )
                topic = self.map_topic(
                    str(row.get(topic_col, "")) if topic_col else ""
                )

                self.qd.save_question(
                    content    = content,
                    topic      = topic,
                    difficulty = difficulty,
                    source     = "kaggle"
                )
                saved += 1

            logger.info(f"Saved {saved} questions from Kaggle CSV")
            return saved

        except Exception as e:
            self.db.rollback()
            raise CustomException(e, sys)

    # ── Source 2: OpenTrivia API ───────────────────────────────────────
    def load_from_opentrivia(self, amount: int = 50) -> int:
        try:
            logger.info(f"Fetching {amount} questions from OpenTrivia API")

            url    = "https://opentdb.com/api.php"
            params = {"amount": amount, "category": 18, "type": "multiple"}
            resp   = requests.get(url, params=params, timeout=10)
            data   = resp.json()

            if data.get("response_code") != 0:
                logger.warning("OpenTrivia API returned non-zero response code")
                return 0

            saved = 0
            for q in data.get("results", []):
                self.qd.save_question(
                    content    = q["question"],
                    topic      = "general_cs",
                    difficulty = self.map_difficulty(q.get("difficulty", "medium")),
                    source     = "opentrivia"
                )
                saved += 1

            logger.info(f"Saved {saved} questions from OpenTrivia")
            return saved

        except Exception as e:
            self.db.rollback()
            raise CustomException(e, sys)

    def close(self):
        self.db.close()