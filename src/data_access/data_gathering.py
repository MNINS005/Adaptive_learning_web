import os
import sys
import requests
import pandas as pd
import numpy as np
import random

from src.logger import logger
from src.exception import CustomException
from src.configuration.db_connection import SessionLocal
from src.data_access.question_data import QuestionData
from src.data_access.attempt_data import AttemptData
from src.entity.db_models import User, Question


class DataGathering:

    def __init__(self):
        self.db = SessionLocal()
        self.qd = QuestionData(self.db)
        self.ad = AttemptData(self.db)

    # ── topic mapper ───────────────────────────────────────────────────
    @staticmethod
    def map_topic(raw: str) -> str:
        tag = str(raw).lower()
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
        return "general_cs"

    # ── difficulty mapper ──────────────────────────────────────────────
    @staticmethod
    def map_difficulty(raw: str) -> float:
        mapping = {"easy": 0.2, "medium": 0.5, "hard": 0.8}
        return mapping.get(str(raw).strip().lower(), 0.5)

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1 — QUESTION SOURCES
    # ══════════════════════════════════════════════════════════════════

    # ── Source 1: Kaggle LeetCode CSV ─────────────────────────────────
    def load_from_kaggle_csv(self, csv_path: str) -> int:
        try:
            logger.info(f"Loading questions from Kaggle CSV: {csv_path}")

            # ── check if already loaded ────────────────────────────────
            existing = self.db.query(Question).filter(
                Question.source == "kaggle"
            ).count()
            if existing > 0:
                logger.warning(f"Kaggle questions already loaded ({existing}) — skipping")
                return existing

            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.lower()

            content_col = next(
                (c for c in df.columns if c in [
                    "title", "question_title", "name", "problem"
                ]), None
            )
            difficulty_col = next(
                (c for c in df.columns if "difficulty" in c), None
            )
            topic_col = next(
                (c for c in df.columns if c in [
                    "topic_tags", "tags", "topics", "category"
                ]), None
            )

            if not content_col:
                raise CustomException("No title column found in CSV", sys)

            saved = 0
            for _, row in df.iterrows():
                content = str(row.get(content_col, "")).strip()
                if not content or content == "nan":
                    continue

                self.qd.save_question(
                    content    = content,
                    topic      = self.map_topic(
                        str(row.get(topic_col, "")) if topic_col else ""
                    ),
                    difficulty = self.map_difficulty(
                        str(row.get(difficulty_col, "medium")) if difficulty_col else "medium"
                    ),
                    source     = "kaggle"
                )
                saved += 1

            logger.info(f"Kaggle CSV loaded: {saved} questions")
            return saved

        except Exception as e:
            self.db.rollback()
            raise CustomException(e, sys)

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2 — ATTEMPT SOURCES
    # ══════════════════════════════════════════════════════════════════

    # ── Source 2: ASSISTments 2009-2010 ───────────────────────────────
    def load_from_assistments(self, csv_path: str, max_students: int = 500) -> int:
        try:
            logger.info(f"Loading ASSISTments dataset: {csv_path}")

            df = pd.read_csv(csv_path, low_memory=False)
            df.columns = df.columns.str.strip().str.lower()

            logger.info(f"Dataset shape: {df.shape}")

            required = ["user_id", "correct"]
            for col in required:
                if col not in df.columns:
                    raise CustomException(f"Missing column {col}", sys)

            skill_col = "skill_name" if "skill_name" in df.columns else "problem_id"

            # ⭐ VERY IMPORTANT — sort by timestamp
            if "order_id" in df.columns:
                df = df.sort_values(["user_id", "order_id"])
            elif "start_time" in df.columns:
                df = df.sort_values(["user_id", "start_time"])

            df = df.dropna(subset=["user_id", skill_col, "correct"])
            df["correct"] = df["correct"].astype(int)

            # ───────── step-1 create questions (skills)
            unique_skills = df[skill_col].unique()
            skill_to_qid = {}

            logger.info(f"Unique skills: {len(unique_skills)}")

            for skill in unique_skills:
                skill = str(skill).strip()

                existing = self.db.query(Question).filter(
                    Question.content == skill,
                    Question.source == "assistments"
                ).first()

                if existing:
                    skill_to_qid[skill] = existing.id
                    continue

                q = self.qd.save_question(
                    content=skill,
                    topic=self.map_topic(skill),
                    difficulty=0.5,
                    source="assistments"
                )
                skill_to_qid[skill] = q.id

            self.db.commit()

            # ───────── step-2 create users
            unique_students = df["user_id"].unique()[:max_students]
            user_map = {}

            for sid in unique_students:
                uname = f"assist_{sid}"

                existing = self.db.query(User).filter(
                    User.username == uname
                ).first()

                if existing:
                    user_map[sid] = existing.id
                    continue

                user = User(
                    username=uname,
                    email=f"{uname}@assistments.com",
                    password="assist123"
                )
                self.db.add(user)
                self.db.flush()
                user_map[sid] = user.id

            self.db.commit()
            logger.info(f"Users created: {len(user_map)}")

            # ───────── step-3 save attempts
            df_sub = df[df["user_id"].isin(unique_students)]
            saved = 0

            for _, row in df_sub.iterrows():
                sid = row["user_id"]
                skill = str(row[skill_col]).strip()

                if sid not in user_map or skill not in skill_to_qid:
                    continue

                time_taken = None
                if "ms_first_response" in df.columns:
                    val = row["ms_first_response"]
                    if not pd.isna(val):
                        time_taken = int(val / 1000)

                self.ad.save_attempt(
                    user_id=user_map[sid],
                    question_id=skill_to_qid[skill],
                    is_correct=bool(row["correct"]),
                    time_taken=time_taken
                )

                saved += 1

                if saved % 10000 == 0:
                    self.db.commit()
                    logger.info(f"Saved {saved} attempts")

            self.db.commit()

            logger.info(f"ASSISTments ingestion complete: {saved} attempts")
            return saved

        except Exception as e:
            self.db.rollback()
            raise CustomException(e, sys)

    