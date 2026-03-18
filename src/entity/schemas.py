from pydantic import BaseModel, EmailStr
from uuid import UUID
from datetime import datetime
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
# AUTH schemas
# ══════════════════════════════════════════════════════════════════════

class UserCreate(BaseModel):
    """
    Used for:
    POST /auth/register
    POST /auth/login
    """
    username: str
    email:    EmailStr
    password: str


class UserOut(BaseModel):
    """
    Returned after register — no password exposed
    """
    id:         UUID
    username:   str
    email:      str
    created_at: datetime

    class Config:
        from_attributes = True   # allows SQLAlchemy model → Pydantic


class LoginResponse(BaseModel):
    """
    Returned after successful login
    """
    access_token: str
    token_type:   str
    user_id:      str
    username:     str


# ══════════════════════════════════════════════════════════════════════
# QUESTION schemas
# ══════════════════════════════════════════════════════════════════════

class QuestionOut(BaseModel):
    """
    Returned when serving a question to user
    """
    question_id: str
    content:     str
    topic:       Optional[str]
    difficulty:  Optional[float]
    user_skill:  Optional[float]   # current user skill level
    source:      Optional[str]     # rl_agent or random_fallback

    class Config:
        from_attributes = True


class QuestionCreate(BaseModel):
    """
    Used for POST /questions — adding new questions
    """
    content:    str
    topic:      str
    difficulty: float
    source:     Optional[str] = "manual"


# ══════════════════════════════════════════════════════════════════════
# ATTEMPT schemas
# ══════════════════════════════════════════════════════════════════════

class AttemptCreate(BaseModel):
    """
    Used for POST /attempts — logging user answer
    """
    user_id:     UUID
    question_id: UUID
    is_correct:  bool
    time_taken:  Optional[int] = None   # seconds


class AttemptOut(BaseModel):
    """
    Returned after logging attempt
    """
    id:           UUID
    is_correct:   bool
    attempted_at: datetime

    class Config:
        from_attributes = True


# ══════════════════════════════════════════════════════════════════════
# USER schemas
# ══════════════════════════════════════════════════════════════════════

class UserStatsOut(BaseModel):
    """
    Returned by GET /users/{user_id}/stats
    """
    user_id:          str
    username:         str
    total_attempts:   int
    correct_attempts: int
    accuracy:         float
    topic_skills:     dict   # {"arrays": 0.8, "trees": 0.6, ...}


class KnowledgeStateOut(BaseModel):
    """
    Returned by GET /users/{user_id}/knowledge-state
    """
    topic:       str
    skill_score: float
    updated_at:  datetime

    class Config:
        from_attributes = True


class KnowledgeStateResponse(BaseModel):
    """
    Full knowledge state response with all topics
    """
    user_id: str
    topics:  list[KnowledgeStateOut]


# ══════════════════════════════════════════════════════════════════════
# TRAINING schemas
# ══════════════════════════════════════════════════════════════════════

class RetrainResponse(BaseModel):
    """
    Returned by POST /train/retrain
    """
    status:  str
    message: str