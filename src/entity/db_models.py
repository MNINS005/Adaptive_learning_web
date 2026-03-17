import uuid
import datetime

from sqlalchemy import (
    Column,
    String,
    Float,
    Boolean,
    Integer,
    DateTime,
    ForeignKey,
    UniqueConstraint
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username   = Column(String(100), unique=True, nullable=False)
    email      = Column(String(255), unique=True, nullable=False)
    password   = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    attempts  = relationship("Attempt",        back_populates="user")
    knowledge = relationship("KnowledgeState", back_populates="user")

    def __repr__(self):
        return f"<User(username={self.username}, email={self.email})>"


class Question(Base):
    __tablename__ = "questions"

    id         = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content    = Column(String,      nullable=False)
    topic      = Column(String(100))
    difficulty = Column(Float)
    source     = Column(String(100))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    attempts = relationship("Attempt", back_populates="question")

    def __repr__(self):
        return f"<Question(topic={self.topic}, difficulty={self.difficulty})>"


class Attempt(Base):
    __tablename__ = "attempts"

    id           = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id      = Column(UUID(as_uuid=True), ForeignKey("users.id"),     nullable=False)
    question_id  = Column(UUID(as_uuid=True), ForeignKey("questions.id"), nullable=False)
    is_correct   = Column(Boolean,  nullable=False)
    time_taken   = Column(Integer)
    attempted_at = Column(DateTime, default=datetime.datetime.utcnow)

    user     = relationship("User",     back_populates="attempts")
    question = relationship("Question", back_populates="attempts")

    def __repr__(self):
        return f"<Attempt(user_id={self.user_id}, is_correct={self.is_correct})>"


class KnowledgeState(Base):
    __tablename__ = "knowledge_states"

    id          = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id     = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    topic       = Column(String(100), nullable=False)
    skill_score = Column(Float)
    updated_at  = Column(DateTime, default=datetime.datetime.utcnow)

    user = relationship("User", back_populates="knowledge")

    __table_args__ = (
        UniqueConstraint("user_id", "topic", name="uq_user_topic"),
    )

    def __repr__(self):
        return f"<KnowledgeState(user_id={self.user_id}, topic={self.topic}, skill={self.skill_score})>"