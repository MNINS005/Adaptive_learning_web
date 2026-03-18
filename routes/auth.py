import os
import sys
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, Depends, HTTPException, status
from passlib.hash import bcrypt
from sqlalchemy.orm import Session

from src.configuration.db_connection import get_db
from src.entity.db_models import User
from src.entity.schemas import UserCreate, UserOut
from src.logger import logger

router     = APIRouter()
SECRET_KEY = os.getenv("SECRET_KEY", "changeme")
ALGORITHM  = "HS256"


def create_token(user_id: str) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


@router.post("/register", response_model=UserOut)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    try:
        if db.query(User).filter(User.email == payload.email).first():
            raise HTTPException(400, "Email already registered")

        user = User(
            username = payload.username,
            email    = payload.email,
            password = bcrypt.hash(payload.password)
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info(f"Registered: {user.username}")
        return user

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(500, str(e))


@router.post("/login")
def login(payload: UserCreate, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.email == payload.email).first()
        if not user or not bcrypt.verify(payload.password, user.password):
            raise HTTPException(401, "Invalid credentials")

        logger.info(f"Login: {user.username}")
        return {
            "access_token" : create_token(user.id),
            "token_type"   : "bearer",
            "user_id"      : str(user.id),
            "username"     : user.username
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))