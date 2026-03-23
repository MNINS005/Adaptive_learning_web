import sys
import os
sys.path.append(os.path.abspath("."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import auth, questions, attempts, users

app = FastAPI(
    title       = "Adaptive Learning Platform",
    description = "DKT + RL based personalized DSA learning",
    version     = "1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

app.include_router(auth.router,      prefix="/auth",      tags=["Auth"])
app.include_router(questions.router, prefix="/questions", tags=["Questions"])
app.include_router(attempts.router,  prefix="/attempts",  tags=["Attempts"])
app.include_router(users.router,     prefix="/users",     tags=["Users"])

@app.get("/")
def root():
    return {
        "message"  : "Adaptive Learning Platform API is running",
        "version"  : "1.0.0",
        "docs"     : "http://localhost:8000/docs"
    }

@app.get("/health")
def health():
    from src.configuration.db_connection import engine
    from sqlalchemy import text
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"

    return {
        "status"    : "ok",
        "database"  : db_status,
        "models"    : "loaded" if _models_loaded() else "not loaded"
    }

def _models_loaded() -> bool:
    try:
        from routes.questions import dkt_model, rl_agent
        return dkt_model is not None and rl_agent is not None
    except Exception:
        return False