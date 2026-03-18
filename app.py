import sys
import os
sys.path.append(os.path.abspath("."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import  questions, attempts, users

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

app.include_router(attempts.router,      prefix="/auth",      tags=["Auth"])
app.include_router(questions.router, prefix="/questions", tags=["Questions"])
app.include_router(attempts.router,  prefix="/attempts",  tags=["Attempts"])
#app.include_router(users.router,     prefix="/users",     tags=["Users"])

@app.get("/")
def root():
    return {"message": "Adaptive Learning Platform is running"}

@app.get("/health")
def health():
    return {"status": "ok"}