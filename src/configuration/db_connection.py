import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, URL
from sqlalchemy.orm import sessionmaker

load_dotenv()

connection_url = URL.create(
    drivername = "postgresql+psycopg2",
    username   = os.getenv("DB_USER",     "postgres"),
    password   = os.getenv("DB_PASSWORD", "yourpassword"),
    host       = os.getenv("DB_HOST",     "localhost"),
    port       = int(os.getenv("DB_PORT", 5432)),
    database   = os.getenv("DB_NAME",     "learning_platform"),
)

engine = create_engine(connection_url, pool_pre_ping=True)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def check_connection():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("DB connection successful!")
    except Exception as e:
        print(f"DB connection failed: {e}")