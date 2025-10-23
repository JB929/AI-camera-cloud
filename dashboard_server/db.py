# db.py
from sqlmodel import SQLModel, create_engine, Session
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FILE = os.path.join(BASE_DIR, "app.db")
DATABASE_URL = f"sqlite:///{DB_FILE}"

# echo=True for SQL debug; set to False in production
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


def init_db():
    """Create DB file and tables if not present."""
    SQLModel.metadata.create_all(engine)


def get_session():
    """Return a new Session (use with context manager)."""
    return Session(engine)

