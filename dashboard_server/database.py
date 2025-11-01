# dashboard_server/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ✅ Ensure /data directory exists (Render's writable directory)
os.makedirs("/data", exist_ok=True)

# ✅ Use a database path in /data (writable on Render)
DB_PATH = os.path.join("/data", "dashboard.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# ✅ Create SQLAlchemy engine and session
engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ✅ Initialize DB (used at startup)
def init_db():
    from dashboard_server.models import Alert
    print("🧠 Checking and creating database tables if needed...")
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables verified/created successfully.")


# ✅ Dependency for FastAPI routes
def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

