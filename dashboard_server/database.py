# dashboard_server/database.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

import os
from pathlib import Path

# ✅ Detect environment automatically
if os.environ.get("RENDER"):
    # On Render: use writable /opt/render/project/src/tmp
    DB_DIR = "/opt/render/project/src/tmp"
else:
    # Local development: use dashboard_server/local_db folder
    BASE_DIR = Path(__file__).resolve().parent
    DB_DIR = BASE_DIR / "local_db"

os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "dashboard.db")

DATABASE_URL = f"sqlite:///{DB_PATH}"

# ✅ Initialize SQLAlchemy engine & session
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ✅ Initialize DB
def init_db():
    from dashboard_server.models import Alert
    print("🧠 Checking and creating database tables if needed...")
    Base.metadata.create_all(bind=engine)
    print(f"✅ Database initialized at: {DB_PATH}")

# ✅ Dependency for FastAPI routes
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

