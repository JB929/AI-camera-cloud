import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# ✅ Use Render tmp directory in cloud, local_db in development
if os.getenv("RENDER"):
    DB_DIR = "/opt/render/project/src/tmp"
else:
    DB_DIR = "dashboard_server/local_db"

os.makedirs(DB_DIR, exist_ok=True)

DB_PATH = os.path.join(DB_DIR, "dashboard.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ✅ Dependency for FastAPI routes that need DB access
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ✅ Optional: one-time database initialization helper
def init_db():
    import dashboard_server.models  # ensure models are imported
    Base.metadata.create_all(bind=engine)

