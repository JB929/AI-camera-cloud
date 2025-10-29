import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ✅ Always use an absolute path — works on both local and Render
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "dashboard.db")

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ✅ Initialize the database (used in main.py)
def init_db():
    import dashboard_server.models  # make sure models are imported
    Base.metadata.create_all(bind=engine)

# ✅ Database session dependency (for routes)
def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

