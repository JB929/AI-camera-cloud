# dashboard_server/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Using a simple SQLite DB — Render will create this file automatically
DATABASE_URL = "sqlite:///./dashboard.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ✅ Initialize the DB — used in main.py
def init_db():
    import dashboard_server.models  # make sure models are imported
    Base.metadata.create_all(bind=engine)


# ✅ This is what auth.py was missing
def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

