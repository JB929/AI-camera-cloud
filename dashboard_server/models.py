from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from dashboard_server.database import Base


# ✅ User model (needed for auth.py)
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)


# ✅ Alert model (used by the camera)
class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    camera_name = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message = Column(String, nullable=True)
    snapshot_path = Column(String, nullable=True)

