from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

# ✅ User table
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True)
    hashed_password = Column(String(255))


# ✅ Camera table
class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(50))
    location = Column(String(100))
    user_id = Column(Integer, ForeignKey("users.id"))

    user = relationship("User", backref="cameras")


# ✅ Alert table
class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    camera_id = Column(Integer, ForeignKey("cameras.id"))
    snapshot_path = Column(String(255))

    camera = relationship("Camera", backref="alerts")

