from sqlalchemy import Column, Integer, String, DateTime
from dashboard_server.database import Base
from datetime import datetime

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    camera_name = Column(String, nullable=False)
    message = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    rtsp_url = Column(String, nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    owner = relationship("User")


class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    message = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    camera_name = Column(String, nullable=False)

