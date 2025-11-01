# dashboard_server/models.py
from sqlalchemy import Column, Integer, String, DateTime
from dashboard_server.database import Base
from datetime import datetime

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    camera_name = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message = Column(String, nullable=True)
    snapshot_url = Column(String, nullable=True)  # ✅ Added this line

    def __repr__(self):
        return f"<Alert(camera='{self.camera_name}', time='{self.timestamp}')>"

