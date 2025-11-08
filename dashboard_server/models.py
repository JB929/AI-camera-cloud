from sqlalchemy import Column, Integer, String, DateTime
from dashboard_server.database import Base
from datetime import datetime

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    camera_name = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    snapshot_url = Column(String) 
    detected_objects = Column(String)

   

