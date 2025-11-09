from sqlalchemy import Column, Integer, String, DateTime
from dashboard_server.database import Base

class Alert(Base):
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    camera_name = Column(String)
    timestamp = Column(String)
    snapshot_url = Column(String) 
    message = Column(String)
    detected_objects = Column(String)

   

