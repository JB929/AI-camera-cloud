from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from dashboard_server.database import Base
from datetime import datetime


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    alerts = relationship("Alert", back_populates="user")


from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from dashboard_server.database import Base

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    camera_name = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    message = Column(String, nullable=True)
    snapshot_path = Column(String, nullable=True)


