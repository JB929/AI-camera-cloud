# models.py
from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    hashed_password: str


class Alert(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    camera: str
    message: str
    timestamp: str  # store as text for readability; can use datetime if preferred
    snapshot_path: Optional[str] = None


class CameraStatus(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    camera: str = Field(index=True, unique=True)
    status: str
    last_seen: Optional[str] = None
    last_snapshot: Optional[str] = None

