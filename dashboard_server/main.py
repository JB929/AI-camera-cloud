from fastapi import FastAPI, Request, Depends, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime
import os

# ✅ Import project modules
from dashboard_server.database import SessionLocal, engine
from dashboard_server.models import Base, Alert
from dashboard_server.auth import get_current_user

# Initialize database
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Camera Cloud")

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static & Templates
app.mount("/static", StaticFiles(directory="dashboard_server/static"), name="static")
templates = Jinja2Templates(directory="dashboard_server/templates")


# 🏠 Root Route
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 📋 Dashboard Page
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    db = SessionLocal()
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    return templates.TemplateResponse("dashboard.html", {"request": request, "alerts": alerts})


# ⚙️ API to Receive Alerts from Detector
@app.post("/api/alerts")
async def receive_alert(camera_name: str = Form(...), timestamp: str = Form(...)):
    db = SessionLocal()
    alert = Alert(camera_name=camera_name, timestamp=timestamp)
    db.add(alert)
    db.commit()
    return {"status": "success", "message": "Alert stored successfully."}


# 📡 API to Fetch Alerts (for dashboard)
@app.get("/api/alerts")
async def get_alerts():
    db = SessionLocal()
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    return [{"camera_name": a.camera_name, "timestamp": a.timestamp.strftime("%Y-%m-%d %H:%M:%S")} for a in alerts]

