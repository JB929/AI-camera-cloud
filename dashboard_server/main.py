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
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head><title>AI Camera Cloud</title></head>
        <body style='font-family: sans-serif; text-align:center; margin-top:50px;'>
            <h1>🚀 AI Camera Cloud Backend is Running!</h1>
            <p>Use <a href="/dashboard">/dashboard</a> to view alerts</p>
            <p>API endpoint: <code>/api/alerts</code></p>
        </body>
    </html>
    """

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
from pydantic import BaseModel

class AlertRequest(BaseModel):
    camera_name: str
    timestamp: str

from pydantic import BaseModel
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from dashboard_server.database import SessionLocal
from dashboard_server.models import Alert, Base
from datetime import datetime

# ✅ Create DB tables if not existing
Base.metadata.create_all(bind=SessionLocal().bind)

app = FastAPI()

# ✅ Define a Pydantic model for JSON body
class AlertRequest(BaseModel):
    camera_name: str
    timestamp: str

# ✅ Define a dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ Corrected POST endpoint
@app.post("/api/alerts")
def receive_alert(request: AlertRequest, db: Session = Depends(get_db)):
    new_alert = Alert(camera_name=request.camera_name, timestamp=request.timestamp)
    db.add(new_alert)
    db.commit()
    print(f"✅ Cloud Alert Saved: {request.camera_name} at {request.timestamp}")
    return {"message": "Alert received successfully!"}

# 📡 API to Fetch Alerts (for dashboard)
@app.get("/api/alerts")
async def get_alerts():
    db = SessionLocal()
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    return [{"camera_name": a.camera_name, "timestamp": a.timestamp.strftime("%Y-%m-%d %H:%M:%S")} for a in alerts]

