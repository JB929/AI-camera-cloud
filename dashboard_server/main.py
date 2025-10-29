from fastapi import FastAPI, Form, File, UploadFile, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime
import os

from dashboard_server.database import SessionLocal, engine
from dashboard_server.models import Base, Alert
from dashboard_server.auth import router as auth_router

# ✅ Initialize database
Base.metadata.create_all(bind=engine)

# ✅ FastAPI app initialization
app = FastAPI(title="AI Camera Cloud", version="2.2")

# ✅ Enable CORS (for both dashboard and camera API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Mount static and templates
templates = Jinja2Templates(directory="dashboard_server/templates")
app.mount("/static", StaticFiles(directory="dashboard_server/static"), name="static")

# ✅ Include authentication routes
app.include_router(auth_router)

# ✅ Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# ✅ Show alerts dashboard
@app.get("/api/alerts", response_class=HTMLResponse)
async def get_alerts(request: Request, db: Session = Depends(get_db)):
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    return templates.TemplateResponse("alerts.html", {"request": request, "alerts": alerts})

# ✅ Receive alerts from camera (Form data)
@app.post("/api/alerts")
def create_alert(
    camera_name: str = Form(...),
    timestamp: str = Form(...),
    message: str = Form(None),
    db: Session = Depends(get_db)
):
    """
    Receives alerts from local AI camera systems (via POST).
    Saves to the database and confirms success.
    """
    try:
        alert = Alert(camera_name=camera_name, timestamp=timestamp, message=message)
        db.add(alert)
        db.commit()
        db.refresh(alert)
        print(f"✅ Alert saved: {camera_name} at {timestamp}")
        return {"status": "ok", "id": alert.id}
    except Exception as e:
        print(f"❌ Error saving alert: {e}")
        return {"status": "error", "detail": str(e)}

