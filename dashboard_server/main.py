from fastapi import FastAPI, Form, File, UploadFile, Depends, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime
import os
from dashboard_server.database import SessionLocal, engine
from dashboard_server.models import Base, Alert
from dashboard_server.auth import router as auth_router, get_current_user

# ✅ Initialize database
Base.metadata.create_all(bind=engine)

# ✅ FastAPI app initialization
app = FastAPI(title="AI Camera Cloud", version="2.1")

# ✅ Enable CORS for API and dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Static and templates setup
templates = Jinja2Templates(directory="dashboard_server/templates")
app.mount("/static", StaticFiles(directory="dashboard_server/static"), name="static")

# ✅ Include auth router
app.include_router(auth_router)

# ✅ Database dependency
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


# ✅ Show all alerts
@app.get("/api/alerts", response_class=HTMLResponse)
async def get_alerts(request: Request, db: Session = Depends(get_db)):
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    return templates.TemplateResponse("alerts.html", {"request": request, "alerts": alerts})


# ✅ Receive alerts from camera (cloud endpoint)
@app.post("/api/alerts")
async def receive_alert(request: Request, db: Session = Depends(get_db)):
    """
    Accepts both JSON or FormData alert payloads from camera app.
    """
    try:
        # Try JSON first
        data = await request.json()
        camera_name = data.get("camera_name")
        timestamp_str = data.get("timestamp")

    except Exception:
        # Fallback to form data
        form = await request.form()
        camera_name = form.get("camera_name")
        timestamp_str = form.get("timestamp")

    # ✅ Parse timestamp into datetime
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        timestamp = datetime.utcnow()

    # ✅ Store alert in database
    new_alert = Alert(
        message=f"Person detected on {camera_name}",
        timestamp=timestamp,
        camera_name=camera_name,
    )
    db.add(new_alert)
    db.commit()

    return JSONResponse({"status": "success", "camera": camera_name})


# ✅ Health check route
@app.get("/health")
def health():
    return {"status": "Server running", "version": "2.1"}

