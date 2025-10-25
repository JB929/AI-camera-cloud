from fastapi import FastAPI, Request, Form, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime
import os

# ✅ Import local modules using absolute paths
from dashboard_server.auth import router as auth_router
from dashboard_server.database import SessionLocal, engine
from dashboard_server.models import Base, Alert

# Initialize DB
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Camera Cloud Dashboard")

# Allow camera client to send alerts
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for UI
app.mount("/static", StaticFiles(directory="dashboard_server/static"), name="static")

templates = Jinja2Templates(directory="dashboard_server/templates")

# --- AUTH ROUTES ---
# (Protected login/dashboard features)
app.include_router(auth_router, prefix="/auth")

# --- PUBLIC ALERT ENDPOINT ---
@app.post("/api/alerts")
async def receive_alert(request: Request):
    """
    ✅ Public endpoint for AI cameras to send alerts.
    Does NOT require authentication.
    """
    try:
        data = await request.json()
    except Exception:
        return {"error": "Invalid JSON"}

    camera_name = data.get("camera_name", "Unknown_Camera")
    timestamp = data.get("timestamp", str(datetime.now()))

    print(f"🚨 ALERT RECEIVED: {camera_name} at {timestamp}")

    # Save to database
    db = SessionLocal()
    new_alert = Alert(camera_name=camera_name, timestamp=timestamp)
    db.add(new_alert)
    db.commit()
    db.close()

    # Optional log
    with open("alerts.log", "a") as f:
        f.write(f"{timestamp} | {camera_name}\n")

    return {"status": "✅ Alert received successfully"}

# --- HOMEPAGE ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

