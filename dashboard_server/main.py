from fastapi import FastAPI, Request, Depends, Form, UploadFile, File 
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime
import os

# ✅ Import modules using absolute paths (Render-safe)
from dashboard_server.auth import router as auth_router, get_current_user
from dashboard_server.database import SessionLocal, engine
from dashboard_server.models import Base, Alert

# Initialize database
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Mount static directory (for CSS, JS, audio)
app.mount("/static", StaticFiles(directory="dashboard_server/static"), name="static")

templates = Jinja2Templates(directory="dashboard_server/templates")

# Include authentication router
app.include_router(auth_router)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(SessionLocal), user: str = Depends(get_current_user)):
    """
    Main dashboard page
    """
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "alerts": alerts,
        "user": user
    })


@app.post("/api/alerts")
async def receive_alert(request: Request):
    try:
        data = await request.json()
    except Exception:
        return {"error": "Invalid JSON"}

    camera_name = data.get("camera_name", "Unknown_Camera")
    timestamp = data.get("timestamp", str(datetime.datetime.now()))

    print(f"🚨 ALERT RECEIVED: {camera_name} at {timestamp}")

    # ✅ Log alert to database (optional)
    db = SessionLocal()
    new_alert = Alert(camera_name=camera_name, timestamp=timestamp)
    db.add(new_alert)
    db.commit()
    db.close()

    return {"status": "✅ Alert received successfully"}

@app.get("/health")
async def health_check():
    return {"status": "ok", "time": datetime.now().isoformat()}
from fastapi import BackgroundTasks, Request
import datetime

@app.post("/api/alerts")
async def receive_alert(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
    except Exception:
        return {"error": "Invalid JSON"}

    camera_name = data.get("camera_name", "Unknown_Camera")
    timestamp = data.get("timestamp", str(datetime.datetime.now()))

    print(f"🚨 ALERT RECEIVED: {camera_name} at {timestamp}")

    # Log to file for testing
    with open("alerts.log", "a") as log:
        log.write(f"{timestamp} | {camera_name}\n")

    return {"status": "✅ Alert received successfully"}

from fastapi import BackgroundTasks
import datetime

@app.post("/api/alerts")
async def receive_alert(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
    except Exception:
        return {"error": "Invalid JSON"}

    camera_name = data.get("camera_name", "Unknown_Camera")
    timestamp = data.get("timestamp", str(datetime.datetime.now()))

    print(f"🚨 ALERT RECEIVED: {camera_name} at {timestamp}")

    # ✅ Optional: Log alerts to your database
    db = SessionLocal()
    new_alert = Alert(camera_name=camera_name, timestamp=timestamp)
    db.add(new_alert)
    db.commit()
    db.close()

    # ✅ Optional: Log to file
    with open("alerts.log", "a") as log:
        log.write(f"{timestamp} | {camera_name}\n")

    return {"status": "✅ Alert received successfully"}

