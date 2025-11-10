from fastapi import FastAPI, Form, File, UploadFile, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime
from pathlib import Path
import os
import cv2

from dashboard_server.models import Base, Alert
from dashboard_server.auth import router as auth_router, get_current_user
from dashboard_server.database import SessionLocal, engine
import os

# ✅ Initialize FastAPI first
app = FastAPI(title="AI Camera Cloud", version="2.5")
print("✅ Loaded main.py from:", __file__)

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# ✅ Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   
    allow_headers=["*"],   
)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

from sqlalchemy import inspect
from dashboard_server.database import Base, engine
from dashboard_server.models import Alert
import os

print("🧠 Checking and synchronizing database schema...")

db_path = "/opt/render/project/src/tmp/dashboard.db"


Base.metadata.create_all(bind=engine)
print("✅ Database tables recreated successfully.")
# ✅ Ensure the database and tables are created on every startup
from dashboard_server.database import Base, engine
from dashboard_server.models import Alert

from pathlib import Path


print("🧠 Checking and creating database tables if needed...")
Base.metadata.create_all(bind=engine)
print("✅ Database tables verified/created successfully.")

# --- TEMPORARY: Force rebuild database on startup ---
from dashboard_server.models import Alert
from dashboard_server.database import engine, Base
import os

db_path = "dashboard_server/dashboard.db"
if os.path.exists(db_path):
    os.remove(db_path)
    print("🧹 Old database deleted, rebuilding...")

Base.metadata.create_all(bind=engine)
print("✅ New database initialized with snapshot_url column.")
# ------------------------------------------------------

# global dictionary to store frames per camera
camera_frames = {}

# ✅ Initialize database correctly
Base.metadata.create_all(bind=engine)

# ✅ CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Templates and static files
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


# ✅ List all alerts with snapshot preview
@app.get("/api/latest_snapshots")
async def latest_snapshots(db: Session = Depends(get_db)):
    """Returns the latest alert per camera (for dashboard grid)."""
    from sqlalchemy import func
    subq = db.query(
        Alert.camera_name,
        func.max(Alert.timestamp).label("latest_time")
    ).group_by(Alert.camera_name).subquery()

    latest = db.query(Alert).join(
        subq,
        (Alert.camera_name == subq.c.camera_name) &
        (Alert.timestamp == subq.c.latest_time)
    ).all()

    return [
        {
            "camera_name": a.camera_name,
            "timestamp": str(a.timestamp),
            "snapshot_url": a.snapshot_url,
        }
        for a in latest
    ]

@app.get("/api/alerts", response_class=HTMLResponse)
async def get_alerts(request: Request, db: Session = Depends(get_db)):
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    return templates.TemplateResponse("alerts.html", {"request": request, "alerts": alerts})


# ✅ Directory for snapshots
UPLOAD_DIR = "dashboard_server/static/snapshots"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ✅ Endpoint to receive alerts + image upload from cameras
from fastapi import FastAPI, Form, File, UploadFile, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
import os
from dashboard_server.database import SessionLocal, get_db
from dashboard_server.models import Alert


UPLOAD_DIR = "dashboard_server/static/snapshots"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/alerts")
async def create_alert(
    camera_name: str = Form(...),
    timestamp: str = Form(...),
    snapshot: UploadFile = File(...),
    message: str = Form(None),
    detected_objects: str = Form(None),
):
    """Receive alerts from edge AI detector, save to DB, and return success."""

    try:
        # Save uploaded snapshot to static directory
        snapshots_dir = "dashboard_server/static/snapshots"
        os.makedirs(snapshots_dir, exist_ok=True)

        filename = f"{camera_name}_{int(datetime.now().timestamp())}.jpg"
        file_path = os.path.join(snapshots_dir, filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await snapshot.read())

        snapshot_url = f"/static/snapshots/{filename}"

        # ✅ Save alert in database
        db = SessionLocal()
        new_alert = Alert(
            camera_name=camera_name,
            timestamp=timestamp,
            snapshot_url=snapshot_url,
            message=message or "Person detected",
            detected_objects=detected_objects or "person",
        )
        db.add(new_alert)
        db.commit()
        db.refresh(new_alert)
        db.close()

        print(f"✅ Alert saved to DB: id={new_alert.id}")
        return {"status": "ok", "id": new_alert.id}

    except Exception as e:
        print(f"❌ Error while creating alert: {e}")
        return {"error": str(e)}






# ✅ Real-time alert broadcast (WebSocket)
from fastapi import WebSocket, WebSocketDisconnect
from typing import List

active_connections: List[WebSocket] = []


@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep connection alive
    except WebSocketDisconnect:
        active_connections.remove(websocket)


def broadcast_alert(alert_data):
    """Push new alerts to all connected dashboards."""
    for connection in active_connections:
        try:
            connection.send_json(alert_data)
        except Exception:
            pass

# ✅ Health check
@app.get("/health")
def health():
    return {"status": "Server running", "version": "2.5"}

def generate_mjpeg(camera_name: str):
    """Generator that yields camera frames as JPEG byte stream."""
    while True:
        if camera_name in camera_frames:
            frame = camera_frames[camera_name]
            ret, jpeg = cv2.imencode('.jpg', frame)
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.05)

@app.get("/video_feed/{camera_name}")
def video_feed(camera_name: str):
    """HTTP MJPEG feed endpoint."""
    return StreamingResponse(generate_mjpeg(camera_name),
                              media_type='multipart/x-mixed-replace; boundary=frame')

# ----------------------------------------------------------
# 🧾 GET RECENT ALERTS — (For dashboard display)
# ----------------------------------------------------------
@app.get("/api/recent_alerts")
async def recent_alerts():
    try:
        db = SessionLocal()
        alerts = db.query(Alert).order_by(Alert.id.desc()).limit(10).all()
        db.close()

        return [
            {
                "camera_name": a.camera_name,
                "timestamp": a.timestamp,
                "snapshot_url": a.snapshot_url,
                "message": a.message,
                "detected_objects": a.detected_objects,
            }
            for a in alerts
        ]
    except Exception as e:
        print(f"❌ Error fetching recent alerts: {e}")
        return {"error": str(e)}

