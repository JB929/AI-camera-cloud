from fastapi import FastAPI, Form, File, UploadFile, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime
import os
from dashboard_server.models import Base, Alert
from dashboard_server.auth import router as auth_router, get_current_user
from dashboard_server.database import SessionLocal, engine, Base
from fastapi.responses import StreamingResponse
import cv2
Base.metadata.create_all(bind=engine)

# global dictionary to store frames per camera
camera_frames = {}

# ✅ Initialize database correctly
Base.metadata.create_all(bind=engine)

# ✅ FastAPI app
app = FastAPI(title="AI Camera Cloud", version="2.5")
print("✅ Loaded main.py from:", __file__)

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


# ✅ Home route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


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
@app.post("/api/alerts")
async def create_alert(
    camera_name: str = Form(...),
    timestamp: str = Form(...),
    message: str = Form(None),
    snapshot: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    """
    Receives alerts from local camera detector and stores them in the database
    (with optional snapshot upload).
    """
    from datetime import datetime
    import traceback, os

    try:
        # ✅ Ensure valid datetime
        try:
            ts = datetime.fromisoformat(timestamp)
        except Exception:
            ts = datetime.utcnow()

        # ✅ Prepare snapshot saving
        snapshot_url = None
        if snapshot:
            folder = "dashboard_server/static/snapshots"
            os.makedirs(folder, exist_ok=True)
            filename = f"{camera_name}_{int(datetime.utcnow().timestamp())}.jpg"
            filepath = os.path.join(folder, filename)

            # Write image to disk
            with open(filepath, "wb") as f:
                f.write(await snapshot.read())

            # Store relative web path
            snapshot_url = f"/static/snapshots/{filename}"

        # ✅ Create and save alert in DB
        alert = Alert(
            camera_name=camera_name,
            timestamp=ts,
            message=message or f"Alert from {camera_name} at {timestamp}",
            snapshot_url=snapshot_url  # <-- Correct field
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)

        # ✅ Return confirmation
        return {
            "status": "ok",
            "id": alert.id,
            "snapshot_url": snapshot_url
        }

    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "detail": str(e)}


        # ✅ Broadcast to connected dashboards
        broadcast_alert({
            "camera_name": alert.camera_name,
            "timestamp": str(alert.timestamp),
            "message": alert.message,
            "snapshot_url": alert.snapshot_url
        })

        return {"status": "ok", "id": alert.id, "snapshot_url": snapshot_url}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "detail": str(e)}


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

