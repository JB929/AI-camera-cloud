# main.py
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime
import os
import shutil

from auth import router as auth_router, get_current_user
from db import init_db, get_session
from models import Alert, CameraStatus, User
from sqlmodel import select

# init DB (creates tables)
init_db()

app = FastAPI()
app.include_router(auth_router)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
SNAPSHOT_DIR = os.path.join(STATIC_DIR, "snapshots")
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATE_DIR)


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, user=Depends(get_current_user)):
    if not user:
        return RedirectResponse(url="/login")

    # fetch camera statuses from DB
    with get_session() as sess:
        stmt = select(CameraStatus)
        cameras = sess.exec(stmt).all()

        # convert to dictionary keyed by camera name for template compatibility
        cam_dict = {}
        for c in cameras:
            cam_dict[c.camera] = {
                "status": c.status,
                "last_seen": c.last_seen,
                "last_snapshot": c.last_snapshot
            }

        # last 20 alerts
        stmt2 = select(Alert).order_by(Alert.id.desc()).limit(20)
        alerts = sess.exec(stmt2).all()
        alerts = [dict(id=a.id, camera=a.camera, message=a.message, timestamp=a.timestamp, snapshot_path=a.snapshot_path) for a in alerts][::-1]

    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "alerts": alerts,
        "camera_status": cam_dict,
        "user": user
    })


@app.post("/api/alert")
async def receive_alert(
    message: str = Form(...),
    camera: str = Form(...),
    timestamp: str = Form(...),
    snapshot: UploadFile = File(None)
):
    snapshot_path = None
    if snapshot is not None:
        filename = f"{camera}_{int(datetime.now().timestamp())}.jpg"
        save_path = os.path.join(SNAPSHOT_DIR, filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(snapshot.file, f)
        snapshot_path = f"/static/snapshots/{filename}"

    # persist Alert and update CameraStatus
    with get_session() as sess:
        a = Alert(camera=camera, message=message, timestamp=timestamp, snapshot_path=snapshot_path)
        sess.add(a)

        # upsert camera status
        stmt = select(CameraStatus).where(CameraStatus.camera == camera)
        cs = sess.exec(stmt).first()
        if cs:
            cs.status = "online"
            cs.last_seen = timestamp
            cs.last_snapshot = snapshot_path
            sess.add(cs)
        else:
            cs = CameraStatus(camera=camera, status="online", last_seen=timestamp, last_snapshot=snapshot_path)
            sess.add(cs)

        sess.commit()

    return JSONResponse({"status": "ok"})


@app.get("/api/alerts")
async def get_alerts():
    with get_session() as sess:
        stmt = select(Alert).order_by(Alert.id.desc()).limit(100)
        rows = sess.exec(stmt).all()
        return [dict(id=r.id, camera=r.camera, message=r.message, timestamp=r.timestamp, snapshot_path=r.snapshot_path) for r in rows[::-1]]


@app.get("/api/status")
async def get_status():
    with get_session() as sess:
        stmt = select(CameraStatus)
        rows = sess.exec(stmt).all()
        return {r.camera: {"status": r.status, "last_seen": r.last_seen, "last_snapshot": r.last_snapshot} for r in rows}

