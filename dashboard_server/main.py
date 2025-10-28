from fastapi import FastAPI, Form, File, UploadFile, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from dashboard_server.database import SessionLocal, engine
from dashboard_server.models import Base, Alert
from dashboard_server.auth import router as auth_router, get_current_user
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os

# ✅ Initialize database
Base.metadata.create_all(bind=engine)

# ✅ FastAPI app initialization
app = FastAPI(title="AI Camera Cloud", version="2.0")

# ✅ Enable CORS (for camera → cloud API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Mount static files & templates (dashboard)
app.mount("/static", StaticFiles(directory="dashboard_server/static"), name="static")
templates = Jinja2Templates(directory="dashboard_server/templates")

# ✅ Include authentication router
app.include_router(auth_router)


@app.get("/", response_class=HTMLResponse)
def home(request):
    return templates.TemplateResponse("index.html", {"request": request})


# ✅ API Endpoint — Receive Alerts
@app.post("/api/alerts")
async def receive_alert(
    camera_name: str = Form(...),
    timestamp: str = Form(...),
    snapshot: UploadFile = File(None),
    db: Session = Depends(SessionLocal)
):
    """
    Receives detection alerts from edge devices (multi_camera_detector).
    Stores them safely in SQLite + saves snapshot if available.
    """
    try:
        # 🕒 Parse timestamp safely
        try:
            ts_obj = datetime.fromisoformat(timestamp)
        except Exception:
            ts_obj = datetime.utcnow()

        # 💾 Save snapshot (if present)
        snapshot_path = None
        if snapshot:
            folder = "dashboard_server/static/snapshots"
            os.makedirs(folder, exist_ok=True)
            snapshot_path = os.path.join(folder, snapshot.filename)
            with open(snapshot_path, "wb") as f:
                f.write(await snapshot.read())

        # 🧠 Store in database
        new_alert = Alert(
            camera_name=camera_name,
            timestamp=ts_obj,
            message=f"Person detected at {ts_obj}"
        )
        db.add(new_alert)
        db.commit()

        print(f"[CLOUD] ✅ Alert stored for {camera_name} at {ts_obj}")
        return JSONResponse(content={"status": "success", "camera": camera_name})

    except Exception as e:
        db.rollback()
        print(f"[CLOUD ERROR] {e}")
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=500)


# ✅ Route to View Alerts Dashboard
@app.get("/alerts", response_class=HTMLResponse)
def show_alerts(request, db: Session = Depends(SessionLocal)):
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    return templates.TemplateResponse("alerts.html", {"request": request, "alerts": alerts})


@app.get("/api/alerts", response_class=JSONResponse)
def get_alerts(db: Session = Depends(SessionLocal)):
    """
    JSON API to get all alerts (for frontend or app integration).
    """
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    data = [
        {"camera_name": a.camera_name, "timestamp": a.timestamp.isoformat(), "message": a.message}
        for a in alerts
    ]
    return JSONResponse(content=data)

