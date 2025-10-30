from fastapi import FastAPI, Form, File, UploadFile, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
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

# ✅ FastAPI app
app = FastAPI(title="AI Camera Cloud", version="2.5")

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
@app.get("/api/alerts", response_class=HTMLResponse)
async def get_alerts(request: Request, db: Session = Depends(get_db)):
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    return templates.TemplateResponse("alerts.html", {"request": request, "alerts": alerts})


# ✅ Endpoint to receive alerts + image upload from cameras
@app.post("/api/alerts")
async def create_alert(
    camera_name: str = Form(...),
    timestamp: str = Form(...),
    message: str = Form(None),
    snapshot: UploadFile = File(None),
    db: Session = Depends(SessionLocal)
):
    try:
        snapshot_path = None

        # ✅ Save uploaded snapshot
        if snapshot:
            os.makedirs("dashboard_server/static/snapshots", exist_ok=True)
            file_ext = os.path.splitext(snapshot.filename)[1]
            filename = f"{camera_name}_{int(datetime.now().timestamp())}{file_ext}"
            snapshot_path = f"dashboard_server/static/snapshots/{filename}"
            with open(snapshot_path, "wb") as f:
                f.write(await snapshot.read())

        # ✅ Convert timestamp correctly
        ts_obj = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

        # ✅ Create new alert entry
        alert = Alert(
            camera_name=camera_name,
            timestamp=ts_obj,
            message=message,
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)

        # ✅ Return public snapshot URL
        snapshot_url = f"/static/snapshots/{os.path.basename(snapshot_path)}" if snapshot_path else None

        return {"status": "ok", "id": alert.id, "snapshot_url": snapshot_url}

    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ✅ Health check
@app.get("/health")
def health():
    return {"status": "Server running", "version": "2.5"}

