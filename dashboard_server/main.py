from fastapi import FastAPI, Form,File,UploadFile Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime
from dashboard_server.database import SessionLocal, engine
from dashboard_server.models import Base, Alert
from dashboard_server.auth import router as auth_router

# ✅ Initialize database
Base.metadata.create_all(bind=engine)

# ✅ FastAPI app initialization
app = FastAPI(title="AI Camera Cloud", version="2.2")

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Templates & Static
templates = Jinja2Templates(directory="dashboard_server/templates")
app.mount("/static", StaticFiles(directory="dashboard_server/static"), name="static")

# ✅ Include Auth Router
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


# ✅ Alerts list page
@app.get("/api/alerts", response_class=HTMLResponse)
async def get_alerts(request: Request, db: Session = Depends(get_db)):
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    return templates.TemplateResponse("alerts.html", {"request": request, "alerts": alerts})

# ✅ Receive alerts from camera (cloud endpoint)
from fastapi import Form

@app.post("/api/alerts")
async def create_alert(
    camera_name: str = Form(...),
    timestamp: str = Form(...),
    message: str = Form(None),
    snapshot: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    """
    Accepts alerts (and optional image snapshot) from local camera detector.
    Stores them in the database and saves the snapshot to /static/snapshots.
    """
    try:
        from datetime import datetime
        ts = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

        # ✅ Handle snapshot file upload
        snapshot_filename = None
        if snapshot:
            snapshots_dir = "dashboard_server/static/snapshots"
            os.makedirs(snapshots_dir, exist_ok=True)
            snapshot_filename = f"{camera_name}_{int(datetime.now().timestamp())}.jpg"
            file_path = os.path.join(snapshots_dir, snapshot_filename)
            with open(file_path, "wb") as f:
                f.write(await snapshot.read())

        # ✅ Save alert entry to DB
        alert = Alert(
            camera_name=camera_name,
            timestamp=ts,
            message=message or f"Alert from {camera_name} at {timestamp}",
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)

        # ✅ Return snapshot path (if any)
        return {
            "status": "ok",
            "id": alert.id,
            "snapshot_url": f"/static/snapshots/{snapshot_filename}" if snapshot_filename else None
        }

    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ✅ Health check
@app.get("/health")
def health():
    return {"status": "running", "version": "2.2"}

