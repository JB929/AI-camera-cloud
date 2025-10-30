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
@app.get("/api/alerts", response_class=HTMLResponse)
async def get_alerts(request: Request, db: Session = Depends(get_db)):
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    return templates.TemplateResponse("alerts.html", {"request": request, "alerts": alerts})


# ✅ Endpoint to receive alerts + image upload from cameras
# ✅ Receive alerts from camera (with optional snapshot)
from fastapi import FastAPI, Form, File, UploadFile, Depends, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
import os
from dashboard_server.database import SessionLocal
from dashboard_server.models import Alert

UPLOAD_DIR = "dashboard_server/static/snapshots"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/alerts")
async def create_alert(
    camera_name: str = Form(...),
    timestamp: str = Form(...),
    message: str = Form(None),
    snapshot: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    """
    Accepts alerts from local camera detector and saves to database.
    """
    try:
        # 🕒 Ensure timestamp stored as a datetime object
        from datetime import datetime
        ts = datetime.fromisoformat(timestamp) if " " in timestamp else datetime.utcnow()

        # 💾 Save snapshot file (if provided)
        snapshot_url = None
        if snapshot:
            folder = "dashboard_server/static/snapshots"
            os.makedirs(folder, exist_ok=True)
            filename = f"{camera_name}_{int(datetime.utcnow().timestamp())}.jpg"
            filepath = os.path.join(folder, filename)
            with open(filepath, "wb") as f:
                f.write(await snapshot.read())
            snapshot_url = f"/static/snapshots/{filename}"

        # 🧠 Create alert entry
        alert = Alert(
            camera_name=camera_name,
            timestamp=ts,
            message=message,
            snapshot_path=snapshot_url
)
        db.add(alert)
        db.commit()
        db.refresh(alert)

        return {
            "status": "ok",
            "id": alert.id,
            "snapshot_url": snapshot_url
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "detail": str(e)}


    try:
        snapshot_url = None
        # Save snapshot if provided
        if snapshot:
            os.makedirs("dashboard_server/static/snapshots", exist_ok=True)
            file_name = f"{camera_name}_{int(datetime.now().timestamp())}.jpg"
            file_path = os.path.join(UPLOAD_DIR, file_name)
            with open(file_path, "wb") as f:
                f.write(await snapshot.read())
            snapshot_url = f"/static/snapshots/{file_name}"

        
        ts = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

        alert = Alert(
            camera_name=camera_name,
            timestamp=ts,
            message=message or f"Alert from {camera_name} at {timestamp}"
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)

        return {"status": "ok", "id": alert.id, "snapshot_url": snapshot_url}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ✅ Health check
@app.get("/health")
def health():
    return {"status": "Server running", "version": "2.5"}

