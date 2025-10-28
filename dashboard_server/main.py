from fastapi import FastAPI, Request, Depends, Form, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime
import os

# ✅ Import project modules
from dashboard_server.auth import router as auth_router
from dashboard_server.database import SessionLocal, engine
from dashboard_server.models import Base, Alert

# ✅ Initialize database
Base.metadata.create_all(bind=engine)

# ✅ Create FastAPI app
app = FastAPI(title="AI Camera Cloud", version="1.0")

# ✅ Allow requests from detector and frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Include auth routes
app.include_router(auth_router)

# ✅ Setup templates and static folders
templates = Jinja2Templates(directory="dashboard_server/templates")
app.mount("/static", StaticFiles(directory="dashboard_server/static"), name="static")


# ============================================================
# ✅ API ROUTES
# ============================================================

@app.post("/api/alerts")
async def receive_alert(
    camera_name: str = Form(...),
    timestamp: str = Form(...),
    snapshot: UploadFile = File(None),
    db: Session = Depends(SessionLocal)
):
    """
    Receive alerts from the AI detector (camera_name, timestamp, optional snapshot)
    and store them in the database + static folder.
    """
    try:
        # Convert timestamp string (HH:MM:SS) → datetime (today's date)
        now = datetime.now()
        timestamp_dt = datetime.combine(now.date(), datetime.strptime(timestamp, "%H:%M:%S").time())

        # Save snapshot image if provided
        snapshot_filename = None
        if snapshot:
            folder = "dashboard_server/static/snapshots"
            os.makedirs(folder, exist_ok=True)
            snapshot_filename = f"{camera_name}_{int(datetime.now().timestamp())}.jpg"
            snapshot_path = os.path.join(folder, snapshot_filename)

            with open(snapshot_path, "wb") as f:
                f.write(await snapshot.read())

        # Create alert record
        new_alert = Alert(
            camera_name=camera_name,
            timestamp=timestamp_dt,
            message=f"Person detected at {timestamp}",
        )
        db.add(new_alert)
        db.commit()
        db.refresh(new_alert)

        print(f"[SERVER] ✅ Alert saved from {camera_name} at {timestamp}")
        return {
            "status": "success",
            "id": new_alert.id,
            "camera_name": camera_name,
            "timestamp": timestamp,
            "snapshot": snapshot_filename
        }

    except Exception as e:
        print(f"[SERVER ERROR] ❌ {str(e)}")
        return {"status": "error", "message": str(e)}


@app.get("/api/alerts")
def get_alerts(db: Session = Depends(SessionLocal)):
    """
    Fetch all saved alerts as JSON.
    """
    alerts = db.query(Alert).order_by(Alert.id.desc()).all()
    return alerts


# ============================================================
# ✅ DASHBOARD ROUTE
# ============================================================

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(SessionLocal)):
    """
    Display recent alerts on a web dashboard.
    """
    alerts = db.query(Alert).order_by(Alert.id.desc()).limit(100).all()
    return templates.TemplateResponse("alerts.html", {"request": request, "alerts": alerts})


# ============================================================
# ✅ ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    """
    Simple API root endpoint.
    """
    return {"message": "✅ AI Camera Cloud Server is Live", "status": "online"}


# ============================================================
# ✅ SERVER LOGGING HELPERS
# ============================================================

@app.on_event("startup")
def on_startup():
    print("🚀 AI Camera Cloud server started successfully on Render.")


@app.on_event("shutdown")
def on_shutdown():
    print("🛑 Server shutting down.")

