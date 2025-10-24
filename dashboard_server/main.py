from fastapi import FastAPI, Request, Depends, Form, UploadFile, File
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


@app.post("/alerts/")
async def receive_alert(
    message: str = Form(...),
    camera: str = Form(...),
    timestamp: str = Form(...),
    snapshot: UploadFile = File(None),
    db: Session = Depends(SessionLocal)
):
    """
    Endpoint to receive alerts from AI detector clients
    """
    snapshot_path = None
    if snapshot:
        os.makedirs("dashboard_server/static/snapshots", exist_ok=True)
        snapshot_path = f"dashboard_server/static/snapshots/{snapshot.filename}"
        with open(snapshot_path, "wb") as f:
            f.write(await snapshot.read())

    alert = Alert(
        message=message,
        camera=camera,
        timestamp=datetime.now(),
        snapshot_path=snapshot_path
    )
    db.add(alert)
    db.commit()

    return {"status": "Alert received", "camera": camera, "message": message}


@app.get("/health")
async def health_check():
    return {"status": "ok", "time": datetime.now().isoformat()}

