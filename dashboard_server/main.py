from fastapi import FastAPI, Form, Depends, Request
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
    db: Session = Depends(get_db)
):
    """
    Accepts alerts from local camera detector and stores them in the SQLite database.
    """
    try:
        # Parse timestamp string into Python datetime
        from datetime import datetime
        ts = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")

        alert = Alert(
            camera_name=camera_name,
            timestamp=ts,
            message=message or f"Alert from {camera_name} at {timestamp}"
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)
        return {"status": "ok", "id": alert.id}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ✅ Health check
@app.get("/health")
def health():
    return {"status": "running", "version": "2.2"}

