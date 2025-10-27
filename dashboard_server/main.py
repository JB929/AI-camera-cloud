from fastapi import FastAPI, Request, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from datetime import datetime
import os

# Import your app modules
from dashboard_server.database import SessionLocal, engine
from dashboard_server.models import Base, Alert

# Create database tables if not already present
Base.metadata.create_all(bind=engine)

# Initialize the FastAPI app
app = FastAPI(title="AI Camera Cloud Server")

# Enable CORS (important for local -> cloud communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict to your local IP if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Jinja2 templates (for web dashboard)
templates = Jinja2Templates(directory="dashboard_server/templates")

# Dependency: get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ✅ Root route — for Render health checks
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head><title>AI Camera Cloud</title></head>
        <body style='font-family: Arial, sans-serif; text-align:center; margin-top:50px;'>
            <h1>🚀 AI Camera Cloud Backend is Running!</h1>
            <p><a href="/dashboard">📊 View Dashboard</a></p>
            <p><a href="/api/alerts">🔗 API Endpoint</a></p>
        </body>
    </html>
    """


# ✅ API endpoint — receive alerts from your detector
@app.post("/api/alerts")
async def receive_alert(request: Request, db: Session = Depends(get_db)):
    """
    Receives alerts from AI detector via POST request.
    Expected JSON body:
    {
        "camera_name": "Front_Yard",
        "timestamp": "13:25:44"
    }
    """
    try:
        data = await request.json()
        camera_name = data.get("camera_name")
        timestamp_str = data.get("timestamp")

        # Validate camera name
        if not camera_name:
            return {"error": "Missing camera_name"}

        # Convert timestamp string to datetime
        try:
            # Parses format like "13:25:44"
            timestamp = datetime.strptime(timestamp_str, "%H:%M:%S")
            # Set today’s date for full datetime
            timestamp = timestamp.replace(
                year=datetime.now().year,
                month=datetime.now().month,
                day=datetime.now().day,
            )
        except Exception:
            timestamp = datetime.utcnow()

        # Create a new alert record
        new_alert = Alert(
            camera_name=camera_name,
            message="Person detected",
            timestamp=timestamp
        )
        db.add(new_alert)
        db.commit()
        db.refresh(new_alert)

        print(f"[SERVER] ✅ Alert received: {camera_name} @ {timestamp}")

        return {"status": "success", "camera_name": camera_name, "timestamp": timestamp.isoformat()}

    except Exception as e:
        print(f"[SERVER ERROR] {e}")
        return {"error": str(e)}


# ✅ Dashboard — shows all saved alerts
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    return templates.TemplateResponse(
        "alerts.html",
        {"request": request, "alerts": alerts}
    )


# ✅ View all alerts as JSON (for debugging)
@app.get("/api/alerts")
def get_alerts(db: Session = Depends(get_db)):
    alerts = db.query(Alert).order_by(Alert.timestamp.desc()).all()
    return [
        {
            "id": a.id,
            "camera_name": a.camera_name,
            "message": a.message,
            "timestamp": a.timestamp.isoformat()
        }
        for a in alerts
    ]

