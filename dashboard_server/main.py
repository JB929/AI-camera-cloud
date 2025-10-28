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

from fastapi import FastAPI, Form, File, UploadFile, Depends, Request, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime
import os

from dashboard_server.database import SessionLocal, engine
from dashboard_server.models import Base, Alert

Base.metadata.create_all(bind=engine)
app = FastAPI()


@app.post("/api/alerts")
async def receive_alert(
    camera_name: str = Form(...),
    timestamp: str = Form(...),
    snapshot: UploadFile = File(None),
    local_kw: str = Form(None),
    local_kw_query: str = Query(None),  # ✅ Accepts from query too
    db: Session = Depends(SessionLocal)
):
    """
    Receives alerts from the camera detector.
    Handles both Form and Query input for local_kw.
    """
    try:
        # ✅ Prefer Form value, fallback to Query if missing
        local_kw_value = local_kw or local_kw_query or "unknown"

        # ✅ Ensure timestamp format is valid
        try:
            timestamp_obj = datetime.fromisoformat(timestamp)
        except ValueError:
            timestamp_obj = datetime.now()

        # ✅ Save snapshot if received
        file_path = None
        if snapshot:
            folder = "snapshots"
            os.makedirs(folder, exist_ok=True)
            file_path = os.path.join(folder, snapshot.filename)
            with open(file_path, "wb") as buffer:
                buffer.write(await snapshot.read())

        # ✅ Save alert in database
        new_alert = Alert(
            camera_name=camera_name,
            timestamp=timestamp_obj,
            message=f"Person detected ({local_kw_value})"
        )
        db.add(new_alert)
        db.commit()

        print(f"[SERVER] Alert stored for {camera_name} at {timestamp_obj}")
        return JSONResponse(content={"status": "success", "camera_name": camera_name})

    except Exception as e:
        db.rollback()
        print(f"[ERROR] {e}")
        return JSONResponse(content={"status": "error", "detail": str(e)}, status_code=500)

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

