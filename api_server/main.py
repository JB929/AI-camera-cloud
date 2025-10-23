from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os
from datetime import datetime

app = FastAPI(title="AI Camera Alerts Dashboard")

ALERTS = []  # Temporary in-memory storage
SNAPSHOT_DIR = "snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

@app.get("/")
async def root():
    return {"message": "AI Monitor Cloud API is running!"}

@app.post("/alerts/")
async def receive_alert(message: str = Form(...), snapshot: UploadFile = None):
    """
    Receives alerts from the AI camera client.
    Saves snapshots and logs the alert.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    snapshot_path = None

    if snapshot:
        filename = f"snapshot_{int(datetime.now().timestamp())}.jpg"
        snapshot_path = os.path.join(SNAPSHOT_DIR, filename)
        with open(snapshot_path, "wb") as f:
            f.write(await snapshot.read())

    alert = {
        "id": len(ALERTS) + 1,
        "timestamp": timestamp,
        "message": message,
        "snapshot_path": snapshot_path,
    }
    ALERTS.append(alert)

    return JSONResponse(content={"status": "success", "alert": alert})

@app.get("/alerts/")
async def get_alerts():
    """Return all saved alerts."""
    return ALERTS

