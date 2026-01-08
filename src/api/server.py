print("🔥🔥🔥 SERVER FILE LOADED:", __file__)

from fastapi import WebSocket, WebSocketDisconnect
from typing import List
import asyncio
import sqlite3 
from src.core.events.event_store import fetch_events
from fastapi import FastAPI
from src.core.config import EVENT_LOG_PATH
from src.core.onvif.discovery import discover_onvif_cameras
from src.core.subscription.enforcement import get_subscription_state
import sys
import os
import time
from collections import deque
from src.core.events.event_store import init_db
from pathlib import Path
from fastapi.staticfiles import StaticFiles
import base64
import time
import threading
from fastapi import Header, HTTPException, Depends
from src.core.relay.state import RELAY_FRAMES, RELAY_LOCK

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, event: dict):
        for ws in list(self.active_connections):
            try:
                await ws.send_json(event)
            except Exception:
                self.disconnect(ws)

manager = ConnectionManager()
# -----------------------------
# APP
# -----------------------------
app = FastAPI()

# -----------------------------
# CONFIG
# -----------------------------
from src.core.config import load_config
from src.core.events.event_store import DB_PATH
print("[DB DEBUG] using DB_PATH =", DB_PATH)


CONFIG = load_config()
DASHBOARD_TOKEN = CONFIG["dashboard"]["token"]

from src.core.events.event_store import init_db
init_db()
# -----------------------------
# AUTH
# -----------------------------
def require_auth(authorization: str = Header(None)):
    if authorization != f"Bearer {DASHBOARD_TOKEN}":
        print("[AUTH FAIL]", authorization)
        raise HTTPException(status_code=403, detail="Unauthorized")

# -----------------------------
# HEALTH
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# -----------------------------
# SNAPSHOTS
# -----------------------------
from fastapi.staticfiles import StaticFiles

app.mount(
    "/snapshots",
    StaticFiles(directory="snapshots"),
    name="snapshots"
)

# ----------------------------------------
# DB HELPERS (PASTE HERE)
# ----------------------------------------
def fetch_events(limit=100):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    sql = """
    SELECT
        event_type,
        camera,
        timestamp,
        pose,
        action,
        snapshot
    FROM events
    ORDER BY id DESC
    LIMIT ?
    """

    cur.execute(sql, (limit,))
    rows = cur.fetchall()
    conn.close()

    events = []
    for r in rows:
        events.append({
            "type": r[0],
            "camera": r[1],
            "timestamp": float(r[2]),
            "payload": {
                "pose": r[3],
                "action": r[4],
                "snapshot": r[5],
            }
        })

    return events

@app.get("/events")
def events_api(
    limit: int = 50,
    _: None = Depends(require_auth),
):
    events = fetch_events(limit=limit)
    return {
        "count": len(events),
        "events": events,
        "tier": CONFIG["subscription"]["tier"],   
    }

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def dashboard():
    print("🔥🔥🔥 DASHBOARD ROUTE HIT")
    
    from src.core.config import load_config
    cfg = load_config()

    refresh_ms = int(cfg["dashboard"]["refresh_seconds"]) * 1000
    token = cfg["dashboard"]["token"]

    html = Path("src/api/templates/dashboard.html").read_text()

    html = html.replace("{{ refresh_ms }}", str(refresh_ms))
    html = html.replace("{{ DASHBOARD_TOKEN }}", token)

    return HTMLResponse(html)

@app.get("/advanced", response_class=HTMLResponse)
def advanced_page():
    html_path = Path("src/api/templates/advanced.html")
    return HTMLResponse(html_path.read_text())

@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)  # keep alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

import threading

def event_stream_loop():
    last_len = 0
    while True:
        events = fetch_events()   # ← LIST, not JSONResponse

        if len(events) > last_len:
            new_events = events[last_len:]
            for e in new_events:
                asyncio.run(manager.broadcast(e))
            last_len = len(events)

        time.sleep(0.5)

threading.Thread(target=event_stream_loop, daemon=True).start()


from src.core.events.event_store import cleanup_old_events
import threading
import time


def cleanup_loop():
    while True:
        try:
            cleanup_old_events()
        except Exception as e:
            print(f"[CLEANUP ERROR] {e}")
        time.sleep(86400)  # run once per day


threading.Thread(target=cleanup_loop, daemon=True).start()

@app.get("/subscription")
def subscription_info():
    return get_subscription_state()

@app.get("/cameras")
def cameras_api():
    CONFIG = load_config()

    cameras = []
    for name, cfg in CONFIG.get("cameras", {}).items():
        cameras.append({
            "name": name,
            "enabled": cfg.get("enabled", True),
            "brand": cfg.get("brand", "generic"),
            "rtsp": cfg.get("rtsp"),
        })

    return {
        "count": len(cameras),
        "cameras": cameras
    }


@app.get("/api/discover-cameras")
def discover_cameras():
    """
    Discover ONVIF cameras on local network
    """
    try:
        cameras = discover_onvif_cameras()
        return {
            "count": len(cameras),
            "cameras": cameras
        }
    except Exception as e:
        return {
            "error": str(e),
            "count": 0,
            "cameras": []
        }

from fastapi import FastAPI
from src.core.onvif.onvif_discovery import discover_onvif_devices

@app.get("/api/onvif/discover")
def onvif_discover():
    devices = discover_onvif_devices()
    return {
        "count": len(devices),
        "cameras": devices
    }

from src.core.onvif.onvif_discovery import discover_cameras_with_rtsp


@app.get("/api/onvif/rtsp")
def onvif_rtsp_builder(
    brand: str = "generic",
    username: str = "admin",
    password: str = "admin",
):
    cams = discover_cameras_with_rtsp(
        brand=brand,
        username=username,
        password=password,
    )

    return {
        "count": len(cams),
        "cameras": cams,
    }

import cv2


@app.post("/api/camera/test")
def test_camera_rtsp(rtsp: str):
    cap = cv2.VideoCapture(rtsp)

    if not cap.isOpened():
        return {"ok": False, "error": "Unable to open stream"}

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        return {"ok": False, "error": "No frames received"}

    return {"ok": True}

from src.core.onvif.onvif_discovery import discover_onvif_cameras
from src.core.config import save_camera_config

@app.get("/cameras")
def cameras_page():
    html_path = Path("src/api/templates/cameras.html")
    return HTMLResponse(html_path.read_text())


@app.get("/api/cameras/discover")
def api_discover_cameras():
    cams = discover_onvif_cameras(timeout=5)
    return {
        "count": len(cams),
        "cameras": cams
    }


from src.core.cameras.rtsp_builder import build_rtsp_url

@app.post("/api/cameras/add")
def api_add_camera(payload: dict):
    """
    payload = {
      name, brand, ip, username, password, port
    }
    """
 
    # 🔒 SUBSCRIPTION CHECK (ADD THIS BLOCK)
    sub = get_subscription_state()

    if not sub["active"]:
        raise HTTPException(
            status_code=403,
            detail=f"Subscription inactive: {sub.get('reason')}"
        )

    max_cams = sub["limits"]["cameras"]

    existing = count_existing_cameras()  
    if existing >= max_cams:
        raise HTTPException(
            status_code=403,
            detail=f"Camera limit reached ({existing}/{max_cams})"
        )


    rtsp = build_rtsp_url(
        brand=payload.get("brand", "generic"),
        ip=payload.get("ip"),
        username=payload.get("username"),
        password=payload.get("password"),
        port=payload.get("port", 554),
    )

    if not rtsp:
        return {"error": "Failed to generate RTSP"}

    save_camera_config({
        "name": payload["name"],
        "brand": payload.get("brand", "generic"),
        "rtsp": rtsp,
        "enabled": True,
    })

    return {"status": "ok", "rtsp": rtsp}

@app.post("/api/cameras/manual")
def api_add_manual_camera(payload: dict):
    """
    payload = {
        name: str,
        rtsp: str,
        enabled: bool
    }
    """

    name = payload.get("name")
    rtsp = payload.get("rtsp")

    if not name or not rtsp:
        return {"error": "name and rtsp required"}

    save_camera_config({
        "name": name,
        "brand": "manual",
        "rtsp": rtsp,
        "enabled": payload.get("enabled", True),
    })

    return {"status": "ok"}

def can_add_camera():
    current = len(CONFIG["cameras"])
    limit = CONFIG["subscription"]["camera_limit"]
    return current < limit

from src.core.subscription.enforcement import get_subscription_state

@app.get("/api/subscription/status")
def subscription_status():
    sub = get_subscription_state()

    return {
        "active": sub["active"],
        "plan": sub["plan"],
        "limits": sub["limits"],
    }

from collections import deque
import base64
import time

from fastapi import Body
import base64
import time

@app.post("/api/relay/frame")
def relay_frame(payload: dict):
    print("🔥 RELAY FRAME RECEIVED:", payload.get("camera"))

    camera = payload.get("camera")
    jpeg_b64 = payload.get("jpeg")
    ts = payload.get("ts")

    if not camera or not jpeg_b64 or not ts:
        return {"error": "invalid payload"}
 
    RELAY_FRAMES[camera] = {
        "jpeg": jpeg_b64,
        "ts": ts
    }
        
    print(f"🔥 RELAY FRAME RECEIVED: {camera}")
    return {"status": "ok"}


from fastapi.responses import Response
import json
import base64

@app.get("/api/relay/latest")
def relay_latest(camera: str):
    data = RELAY_FRAMES.get(camera)

    if not data:
        raise HTTPException(status_code=404, detail="No frame")

    return {
        "camera": camera,
        "ts": data["ts"],
        "jpeg": data["jpeg"]  # base64 string ONLY
    }

@app.get("/api/relay/status")
def relay_status():
    now = time.time()
    out = {}

    for cam, data in RELAY_FRAMES.items():
        if not isinstance(data, dict):
            continue

        ts = data.get("ts")
        if not ts:
            continue

        out[cam] = {
            "age": round(now - ts, 3)
        }

    return out

from fastapi import HTTPException
from fastapi import UploadFile, Form, File

@app.post("/ingest/frame")
async def ingest_frame(
    frame: UploadFile = File(...),
    camera: str = Form(...),
):
    data = await frame.read()
    FRAME_QUEUE[camera] = data
    return {"status": "ok"}

def prune_old_events():
    days = CONFIG["subscription"]["retention_days"]
    cutoff = time.time() - days * 86400

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
    conn.commit()

@app.middleware("http")
async def security_headers(request, call_next):
    resp = await call_next(request)
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    return resp
