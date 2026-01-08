# src/core/storage/event_store.py

import sqlite3
import os
import threading
import time

DB_PATH = os.environ.get(
    "AI_CAMERA_DB",
    os.path.abspath("data/events.db")
)

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

_db_lock = threading.Lock()


def get_connection():
    return sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )


def init_db():
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            camera TEXT NOT NULL,
            timestamp REAL NOT NULL,
            pose TEXT,
            action TEXT,
            snapshot TEXT,
            payload TEXT
        )
        """)

        cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_events_time
        ON events(timestamp DESC)
        """)

        conn.commit()
        conn.close()


def insert_event(
    event_type: str,
    camera: str,
    timestamp: float,
    pose: str = None,
    action: str = None,
    snapshot: str = None,
    payload: str = None,
):
    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute("""
        INSERT INTO events (
            event_type,
            camera,
            timestamp,
            pose,
            action,
            snapshot,
            payload
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            event_type,
            camera,
            timestamp,
            pose,
            action,
            snapshot,
            payload
        ))

        conn.commit()
        conn.close()


def fetch_events(
    limit: int = 50,
    camera: str = None,
    event_type: str = None
):
    query = "SELECT event_type, camera, timestamp, pose, action, snapshot FROM events"
    clauses = []
    params = []

    if camera:
        clauses.append("camera = ?")
        params.append(camera)

    if event_type:
        clauses.append("event_type = ?")
        params.append(event_type)

    if clauses:
        query += " WHERE " + " AND ".join(clauses)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    with _db_lock:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        conn.close()

    events = []
    for r in rows:
        events.append({
            "type": r[0],
            "camera": r[1],
            "timestamp": r[2],
            "payload": {
                "pose": r[3],
                "action": r[4],
                "snapshot": r[5],
            }
        })

    return events

