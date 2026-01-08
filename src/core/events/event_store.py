# src/core/events/event_store.py

import sqlite3
import os
import time
import threading
DB_WRITE_LOCK = threading.Lock()
from src.core.config import load_config
CONFIG = load_config()

# -------------------------------------------------
# DATABASE PATH (ABSOLUTE, SINGLE SOURCE OF TRUTH)
# -------------------------------------------------
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

DB_PATH = os.path.join(BASE_DIR, "data", "events.db")

print("[EVENT_STORE] DB_PATH =", DB_PATH)

# -------------------------------------------------
# DB CONNECTION (THREAD-SAFE)
# -------------------------------------------------
def get_db_connection():
    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False,
        timeout=30
    )
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


# -------------------------------------------------
# DB INITIALIZATION
# -------------------------------------------------
def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            camera TEXT,
            timestamp REAL,
            pose TEXT,
            action TEXT,
            snapshot TEXT,
            payload TEXT
        )
    """)

    conn.commit()
    conn.close()


# -------------------------------------------------
# INSERT EVENT
# -------------------------------------------------
def insert_event(
    type,
    camera,
    timestamp,
    pose,
    action,
    snapshot,
    payload,
):
    with DB_WRITE_LOCK:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO events (event_type, camera, timestamp, pose, action, snapshot, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            type,
            camera,
            timestamp,
            pose,
            action,
            snapshot,
            payload,
        ))

        conn.commit()
        conn.close()

        print(f"[SQL] INSERTED {type} | {camera}")


# -------------------------------------------------
# READ EVENTS
# -------------------------------------------------
def fetch_events(limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT type, camera, timestamp, pose, action, snapshot
        FROM events
        ORDER BY timestamp DESC
        LIMIT ?
    """, (limit,))

    rows = c.fetchall()
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


EVENT_RETENTION_DAYS = 30
SNAPSHOT_RETENTION_DAYS = 30


def cleanup_old_events():
    """
    Deletes old events from DB and removes their snapshots from disk.
    Safe to run repeatedly.
    """
    cutoff_ts = time.time() - (EVENT_RETENTION_DAYS * 86400)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Fetch snapshots to delete
    cur.execute(
        "SELECT snapshot FROM events WHERE timestamp < ? AND snapshot IS NOT NULL",
        (cutoff_ts,),
    )
    rows = cur.fetchall()

    # Delete DB rows
    cur.execute(
        "DELETE FROM events WHERE timestamp < ?",
        (cutoff_ts,),
    )

    conn.commit()
    conn.close()

    # Delete snapshot files
    for (snap,) in rows:
        try:
            if snap and os.path.exists(snap):
                os.remove(snap)
        except Exception:
            pass

    print(f"[CLEANUP] Old events & snapshots cleaned")
