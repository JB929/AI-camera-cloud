# src/core/events/event_bus.py
import time
import threading
from src.core.events.event_store import insert_event

from src.core.events.event_store import init_db
init_db()

import requests
import time
from src.core.config import load_config

CONFIG = load_config()
DB_PATH = CONFIG["paths"]["events_db"]

print("[EVENT BUS] DB PATH =", DB_PATH)

SERVER_BASE_URL = CONFIG["server"]["base_url"]

def emit_event(event_type, camera, payload):
    """
    Central event emitter.
    Stores event in SQLite via event_store.
    """

    timestamp = time.time()

    print(f"[EVENT] {event_type} | {camera}")

    def _store():
        try:
            insert_event(
                type=event_type,
                camera=camera,
                timestamp=timestamp,
                pose=payload.get("pose"),
                action=payload.get("action"),
                snapshot=payload.get("snapshot"),
                payload=str(payload),
            )
            print(f"[SQL] INSERTED {event_type} | {camera}")
        except Exception as e:
            print(f"[EVENT STORE ERROR] {e}")

    # run DB write in background (non-blocking)
    threading.Thread(target=_store, daemon=True).start()

