# src/core/alerts/alert_router.py

import time

# --- CONFIG ---
ALERT_COOLDOWN_SEC = 60  # per camera per event
ENABLE_ALERTS = True

# state: (camera, event_type) -> last_sent_ts
_last_alert_sent = {}


def should_send_alert(camera, event_type):
    now = time.time()
    key = (camera, event_type)

    last = _last_alert_sent.get(key, 0)
    if now - last < ALERT_COOLDOWN_SEC:
        return False

    _last_alert_sent[key] = now
    return True


def route_alert(event):
    """
    event = {
        type, camera, timestamp, payload
    }
    """
    if not ENABLE_ALERTS:
        return

    event_type = event["type"]
    camera = event["camera"]

    # 🔥 only alert on important events
    if event_type not in (
        "PERSON_ENTER",
        "PERSON_EXIT",
        "FALL",
    ):
        return

    if not should_send_alert(camera, event_type):
        return

    # 👉 send via email (Phase 7A)
    try:
        from src.core.alerts.email_alert import send_email_alert
        send_email_alert(event)
    except Exception as e:
        print(f"[ALERT ERROR] email failed: {e}")

