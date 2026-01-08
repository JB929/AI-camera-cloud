# src/core/alerts/alert_throttle.py
import time

ALERT_COOLDOWNS = {
    "PERSON_DETECTED": 60,
    "PERSON_ENTER": 120,
    "PERSON_EXIT": 60,
    "ACTION": 90,
    "FALL": 300,
}

_last_alert_ts = {}


def can_send_alert(camera, alert_type):
    now = time.time()
    key = f"{camera}:{alert_type}"

    cooldown = ALERT_COOLDOWNS.get(alert_type, 60)
    last_ts = _last_alert_ts.get(key, 0)

    if now - last_ts < cooldown:
        return False

    _last_alert_ts[key] = now
    return True

