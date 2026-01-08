import json
import time
from datetime import datetime, timedelta

LICENSE_PATH = "license.json"

def load_license():
    try:
        with open(LICENSE_PATH, "r") as f:
            lic = json.load(f)
    except Exception:
        return None, "NO_LICENSE"

    expires = datetime.fromisoformat(lic["expires_at"])
    grace = lic.get("grace_days", 0)

    if datetime.utcnow() > expires + timedelta(days=grace):
        return None, "EXPIRED"

    return lic, "OK"
