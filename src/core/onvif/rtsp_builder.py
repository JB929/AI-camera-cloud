from urllib.parse import urlparse


def extract_ip_from_xaddr(xaddr: str) -> str | None:
    try:
        parsed = urlparse(xaddr)
        return parsed.hostname
    except Exception:
        return None

RTSP_TEMPLATES = {
    "hikvision": "rtsp://{user}:{pwd}@{ip}:554/Streaming/Channels/101",
    "cpplus":    "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel=1&subtype=0",
    "dahua":     "rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel=1&subtype=0",
    "tp-link":   "rtsp://{user}:{pwd}@{ip}:554/stream1",
    "generic":   "rtsp://{user}:{pwd}@{ip}:554/stream1",
}

def build_rtsp_url(
    ip: str,
    brand: str = "generic",
    username: str = "admin",
    password: str = "admin",
):
    tpl = RTSP_TEMPLATES.get(brand.lower(), RTSP_TEMPLATES["generic"])
    return tpl.format(
        ip=ip,
        user=username,
        pwd=password,
    )
