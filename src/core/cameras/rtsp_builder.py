# src/core/camera/rtsp_builder.py

from typing import Optional

RTSP_TEMPLATES = {
    "hikvision": "rtsp://{username}:{password}@{ip}:554/Streaming/Channels/{channel}01",
    "dahua": "rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel={channel}&subtype=0",
    "cpplus": "rtsp://{username}:{password}@{ip}:554/cam/realmonitor?channel={channel}&subtype=0",
    "ezviz": "rtsp://{username}:{password}@{ip}:554/h264_stream",
    "tplink": "rtsp://{username}:{password}@{ip}:554/stream1",
    "generic": "rtsp://{username}:{password}@{ip}:554/stream1",
}


from typing import Optional

def build_rtsp_url(
    brand: str,
    ip: str,
    username: Optional[str] = None,
    password: Optional[str] = None,
    port: int = 554,
) -> Optional[str]:
    """
    Build RTSP URL from brand + credentials.
    """

    if not ip:
        return None

    brand = (brand or "generic").lower()

    auth = ""
    if username and password:
        auth = f"{username}:{password}@"

    # CP Plus / Hikvision
    if brand in ("hikvision", "cpplus", "cp-plus"):
        return f"rtsp://{auth}{ip}:{port}/Streaming/Channels/101"

    # Dahua
    if brand in ("dahua",):
        return f"rtsp://{auth}{ip}:{port}/cam/realmonitor?channel=1&subtype=0"

    # TP-Link / Tapo
    if brand in ("tplink", "tapo"):
        return f"rtsp://{auth}{ip}:{port}/stream1"

    # Generic fallback
    return f"rtsp://{auth}{ip}:{port}/"
