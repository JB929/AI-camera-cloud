# src/core/cameras/rtsp_templates.py

RTSP_TEMPLATES = {
    "hikvision": [
        "rtsp://{user}:{password}@{ip}:554/Streaming/Channels/101",
        "rtsp://{user}:{password}@{ip}:554/Streaming/Channels/102",
    ],
    "dahua": [
        "rtsp://{user}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype=0",
    ],
    "cp_plus": [
        "rtsp://{user}:{password}@{ip}:554/cam/realmonitor?channel=1&subtype=0",
    ],
    "ezviz": [
        "rtsp://{user}:{password}@{ip}:554/h264_stream",
    ],
    "generic": [
        "rtsp://{user}:{password}@{ip}:554/stream1",
        "rtsp://{user}:{password}@{ip}:554/live",
    ],
}

