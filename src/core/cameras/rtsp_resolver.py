import cv2
from src.core.cameras.rtsp_templates import RTSP_TEMPLATES

def resolve_rtsp_from_onvif(brand, ip, username=None, password=None, port=554):
    brand = brand.lower()
    templates = RTSP_TEMPLATES.get(brand, RTSP_TEMPLATES["generic"])

    for tmpl in templates:
        try:
            rtsp = tmpl.format(
                user=username or "",
                password=password or "",
                ip=ip,
                port=port
            )

            cap = cv2.VideoCapture(rtsp)
            if cap.isOpened():
                ret, _ = cap.read()
                cap.release()
                if ret:
                    print(f"[RTSP] Working stream found: {rtsp}")
                    return rtsp

        except Exception:
            continue

    return None

