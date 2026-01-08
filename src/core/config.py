import os
from pathlib import Path
import yaml

# Project root directory
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

EVENT_LOG_PATH = os.path.join(BASE_DIR, "events.jsonl")


# -----------------------------
# NEW: YAML CONFIG SUPPORT
# -----------------------------
_CONFIG = None

def load_config():
    """
    Load config.yaml once and cache it.
    """
    global _CONFIG
    if _CONFIG is None:
        config_path = Path(BASE_DIR) / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"config.yaml not found at {config_path}")

        with open(config_path, "r") as f:
            _CONFIG = yaml.safe_load(f)

    return _CONFIG


def get_camera_config(camera_name: str) -> dict:
    """
    Returns camera-specific config with system defaults as fallback
    """
    cam_cfg = _CONFIG.get("cameras", {}).get(camera_name, {})

    return {
        "enabled": cam_cfg.get("enabled", True),

        "active_hours": cam_cfg.get("active_hours"),

        "presence": {
            "confirm_frames": cam_cfg.get("presence", {}).get(
                "presence_confirm_frames",
                _CONFIG["presence"]["presence_confirm_frames"]
            ),
            "absence_frames": cam_cfg.get("presence", {}).get(
                "absence_confirm_frames",
                _CONFIG["presence"]["absence_confirm_frames"]
            ),
            "cooldown_seconds": cam_cfg.get("presence", {}).get(
                "cooldown_seconds",
                _CONFIG["presence"]["cooldown_seconds"]
            ),
        },


        "motion": {
            "idle_eps": cam_cfg.get("motion", {}).get(
                "idle_eps", _CONFIG["motion"]["idle_eps"]
            ),
            "moving_threshold": cam_cfg.get("motion", {}).get(
                "moving_threshold", _CONFIG["motion"]["moving_threshold"]
            ),
        },

        "alerts": cam_cfg.get("alerts", {})
    }

def save_camera_config(camera: dict):
    """
    Persist a new camera into config.yaml
    camera = {
        name, brand, rtsp, enabled
    }
    """
    global _CONFIG

    if _CONFIG is None:
        load_config()

    cams = _CONFIG.setdefault("cameras", {})

    cams[camera["name"]] = {
        "enabled": camera.get("enabled", True),
        "brand": camera.get("brand", "generic"),
        "rtsp": camera["rtsp"],
    }

    config_path = Path(BASE_DIR) / "config.yaml"

    with open(config_path, "w") as f:
        yaml.safe_dump(_CONFIG, f, sort_keys=False)

    print(f"[CONFIG] Camera saved: {camera['name']}")
