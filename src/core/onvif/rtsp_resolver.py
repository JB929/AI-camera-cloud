# src/core/onvif/rtsp_resolver.py

def resolve_rtsp_url(
    brand: str,
    ip: str,
    username: str | None = None,
    password: str | None = None,
    port: int = 554,
    channel: int = 1,
    stream: int = 1,
):
    """
    Generate RTSP URL based on camera brand.
    """

    auth = ""
    if username and password:
        auth = f"{username}:{password}@"

    brand = brand.lower()

    if brand.lower() in ["cpplus", "hikvision"]:
        if username and password:
            return (
                f"rtsp://{username}:{password}@{ip}:{port}"
                f"/Streaming/Channels/{channel}{stream:02d}"
            )
        else:
            return (
                f"rtsp://{ip}:{port}"
                f"/Streaming/Channels/{channel}{stream:02d}"
            )

    elif brand in ("dahua", "imou"):
        return (
            f"rtsp://{auth}{ip}:{port}"
            f"/cam/realmonitor?channel={channel}&subtype={stream-1}"
        )

    elif brand in ("tp-link", "tplink", "tapo"):
        return f"rtsp://{auth}{ip}:{port}/stream1"

    elif brand == "generic":
        return f"rtsp://{auth}{ip}:{port}/"

    else:
        raise ValueError(f"Unsupported camera brand: {brand}")
