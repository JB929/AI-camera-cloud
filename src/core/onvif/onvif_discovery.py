from wsdiscovery import WSDiscovery
from wsdiscovery.scope import Scope
import socket

def discover_onvif_devices(timeout=5):
    """
    Discover ONVIF-compatible devices on the local network.
    Returns a list of devices with xaddr and basic metadata.
    """

    devices = []
    wsd = None

    try:
        wsd = WSDiscovery()
        wsd.start()

        services = wsd.searchServices(
            scopes=[Scope("onvif://www.onvif.org")],
            timeout=timeout
        )

        for service in services:
            xaddrs = service.getXAddrs()
            if not xaddrs:
                continue

            for xaddr in xaddrs:
                devices.append({
                    "xaddr": xaddr,
                    "types": str(service.getTypes()),
                    "scopes": [str(s) for s in service.getScopes()],
                })

    except Exception as e:
        print("[ONVIF WARNING]", e)

    finally:
        if wsd:
            wsd.stop()

    return devices

from src.core.onvif.rtsp_builder import extract_ip_from_xaddr, build_rtsp_url


def discover_cameras_with_rtsp(
    brand="generic",
    username="admin",
    password="admin",
):
    devices = discover_onvif_devices()
    cameras = []

    for dev in devices:
        ip = extract_ip_from_xaddr(dev["xaddr"])
        if not ip:
            continue

        rtsp = build_rtsp_url(
            ip=ip,
            brand=brand,
            username=username,
            password=password,
        )

        cameras.append({
            "ip": ip,
            "rtsp": rtsp,
            "xaddr": dev["xaddr"],
        })

    return cameras

def get_rtsp_from_onvif(ip, username=None, password=None):
    try:
        cam = ONVIFCamera(ip, 80, username, password)
        media = cam.create_media_service()
        profiles = media.GetProfiles()

        if not profiles:
            return None

        token = profiles[0].token
        stream = media.GetStreamUri({
            "StreamSetup": {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"}
            },
            "ProfileToken": token
        })

        return stream.Uri

    except Exception as e:
        return None

def discover_onvif_cameras(timeout=5):
    """
    Stable public API for ONVIF discovery
    """
    return discover_onvif_devices(timeout=timeout)
