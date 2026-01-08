# src/core/onvif/discovery.py

from wsdiscovery.discovery import ThreadedWSDiscovery as WSDiscovery
from wsdiscovery import QName
from urllib.parse import urlparse

def discover_onvif_cameras(timeout=5):
    """
    Discover ONVIF-compatible cameras on local network.
    Returns list of dicts with ip + xaddr.
    """
    wsd = WSDiscovery()
    wsd.start()

    services = wsd.searchServices(
        types=[QName("http://www.onvif.org/ver10/network/wsdl", "NetworkVideoTransmitter")],
        timeout=timeout
    )

    cameras = []

    for service in services:
        for xaddr in service.getXAddrs():
            parsed = urlparse(xaddr)
            if parsed.hostname:
                cameras.append({
                    "ip": parsed.hostname,
                    "xaddr": xaddr
                })

    wsd.stop()
    return cameras
