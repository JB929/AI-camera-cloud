PLANS = {
    "FREE": {
        "cameras": 1,
        "retention_days": 1,
        "features": {
            "presence": True,
            "actions": False,
            "alerts": False,
        },
    },
    "PRO": {
        "cameras": 4,
        "retention_days": 14,
        "features": {
            "presence": True,
            "actions": True,
            "alerts": True,
        },
    },
    "ENTERPRISE": {
        "cameras": 999,
        "retention_days": 90,
        "features": {
            "presence": True,
            "actions": True,
            "alerts": True,
        },
    },
}
