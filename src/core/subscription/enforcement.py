from src.core.subscription.plans import PLANS
from src.core.subscription.license import load_license

def get_subscription_state():
    lic, status = load_license()

    if status != "OK":
        return {
            "active": False,
            "reason": status,
        }

    plan = PLANS.get(lic["plan"])
    if not plan:
        return {
            "active": False,
            "reason": "INVALID_PLAN",
        }

    return {
        "active": True,
        "plan": lic["plan"],
        "limits": plan,
    }
