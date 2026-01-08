# src/core/alerts/email_alert.py

import smtplib
from email.message import EmailMessage
import os
import datetime

# 🔐 CONFIG (use env vars in production)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

EMAIL_FROM = os.environ.get("ALERT_EMAIL_FROM")
EMAIL_PASS = os.environ.get("ALERT_EMAIL_PASS")
EMAIL_TO   = os.environ.get("ALERT_EMAIL_TO")

if not EMAIL_FROM or not EMAIL_PASS:
    print("[EMAIL] disabled (missing credentials)")
    return

def send_email_alert(event):
    if not EMAIL_FROM or not EMAIL_PASS or not EMAIL_TO:
        print("[ALERT] Email not configured, skipping")
        return

    msg = EmailMessage()

    ts = datetime.datetime.fromtimestamp(
        event["timestamp"]
    ).strftime("%Y-%m-%d %H:%M:%S")

    subject = f"[ALERT] {event['type']} — {event['camera']}"
    body = f"""
🚨 AI Camera Alert

Event: {event['type']}
Camera: {event['camera']}
Time: {ts}

Details:
{event.get('payload', {})}
"""

    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASS)
        server.send_message(msg)

    print(f"[ALERT] Email sent for {event['type']} | {event['camera']}")

