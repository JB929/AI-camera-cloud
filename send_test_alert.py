import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder

# ✅ Correct cloud API endpoint
url = "https://ai-camera-cloud.onrender.com/api/alerts"

# ✅ Prepare multipart form data correctly
m = MultipartEncoder(
    fields={
        "camera_name": "Front_Yard",
        "timestamp": "2025-10-30 19:45:00",
        "message": "Test alert with proper multipart encoding",
        "snapshot": ("test.jpg", open("testimage1.jpg", "rb"), "image/jpg")
    }
)

# ✅ Send the request
response = requests.post(url, data=m, headers={'Content-Type': m.content_type})

print("Status Code:", response.status_code)
print("Response:", response.text)

