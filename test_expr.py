import requests
import json

path = 'uploads/EYE3.mp4'
url = 'http://127.0.0.1:8000/analyze'

print(f"Testing: {path}")
with open(path, 'rb') as f:
    files = {'file': ('EYE3.mp4', f, 'video/mp4')}
    resp = requests.post(url, files=files, timeout=600)

r = resp.json()
with open('response.json', 'w') as out:
    json.dump(r, out, indent=2)
print("Saved to response.json")
