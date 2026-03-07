import os
import requests
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
api_url = os.getenv("API_URL", "https://findmyforce.online")

headers = {"X-API-Key": api_key}

def test_connection():
    print(f"Testing connection to {api_url} with key {api_key[:5]}...")
    
    # Try public status
    try:
        resp = requests.get(f"{api_url}/api/status", timeout=5)
        print(f"Status: {resp.status_code}")
    except Exception as e:
        print(f"Status failed: {e}")

    # Try authenticated score
    try:
        resp = requests.get(f"{api_url}/scores/me", headers=headers, timeout=5)
        print(f"Score: {resp.status_code} - {resp.text}")
    except Exception as e:
        print(f"Score failed: {e}")

if __name__ == "__main__":
    test_connection()
