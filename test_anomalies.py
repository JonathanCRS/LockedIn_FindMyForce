import time
import requests
import json
import numpy as np
from dotenv import load_dotenv
import os
import sys

# We need the classifier to get features
sys.path.append(os.getcwd())
try:
    from classifier.signal_classifier import SignalClassifier, extract_features
except Exception as e:
    print(f"Error loading classifier: {e}")
    sys.exit(1)

load_dotenv()
api_key = os.getenv("API_KEY")
api_url = os.getenv("API_URL", "https://findmyforce.online")

def test_feed():
    clf = SignalClassifier()
    clf.load()
    
    headers = {"X-API-Key": api_key, "Accept": "text/event-stream"}
    try:
        with requests.get(f"{api_url}/feed/stream", headers=headers, stream=True, timeout=10) as resp:
            print("Connected to stream...")
            anoms = 0
            for line in resp.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    obs = json.loads(line[6:].strip())
                    iq = obs["iq_snapshot"]
                    res = clf.predict(iq)
                    
                    if res["is_anomaly"] or res["label"] == "unknown":
                        f = res.get("features", {})
                        if f.get('spectral_flatness', 0) < 0.3 and f.get('duty_cycle', 0) < 0.9:
                            print(f"PULSED RADAR ANOMALY:")
                            print(f"  Duty Cycle: {f.get('duty_cycle', 0):.3f}")
                            print(f"  Flatness  : {f.get('spectral_flatness', 0):.3f}")
                            print(f"  ASK Ratio : {f.get('ask_ratio', 0):.3f}")
                            print(f"  Freq Linearity : {f.get('freq_linearity', 0):.3f}")
                            print(f"  ZCR Amp   : {f.get('zcr_amp', 0):.3f}")
                            print(f"  Phase Std : {f.get('phase_std', 0):.3f}")
                            print("-------------")
                            anoms += 1
                            if anoms >= 20:
                                break
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_feed()
