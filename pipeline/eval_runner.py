"""
Evaluation Runner for Find My Force.
Fetches the official evaluation dataset, runs the classifier (with hostile heuristics),
performs geolocation, and submits the final payload to the scoring endpoint.
"""

import os
import time
import requests
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from classifier import SignalClassifier
from pipeline.geolocator import GeolocatorEngine

logger = logging.getLogger(__name__)

# Hostile & Civilian labels for anomaly mapping
HOSTILE_LABELS = {
    "Airborne-detection", # Pulsed
    "Airborne-range",     # Pulsed
    "Air-Ground-MTI",     # Pulsed
    "EW-Jammer",          # Jamming
}
CIVILIAN_LABELS = {
    "AM radio",           # AM-DSB
}

def guess_hostile_type(features: dict) -> str:
    """
    Heuristically map an 'unknown' anomaly to a specific hostile/civilian label
    based on the extracted IQ features.
    """
    duty_cycle = features.get("duty_cycle", 1.0)
    flatness = features.get("spectral_flatness", 0.0)
    ask_ratio = features.get("ask_ratio", 0.0)

    # Pulsed signals (Radars) have low duty cycle (lots of silent gaps)
    if duty_cycle < 0.8:
        return "Airborne-detection"
    
    # Jammers are broad-band noise, which means high spectral flatness
    if flatness > 0.2:
        return "EW-Jammer"
    
    # If it's continuous (high duty cycle) and localized in frequency (low flatness),
    # it's very likely civilian AM radio.
    return "AM radio"

def run_evaluation_pipeline():
    """Main evaluation execution flow."""
    load_dotenv()
    api_key = os.getenv("API_KEY")
    api_url = os.getenv("API_URL", "https://findmyforce.online")

    if not api_key:
        logger.error("API_KEY not found in .env file!")
        return

    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json"
    }

    logger.info("Initializing models...")
    classifier = SignalClassifier()
    if not classifier.load():
        logger.error("Failed to load classifier! Run 'python3 main.py train' first.")
        return

    # We will instantiate the geolocator to grab the config, 
    # but actual evaluation feed doesn't group obs for us, so we 
    # just do single-receiver distance estimates if TDoA isn't possible.
    # Initialize Geolocator by fetching configs first
    try:
        receivers = []
        path_loss = None
        
        recv_resp = requests.get(f"{api_url}/config/receivers", headers=headers, timeout=10)
        recv_resp.raise_for_status()
        recv_data = recv_resp.json()
        
        # We need the ReceiverInfo dataclass format
        from pipeline.geolocator import ReceiverInfo, PathLossModel
        for r in recv_data.get("receivers", []):
            receivers.append(ReceiverInfo(r["receiver_id"], r["latitude"], r["longitude"], r.get("sensitivity_dbm", -120.0), getattr(r, "timing_accuracy_ns", 10.0)))
            
        pl_resp = requests.get(f"{api_url}/config/pathloss", headers=headers, timeout=10)
        pl_resp.raise_for_status()
        pl_data = pl_resp.json()
        path_loss = PathLossModel(pl_data["rssi_ref_dbm"], pl_data["d_ref_m"], pl_data["path_loss_exponent"], getattr(pl_data, "rssi_noise_std_db", 2.0))
        
        geo = GeolocatorEngine(receivers, path_loss)
    except Exception as e:
        logger.error(f"Failed to initialize GeolocatorEngine: {e}")
        return

    logger.info(f"Fetching evaluation dataset from {api_url}/evaluate/observations...")
    try:
        eval_resp = requests.get(f"{api_url}/evaluate/observations", headers=headers, timeout=30)
        eval_resp.raise_for_status()
        eval_data = eval_resp.json()
        eval_obs = eval_data.get("observations", [])
    except Exception as e:
        logger.error(f"Failed to fetch evaluation data: {e}")
        return

    logger.info(f"Retrieved {len(eval_obs)} observations for scoring.")

    submissions = []
    logger.info("Classifying observations...")
    
    # Process observations in batches to speed up feature extraction
    for idx, obs in enumerate(eval_obs):
        obs_id = obs["observation_id"]
        recv_id = obs["receiver_id"]
        iq = obs["iq_snapshot"]

        # Run classifier
        clf_result = classifier.predict(iq)
        
        # Determine final label
        final_label = clf_result["label"]
        if clf_result["is_anomaly"] or final_label == "unknown":
            final_label = guess_hostile_type(clf_result.get("features", {}))

        # Basic Single-Receiver Geolocation
        lat, lon = None, None
        rssi = obs.get("rssi_dbm")
        if rssi is not None and recv_id in geo.receivers:
            recv = geo.receivers[recv_id]
            # Estimate distance
            distance = geo._rssi_to_distance(rssi)
            # Default to the receiver's location as a highly naive estimate
            # (In a real system we'd associate obs across receivers, but the eval feed is flat)
            lat, lon = recv["lat"], recv["lon"]

        submissions.append({
            "observation_id": obs_id,
            "classification_label": final_label,
            "confidence": clf_result["confidence"],
            "estimated_latitude": lat,
            "estimated_longitude": lon,
        })

        if (idx + 1) % 500 == 0:
            logger.info(f"Processed {idx + 1}/{len(eval_obs)} observations...")

    logger.info(f"Submitting {len(submissions)} classifications for official scoring...")
    
    try:
        score_resp = requests.post(
            f"{api_url}/evaluate/submit",
            headers=headers,
            json={"submissions": submissions},
            timeout=60
        )
        score_resp.raise_for_status()
        result = score_resp.json()
        
        logger.info("\n=== EVALUATION RESULTS ===")
        logger.info(f"Attempt #{result.get('attempt_number', 1)}")
        logger.info(f"Coverage: {result.get('coverage', 0):.1f}%")
        logger.info(f"Total Score: {result.get('total_score', 0):.1f} / 100")
        logger.info(f"  - Classification: {result.get('classification_score', 0):.1f}")
        logger.info(f"  - Geolocation:    {result.get('geolocation_score', 0):.1f}")
        logger.info(f"  - Novelty:        {result.get('novelty_score', 0):.1f}")
        logger.info(f"Best Total Score:   {result.get('best_total_score', 0):.1f}")
        logger.info("========================\n")
        
    except Exception as e:
        logger.error(f"Failed to submit evaluation: {e}")
        if hasattr(e, 'response') and e.response:
            logger.error(f"Server response: {e.response.text}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_evaluation_pipeline()
