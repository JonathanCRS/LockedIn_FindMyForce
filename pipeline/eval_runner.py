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
    # 1. Jammers are broad-band white noise (high spectral flatness)
    if flatness > 0.3:
        return "EW-Jammer"
        
    # 2. AM Radio has a continuous carrier wave (very high duty cycle > 0.90)
    if duty_cycle > 0.90:
        return "AM radio"
        
    # 3. Pulsed radars have silent gaps between pings (duty cycle < 0.85)
    # The different radar types (Airborne-detection, Airborne-range) operate at
    # different pulse widths (duty cycles ~0.50 vs ~0.70). 
    # Since we just need to flag them as Hostile, we map them all to Airborne-detection.
    return "Airborne-detection"

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
    logger.info("Classifying and associating observations...")
    
    from pipeline.associator import ObservationAssociator
    associator = ObservationAssociator()
    
    # Process and build temporal observation groups
    for idx, obs in enumerate(eval_obs):
        # Run classifier
        clf_result = classifier.predict(obs.get("iq_snapshot", []))
        
        # Determine final label
        final_label = clf_result["label"]
        if clf_result.get("is_anomaly") or final_label == "unknown":
            final_label = guess_hostile_type(clf_result.get("features", {}))
            
        # Update classification dict so associator can use the final mapped heuristic label
        clf_result["label"] = final_label
        
        # Feed into associator
        associator.add_observation(obs, clf_result)

        if (idx + 1) % 500 == 0:
            logger.info(f"Processed {idx + 1}/{len(eval_obs)} initial classifications...")

    # Force all remaining observations in the buffer into groups
    groups = associator.flush_all()
    logger.info(f"Formed {len(groups)} distinct emitter groups from {len(eval_obs)} observations.")

    logger.info("Geolocating groups and preparing final payload...")
    for group in groups:
        # Geolocate the entire group using TDoA or multi-receiver RSSI
        geo_result = geo.geolocate(group.observations)
        
        # Fallback to None if geolocation fails completely
        lat = geo_result.latitude if geo_result else None
        lon = geo_result.longitude if geo_result else None
        
        # All individual observations within this group share the same predicted coordinates 
        # and benefit from the group's highest-confidence classification label
        for obs in group.observations:
            submissions.append({
                "observation_id": obs["observation_id"],
                "classification_label": group.classification_label,
                "confidence": group.classification_confidence,
                "estimated_latitude": lat,
                "estimated_longitude": lon,
            })

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
