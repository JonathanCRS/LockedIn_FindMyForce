"""
Find My Force — Main Entry Point
Orchestrates training and starts the dashboard server.
"""

import os
import sys
import logging
import argparse
import threading
import time
from pathlib import Path

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def cmd_server(args):
    """Start the dashboard server."""
    os.environ.setdefault("PORT", str(args.port))
    print(f"""
╔══════════════════════════════════════════════════════════╗
║           FIND MY FORCE — RF COP Dashboard               ║
║                                                          ║
║  Dashboard:  http://localhost:{args.port:<5}                   ║
║  API:        http://localhost:{args.port:<5}/api/status          ║
╚══════════════════════════════════════════════════════════╝
""")
    from server import app, socketio, initialize_system
    threading.Thread(target=initialize_system, daemon=True).start()
    socketio.run(app, host="0.0.0.0", port=args.port,
                 debug=args.debug, allow_unsafe_werkzeug=True)


def cmd_train(args):
    """Train the classifier on HDF5 data."""
    from classifier import SignalClassifier, load_training_data

    data_dir = ROOT_DIR / "data"
    hdf5_files = list(data_dir.glob("*.h5")) + list(data_dir.glob("*.hdf5"))

    if not hdf5_files:
        print(f"ERROR: No HDF5 files found in {data_dir}/")
        print("Download the training data and place it in the data/ directory.")
        sys.exit(1)

    hdf5_path = str(hdf5_files[0])
    print(f"Training on: {hdf5_path}")

    clf = SignalClassifier()
    X, y = load_training_data(hdf5_path)
    metrics = clf.train(X, y)
    clf.save()

    print("\n=== Training Results ===")
    print(f"F1 (macro):    {metrics['f1_macro']:.4f} ({metrics['f1_macro']*100:.1f}%)")
    print(f"Samples:       {metrics['n_samples']}")
    print(f"Classes:       {', '.join(metrics['classes'])}")
    print("\nPer-class scores:")
    for cls, scores in metrics.get('per_class', {}).items():
        print(f"  {cls:<25} F1={scores.get('f1-score', 0):.3f}  P={scores.get('precision', 0):.3f}  R={scores.get('recall', 0):.3f}")


def cmd_stream(args):
    """Connect to the live feed and print observations (debug mode)."""
    import json
    import requests

    api_url = os.getenv("API_URL", "https://findmyforce.online")
    api_key = os.getenv("API_KEY", "")

    if not api_key:
        print("ERROR: No API_KEY set in .env file")
        sys.exit(1)

    print(f"Connecting to {api_url}/feed/stream ...")

    resp = requests.get(
        f"{api_url}/feed/stream",
        headers={"X-API-Key": api_key},
        stream=True,
        timeout=60,
    )

    if resp.status_code != 200:
        print(f"ERROR: {resp.status_code} {resp.text}")
        sys.exit(1)

    print("Connected! Streaming observations (Ctrl+C to stop):\n")
    count = 0
    for line in resp.iter_lines(decode_unicode=True):
        if line and line.startswith("data: "):
            data = json.loads(line[6:])
            obs_id = data.get("observation_id", "?")
            rx = data.get("receiver_id", "?")
            rssi = data.get("rssi_dbm", 0)
            snr = data.get("snr_estimate_db", 0)
            count += 1
            print(f"[{count:04d}] {obs_id} | RX={rx} | RSSI={rssi:.1f}dBm | SNR={snr:.1f}dB")


def cmd_score(args):
    """Fetch and display team score."""
    from pipeline.feed_consumer import get_score
    score = get_score()
    if not score:
        print("Could not fetch score. Check API key and server status.")
        return

    print(f"\n{'='*50}")
        
    try:
        resp = requests.get(f"{api_url}/scores/me", headers={"X-API-Key": api_key})
        resp.raise_for_status()
        result = resp.json()
        
        print("\n=== CURRENT SCORE ===")
        print(f"Total: {result.get('total_score', 0):.1f}")
        print(f"Classification: {result.get('classification_score', 0):.1f}")
        print(f"Geolocation: {result.get('geolocation_score', 0):.1f}")
        print(f"Novelty: {result.get('novelty_score', 0):.1f}")
        print("=====================\n")
    except Exception as e:
        logger.error(f"Failed to fetch score: {e}")

def run_eval():
    """Run the official evaluation submission pipeline."""
    from pipeline.eval_runner import run_evaluation_pipeline
    logger.info("Starting evaluation pipeline...")
    run_evaluation_pipeline()

def main():
    parser = argparse.ArgumentParser(prog="findmyforce", description="Find My Force RF COP System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Server command
    parser_server = subparsers.add_parser("server", help="Start the dashboard server")
    parser_server.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser_server.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Train command
    parser_train = subparsers.add_parser("train", help="Train the ML classifier")

    # Stream command
    parser_stream = subparsers.add_parser("stream", help="Stream live observations (debug)")

    # Score command
    parser_score = subparsers.add_parser("score", help="Fetch team score")

    # Eval command
    parser_eval = subparsers.add_parser("eval", help="Run the official evaluation submission pipeline")

    args = parser.parse_args()

    if args.command == "server":
        cmd_server(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "stream":
        cmd_stream(args)
    elif args.command == "score":
        get_score()
    elif args.command == "eval":
        run_eval()


if __name__ == "__main__":
    main()
