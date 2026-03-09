# 🛰️ Find My Force — RF COP System
> **🥈 2nd Place Winner — UBC Defence Tech Hackathon 2026**

A high-performance RF Signal Classification and Geolocation Common Operating Picture (COP) developed for the **LockedIn** challenge.

## 🛠️ Core Technologies
- **Deep Learning**: 1D-CNN (PyTorch) for raw IQ feature extraction
- **Machine Learning**: Scikit-Learn Hybrid Ensemble (HistGradientBoosting + MLP + OC-SVM)
- **Geolocation**: Multi-receiver TDoA & RSSI Trilateration with Kalman Filter smoothing
- **Backend**: Flask + Socket.IO for real-time data orchestration
- **Frontend**: Leaflet.js + custom tactical HUD (Glassmorphism & Neon UI)

## ⚡ Quick Start

```bash
# 1. Set up virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Set your API key
echo "API_KEY=your-team-api-key" >> .env

# 3. Place training data in data/ directory
#    Download the HDF5 file and put it at data/training.h5

# 4. Train the Hybrid Classifier
python3 main.py train

# 5. Start the tactical dashboard
python3 main.py server --port 5050
# → Open http://localhost:5050
```

## 🏗️ Architecture

```
LockedIn_FindMyForce/
├── classifier/
│   ├── signal_classifier.py    # Hybrid Engine: Deep 1D-CNN + Expert Features + Ensemble
│   └── __init__.py
├── pipeline/
│   ├── geolocator.py           # RSSI & TDoA trilateration + Kalman filtering
│   ├── track_manager.py        # Persistent track lifecycle (Tentative -> Confirmed)
│   ├── associator.py           # Multi-receiver pulse grouping (Temporal/IQ Similarity)
│   ├── feed_consumer.py        # Real-time SSE stream consumption
│   └── __init__.py
├── dashboard/
│   ├── index.html              # Tactical HUD (Tailwind + Custom CSS)
│   ├── style.css               # Premium "Find My Force" HUD styling
│   └── app.js                  # Socket.IO & Leaflet map orchestration
├── data/                       # RF training data (HDF5)
├── models/                     # Saved model artifacts (.joblib + CNN weights)
├── server.py                   # High-concurrency Flask/Socket.IO backend
├── main.py                     # Unified CLI entry point
├── requirements.txt
└── .env                        # Configuration (Team API Key)
```

## CLI Commands

```bash
# Start dashboard server
python3 main.py server [--port 5050] [--debug]

# Train classifier on HDF5 data
python3 main.py train

# Stream live feed observations (debug)
python3 main.py stream

# Fetch team score
python3 main.py score
```

## ML Pipeline

### Hybrid Signal Classifier
- **Deep 1D-CNN**: Extracts 64-dimensional latent features directly from raw 256-sample IQ snapshots. Architecture includes multiple Conv1D layers, BatchNorm, and Dropout for robust representation learning.
- **Feature Extraction (Manual)**: 42 expert-engineered features (spectral entropy, kurtosis, Phase Histogram, LPC coefficients, etc.).
- **Hybrid Ensemble**: Concatenates deep CNN features with manual features, followed by a **Voting Ensemble** of HistGradientBoosting and MLP.
- **One-Class SVM**: Trained on high-dimensional hybrid features of friendly data — detects hostile/civilian signals as anomalies.

### Training Procedure
1. Load HDF5 data → extract raw IQ and 42 manual feature vectors.
2. Train 1D-CNN on raw IQ to minimize Cross-Entropy loss for known classes.
3. Freeze CNN and use as a deep feature extractor.
4. Scale combined manual + deep features (Quantile Transformer).
5. Train the Multi-class Ensemble on the hybrid feature space.
6. Calibrate anomaly threshold at 15th percentile of friendly OOD scores.
7. Save hybrid model bundle to `models/classifier.joblib`.

## Geolocation Engine

### RSSI Trilateration
- Converts RSSI to distance via path-loss model: `d = d_ref × 10^((RSSI_ref - RSSI) / (10 × n))`
- Nonlinear least-squares optimization (scipy `least_squares`)
- SNR-weighted receiver contributions
- Fallback to 2-receiver and single-receiver modes

### TDoA Multilateration
- Uses time-of-arrival differences to compute hyperbolic position fixes
- Linearized initial estimate, then nonlinear refinement
- Requires ≥3 receivers with valid ToA

### Hybrid Fusion
- 70% TDoA + 30% RSSI weighted position average when both methods available
- GDOP (Geometric Dilution of Precision) estimation

### Kalman Filter Tracking
- State: [x, y, vx, vy] (position + velocity)
- Constant velocity motion model
- Measurement update weighted by geolocation uncertainty

## Track Management

- **TENTATIVE** → **CONFIRMED** after 2 updates
- **CONFIRMED** → **COASTING** after 20s without updates
- **COASTING** → **LOST** after 60s
- Kalman-smoothed position estimates
- Classification confidence EMA fusion across observations

## Observation Association

Groups independent receiver observations into single-emitter groups using:
1. **Temporal gate**: observations within 500ms window
2. **Classification gate**: same signal type (or at least one "unknown")
3. **IQ similarity gate**: cosine similarity > 0.5

## Scoring Strategy

- **Classification (40%)**: Correctly identifies friendly types AND detects hostile/civilian as anomalies
- **Geolocation (30%)**: RSSI + TDoA hybrid with Kalman smoothing minimizes CEP
- **Novelty (30%)**: OC-SVM flags unknowns; use intelligence briefing labels for hostile sub-types

### Hostile Signal Labels (for eval submission)
- `Airborne-detection` — Airborne surveillance radar
- `Airborne-range` — Airborne range-finding radar
- `Air-Ground-MTI` — Air-to-ground moving target indicator
- `EW-Jammer` — Electronic warfare / broadband jammer
- `AM radio` — Commercial AM radio broadcast

## Dashboard Features

- **Leaflet map** with dark tactical overlay
- Real-time track markers color-coded by affiliation (green=friendly, red=hostile, orange=unknown, blue=civilian)
- Uncertainty radius circles around each track
- Track history path polylines
- Receiver station markers with tooltip details
- Live observation feed panel
- Track detail panel with classification confidence, GDOP, velocity
- Track filter by affiliation
- Score display with per-component breakdown
- One-click eval submission button
- Socket.IO real-time updates (<100ms latency)
