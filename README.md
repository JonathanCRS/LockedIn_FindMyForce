# LockedIn_FindMyForce
Find My Force pipeline for RF classification, emitter geolocation, track management, and live COP visualization.

## System Overview
This project implements the full hackathon flow:
1. Data tidying from `training_data.hdf5`
2. RF classifier training on friendly labels
3. Unknown-signal handling with faction/type taxonomy
4. Observation association across receivers
5. RSSI-based geolocation + uncertainty
6. Track lifecycle management
7. Live web COP (Leaflet map + realtime table)

### Compact Technical Diagram
```text
        training_data.hdf5
               |
               v
      [training.py] tidy dataset
      - training_tidy.csv
      - training_vectors.npz
               |
               v
 [baseline_classifier.py] RandomForest(257 features)
 p_max = max P(class|x)
 unknown if p_max < tau
               |
               v
      Live stream /feed/stream
               |
               v
   classify each observation (friendly/unknown)
   unknown -> hostile/civilian subtype via IQ heuristics
               |
               v
 association window (time-based grouping across RX)
               |
               v
 geolocation (RSSI -> distance):
 d = d_ref * 10^((RSSI_ref - RSSI)/(10*n))
 solve (x,y): min Σ wi (||[x,y]-[xi,yi]|| - di)^2
 (damped Gauss-Newton WLS)
               |
               v
 uncertainty ~= RMSE(residuals) + 2*sigma_rssi
               |
               v
 track manager (EMA smoothing, staleness pruning)
 lat_t = (1-a)lat_(t-1) + a*lat_new
 lon_t = (1-a)lon_(t-1) + a*lon_new
               |
               v
        COP (Leaflet frontend)
  map markers + uncertainty rings + live table
```

## Repository Structure
- `training.py`: Converts raw HDF5 into tidy training artifacts.
- `baseline_classifier.py`: Trains the classifier (`RandomForest`/`ExtraTrees`) and saves model bundle.
- `cop_live.py`: Live Stage-2 backend (stream ingest, classify, associate, geolocate, track, API).
- `templates/cop.html`: COP page template.
- `static/cop.css`: COP styling.
- `static/cop.js`: COP frontend map/table logic.
- `data/training_tidy.csv`: Tidy feature table.
- `data/training_vectors.npz`: Full 256-element IQ vectors.
- `models/baseline_model.joblib`: Trained model bundle used by live COP.

## Stage 1: Data Preparation
Method:
- Parse HDF5 tuple keys into labels and metadata:
  - `modulation`, `emitter_type`, `snr_db`, `source_index`
- Validate vectors (`float32`, finite, non-empty)
- Export:
  - tabular stats (`mean/std/min/max/energy`)
  - raw vectors for model training

Command:
```powershell
py -3 .\training.py --input training_data.hdf5 --output-dir data
```

Expected output:
- `data/training_tidy.csv` (~41k rows + header)
- `data/training_vectors.npz`

## Stage 1: Classification Method
Model:
- `RandomForestClassifier` on full IQ feature vector:
  - 256 IQ values + SNR feature
- Saved in `models/baseline_model.joblib`

Unknown detection:
- Confidence-based threshold from train-set probability quantile
- Low-confidence predictions are marked as out-of-distribution (`unknown`)

Command:
```powershell
py -3 .\baseline_classifier.py train --model-type randomforest --input-csv data/training_tidy.csv --vectors-npz data/training_vectors.npz --feature-mode full --output-dir models
```

## Friendly / Hostile / Civilian Categorization
Friendly (from training modulation mapping):
- `FMCW -> Radar-Altimeter`
- `BPSK -> Satcom`
- `ASK -> short-range`

Unknown split:
- `hostile` vs `civilian` via signal heuristics (amplitude/phase/spectral features)
- Hostile subtypes:
  - `Airborne-detection`
  - `Airborne-range`
  - `Air-Ground-MTI`
  - `EW-Jammer`
- Civilian subtype:
  - `AM radio` (`AM-DSB`)

Implementation note:
- Hostile/civilian subtype logic is heuristic because hostile/civilian labels are not in training data.

## Stage 2: Observation Association
Method:
- Stream observations from `/feed/stream`
- Group observations in a short time window (`assoc-window-ms`)
- Avoid duplicate receiver entries in same group
- Finalize groups by elapsed ingest time
- Supports 2-receiver and 3+ receiver groups

## Stage 2: Geolocation
Method:
- Load receiver geometry from `/config/receivers`
- Load propagation params from `/config/pathloss`
- Convert RSSI to distance using path-loss model
- Estimate source with weighted nonlinear least squares (local XY)
- Convert back to lat/lon
- Report uncertainty from residual error
- 2-receiver fallback uses weighted centroid with larger uncertainty

### Mathematical Details
Coordinate model:
- Receivers are given in latitude/longitude.
- For local optimization, we use a local tangent-plane approximation around group centroid `(lat0, lon0)`:
  - `x = (lon - lon0) * 111320 * cos(lat0)`
  - `y = (lat - lat0) * 111320`

Path-loss to distance:
- Using scenario parameters `rssi_ref_dbm`, `d_ref_m`, `path_loss_exponent = n`:
  - `RSSI = RSSI_ref - 10*n*log10(d/d_ref)`
  - Rearranged:
  - `d = d_ref * 10^((RSSI_ref - RSSI)/(10*n))`

Optimization objective (3+ receivers):
- For receiver `i` at `(xi, yi)` with estimated distance `di`, unknown emitter `(x, y)`:
  - `ri(x,y) = sqrt((x-xi)^2 + (y-yi)^2) - di`
- Weighted least squares objective:
  - `min Σ wi * ri(x,y)^2`
- We solve iteratively with Gauss-Newton style updates:
  - Jacobian row:
  - `Ji = [ (x-xi)/ri_geom , (y-yi)/ri_geom ]`, where `ri_geom = sqrt((x-xi)^2 + (y-yi)^2)`
  - Normal equation step:
  - `(J^T W J + λI) Δ = J^T W r`
  - Update:
  - `[x, y] <- [x, y] - Δ`
  - Small damping `λ` stabilizes inversion.
- Initialization:
  - weighted average of receiver coordinates (weights from RSSI strength).

Receiver weighting:
- Stronger signals are trusted more:
  - `wi = max(0.1, 10^((RSSI_i + 120)/20))`

Uncertainty estimate:
- After convergence:
  - residual vector `r = [ri]`
  - `RMSE = sqrt(mean(r^2))`
- Reported uncertainty radius:
  - `uncertainty_m = RMSE + 2 * rssi_noise_std_db`
- This is an operational confidence proxy (not full covariance ellipse).

2-receiver fallback:
- Exact trilateration is underconstrained with 2 circles under noise.
- We use weighted midpoint in lat/lon:
  - `lat = (w1*lat1 + w2*lat2)/(w1+w2)`
  - `lon = (w1*lon1 + w2*lon2)/(w1+w2)`
- Assign conservative fixed uncertainty (`~700 m`).

Algorithm used in code:
- `Weighted nonlinear least squares` (Gauss-Newton style, damped, fixed iterations).
- Not pure closed-form trilateration; more robust under noisy RSSI.

## Track Management
Method:
- Create/update tracks from finalized groups
- Match by faction + signal type + spatial proximity
- Smooth position/confidence over time
- Age out stale tracks

### Track Update Mathematics
Association to existing track:
- Candidate track must satisfy:
  - same `faction`
  - same `signal_type`
  - distance to new estimate `< track_match_m`

Distance metric:
- Haversine distance between track position and new estimate.

Smoothing:
- Exponential moving average with factor `alpha = 0.35`:
  - `lat_t = (1-alpha)*lat_{t-1} + alpha*lat_new`
  - `lon_t = (1-alpha)*lon_{t-1} + alpha*lon_new`
  - `conf_t = (1-alpha)*conf_{t-1} + alpha*conf_new`
  - same for uncertainty.

Staleness:
- Track removed if:
  - `now - last_seen > track_stale_s`

## Classification / Unknown Decision Math
Friendly classifier:
- `RandomForestClassifier` on `257` features:
  - 256 IQ samples + 1 SNR feature.

Unknown detection:
- Let `p_max = max_k P(class=k | x)` from classifier softmax-like output.
- Threshold `tau` is chosen as a train-set quantile:
  - `tau = Quantile(p_max_train, q)` where `q = unknown_quantile`
- Decision:
  - if `p_max < tau` -> `unknown`
  - else -> friendly class argmax.

Taxonomy refinement (unknown -> hostile/civilian subtype):
- Hand-crafted signal features from IQ:
  - amplitude crest factor
  - kurtosis
  - duty-cycle proxy
  - zero-crossing rate
  - phase-derivative std
  - spectral flatness / high-frequency energy ratio
- Rule-based decision boundaries map unknown into:
  - hostile subtypes (`Airborne-detection`, `Airborne-range`, `Air-Ground-MTI`, `EW-Jammer`)
  - civilian subtype (`AM radio`).

## Live COP Frontend
Features:
- Leaflet live map with receiver nodes and track markers
- Uncertainty circles around track estimates
- Tactical coloring by faction:
  - Friendly: green
  - Hostile: red
  - Civilian: orange
- Realtime latest-observations table
- Track popups include faction, signal type, modulation, assessment, confidence

## Run Live COP
From project root:

```powershell
cd /d "D:\Hackathon pertama\LockedIn_FindMyForce"
```

Set API key:

```powershell
$env:FINDMYFORCE_API_KEY="YOUR_API_KEY"
```

Start server:

```powershell
py -3 .\cop_live.py --api-key $env:FINDMYFORCE_API_KEY --host localhost --port 8080
```

Open:
- `http://localhost:8080`

## Useful Tuning Flags
`cop_live.py`:
- `--assoc-window-ms` association tolerance window
- `--group-finalize-ms` group finalization delay
- `--track-match-m` distance threshold to merge with existing track
- `--track-stale-s` track aging timeout

Example:
```powershell
py -3 .\cop_live.py --api-key $env:FINDMYFORCE_API_KEY --assoc-window-ms 350 --track-stale-s 120
```

## Troubleshooting
- `can't open file ...cop_live.py`: run from repo folder or pass absolute path.
- Blank/old UI: hard refresh browser (`Ctrl + F5`).
- No model file: retrain with `baseline_classifier.py train`.
- Map updates but no tracks: increase `--assoc-window-ms` (e.g., 350).
