"""
Signal Classifier Module
Trains on labeled IQ waveform data and classifies signals.
Supports both known friendly signal classification and
out-of-distribution (hostile/civilian) anomaly detection.
"""

import numpy as np
import joblib
import os
import logging
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.covariance import EllipticEnvelope
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# ─── Signal catalog ───────────────────────────────────────────────────────────
FRIENDLY_LABELS = {
    "Radar-Altimeter",
    "Satcom",
    "short-range",
}

# Note: The server uses these exact labels (case-insensitive)
HOSTILE_LABELS = {
    "Airborne-detection",
    "Airborne-range",
    "Air-Ground-MTI",
    "EW-Jammer",
}

CIVILIAN_LABELS = {
    "AM radio",
}

ALL_KNOWN_LABELS = FRIENDLY_LABELS | HOSTILE_LABELS | CIVILIAN_LABELS

MODEL_DIR = Path(__file__).parent.parent / "models"


def extract_features(iq_snapshot: list) -> np.ndarray:
    """
    Extract rich feature vector from 256-element IQ snapshot.
    Elements 0-127: I components
    Elements 128-255: Q components
    Sample rate: 10 MS/s
    """
    iq = np.array(iq_snapshot, dtype=np.float32)
    if len(iq) != 256:
        iq = np.pad(iq, (0, max(0, 256 - len(iq))))[:256]

    I = iq[:128]
    Q = iq[128:]

    # Complex representation
    z = I + 1j * Q

    # ── Amplitude/envelope features ─────────────────────────────────
    amplitude = np.abs(z)
    amp_mean = np.mean(amplitude)
    amp_std = np.std(amplitude)
    amp_max = np.max(amplitude)
    amp_min = np.min(amplitude)
    amp_range = amp_max - amp_min
    amp_skew = _skewness(amplitude)
    amp_kurt = _kurtosis(amplitude)
    crest_factor = amp_max / (amp_mean + 1e-10)

    # ── Phase features ───────────────────────────────────────────────
    phase = np.angle(z)
    phase_diff = np.diff(np.unwrap(phase))
    phase_mean = np.mean(phase)
    phase_std = np.std(phase)
    phase_diff_std = np.std(phase_diff)
    phase_diff_mean = np.mean(phase_diff)

    # ── Instantaneous frequency ──────────────────────────────────────
    inst_freq = phase_diff / (2 * np.pi * 1e-7)  # Hz (dt = 1/10MHz = 100ns)
    freq_mean = np.mean(inst_freq)
    freq_std = np.std(inst_freq)
    freq_range = np.ptp(inst_freq)

    # ── Power / energy ───────────────────────────────────────────────
    power = amplitude ** 2
    total_power = np.sum(power)
    i_power = np.sum(I ** 2)
    q_power = np.sum(Q ** 2)
    power_ratio = i_power / (q_power + 1e-10)

    # ── Spectral features (FFT) ──────────────────────────────────────
    spectrum = np.abs(np.fft.fft(z)) ** 2
    spectrum_norm = spectrum / (np.sum(spectrum) + 1e-10)
    freqs = np.fft.fftfreq(128, d=1e-7)

    spec_mean = np.mean(spectrum_norm)
    spec_std = np.std(spectrum_norm)
    spec_entropy = -np.sum(spectrum_norm * np.log2(spectrum_norm + 1e-10))
    peak_freq_idx = np.argmax(spectrum_norm[:64])
    spectral_centroid = np.sum(np.arange(len(spectrum_norm)) * spectrum_norm) / (np.sum(spectrum_norm) + 1e-10)
    spectral_flatness = np.exp(np.mean(np.log(spectrum + 1e-10))) / (np.mean(spectrum) + 1e-10)

    # Top 5 spectral peaks
    top5_peaks = np.sort(spectrum_norm[:64])[-5:][::-1]

    # ── Pulsed signal features ───────────────────────────────────────
    threshold = amp_mean * 0.5
    above = amplitude > threshold
    duty_cycle = np.mean(above)
    # Zero crossings of amplitude envelope (proxy for pulse transitions)
    amp_centered = amplitude - amp_mean
    zcr_amp = np.sum(np.diff(np.sign(amp_centered)) != 0) / len(amplitude)

    # ── BPSK detection (phase transitions of ~180°) ──────────────────
    phase_jumps_180 = np.sum(np.abs(phase_diff) > np.pi * 0.7) / len(phase_diff)

    # ── FMCW detection (linear frequency sweep) ──────────────────────
    freq_linearity = np.corrcoef(np.arange(len(inst_freq)), inst_freq)[0, 1] if len(inst_freq) > 1 else 0.0

    # ── ASK detection (amplitude on/off pattern) ─────────────────────
    ask_ratio = amp_std / (amp_mean + 1e-10)

    # ── Higher-order statistics ──────────────────────────────────────
    i_std = np.std(I)
    q_std = np.std(Q)
    iq_corr = np.corrcoef(I, Q)[0, 1] if i_std > 0 and q_std > 0 else 0.0

    # ── Zero-crossing rate of raw I and Q ────────────────────────────
    zcr_i = np.sum(np.diff(np.sign(I)) != 0) / len(I)
    zcr_q = np.sum(np.diff(np.sign(Q)) != 0) / len(Q)

    # ── Noise floor estimate ─────────────────────────────────────────
    sorted_amp = np.sort(amplitude)
    noise_floor = np.mean(sorted_amp[:16])  # Bottom 12.5% = noise

    features = np.array([
        # Amplitude stats (8)
        amp_mean, amp_std, amp_max, amp_min, amp_range,
        amp_skew, amp_kurt, crest_factor,
        # Phase stats (5)
        phase_mean, phase_std,
        phase_diff_std, phase_diff_mean, phase_jumps_180,
        # Frequency stats (4)
        freq_mean, freq_std, freq_range, freq_linearity,
        # Power stats (4)
        total_power, i_power, q_power, power_ratio,
        # Spectral stats (6)
        spec_mean, spec_std, spec_entropy,
        spectral_centroid, spectral_flatness, peak_freq_idx,
        # Top 5 spectral peaks (5)
        *top5_peaks,
        # Pulsed/modulation features (4)
        duty_cycle, zcr_amp, ask_ratio, freq_linearity,
        # IQ correlation stats (4)
        i_std, q_std, iq_corr, noise_floor,
        # ZCR (2)
        zcr_i, zcr_q,
    ], dtype=np.float32)

    return features


def _skewness(x: np.ndarray) -> float:
    """Compute skewness of array."""
    mu = np.mean(x)
    sig = np.std(x)
    if sig == 0:
        return 0.0
    return float(np.mean(((x - mu) / sig) ** 3))


def _kurtosis(x: np.ndarray) -> float:
    """Compute kurtosis of array."""
    mu = np.mean(x)
    sig = np.std(x)
    if sig == 0:
        return 0.0
    return float(np.mean(((x - mu) / sig) ** 4)) - 3.0


class SignalClassifier:
    """
    Two-stage signal classifier:
    1. Friendly classifier: identifies known friendly signal types
    2. Anomaly detector: flags out-of-distribution (hostile/civilian) signals
    """

    def __init__(self):
        self.friendly_classifier = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.anomaly_detector = None
        self.is_trained = False
        self._ood_threshold = -0.1  # OneClassSVM decision threshold
        MODEL_DIR.mkdir(exist_ok=True)

    def train(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Train on labeled friendly IQ data.
        X: (N, feature_dim) feature matrix
        y: (N,) string labels
        Returns training metrics dict.
        """
        logger.info(f"Training on {len(X)} samples, {len(np.unique(y))} classes")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Encode labels
        y_enc = self.label_encoder.fit_transform(y)

        # Split for evaluation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        # Train multi-class friendly classifier (Random Forest + GB ensemble)
        self.friendly_classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.friendly_classifier.fit(X_tr, y_tr)

        # Evaluate
        y_pred = self.friendly_classifier.predict(X_val)
        f1 = f1_score(y_val, y_pred, average="macro")
        report = classification_report(
            y_val, y_pred,
            target_names=self.label_encoder.classes_,
            output_dict=True
        )
        logger.info(f"Friendly classifier F1 (macro): {f1:.3f}")

        # Train One-Class SVM anomaly detector on friendly data only
        # This learns the "friendly" manifold; unknown signals will be rejected
        self.anomaly_detector = OneClassSVM(
            kernel="rbf",
            nu=0.05,  # 5% expected outlier fraction
            gamma="scale",
        )
        self.anomaly_detector.fit(X_tr)

        # Calibrate threshold on validation set
        scores = self.anomaly_detector.decision_function(X_val)
        # Friendly samples should be +; we set threshold at 5th percentile
        self._ood_threshold = float(np.percentile(scores, 10))
        logger.info(f"OOD threshold calibrated at: {self._ood_threshold:.4f}")

        self.is_trained = True
        return {
            "f1_macro": round(f1, 4),
            "n_samples": len(X),
            "classes": list(self.label_encoder.classes_),
            "per_class": {k: v for k, v in report.items() if k in self.label_encoder.classes_},
        }

    def predict(self, iq_snapshot: list) -> dict:
        """
        Classify a single IQ snapshot.
        Returns dict with label, confidence, is_friendly, is_anomaly.
        """
        features = extract_features(iq_snapshot)
        return self.predict_features(features.reshape(1, -1))[0]

    def predict_features(self, X: np.ndarray) -> list:
        """
        Classify a batch of pre-extracted features.
        X: (N, feature_dim)
        Returns list of dicts.
        """
        if not self.is_trained:
            return [self._unknown_result() for _ in range(len(X))]

        X_scaled = self.scaler.transform(X)

        # Anomaly detection
        ood_scores = self.anomaly_detector.decision_function(X_scaled)
        is_anomaly = ood_scores < self._ood_threshold

        # Friendly classification (always run to get probabilities)
        proba = self.friendly_classifier.predict_proba(X_scaled)
        pred_idx = np.argmax(proba, axis=1)
        pred_labels = self.label_encoder.inverse_transform(pred_idx)
        confidences = proba[np.arange(len(proba)), pred_idx]

        results = []
        for i in range(len(X)):
            # Reconstruct feature dict for heuristic classification
            feat = X[i]
            features_dict = {
                "amp_std": feat[1],
                "freq_mean": feat[13],
                "total_power": feat[17],
                "spectral_flatness": feat[25],
                "duty_cycle": feat[32],
                "ask_ratio": feat[34],
                "freq_linearity": feat[35]
            }

            if is_anomaly[i]:
                # Out-of-distribution: likely hostile or civilian
                # Use confidence to determine how "unknown" it is
                friendly_conf = float(confidences[i])
                ood_conf = float(1.0 - (ood_scores[i] - self._ood_threshold) /
                                  (abs(self._ood_threshold) + 1e-10))
                ood_conf = max(0.5, min(0.99, ood_conf))

                results.append({
                    "label": "unknown",
                    "confidence": round(ood_conf, 3),
                    "is_friendly": False,
                    "is_anomaly": True,
                    "friendly_guess": str(pred_labels[i]),
                    "friendly_confidence": round(friendly_conf, 3),
                    "ood_score": round(float(ood_scores[i]), 4),
                    "features": features_dict,
                })
            else:
                results.append({
                    "label": str(pred_labels[i]),
                    "confidence": round(float(confidences[i]), 3),
                    "is_friendly": True,
                    "is_anomaly": False,
                    "friendly_guess": str(pred_labels[i]),
                    "friendly_confidence": round(float(confidences[i]), 3),
                    "ood_score": round(float(ood_scores[i]), 4),
                    "features": features_dict,
                })

        return results

    def _unknown_result(self) -> dict:
        return {
            "label": "unknown",
            "confidence": 0.5,
            "is_friendly": False,
            "is_anomaly": True,
            "friendly_guess": None,
            "friendly_confidence": 0.0,
            "ood_score": 0.0,
        }

    def save(self, path: str = None):
        """Save model to disk."""
        if path is None:
            path = str(MODEL_DIR / "classifier.joblib")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "friendly_classifier": self.friendly_classifier,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "anomaly_detector": self.anomaly_detector,
            "ood_threshold": self._ood_threshold,
            "is_trained": self.is_trained,
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str = None) -> bool:
        """Load model from disk. Returns True if successful."""
        if path is None:
            path = str(MODEL_DIR / "classifier.joblib")
        if not os.path.exists(path):
            logger.warning(f"No saved model found at {path}")
            return False
        data = joblib.load(path)
        self.friendly_classifier = data["friendly_classifier"]
        self.scaler = data["scaler"]
        self.label_encoder = data["label_encoder"]
        self.anomaly_detector = data["anomaly_detector"]
        self._ood_threshold = data["ood_threshold"]
        self.is_trained = data["is_trained"]
        logger.info(f"Model loaded from {path}")
        return True


def load_training_data(hdf5_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load labeled IQ training data from HDF5 file.
    The HDF5 file is expected to have datasets where the key is a string
    representation of a tuple (label, instance_id), e.g., "('802.11a', 'instance_0')".
    Returns (X_features, y_labels) arrays.
    """
    import h5py
    import ast # For safely evaluating string-encoded tuples
    logger.info(f"Loading training data from {hdf5_path}")

    X_list = []
    y_list = []

    with h5py.File(hdf5_path, "r") as f:
        for key in f.keys():
            try:
                # Keys are 4-element tuples: (modulation, label, snr_int, sample_idx)
                # e.g. "('fmcw', 'Radar-Altimeter', 8, 0)"
                t = ast.literal_eval(key)
                if not isinstance(t, tuple) or len(t) < 2:
                    continue
                label = str(t[1])
            except (ValueError, SyntaxError):
                continue

            dataset = f[key]
            data = dataset[()]

            if data.ndim == 2 and data.shape[1] == 256:
                # Matrix of samples
                for sample in data:
                    X_list.append(sample)
                    y_list.append(label)
            elif data.ndim == 1 and len(data) == 256:
                # Single sample
                X_list.append(data)
                y_list.append(label)
            else:
                logger.warning(f"Skipping dataset '{key}': unexpected shape {data.shape}")

    if not X_list:
        raise ValueError("No data found in HDF5 file")

    X_raw = np.array(X_list, dtype=np.float32)
    y_raw = np.array(y_list, dtype=str)

    # Extract features
    logger.info(f"Extracting features from {len(X_raw)} samples...")
    X_feat = np.array([extract_features(x) for x in X_raw])

    logger.info(f"Loaded {len(X_feat)} samples, shape={X_feat.shape}")
    logger.info(f"Class distribution: {dict(zip(*np.unique(y_raw, return_counts=True)))}")


    return X_feat, y_raw
