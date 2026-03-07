import argparse
import json
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import requests
from flask import Flask, jsonify, render_template


@dataclass
class Observation:
    observation_id: str
    event_timestamp: datetime
    ingest_timestamp: datetime
    receiver_id: str
    rssi_dbm: float
    snr_db: float
    iq_snapshot: np.ndarray
    predicted_label: str
    known_label: str
    confidence: float
    unknown_score: float
    faction: str
    signal_type: str
    modulation: str
    assessment: str


@dataclass
class Group:
    group_id: int
    class_key: str
    created_at: datetime
    last_at: datetime
    observations: list[Observation] = field(default_factory=list)


@dataclass
class Track:
    track_id: str
    label: str
    faction: str
    modulation: str
    assessment: str
    lat: float
    lon: float
    uncertainty_m: float
    confidence: float
    last_seen: datetime
    hit_count: int


def parse_iso(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


def ensure_utc_now() -> datetime:
    return datetime.now(timezone.utc)


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def ll_to_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    lat_scale = 111320.0
    lon_scale = 111320.0 * math.cos(math.radians(lat0))
    x = (lon - lon0) * lon_scale
    y = (lat - lat0) * lat_scale
    return x, y


def xy_to_ll(x: float, y: float, lat0: float, lon0: float) -> tuple[float, float]:
    lat_scale = 111320.0
    lon_scale = 111320.0 * math.cos(math.radians(lat0))
    lat = lat0 + y / lat_scale
    lon = lon0 + x / max(lon_scale, 1e-9)
    return lat, lon


class LiveCOP:
    def __init__(
        self,
        api_base: str,
        api_key: str,
        model_path: str,
        assoc_window_ms: int,
        group_finalize_ms: int,
        track_match_m: float,
        track_stale_s: float,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.assoc_window_s = assoc_window_ms / 1000.0
        self.group_finalize_s = group_finalize_ms / 1000.0
        self.track_match_m = track_match_m
        self.track_stale_s = track_stale_s

        self.bundle: dict[str, Any] = joblib.load(model_path)
        self.model = self.bundle["model"]
        self.feature_mode = self.bundle.get("feature_mode", "full")
        self.unknown_threshold = float(self.bundle.get("unknown_threshold", 0.5))

        self.lock = threading.Lock()
        self.running = False
        self.worker: threading.Thread | None = None

        self.receivers: dict[str, dict[str, Any]] = {}
        self.pathloss: dict[str, float] = {}

        self.group_seq = 1
        self.track_seq = 1
        self.groups: list[Group] = []
        self.tracks: list[Track] = []
        self.latest_obs: list[dict[str, Any]] = []

        self.friendly_map = {
            "fmcw": ("friendly", "Radar-Altimeter", "FMCW", "Navigation radar on friendly UAVs"),
            "bpsk": ("friendly", "Satcom", "BPSK", "Satellite communications link"),
            "ask": ("friendly", "short-range", "ASK", "Low-power telemetry from friendly UGVs"),
        }

    def headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key}

    def load_config(self) -> None:
        r1 = requests.get(f"{self.api_base}/config/receivers", headers=self.headers(), timeout=15)
        r1.raise_for_status()
        data = r1.json()
        self.receivers = {item["receiver_id"]: item for item in data["receivers"]}

        r2 = requests.get(f"{self.api_base}/config/pathloss", headers=self.headers(), timeout=15)
        r2.raise_for_status()
        self.pathloss = r2.json()

    def start(self) -> None:
        self.load_config()
        self.running = True
        self.worker = threading.Thread(target=self._run_stream_loop, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.running = False
        if self.worker is not None:
            self.worker.join(timeout=3)

    def _classify(self, iq_snapshot: np.ndarray, snr_db: float) -> tuple[str, str, float, float]:
        iq = iq_snapshot.astype(np.float32).reshape(1, -1)
        if self.feature_mode == "full":
            x = np.concatenate([iq, np.asarray([[snr_db]], dtype=np.float32)], axis=1)
        else:
            mean = float(np.mean(iq_snapshot))
            std = float(np.std(iq_snapshot))
            min_v = float(np.min(iq_snapshot))
            max_v = float(np.max(iq_snapshot))
            energy = float(np.mean(np.square(iq_snapshot)))
            x = np.asarray([[snr_db, mean, std, min_v, max_v, energy]], dtype=np.float32)

        proba = self.model.predict_proba(x)[0]
        best_idx = int(np.argmax(proba))
        known_label = str(self.model.classes_[best_idx])
        conf = float(proba[best_idx])
        is_unknown = conf < self.unknown_threshold
        pred = "unknown" if is_unknown else known_label
        unknown_score = float(self.unknown_threshold / max(conf, 1e-6))
        return pred, known_label, conf, unknown_score

    def _extract_features(self, iq_snapshot: np.ndarray) -> dict[str, float]:
        i = iq_snapshot[:128]
        q = iq_snapshot[128:]
        amp = np.sqrt(i * i + q * q)

        mu = float(np.mean(amp))
        sigma = float(np.std(amp) + 1e-6)
        crest = float(np.max(amp) / max(mu, 1e-6))
        kurt = float(np.mean(((amp - mu) / sigma) ** 4))
        duty = float(np.mean(amp > (mu + sigma)))

        zc_i = float(np.mean(np.sign(i[1:]) != np.sign(i[:-1])))

        phase = np.unwrap(np.arctan2(q, i))
        phase_std = float(np.std(np.diff(phase)))

        spec = np.abs(np.fft.rfft(amp - np.mean(amp))) ** 2
        spec = spec + 1e-12
        spec_n = spec / np.sum(spec)
        flatness = float(np.exp(np.mean(np.log(spec_n))) / np.mean(spec_n))
        high_ratio = float(np.sum(spec_n[len(spec_n) // 2 :]))

        return {
            "crest": crest,
            "kurtosis": kurt,
            "duty": duty,
            "zc_i": zc_i,
            "phase_std": phase_std,
            "flatness": flatness,
            "high_ratio": high_ratio,
        }

    def _taxonomy(self, pred: str, known: str, snr_db: float, unknown_score: float, iq_snapshot: np.ndarray) -> tuple[str, str, str, str]:
        if pred != "unknown" and known in self.friendly_map:
            return self.friendly_map[known]

        f = self._extract_features(iq_snapshot)

        # Civilian AM profile: low phase dynamics, low high-frequency energy, moderate crest.
        is_civilian_am = (
            f["phase_std"] < 0.55
            and f["high_ratio"] < 0.33
            and 1.1 < f["crest"] < 3.0
            and snr_db < 18.0
        )
        if is_civilian_am and unknown_score < 1.8:
            return ("civilian", "AM radio", "AM-DSB", "Commercial AM radio broadcast")

        # Hostile heuristics from intelligence profile.
        if f["flatness"] > 0.72 and f["high_ratio"] > 0.52 and f["phase_std"] > 0.9:
            return ("hostile", "EW-Jammer", "Jamming", "Electronic warfare / broadband jammer")

        if f["crest"] > 3.0 or f["duty"] < 0.30:
            if f["zc_i"] > 0.34:
                return ("hostile", "Air-Ground-MTI", "Pulsed", "Air-to-ground moving target indicator radar")
            if snr_db >= 7.0:
                return ("hostile", "Airborne-range", "Pulsed", "Airborne range-finding radar")
            return ("hostile", "Airborne-detection", "Pulsed", "Airborne surveillance radar")

        if unknown_score >= 1.6:
            return ("hostile", "Airborne-detection", "Pulsed", "Airborne surveillance radar")
        return ("civilian", "AM radio", "AM-DSB", "Commercial AM radio broadcast")

    def _estimate_position(self, obs_list: list[Observation]) -> tuple[float, float, float] | None:
        usable = [o for o in obs_list if o.receiver_id in self.receivers]
        if len(usable) < 2:
            return None

        # Fallback for 2-receiver groups: weighted centroid with larger uncertainty.
        if len(usable) == 2:
            p1 = self.receivers[usable[0].receiver_id]
            p2 = self.receivers[usable[1].receiver_id]
            w1 = max(0.1, 10.0 ** ((usable[0].rssi_dbm + 120.0) / 20.0))
            w2 = max(0.1, 10.0 ** ((usable[1].rssi_dbm + 120.0) / 20.0))
            lat = (p1["latitude"] * w1 + p2["latitude"] * w2) / (w1 + w2)
            lon = (p1["longitude"] * w1 + p2["longitude"] * w2) / (w1 + w2)
            return float(lat), float(lon), 700.0

        rx_points = [(self.receivers[o.receiver_id]["latitude"], self.receivers[o.receiver_id]["longitude"], o.rssi_dbm) for o in usable]
        lat0 = float(np.mean([p[0] for p in rx_points]))
        lon0 = float(np.mean([p[1] for p in rx_points]))

        rssi_ref = float(self.pathloss.get("rssi_ref_dbm", -30.0))
        d_ref = float(self.pathloss.get("d_ref_m", 1.0))
        n = float(self.pathloss.get("path_loss_exponent", 2.8))
        noise_std = float(self.pathloss.get("rssi_noise_std_db", 3.0))

        pts = []
        dists = []
        weights = []
        for lat, lon, rssi in rx_points:
            x, y = ll_to_xy(lat, lon, lat0, lon0)
            d = d_ref * (10.0 ** ((rssi_ref - rssi) / (10.0 * n)))
            pts.append((x, y))
            dists.append(d)
            weights.append(max(0.1, 10.0 ** ((rssi + 120.0) / 20.0)))

        pts_arr = np.asarray(pts, dtype=np.float64)
        dists_arr = np.asarray(dists, dtype=np.float64)
        w_arr = np.asarray(weights, dtype=np.float64)

        x = np.average(pts_arr[:, 0], weights=w_arr)
        y = np.average(pts_arr[:, 1], weights=w_arr)

        for _ in range(12):
            dx = x - pts_arr[:, 0]
            dy = y - pts_arr[:, 1]
            model_d = np.sqrt(dx * dx + dy * dy)
            model_d = np.where(model_d < 1.0, 1.0, model_d)
            residual = model_d - dists_arr

            j = np.column_stack([dx / model_d, dy / model_d])
            wj = j * w_arr[:, None]
            h = wj.T @ j
            g = wj.T @ residual
            try:
                step = np.linalg.solve(h + np.eye(2) * 1e-3, g)
            except np.linalg.LinAlgError:
                break
            x -= step[0]
            y -= step[1]
            if float(np.linalg.norm(step)) < 0.5:
                break

        final_dx = x - pts_arr[:, 0]
        final_dy = y - pts_arr[:, 1]
        final_model_d = np.sqrt(final_dx * final_dx + final_dy * final_dy)
        resid = final_model_d - dists_arr
        uncertainty = float(np.sqrt(np.mean(resid * resid)) + noise_std * 2.0)

        lat, lon = xy_to_ll(float(x), float(y), lat0, lon0)
        return lat, lon, uncertainty

    def _update_tracks(
        self,
        label: str,
        faction: str,
        modulation: str,
        assessment: str,
        conf: float,
        lat: float,
        lon: float,
        uncertainty_m: float,
        seen_at: datetime,
    ) -> Track:
        best_idx = -1
        best_dist = float("inf")
        for i, tr in enumerate(self.tracks):
            age = (seen_at - tr.last_seen).total_seconds()
            if age > self.track_stale_s:
                continue
            if tr.label != label or tr.faction != faction:
                continue
            d = haversine_m(lat, lon, tr.lat, tr.lon)
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx >= 0 and best_dist <= self.track_match_m:
            tr = self.tracks[best_idx]
            alpha = 0.35
            tr.lat = (1.0 - alpha) * tr.lat + alpha * lat
            tr.lon = (1.0 - alpha) * tr.lon + alpha * lon
            tr.uncertainty_m = (1.0 - alpha) * tr.uncertainty_m + alpha * uncertainty_m
            tr.confidence = (1.0 - alpha) * tr.confidence + alpha * conf
            tr.last_seen = seen_at
            tr.hit_count += 1
            return tr

        tid = f"T-{self.track_seq:04d}"
        self.track_seq += 1
        tr = Track(
            track_id=tid,
            label=label,
            faction=faction,
            modulation=modulation,
            assessment=assessment,
            lat=lat,
            lon=lon,
            uncertainty_m=uncertainty_m,
            confidence=conf,
            last_seen=seen_at,
            hit_count=1,
        )
        self.tracks.append(tr)
        return tr

    def _finalize_old_groups(self, now: datetime) -> None:
        keep: list[Group] = []
        for g in self.groups:
            age = (now - g.last_at).total_seconds()
            if age < self.group_finalize_s:
                keep.append(g)
                continue

            if len(g.observations) >= 2:
                fix = self._estimate_position(g.observations)
                if fix is not None:
                    lat, lon, unc = fix
                    conf = float(np.mean([o.confidence for o in g.observations]))
                    weighted_labels: dict[str, float] = {}
                    for o in g.observations:
                        weighted_labels[o.predicted_label] = weighted_labels.get(o.predicted_label, 0.0) + o.confidence
                    chosen_label = max(weighted_labels, key=weighted_labels.get)
                    weighted_types: dict[str, float] = {}
                    weighted_factions: dict[str, float] = {}
                    weighted_mods: dict[str, float] = {}
                    weighted_assess: dict[str, float] = {}
                    for o in g.observations:
                        weighted_types[o.signal_type] = weighted_types.get(o.signal_type, 0.0) + o.confidence
                        weighted_factions[o.faction] = weighted_factions.get(o.faction, 0.0) + o.confidence
                        weighted_mods[o.modulation] = weighted_mods.get(o.modulation, 0.0) + o.confidence
                        weighted_assess[o.assessment] = weighted_assess.get(o.assessment, 0.0) + o.confidence
                    chosen_type = max(weighted_types, key=weighted_types.get)
                    chosen_faction = max(weighted_factions, key=weighted_factions.get)
                    chosen_mod = max(weighted_mods, key=weighted_mods.get)
                    chosen_assessment = max(weighted_assess, key=weighted_assess.get)
                    seen_at = max(o.ingest_timestamp for o in g.observations)
                    self._update_tracks(chosen_type, chosen_faction, chosen_mod, chosen_assessment, conf, lat, lon, unc, seen_at)

        self.groups = keep

    def _attach_to_group(self, obs: Observation) -> None:
        chosen: Group | None = None
        best_dt = 1e9

        for g in self.groups:
            if any(x.receiver_id == obs.receiver_id for x in g.observations):
                continue
            dt = abs((obs.ingest_timestamp - g.last_at).total_seconds())
            if dt <= self.assoc_window_s and dt < best_dt:
                best_dt = dt
                chosen = g

        if chosen is None:
            g = Group(
                group_id=self.group_seq,
                class_key=obs.predicted_label,
                created_at=obs.ingest_timestamp,
                last_at=obs.ingest_timestamp,
                observations=[obs],
            )
            self.group_seq += 1
            self.groups.append(g)
            return

        chosen.observations.append(obs)
        chosen.last_at = obs.ingest_timestamp

    def _run_stream_loop(self) -> None:
        while self.running:
            try:
                with requests.get(
                    f"{self.api_base}/feed/stream",
                    headers=self.headers(),
                    stream=True,
                    timeout=60,
                ) as resp:
                    resp.raise_for_status()
                    for raw in resp.iter_lines(decode_unicode=True):
                        if not self.running:
                            break
                        if not raw or not raw.startswith("data: "):
                            continue
                        payload = raw[6:].strip()
                        if not payload:
                            continue

                        data = json.loads(payload)
                        event_ts = parse_iso(str(data.get("timestamp", ensure_utc_now().isoformat())))
                        ingest_ts = ensure_utc_now()
                        rid = str(data.get("receiver_id", ""))
                        iq = np.asarray(data.get("iq_snapshot", []), dtype=np.float32).reshape(-1)
                        if rid == "" or iq.size != 256:
                            continue

                        rssi = float(data.get("rssi_dbm", -120.0))
                        snr = float(data.get("snr_estimate_db", 0.0))
                        pred, known, conf, unknown_score = self._classify(iq, snr)
                        faction, signal_type, modulation, assessment = self._taxonomy(pred, known, snr, unknown_score, iq)

                        obs = Observation(
                            observation_id=str(data.get("observation_id", "")),
                            event_timestamp=event_ts,
                            ingest_timestamp=ingest_ts,
                            receiver_id=rid,
                            rssi_dbm=rssi,
                            snr_db=snr,
                            iq_snapshot=iq,
                            predicted_label=pred,
                            known_label=known,
                            confidence=conf,
                            unknown_score=unknown_score,
                            faction=faction,
                            signal_type=signal_type,
                            modulation=modulation,
                            assessment=assessment,
                        )

                        with self.lock:
                            self._attach_to_group(obs)
                            self._finalize_old_groups(ingest_ts)

                            self.latest_obs.append(
                                {
                                    "observation_id": obs.observation_id,
                                    "timestamp": obs.event_timestamp.isoformat(),
                                    "receiver_id": obs.receiver_id,
                                    "predicted_label": obs.predicted_label,
                                    "known_label": obs.known_label,
                                    "confidence": obs.confidence,
                                    "unknown_score": obs.unknown_score,
                                    "faction": obs.faction,
                                    "signal_type": obs.signal_type,
                                    "modulation": obs.modulation,
                                    "assessment": obs.assessment,
                                    "rssi_dbm": obs.rssi_dbm,
                                    "snr_db": obs.snr_db,
                                }
                            )
                            self.latest_obs = self.latest_obs[-120:]

                            cutoff = ensure_utc_now().timestamp() - self.track_stale_s
                            self.tracks = [t for t in self.tracks if t.last_seen.timestamp() >= cutoff]
            except Exception as exc:
                print(f"stream error: {exc}; reconnecting in 2s")
                time.sleep(2)

    def state_snapshot(self) -> dict[str, Any]:
        with self.lock:
            rx = list(self.receivers.values())
            tracks = [
                {
                    "track_id": t.track_id,
                    "label": t.label,
                    "faction": t.faction,
                    "modulation": t.modulation,
                    "assessment": t.assessment,
                    "lat": t.lat,
                    "lon": t.lon,
                    "uncertainty_m": t.uncertainty_m,
                    "confidence": t.confidence,
                    "last_seen": t.last_seen.isoformat(),
                    "hit_count": t.hit_count,
                }
                for t in self.tracks
            ]
            obs = list(self.latest_obs)

        return {
            "server_time": ensure_utc_now().isoformat(),
            "receiver_count": len(rx),
            "track_count": len(tracks),
            "receivers": rx,
            "tracks": tracks,
            "latest_observations": obs,
        }


def make_app(cop: LiveCOP) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> str:
        return render_template("cop.html")

    @app.get("/state")
    def state() -> Any:
        return jsonify(cop.state_snapshot())

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Stage-2 COP: association + geolocation + map")
    parser.add_argument("--api-base", default="https://findmyforce.online")
    parser.add_argument("--api-key", default=os.getenv("FINDMYFORCE_API_KEY"))
    parser.add_argument("--model", default="models/baseline_model.joblib")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--assoc-window-ms", type=int, default=250)
    parser.add_argument("--group-finalize-ms", type=int, default=450)
    parser.add_argument("--track-match-m", type=float, default=650.0)
    parser.add_argument("--track-stale-s", type=float, default=90.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise SystemExit("Missing API key. Set FINDMYFORCE_API_KEY or pass --api-key.")

    cop = LiveCOP(
        api_base=args.api_base,
        api_key=args.api_key,
        model_path=args.model,
        assoc_window_ms=args.assoc_window_ms,
        group_finalize_ms=args.group_finalize_ms,
        track_match_m=args.track_match_m,
        track_stale_s=args.track_stale_s,
    )
    cop.start()

    app = make_app(cop)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

