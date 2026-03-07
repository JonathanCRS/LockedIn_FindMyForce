import argparse
import csv
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def load_rows(csv_path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    sample_ids: list[str] = []
    labels: list[str] = []
    summary: list[list[float]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sample_ids.append(row["sample_id"])
            labels.append(row["modulation"])
            summary.append(
                [
                    float(row["snr_db"]),
                    float(row["mean"]),
                    float(row["std"]),
                    float(row["minimum"]),
                    float(row["maximum"]),
                    float(row["energy"]),
                ]
            )

    return sample_ids, np.asarray(labels, dtype=str), np.asarray(summary, dtype=np.float32)


def build_features(
    sample_ids: list[str], summary: np.ndarray, feature_mode: str, vectors_npz_path: Path | None
) -> np.ndarray:
    if feature_mode == "summary":
        return summary

    if vectors_npz_path is None:
        raise ValueError("--vectors-npz is required when feature mode is 'full'.")

    vectors_npz = np.load(vectors_npz_path)
    vectors = np.stack([vectors_npz[sid].astype(np.float32) for sid in sample_ids], axis=0)

    # Keep SNR signal quality context as an extra feature.
    snr_col = summary[:, :1]
    return np.concatenate([vectors, snr_col], axis=1)


def train(args: argparse.Namespace) -> None:
    sample_ids, labels, summary = load_rows(Path(args.input_csv))

    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]
        labels = labels[: args.limit]
        summary = summary[: args.limit]

    x = build_features(sample_ids, summary, args.feature_mode, Path(args.vectors_npz) if args.vectors_npz else None)

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        labels,
        test_size=args.val_size,
        random_state=42,
        stratify=labels,
    )

    if args.model_type == "randomforest":
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        model = ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)

    # Lower confidence than this threshold will be flagged as unknown.
    train_proba = model.predict_proba(x_train)
    max_train_proba = np.max(train_proba, axis=1)
    unknown_threshold = float(np.quantile(max_train_proba, args.unknown_quantile))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = out_dir / "baseline_model.joblib"

    joblib.dump(
        {
            "model": model,
            "feature_mode": args.feature_mode,
            "unknown_threshold": unknown_threshold,
        },
        bundle_path,
    )

    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Model type: {args.model_type}")
    print(f"Unknown threshold (max proba): {unknown_threshold:.4f}")
    print(f"Saved model bundle: {bundle_path}")


def predict(args: argparse.Namespace) -> None:
    bundle: dict[str, Any] = joblib.load(args.model)
    model = bundle["model"]
    feature_mode = bundle["feature_mode"]
    unknown_threshold = float(bundle["unknown_threshold"])

    sample_ids, labels, summary = load_rows(Path(args.input_csv))
    if args.limit is not None:
        sample_ids = sample_ids[: args.limit]
        labels = labels[: args.limit]
        summary = summary[: args.limit]

    x = build_features(sample_ids, summary, feature_mode, Path(args.vectors_npz) if args.vectors_npz else None)

    proba = model.predict_proba(x)
    pred_idx = np.argmax(proba, axis=1)
    max_proba = np.max(proba, axis=1)

    for i in range(min(args.show, len(labels))):
        known_label = model.classes_[pred_idx[i]]
        is_unknown = max_proba[i] < unknown_threshold
        final_label = "unknown" if is_unknown else known_label
        print(
            f"i={i} truth={labels[i]} pred={final_label} known={known_label} "
            f"conf={max_proba[i]:.3f} unknown_score={(unknown_threshold / max(max_proba[i], 1e-6)):.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stronger RF baseline classifier with unknown detection.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--input-csv", default="data/training_tidy.csv")
    p_train.add_argument("--vectors-npz", default="data/training_vectors.npz")
    p_train.add_argument("--model-type", choices=["randomforest", "extratrees"], default="randomforest")
    p_train.add_argument("--feature-mode", choices=["full", "summary"], default="full")
    p_train.add_argument("--output-dir", default="models")
    p_train.add_argument("--n-estimators", type=int, default=300)
    p_train.add_argument("--val-size", type=float, default=0.2)
    p_train.add_argument("--unknown-quantile", type=float, default=0.02)
    p_train.add_argument("--limit", type=int, default=None)
    p_train.set_defaults(func=train)

    p_pred = sub.add_parser("predict")
    p_pred.add_argument("--input-csv", default="data/training_tidy.csv")
    p_pred.add_argument("--vectors-npz", default="data/training_vectors.npz")
    p_pred.add_argument("--model", default="models/baseline_model.joblib")
    p_pred.add_argument("--limit", type=int, default=20)
    p_pred.add_argument("--show", type=int, default=10)
    p_pred.set_defaults(func=predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
