import argparse
import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


@dataclass
class CleanSample:
    sample_id: str
    modulation: str
    emitter_type: str
    snr_db: int
    source_index: int
    vector_length: int
    mean: float
    std: float
    minimum: float
    maximum: float
    energy: float

    def to_row(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "modulation": self.modulation,
            "emitter_type": self.emitter_type,
            "snr_db": self.snr_db,
            "source_index": self.source_index,
            "vector_length": self.vector_length,
            "mean": self.mean,
            "std": self.std,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "energy": self.energy,
        }


def parse_key(raw_key: str) -> tuple[str, str, int, int] | None:
    """Parse tuple-like HDF5 key: ('ask', 'short-range', -10, 0)."""
    try:
        parsed = ast.literal_eval(raw_key)
    except (SyntaxError, ValueError):
        return None

    if not isinstance(parsed, tuple) or len(parsed) != 4:
        return None

    modulation, emitter_type, snr_db, source_index = parsed

    try:
        return str(modulation), str(emitter_type), int(snr_db), int(source_index)
    except (TypeError, ValueError):
        return None


def clean_vector(raw_vector: Any) -> np.ndarray | None:
    vector = np.asarray(raw_vector, dtype=np.float32).reshape(-1)
    if vector.size == 0:
        return None
    if not np.isfinite(vector).all():
        return None
    return vector


def clean_training_hdf5(input_path: Path, output_dir: Path, limit: int | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tidy_csv = output_dir / "training_tidy.csv"
    vectors_npz = output_dir / "training_vectors.npz"

    cleaned = 0
    skipped = 0
    vector_store: dict[str, np.ndarray] = {}

    fieldnames = [
        "sample_id",
        "modulation",
        "emitter_type",
        "snr_db",
        "source_index",
        "vector_length",
        "mean",
        "std",
        "minimum",
        "maximum",
        "energy",
    ]

    with h5py.File(input_path, "r") as h5_file, tidy_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for raw_key in h5_file.keys():
            key_parts = parse_key(raw_key)
            if key_parts is None:
                skipped += 1
                continue

            modulation, emitter_type, snr_db, source_index = key_parts

            vector = clean_vector(h5_file[raw_key][()])
            if vector is None:
                skipped += 1
                continue

            sample_id = f"{modulation}|{emitter_type}|{snr_db}|{source_index}"
            sample = CleanSample(
                sample_id=sample_id,
                modulation=modulation,
                emitter_type=emitter_type,
                snr_db=snr_db,
                source_index=source_index,
                vector_length=int(vector.size),
                mean=float(np.mean(vector)),
                std=float(np.std(vector)),
                minimum=float(np.min(vector)),
                maximum=float(np.max(vector)),
                energy=float(np.mean(np.square(vector))),
            )

            writer.writerow(sample.to_row())
            vector_store[sample_id] = vector
            cleaned += 1

            if cleaned % 1000 == 0:
                print(f"cleaned={cleaned} skipped={skipped}")

            if limit is not None and cleaned >= limit:
                break

    np.savez_compressed(vectors_npz, **vector_store)
    print(f"Saved tidy table: {tidy_csv}")
    print(f"Saved vectors: {vectors_npz}")
    print(f"Final counts -> cleaned: {cleaned}, skipped: {skipped}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tidy RF training_data.hdf5 into model-ready artifacts.")
    parser.add_argument("--input", default="training_data.hdf5", help="Input HDF5 file path.")
    parser.add_argument("--output-dir", default="data", help="Directory for output files.")
    parser.add_argument("--limit", type=int, default=None, help="Stop after N cleaned samples.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    clean_training_hdf5(input_path=input_path, output_dir=Path(args.output_dir), limit=args.limit)


if __name__ == "__main__":
    main()
