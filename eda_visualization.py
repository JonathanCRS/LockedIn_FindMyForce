import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


FEATURE_COLS = ["snr_db", "mean", "std", "minimum", "maximum", "energy"]


def save_summary(df: pd.DataFrame, out_dir: Path) -> None:
    summary = {
        "rows": int(len(df)),
        "columns": df.columns.tolist(),
        "modulation_counts": df["modulation"].value_counts().to_dict(),
        "emitter_counts": df["emitter_type"].value_counts().to_dict(),
        "snr_min": float(df["snr_db"].min()),
        "snr_max": float(df["snr_db"].max()),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    df.describe(include="all").to_csv(out_dir / "dataset_describe.csv")


def plot_class_counts(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="modulation", order=df["modulation"].value_counts().index, palette="Set2")
    plt.title("Sample Count by Modulation")
    plt.tight_layout()
    plt.savefig(out_dir / "01_modulation_counts.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="emitter_type", order=df["emitter_type"].value_counts().index, palette="Set3")
    plt.title("Sample Count by Emitter Type")
    plt.tight_layout()
    plt.savefig(out_dir / "02_emitter_counts.png", dpi=180)
    plt.close()


def plot_snr_views(df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    sns.histplot(df["snr_db"], bins=20, kde=True, color="#3b82f6")
    plt.title("SNR Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "03_snr_hist.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df, x="modulation", y="snr_db", palette="Set2")
    plt.title("SNR by Modulation")
    plt.tight_layout()
    plt.savefig(out_dir / "04_snr_by_modulation_box.png", dpi=180)
    plt.close()


def plot_feature_correlations(df: pd.DataFrame, out_dir: Path) -> None:
    corr = df[FEATURE_COLS].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "05_feature_correlation.png", dpi=180)
    plt.close()


def plot_feature_scatter(df: pd.DataFrame, out_dir: Path, sample_size: int) -> None:
    sampled = df.sample(min(sample_size, len(df)), random_state=42)

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sampled, x="std", y="energy", hue="modulation", s=20, alpha=0.8)
    plt.title("Std vs Energy")
    plt.tight_layout()
    plt.savefig(out_dir / "06_std_vs_energy.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sampled, x="minimum", y="maximum", hue="modulation", s=20, alpha=0.8)
    plt.title("Minimum vs Maximum")
    plt.tight_layout()
    plt.savefig(out_dir / "07_min_vs_max.png", dpi=180)
    plt.close()


def plot_pca_summary(df: pd.DataFrame, out_dir: Path, sample_size: int) -> None:
    sampled = df.sample(min(sample_size, len(df)), random_state=42).copy()
    x = sampled[FEATURE_COLS].to_numpy(dtype=np.float32)
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(x)
    sampled["pc1"] = z[:, 0]
    sampled["pc2"] = z[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=sampled, x="pc1", y="pc2", hue="modulation", s=18, alpha=0.8)
    plt.title(f"PCA (Summary Features) - Explained Var: {pca.explained_variance_ratio_.sum():.2f}")
    plt.tight_layout()
    plt.savefig(out_dir / "08_pca_summary_features.png", dpi=180)
    plt.close()


def train_eval_summary_model(df: pd.DataFrame, out_dir: Path) -> None:
    x = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["modulation"].to_numpy(dtype=str)

    x_train, x_test, y_train, y_test, _snr_train, snr_test = train_test_split(
        x, y, df["snr_db"].to_numpy(), test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    (out_dir / "09_model_eval_report.txt").write_text(f"Accuracy: {acc:.4f}\n\n{report}", encoding="utf-8")

    labels = sorted(np.unique(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Blues", annot=True, fmt=".2f")
    plt.title("Normalized Confusion Matrix (Summary-Feature RF)")
    plt.tight_layout()
    plt.savefig(out_dir / "10_confusion_matrix.png", dpi=180)
    plt.close()

    bins = np.arange(-20, 20, 2)
    snr_bucket = pd.cut(snr_test, bins=bins, include_lowest=True)
    snr_df = pd.DataFrame({"snr_bin": snr_bucket.astype(str), "ok": (y_test == y_pred).astype(int)})
    snr_acc = snr_df.groupby("snr_bin", observed=False)["ok"].mean().reset_index()
    snr_acc.to_csv(out_dir / "11_accuracy_by_snr_bin.csv", index=False)

    plt.figure(figsize=(11, 4))
    sns.barplot(data=snr_acc, x="snr_bin", y="ok", color="#22c55e")
    plt.xticks(rotation=70)
    plt.ylim(0, 1)
    plt.title("Validation Accuracy by SNR Bin")
    plt.tight_layout()
    plt.savefig(out_dir / "11_accuracy_by_snr_bin.png", dpi=180)
    plt.close()


def load_vector_sample(df: pd.DataFrame, vectors_npz_path: Path, max_vectors: int) -> tuple[pd.DataFrame, np.ndarray]:
    sampled = df.sample(min(max_vectors, len(df)), random_state=42).copy()
    npz = np.load(vectors_npz_path)
    vectors = np.stack([npz[sid].astype(np.float32) for sid in sampled["sample_id"].tolist()], axis=0)
    return sampled, vectors


def plot_waveforms_and_spectra(df: pd.DataFrame, vectors_npz_path: Path, out_dir: Path) -> None:
    npz = np.load(vectors_npz_path)
    mods = sorted(df["modulation"].unique().tolist())

    plt.figure(figsize=(10, 6))
    for mod in mods:
        sid = df[df["modulation"] == mod]["sample_id"].iloc[0]
        v = npz[sid]
        plt.plot(v, label=mod, linewidth=1.4)
    plt.title("Example IQ Snapshot per Modulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "12_example_waveforms.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    for mod in mods:
        sids = df[df["modulation"] == mod]["sample_id"].head(200).tolist()
        vs = np.stack([npz[sid] for sid in sids], axis=0).astype(np.float32)
        p = np.abs(np.fft.rfft(vs - vs.mean(axis=1, keepdims=True), axis=1)) ** 2
        p_mean = p.mean(axis=0)
        p_mean = p_mean / (p_mean.max() + 1e-9)
        plt.plot(p_mean, label=mod, linewidth=1.5)
    plt.title("Average Normalized Spectrum by Modulation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "13_average_spectrum_by_modulation.png", dpi=180)
    plt.close()


def plot_vector_pca(df: pd.DataFrame, vectors_npz_path: Path, out_dir: Path, max_vectors: int) -> None:
    sampled, vectors = load_vector_sample(df, vectors_npz_path, max_vectors)
    x = vectors
    x = (x - x.mean(axis=0)) / (x.std(axis=0) + 1e-6)

    pca = PCA(n_components=2, random_state=42)
    z = pca.fit_transform(x)
    sampled["pc1"] = z[:, 0]
    sampled["pc2"] = z[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=sampled, x="pc1", y="pc2", hue="modulation", s=16, alpha=0.75)
    plt.title(f"PCA (Raw IQ Vectors) - Explained Var: {pca.explained_variance_ratio_.sum():.2f}")
    plt.tight_layout()
    plt.savefig(out_dir / "14_pca_raw_vectors.png", dpi=180)
    plt.close()


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)

    save_summary(df, out_dir)
    plot_class_counts(df, out_dir)
    plot_snr_views(df, out_dir)
    plot_feature_correlations(df, out_dir)
    plot_feature_scatter(df, out_dir, args.sample_size)
    plot_pca_summary(df, out_dir, args.sample_size)
    train_eval_summary_model(df, out_dir)

    if args.vectors_npz and Path(args.vectors_npz).exists():
        plot_waveforms_and_spectra(df, Path(args.vectors_npz), out_dir)
        plot_vector_pca(df, Path(args.vectors_npz), out_dir, args.max_vector_samples)

    print(f"EDA finished. Output folder: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RF data analysis and visualization plots.")
    parser.add_argument("--input-csv", default="data/training_tidy.csv")
    parser.add_argument("--vectors-npz", default="data/training_vectors.npz")
    parser.add_argument("--output-dir", default="eda_outputs")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--max-vector-samples", type=int, default=4000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
