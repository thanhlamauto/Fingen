from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.environ.get("FINGEN_WORK_ROOT", Path.home()))
FIG_ROOT = REPO / "figures"

DB_RUNS = {
    "DB1": WORK_ROOT / "outputs/fvc2004_db1a_lora_stage2_synth500x10_20260615/images/challengers/DB1A/roll/png",
    "DB2": WORK_ROOT / "outputs/fvc2004_db2a_lora_stage2_synth500x10_20260617/images/challengers/DB2A/roll/png",
    "DB3": WORK_ROOT / "outputs/fvc2004_db3a_lora_stage2_synth500x10_20260617/images/challengers/DB3A/roll/png",
    "DB4": WORK_ROOT / "outputs/fvc2004_db4a_lora_stage2_synth500x10_20260617/images/challengers/DB4A/roll/png",
}


def common_filenames() -> list[str]:
    sets = []
    for root in DB_RUNS.values():
        sets.append({path.name for path in root.glob("*.png")})
    return sorted(set.intersection(*sets))


def local_std(gray: np.ndarray, ksize: int = 9) -> np.ndarray:
    gray_f = gray.astype(np.float32) / 255.0
    mean = cv2.blur(gray_f, (ksize, ksize))
    mean_sq = cv2.blur(gray_f * gray_f, (ksize, ksize))
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def image_descriptor(path: Path) -> np.ndarray:
    gray = np.asarray(Image.open(path).convert("L").resize((128, 128), Image.Resampling.BILINEAR), dtype=np.uint8)
    img_small = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    std_small = cv2.resize(local_std(gray), (32, 32), interpolation=cv2.INTER_AREA)
    mask = (gray < 245).astype(np.float32)
    mask_small = cv2.resize(mask, (32, 32), interpolation=cv2.INTER_AREA)
    hist = np.histogram(gray, bins=32, range=(0, 255), density=True)[0].astype(np.float32)
    std_hist = np.histogram(local_std(gray), bins=16, range=(0, 0.35), density=True)[0].astype(np.float32)
    stats = np.asarray(
        [
            float(gray.mean() / 255.0),
            float(gray.std() / 255.0),
            float(mask.mean()),
            float(local_std(gray).mean()),
            float(local_std(gray).std()),
        ],
        dtype=np.float32,
    )
    return np.concatenate(
        [
            img_small.reshape(-1),
            std_small.reshape(-1),
            mask_small.reshape(-1),
            hist,
            std_hist,
            stats,
        ]
    ).astype(np.float32)


def cluster_summary(coords: np.ndarray, labels: list[str]) -> dict:
    out = {}
    label_arr = np.asarray(labels)
    for db in sorted(set(labels)):
        pts = coords[label_arr == db]
        center = pts.mean(axis=0)
        intra = np.sqrt(((pts - center) ** 2).sum(axis=1)).mean()
        nearest = min(
            np.linalg.norm(center - coords[label_arr == other].mean(axis=0))
            for other in sorted(set(labels))
            if other != db
        )
        out[db] = {
            "n": int(pts.shape[0]),
            "mean_intra_radius": float(intra),
            "nearest_center_distance": float(nearest),
            "separation_ratio": float(nearest / max(intra, 1e-6)),
        }
    return out


def plot_tsne(coords: np.ndarray, labels: list[str], out_path: Path, title: str) -> None:
    colors = {
        "DB1": "#2B6CB0",
        "DB2": "#D53F8C",
        "DB3": "#2F855A",
        "DB4": "#DD6B20",
    }
    markers = {"DB1": "o", "DB2": "s", "DB3": "^", "DB4": "D"}
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=180)
    label_arr = np.asarray(labels)
    for db in sorted(set(labels)):
        pts = coords[label_arr == db]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=16,
            c=colors.get(db, "gray"),
            marker=markers.get(db, "o"),
            label=db,
            alpha=0.78,
            linewidths=0,
        )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.grid(True, color="#eeeeee", linewidth=0.6)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-db", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--out-prefix", default="fvc_lora_tsne")
    args = parser.parse_args()

    names = common_filenames()
    if len(names) < args.samples_per_db:
        raise ValueError(f"Only {len(names)} common filenames found")
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(names, args.samples_per_db))

    features = []
    labels = []
    rows = []
    for db, root in DB_RUNS.items():
        for name in chosen:
            path = root / name
            features.append(image_descriptor(path))
            labels.append(db)
            rows.append({"db": db, "filename": name, "path": str(path)})

    x = np.stack(features, axis=0)
    x = StandardScaler().fit_transform(x)
    pca_dims = min(50, x.shape[0] - 1, x.shape[1])
    x_pca = PCA(n_components=pca_dims, random_state=args.seed).fit_transform(x)
    coords = TSNE(
        n_components=2,
        perplexity=35,
        learning_rate="auto",
        init="pca",
        random_state=args.seed,
        max_iter=1500,
    ).fit_transform(x_pca)

    for row, coord in zip(rows, coords):
        row["tsne_x"] = float(coord[0])
        row["tsne_y"] = float(coord[1])

    sil = float(silhouette_score(coords, labels))
    summary = {
        "samples_per_db": int(args.samples_per_db),
        "total_samples": int(len(rows)),
        "embedding": "48x48 grayscale + 32x32 local-contrast + 32x32 foreground-mask + intensity/local-contrast histograms",
        "pca_dims": int(pca_dims),
        "tsne_perplexity": 35,
        "silhouette_2d": sil,
        "clusters": cluster_summary(coords, labels),
    }

    csv_path = FIG_ROOT / f"{args.out_prefix}.csv"
    json_path = FIG_ROOT / f"{args.out_prefix}_summary.json"
    png_path = FIG_ROOT / f"{args.out_prefix}.png"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    plot_tsne(coords, labels, png_path, "t-SNE of FVC DB1-DB4 LoRA Synthetic Styles")

    print(json.dumps(summary, indent=2))
    print(csv_path)
    print(json_path)
    print(png_path)


if __name__ == "__main__":
    main()
