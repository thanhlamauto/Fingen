from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


REPO = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.environ.get("FINGEN_WORK_ROOT", Path.home()))
LIGHTGLUE_PATH = Path(
    os.environ.get("LIGHTGLUE_PATH", REPO / "LightGlue/lightglue/lightglue.py")
)
FIG_ROOT = REPO / "figures"

DB_RUNS = {
    "DB1": WORK_ROOT / "outputs/fvc2004_db1a_lora_stage2_synth500x10_20260615",
    "DB2": WORK_ROOT / "outputs/fvc2004_db2a_lora_stage2_synth500x10_20260617",
    "DB3": WORK_ROOT / "outputs/fvc2004_db3a_lora_stage2_synth500x10_20260617",
    "DB4": WORK_ROOT / "outputs/fvc2004_db4a_lora_stage2_synth500x10_20260617",
}


def load_lightglue_class():
    spec = importlib.util.spec_from_file_location("lightglue_direct", LIGHTGLUE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {LIGHTGLUE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.LightGlue


def rootsift(desc: torch.Tensor) -> torch.Tensor:
    desc = torch.nn.functional.normalize(desc, p=1, dim=-1, eps=1e-6)
    desc = desc.clamp(min=1e-6).sqrt()
    return torch.nn.functional.normalize(desc, p=2, dim=-1, eps=1e-6)


def extract_sift(path: Path, max_keypoints: int) -> dict[str, torch.Tensor]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    sift = cv2.SIFT_create(
        nfeatures=max_keypoints,
        contrastThreshold=0.0066667,
        edgeThreshold=10,
        nOctaveLayers=4,
    )
    kps, desc = sift.detectAndCompute(image, None)
    if desc is None or not kps:
        pts = np.zeros((0, 2), dtype=np.float32)
        desc = np.zeros((0, 128), dtype=np.float32)
        scales = np.zeros((0,), dtype=np.float32)
        oris = np.zeros((0,), dtype=np.float32)
    else:
        pts = np.asarray([kp.pt for kp in kps], dtype=np.float32)
        scales = np.asarray([kp.size for kp in kps], dtype=np.float32)
        oris = np.deg2rad(np.asarray([kp.angle for kp in kps], dtype=np.float32))
        desc = desc.astype(np.float32)
    return {
        "keypoints": torch.from_numpy(pts)[None],
        "descriptors": rootsift(torch.from_numpy(desc))[None],
        "scales": torch.from_numpy(scales)[None],
        "oris": torch.from_numpy(oris)[None],
        "image_size": torch.tensor([[image.shape[1], image.shape[0]]], dtype=torch.float32),
    }


def paired_paths(run_root: Path) -> list[tuple[Path, Path]]:
    hint_root = run_root / "pose_aligned_hints"
    synth_root = run_root / "images/challengers"
    pairs = []
    for hint_path in sorted(hint_root.glob("**/*.png")):
        rel = hint_path.relative_to(hint_root)
        synth_path = synth_root / rel
        if synth_path.exists():
            pairs.append((hint_path, synth_path))
    return pairs


def run_match(
    matcher,
    hint_path: Path,
    synth_path: Path,
    max_keypoints: int,
) -> dict:
    feats0 = extract_sift(hint_path, max_keypoints)
    feats1 = extract_sift(synth_path, max_keypoints)
    with torch.inference_mode():
        out = matcher({"image0": feats0, "image1": feats1})
    matches = out["matches"][0].detach().cpu()
    scores = out["scores"][0].detach().cpu()
    return {
        "hint_path": hint_path,
        "synth_path": synth_path,
        "keypoints0": int(feats0["keypoints"].shape[1]),
        "keypoints1": int(feats1["keypoints"].shape[1]),
        "matches": int(matches.shape[0]),
        "mean_score": float(scores.mean().item()) if scores.numel() else 0.0,
        "points0": feats0["keypoints"][0].detach().cpu().numpy(),
        "points1": feats1["keypoints"][0].detach().cpu().numpy(),
        "match_indices": matches.numpy(),
    }


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def fit_image(path: Path, max_side: int) -> tuple[Image.Image, float]:
    img = Image.open(path).convert("RGB")
    scale = min(max_side / img.width, max_side / img.height)
    new_size = (max(1, int(round(img.width * scale))), max(1, int(round(img.height * scale))))
    return img.resize(new_size, Image.Resampling.LANCZOS), scale


def draw_pair_panel(db: str, result: dict, max_lines: int = 90) -> Image.Image:
    max_side = 170
    gap = 20
    title_h = 30
    pad = 8
    img0, s0 = fit_image(result["hint_path"], max_side)
    img1, s1 = fit_image(result["synth_path"], max_side)
    h = title_h + max(img0.height, img1.height) + 2 * pad
    w = img0.width + img1.width + gap + 2 * pad
    panel = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(panel)
    draw.text(
        (pad, 6),
        f"{db}: {result['matches']} matches",
        font=font(16, bold=True),
        fill=(0, 0, 0),
    )
    y0 = title_h + pad
    x0 = pad
    x1 = pad + img0.width + gap
    panel.paste(img0, (x0, y0))
    panel.paste(img1, (x1, y0))
    draw.rectangle([x0, y0, x0 + img0.width - 1, y0 + img0.height - 1], outline=(215, 215, 215))
    draw.rectangle([x1, y0, x1 + img1.width - 1, y0 + img1.height - 1], outline=(215, 215, 215))

    matches = result["match_indices"]
    if matches.shape[0] > max_lines:
        idx = np.linspace(0, matches.shape[0] - 1, max_lines).round().astype(int)
        matches = matches[idx]
    pts0 = result["points0"]
    pts1 = result["points1"]
    for a, b in matches:
        p0 = pts0[int(a)] * s0 + np.asarray([x0, y0])
        p1 = pts1[int(b)] * s1 + np.asarray([x1, y0])
        draw.line([tuple(p0), tuple(p1)], fill=(20, 150, 70), width=1)
    return panel


def make_contact_sheet(examples: dict[str, dict], out_path: Path) -> None:
    panels = [draw_pair_panel(db, examples[db]) for db in sorted(examples)]
    cols = 2
    gap = 16
    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    canvas = Image.new("RGB", (cols * w + (cols + 1) * gap, 2 * h + 3 * gap), "white")
    for i, panel in enumerate(panels):
        x = gap + (i % cols) * (w + gap)
        y = gap + (i // cols) * (h + gap)
        canvas.paste(panel, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def summarize(rows: list[dict]) -> dict[str, dict]:
    summary = {}
    for db in sorted({row["db"] for row in rows}):
        vals = np.asarray([row["matches"] for row in rows if row["db"] == db], dtype=np.float32)
        scores = np.asarray([row["mean_score"] for row in rows if row["db"] == db], dtype=np.float32)
        summary[db] = {
            "n_pairs": int(vals.size),
            "mean_matches": float(vals.mean()),
            "std_matches": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
            "median_matches": float(np.median(vals)),
            "mean_score": float(scores.mean()),
        }
    all_vals = np.asarray([row["matches"] for row in rows], dtype=np.float32)
    summary["ALL"] = {
        "n_pairs": int(all_vals.size),
        "mean_matches": float(all_vals.mean()),
        "std_matches": float(all_vals.std(ddof=1)) if all_vals.size > 1 else 0.0,
        "median_matches": float(np.median(all_vals)),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-per-db", type=int, default=50)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument("--out-prefix", default="lightglue_fvc_lora")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    LightGlue = load_lightglue_class()
    matcher = LightGlue(
        features="sift",
        n_layers=6,
        depth_confidence=0.9,
        width_confidence=0.95,
    ).eval()

    rows: list[dict] = []
    full_results: dict[str, list[dict]] = {}
    for db, run_root in DB_RUNS.items():
        pairs = paired_paths(run_root)
        if len(pairs) < args.pairs_per_db:
            raise ValueError(f"{db} has only {len(pairs)} paired hint/output images")
        chosen = rng.sample(pairs, args.pairs_per_db)
        db_results = []
        for idx, (hint_path, synth_path) in enumerate(chosen, start=1):
            result = run_match(matcher, hint_path, synth_path, args.max_keypoints)
            rel = hint_path.relative_to(run_root / "pose_aligned_hints")
            row = {
                "db": db,
                "pair": str(rel),
                "hint_path": str(hint_path),
                "synth_path": str(synth_path),
                "keypoints_hint": result["keypoints0"],
                "keypoints_synth": result["keypoints1"],
                "matches": result["matches"],
                "mean_score": result["mean_score"],
            }
            rows.append(row)
            db_results.append({**result, **row})
            print(f"{db} {idx:03d}/{args.pairs_per_db}: {row['matches']} matches")
        full_results[db] = db_results

    summary = summarize(rows)
    csv_path = FIG_ROOT / f"{args.out_prefix}_matches.csv"
    json_path = FIG_ROOT / f"{args.out_prefix}_summary.json"
    png_path = FIG_ROOT / f"{args.out_prefix}_matching.png"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    examples = {}
    for db, db_results in full_results.items():
        target = summary[db]["median_matches"]
        examples[db] = min(db_results, key=lambda row: abs(row["matches"] - target))
    make_contact_sheet(examples, png_path)

    print(json.dumps(summary, indent=2))
    print(csv_path)
    print(json_path)
    print(png_path)


if __name__ == "__main__":
    main()
