from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.environ.get("FINGEN_WORK_ROOT", Path.home()))
IMPOSE_ROOT = WORK_ROOT / "IMPOSEStage1"
if str(IMPOSE_ROOT) not in sys.path:
    sys.path.insert(0, str(IMPOSE_ROOT))

from jax_stage2.tools.generate_stage1_stage2_identity_dataset import (  # noqa: E402
    _align_ridge_to_pose_mask,
    _extract_pose_mask,
    _load_gray_u8,
)


OUT_ROOT = WORK_ROOT / "outputs/sd302a_to_fvc_db1_lora_posemask_samples_figure5"
PREP_ROOT = OUT_ROOT / "posemasked_inputs"
FIG_ROOT = REPO / "figures"
SD302A_ROOT = WORK_ROOT / "kaggle_downloads/fingerprint_datasets/sd302a/images/challengers"
RIDGE_ROOT = WORK_ROOT / "data/ridge_conditions_sauvola_w11_k0007"
DB1_ROOT = WORK_ROOT / "data/fvc2004_db1a_png512"

ROWS = [
    {
        "label": "SD302A A-2373",
        "source_rel": "A/roll/png/00002373_A_roll_04.png",
        "db1_ref": "82_5",
    },
    {
        "label": "SD302A A-2555",
        "source_rel": "A/roll/png/00002555_A_roll_08.png",
        "db1_ref": "84_8",
    },
    {
        "label": "SD302A A-2455",
        "source_rel": "A/roll/png/00002455_A_roll_05.png",
        "db1_ref": "68_5",
    },
]

HEADERS = [
    "SD302A real",
    "Extracted ridge",
    "DB1 pose mask",
    "Warped ridge",
    "DB1 LoRA synth",
]


def stem_for(row: dict[str, str]) -> str:
    source_stem = Path(row["source_rel"]).stem
    return f"{source_stem}__db1_{row['db1_ref']}"


def paths_for(row: dict[str, str]) -> dict[str, Path]:
    stem = stem_for(row)
    return {
        "source_real": SD302A_ROOT / row["source_rel"],
        "source_ridge": RIDGE_ROOT / row["source_rel"],
        "target_real": DB1_ROOT / f"{row['db1_ref']}.png",
        "target_mask": PREP_ROOT / "masks" / f"{stem}.png",
        "aligned_ridge": PREP_ROOT / "conditions" / f"{stem}.png",
        "synth": OUT_ROOT / stem / "00000.png",
    }


def pose_args() -> SimpleNamespace:
    return SimpleNamespace(
        pose_mask_mode="intensity",
        pose_background_threshold=250,
        pose_mask_min_area_frac=0.002,
        pose_source_mask_min_area_frac=0.002,
        pose_local_std_window=51,
        pose_local_std_threshold=6.0,
        pose_mask_close=31,
        pose_mask_dilate=9,
        pose_min_intersection_frac=0.01,
        pose_align_transform="bounded_similarity",
        pose_scale_basis="bbox_min",
        pose_scale_min=0.75,
        pose_scale_max=1.25,
    )


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = np.where(mask.astype(bool), 35, 255).astype(np.uint8)
    Image.fromarray(mask_img, mode="L").save(path)


def prepare_posemasked_inputs() -> None:
    args = pose_args()
    manifest = []
    for row in ROWS:
        paths = paths_for(row)
        source_real = _load_gray_u8(paths["source_real"], 512)
        source_ridge = _load_gray_u8(paths["source_ridge"], 512)
        target_real = _load_gray_u8(paths["target_real"], 512)
        source_mask = _extract_pose_mask(
            source_real,
            args,
            min_area_frac=args.pose_source_mask_min_area_frac,
        )
        target_mask = _extract_pose_mask(target_real, args)
        aligned, info = _align_ridge_to_pose_mask(source_ridge, source_mask, target_mask, args)

        paths["aligned_ridge"].parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(aligned, mode="L").save(paths["aligned_ridge"])
        save_mask(target_mask, paths["target_mask"])
        manifest.append(
            {
                "label": row["label"],
                "source_rel": row["source_rel"],
                "db1_ref": row["db1_ref"],
                "condition_path": str(paths["aligned_ridge"]),
                "target_mask_path": str(paths["target_mask"]),
                "alignment": info,
            }
        )

    PREP_ROOT.mkdir(parents=True, exist_ok=True)
    (PREP_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def foreground_bbox(img: Image.Image, threshold: int = 248, pad: int = 20) -> tuple[int, int, int, int]:
    gray = img.convert("L")
    mask = gray.point(lambda p: 255 if p < threshold else 0)
    bbox = mask.getbbox()
    if bbox is None:
        return (0, 0, img.width, img.height)
    left, top, right, bottom = bbox
    return (
        max(0, left - pad),
        max(0, top - pad),
        min(img.width, right + pad),
        min(img.height, bottom + pad),
    )


def make_tile(path: Path, size: int) -> Image.Image:
    img = Image.open(path).convert("L")
    img = img.crop(foreground_bbox(img))
    img = ImageOps.autocontrast(img, cutoff=0.2)
    img.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("L", (size, size), 255)
    tile.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return tile.convert("RGB")


def centered_text(
    draw: ImageDraw.ImageDraw,
    xywh: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> None:
    x, y, w, h = xywh
    bbox = draw.textbbox((0, 0), text, font=fnt)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((x + (w - tw) / 2, y + (h - th) / 2 - 1), text, font=fnt, fill=fill)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    prepare_posemasked_inputs()
    if args.prepare_only:
        print(PREP_ROOT)
        return

    tile = 150
    gap = 10
    header_h = 36
    row_label_w = 112
    bottom_pad = 12
    cols = len(HEADERS)
    rows = len(ROWS)

    width = row_label_w + cols * tile + (cols + 1) * gap
    height = header_h + rows * tile + (rows + 1) * gap + bottom_pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(15, bold=True)
    row_font = font(13, bold=True)

    for col, title in enumerate(HEADERS):
        x = row_label_w + gap + col * (tile + gap)
        centered_text(draw, (x, 0, tile, header_h), title, header_font)

    for row_idx, row in enumerate(ROWS):
        y = header_h + gap + row_idx * (tile + gap)
        centered_text(draw, (0, y, row_label_w, tile), row["label"], row_font)
        row_paths = paths_for(row)
        display_paths = [
            row_paths["source_real"],
            row_paths["source_ridge"],
            row_paths["target_mask"],
            row_paths["aligned_ridge"],
            row_paths["synth"],
        ]
        for col, path in enumerate(display_paths):
            x = row_label_w + gap + col * (tile + gap)
            tile_img = make_tile(path, tile - 10)
            canvas.paste(tile_img, (x + 5, y + 5))
            draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(210, 210, 210), width=1)

    out_path = FIG_ROOT / "sd302a_to_fvc_db1_lora_overview.png"
    canvas.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
