#!/usr/bin/env python3
"""Build a compact visual sheet for the DB1 Stage-1/Stage-2 demo run."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANEL = 150
LABEL_H = 28
GAP = 12
MARGIN = 24
TITLE_H = 52


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_image(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("L")


def fit_panel(path: Path | None = None, image: Image.Image | None = None) -> Image.Image:
    if image is None:
        if path is None:
            raise ValueError("Either path or image is required")
        image = load_image(path)
    image = image.convert("L")
    image.thumbnail((PANEL, PANEL), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (PANEL, PANEL), "white")
    x = (PANEL - image.width) // 2
    y = (PANEL - image.height) // 2
    canvas.paste(Image.merge("RGB", (image, image, image)), (x, y))
    return canvas


def draw_panel(
    canvas: Image.Image,
    x: int,
    y: int,
    label: str,
    path: Path | None = None,
    image: Image.Image | None = None,
) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        [x, y, x + PANEL - 1, y + LABEL_H + PANEL - 1],
        radius=4,
        outline=(210, 210, 210),
        width=1,
        fill=(252, 252, 252),
    )
    draw.text((x + 8, y + 6), label, fill=(20, 20, 20), font=font(13, bold=True))
    canvas.paste(fit_panel(path=path, image=image), (x, y + LABEL_H))


def draw_text_panel(canvas: Image.Image, x: int, y: int, title: str, lines: list[str]) -> None:
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle(
        [x, y, x + PANEL - 1, y + LABEL_H + PANEL - 1],
        radius=4,
        outline=(210, 210, 210),
        width=1,
        fill=(252, 252, 252),
    )
    draw.text((x + 8, y + 6), title, fill=(20, 20, 20), font=font(13, bold=True))
    body_y = y + LABEL_H + 16
    for line in lines:
        draw.text((x + 10, body_y), line, fill=(50, 50, 50), font=font(12))
        body_y += 21


def mask_image(mask_arr: np.ndarray) -> Image.Image:
    arr = (mask_arr.astype(np.uint8) * 255)
    return Image.fromarray(arr, mode="L")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: make_demo_stage1_stage2_montage.py OUTDIR")

    outdir = Path(sys.argv[1]).expanduser()
    manifest = read_csv(outdir / "manifest.csv")
    if not manifest:
        raise RuntimeError(f"No rows in {outdir / 'manifest.csv'}")

    manifest.sort(key=lambda row: int(row["impression_index"]))
    identity_id = manifest[0]["identity_id"]
    sensor = manifest[0]["target_sensor"]

    stage1_path = outdir / "stage1_identities" / f"{identity_id}.png"
    ridge_path = outdir / "identity_ridges" / f"{identity_id}.png"
    mask_npz = outdir / "pose_mask_library" / f"{sensor}_masks_512.npz"
    masks = np.load(mask_npz)["masks"].astype(bool)

    records = read_jsonl(outdir / "pose_alignment_manifest.jsonl")
    record_by_rel = {
        str(Path(row.get("hint_path", "")).relative_to(outdir / "pose_aligned_hints")): row
        for row in records
        if row.get("hint_path")
    }

    visual_dir = outdir / "demo_visuals"
    mask_dir = visual_dir / "target_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in manifest:
        rel = row["output_relpath"]
        rec = record_by_rel.get(rel)
        if rec is None:
            raise RuntimeError(f"No pose record found for {rel}")
        mask_idx = int(rec["target_mask_index"])
        mask = mask_image(masks[mask_idx])
        mask_path = mask_dir / f"{Path(rel).stem}_mask.png"
        mask.save(mask_path)
        rows.append(
            {
                "index": int(row["impression_index"]),
                "mask_path": mask_path,
                "hint_path": Path(rec["hint_path"]),
                "output_path": outdir / "images" / "challengers" / rel,
            }
        )

    cols = 6
    panel_h = LABEL_H + PANEL
    width = MARGIN * 2 + cols * PANEL + (cols - 1) * GAP
    height = MARGIN + TITLE_H + panel_h + GAP + 5 * panel_h + 4 * GAP + MARGIN
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (MARGIN, MARGIN),
        "DB1 LoRA Demo: Stage 1 identity -> ridge/mask augmentation -> 10 Stage 2 instances",
        fill=(15, 15, 15),
        font=font(18, bold=True),
    )

    xs = [MARGIN + i * (PANEL + GAP) for i in range(cols)]
    y = MARGIN + TITLE_H
    first = rows[0]
    draw_panel(canvas, xs[0], y, "Stage 1", path=stage1_path)
    draw_panel(canvas, xs[1], y, "Extracted ridge", path=ridge_path)
    draw_panel(canvas, xs[2], y, "DB1 mask 01", path=first["mask_path"])
    draw_panel(canvas, xs[3], y, "Aug ridge 01", path=first["hint_path"])
    draw_panel(canvas, xs[4], y, "Stage 2 01", path=first["output_path"])
    draw_text_panel(
        canvas,
        xs[5],
        y,
        "Run",
        [
            f"identity: {identity_id}",
            f"sensor: {sensor}",
            "LoRA: DB1 rank 8",
            "instances: 10",
        ],
    )

    y += panel_h + GAP
    for block in range(5):
        left = rows[2 * block]
        right = rows[2 * block + 1]
        draw_panel(canvas, xs[0], y, f"mask {left['index']:02d}", path=left["mask_path"])
        draw_panel(canvas, xs[1], y, f"aug ridge {left['index']:02d}", path=left["hint_path"])
        draw_panel(canvas, xs[2], y, f"stage2 {left['index']:02d}", path=left["output_path"])
        draw_panel(canvas, xs[3], y, f"mask {right['index']:02d}", path=right["mask_path"])
        draw_panel(canvas, xs[4], y, f"aug ridge {right['index']:02d}", path=right["hint_path"])
        draw_panel(canvas, xs[5], y, f"stage2 {right['index']:02d}", path=right["output_path"])
        y += panel_h + GAP

    output = visual_dir / "db1_stage1_stage2_demo.png"
    canvas.save(output)
    print(f"Saved montage: {output}")
    print(f"Saved target masks: {mask_dir}")


if __name__ == "__main__":
    main()
