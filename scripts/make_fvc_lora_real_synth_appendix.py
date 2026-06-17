from __future__ import annotations

import json
import os
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(os.environ.get("FINGEN_WORK_ROOT", Path.home()))
FIG_ROOT = REPO / "figures"

DB_SPECS = {
    "DB1": {
        "synthetic": WORK_ROOT
        / "outputs/fvc2004_db1a_lora_stage2_synth500x10_20260615/images/challengers/DB1A/roll/png",
        "real": WORK_ROOT / "data/fvc2004_db1a_png512",
    },
    "DB2": {
        "synthetic": WORK_ROOT
        / "outputs/fvc2004_db2a_lora_stage2_synth500x10_20260617/images/challengers/DB2A/roll/png",
        "real": WORK_ROOT / "data/fvc2004_db2a_png512",
    },
    "DB3": {
        "synthetic": WORK_ROOT
        / "outputs/fvc2004_db3a_lora_stage2_synth500x10_20260617/images/challengers/DB3A/roll/png",
        "real": WORK_ROOT / "data/fvc2004_db3a_png512",
    },
    "DB4": {
        "synthetic": WORK_ROOT
        / "outputs/fvc2004_db4a_lora_stage2_synth500x10_20260617/images/challengers/DB4A/roll/png",
        "real": WORK_ROOT / "data/fvc2004_db4a_png512",
    },
}

SAMPLES_PER_SPLIT = 32
GRID_COLS = 8
GRID_ROWS = 4
TILE = 165
GAP = 10
CANVAS_W = 1500
CANVAS_H = 1985


def natural_key(path: Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.name)]


def identity_key(path: Path) -> str:
    if path.name.startswith("synth"):
        return path.stem.split("_roll_")[0]
    return path.stem.split("_")[0]


def image_score(path: Path) -> float:
    gray = Image.open(path).convert("L").resize((96, 96))
    if hasattr(gray, "get_flattened_data"):
        pixels = list(gray.get_flattened_data())
    else:
        pixels = list(gray.getdata())
    fg_pixels = [pix for pix in pixels if pix < 245]
    fg = len(fg_pixels) / len(pixels)
    if fg < 0.04:
        return -1.0
    mean = sum(fg_pixels) / len(fg_pixels)
    variance = sum((pix - mean) ** 2 for pix in fg_pixels) / len(fg_pixels)
    contrast = (variance ** 0.5) / 255.0
    foreground_bonus = 1.0 - min(abs(fg - 0.35) / 0.35, 1.0)
    return contrast + 0.25 * foreground_bonus


def one_sample_per_identity(paths: list[Path], max_candidates: int = 420) -> list[Path]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(identity_key(path), []).append(path)

    identities = sorted(grouped, key=lambda name: natural_key(Path(f"{name}.png")))
    if len(identities) > max_candidates:
        step = len(identities) / max_candidates
        identities = [identities[int(i * step)] for i in range(max_candidates)]

    candidates = []
    for identity in identities:
        samples = sorted(grouped[identity], key=natural_key)
        candidates.append(samples[len(samples) // 2])
    return candidates


def pick_examples(root: Path, count: int = SAMPLES_PER_SPLIT) -> list[Path]:
    paths = sorted(root.glob("*.png"), key=natural_key)
    candidates = one_sample_per_identity(paths)
    scored = [(image_score(path), path) for path in candidates]
    scored = [(score, path) for score, path in scored if score > 0]
    scored.sort(key=lambda item: item[0], reverse=True)

    pool_size = min(len(scored), max(count * 4, count))
    pool = scored[:pool_size]
    if len(pool) < count:
        raise RuntimeError(f"Could not pick {count} examples from {root}")

    if count == 1:
        return [pool[0][1]]

    step = (len(pool) - 1) / (count - 1)
    picked = [pool[round(i * step)][1] for i in range(count)]
    return picked


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: tuple[int, int, int] = (25, 25, 25),
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    x = box[0] + (box[2] - box[0] - (bbox[2] - bbox[0])) // 2
    y = box[1] + (box[3] - box[1] - (bbox[3] - bbox[1])) // 2
    draw.text((x, y), text, font=font, fill=fill)


def make_tile(path: Path, size: int = TILE) -> Image.Image:
    image = Image.open(path).convert("L")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size), "white")
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    tile.paste(Image.merge("RGB", (image, image, image)), (x, y))
    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, 0, size - 1, size - 1), outline=(205, 205, 205), width=1)
    return tile


def draw_grid(canvas: Image.Image, paths: list[Path], y0: int) -> None:
    grid_w = GRID_COLS * TILE + (GRID_COLS - 1) * GAP
    x0 = (CANVAS_W - grid_w) // 2
    for idx, path in enumerate(paths):
        row, col = divmod(idx, GRID_COLS)
        x = x0 + col * (TILE + GAP)
        y = y0 + row * (TILE + GAP)
        canvas.paste(make_tile(path), (x, y))


def make_page(db: str, synthetic: list[Path], real: list[Path]) -> Image.Image:
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), "white")
    draw = ImageDraw.Draw(canvas)

    title_font = load_font(36, bold=True)
    section_font = load_font(26, bold=True)

    draw_centered_text(
        draw,
        (0, 30, CANVAS_W, 88),
        f"FVC 2004 {db}: LoRA synthetic vs real {db}_A",
        title_font,
    )

    synthetic_heading_y = 105
    synthetic_grid_y = 145
    real_heading_y = 1010
    real_grid_y = 1050

    draw_centered_text(
        draw,
        (0, synthetic_heading_y, CANVAS_W, synthetic_heading_y + 36),
        f"Synthetic generated by the {db} Stage-2 LoRA checkpoint",
        section_font,
        fill=(20, 20, 20),
    )
    draw_grid(canvas, synthetic, synthetic_grid_y)

    draw_centered_text(
        draw,
        (0, real_heading_y, CANVAS_W, real_heading_y + 36),
        f"Real {db}_A impressions",
        section_font,
        fill=(20, 20, 20),
    )
    draw_grid(canvas, real, real_grid_y)

    return canvas


def main() -> None:
    metadata: dict[str, dict[str, list[str]]] = {}
    FIG_ROOT.mkdir(parents=True, exist_ok=True)

    for db, spec in DB_SPECS.items():
        synthetic = pick_examples(spec["synthetic"])
        real = pick_examples(spec["real"])

        page = make_page(db, synthetic, real)
        out_path = FIG_ROOT / f"fvc_lora_{db.lower()}_real_synth_page.png"
        page.save(out_path)

        metadata[db] = {
            "synthetic": [str(path.relative_to(WORK_ROOT)) for path in synthetic],
            "real": [str(path.relative_to(WORK_ROOT)) for path in real],
            "figure": str(out_path.relative_to(REPO)),
        }
        print(f"Wrote {out_path}")

    meta_path = FIG_ROOT / "fvc_lora_real_synth_appendix.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
