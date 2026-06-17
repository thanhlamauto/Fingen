from __future__ import annotations

import csv
import json
import math
import random
import statistics
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


ROOT = Path("/home/nguyenthanhlam")
FIG_ROOT = ROOT / "Fingen/figures"
IMG_ROOT = ROOT / "kaggle_downloads/fingerprint_datasets/sd302a/images/challengers"
SAUVOLA_ROOT = ROOT / "data/ridge_conditions_sauvola_w11_k0007"

SENSORS = list("ABCDEFGH")
SEED = 20260617
PER_SENSOR = 20
CANNY_LOW = 100
CANNY_HIGH = 200
FOREGROUND_THRESHOLD = 248
SMALL_COMPONENT_PX = 32

OUT_FIG = FIG_ROOT / "ablation_sauvola_vs_canny.png"
OUT_CSV = FIG_ROOT / "ablation_sauvola_vs_canny.csv"
OUT_JSON = FIG_ROOT / "ablation_sauvola_vs_canny.json"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


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


def foreground_bbox(img: Image.Image, pad: int = 22) -> tuple[int, int, int, int]:
    gray = img.convert("L")
    mask = gray.point(lambda p: 255 if p < FOREGROUND_THRESHOLD else 0)
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


def make_tile(img: Image.Image, bbox: tuple[int, int, int, int], size: int, enhance: bool) -> Image.Image:
    tile_img = img.convert("L").crop(bbox)
    if enhance:
        tile_img = ImageOps.autocontrast(tile_img, cutoff=0.2)
    resample = Image.Resampling.LANCZOS if enhance else Image.Resampling.NEAREST
    tile_img.thumbnail((size, size), resample)
    tile = Image.new("L", (size, size), 255)
    tile.paste(tile_img, ((size - tile_img.width) // 2, (size - tile_img.height) // 2))
    return tile.convert("RGB")


def load_condition_pair(img_path: Path, sauvola_path: Path) -> tuple[Image.Image, Image.Image]:
    sauvola = Image.open(sauvola_path).convert("L")
    gray = Image.open(img_path).convert("L")
    if gray.size != sauvola.size:
        gray = gray.resize(sauvola.size, Image.Resampling.BILINEAR)
    return gray, sauvola


def active_to_image(active: bytearray, width: int, height: int) -> Image.Image:
    data = bytearray(len(active))
    for i, value in enumerate(active):
        data[i] = 0 if value else 255
    return Image.frombytes("L", (width, height), bytes(data))


def canny_edges(gray: Image.Image, low: int = CANNY_LOW, high: int = CANNY_HIGH) -> bytearray:
    gray = ImageOps.autocontrast(gray.convert("L"), cutoff=0.2).filter(ImageFilter.GaussianBlur(1.0))
    width, height = gray.size
    pixels = gray.tobytes()
    total = width * height
    magnitude = [0.0] * total
    direction = bytearray(total)

    for y in range(1, height - 1):
        row = y * width
        for x in range(1, width - 1):
            idx = row + x
            a = pixels[idx - width - 1]
            b = pixels[idx - width]
            c = pixels[idx - width + 1]
            d = pixels[idx - 1]
            f = pixels[idx + 1]
            g = pixels[idx + width - 1]
            h = pixels[idx + width]
            k = pixels[idx + width + 1]

            gx = -a + c - 2 * d + 2 * f - g + k
            gy = -a - 2 * b - c + g + 2 * h + k
            mag = math.hypot(gx, gy)
            magnitude[idx] = mag

            abs_x = abs(gx)
            abs_y = abs(gy)
            if abs_y * 2 <= abs_x:
                direction[idx] = 0
            elif abs_x * 2 <= abs_y:
                direction[idx] = 2
            elif gx * gy > 0:
                direction[idx] = 1
            else:
                direction[idx] = 3

    keep = bytearray(total)
    for y in range(1, height - 1):
        row = y * width
        for x in range(1, width - 1):
            idx = row + x
            mag = magnitude[idx]
            if mag < low:
                continue
            sector = direction[idx]
            if sector == 0:
                is_max = mag >= magnitude[idx - 1] and mag >= magnitude[idx + 1]
            elif sector == 2:
                is_max = mag >= magnitude[idx - width] and mag >= magnitude[idx + width]
            elif sector == 1:
                is_max = mag >= magnitude[idx - width - 1] and mag >= magnitude[idx + width + 1]
            else:
                is_max = mag >= magnitude[idx - width + 1] and mag >= magnitude[idx + width - 1]
            if is_max:
                keep[idx] = 1

    edges = bytearray(total)
    stack: list[int] = []
    for idx, value in enumerate(keep):
        if value and magnitude[idx] >= high:
            edges[idx] = 1
            stack.append(idx)

    while stack:
        idx = stack.pop()
        y, x = divmod(idx, width)
        for yy in (y - 1, y, y + 1):
            if yy < 0 or yy >= height:
                continue
            base = yy * width
            for xx in (x - 1, x, x + 1):
                if xx < 0 or xx >= width or (xx == x and yy == y):
                    continue
                n_idx = base + xx
                if keep[n_idx] and not edges[n_idx]:
                    edges[n_idx] = 1
                    stack.append(n_idx)

    return edges


def foreground_mask(gray: Image.Image) -> bytearray:
    return bytearray(1 if value < FOREGROUND_THRESHOLD else 0 for value in gray.convert("L").tobytes())


def binary_active(img: Image.Image) -> bytearray:
    return bytearray(1 if value < 128 else 0 for value in img.convert("L").tobytes())


def interior_support(active: bytearray, fg: bytearray, width: int, height: int) -> int:
    count = 0
    for y in range(1, height - 1):
        row = y * width
        for x in range(1, width - 1):
            idx = row + x
            if not (active[idx] and fg[idx]):
                continue
            ok = True
            for yy in (y - 1, y, y + 1):
                base = yy * width
                for xx in (x - 1, x, x + 1):
                    if not active[base + xx]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                count += 1
    return count


def component_stats(
    active: bytearray,
    fg: bytearray,
    width: int,
    height: int,
) -> tuple[int, int, int]:
    total = width * height
    visited = bytearray(total)
    components = 0
    small_components = 0
    largest_component = 0

    for idx in range(total):
        if visited[idx] or not (active[idx] and fg[idx]):
            continue
        components += 1
        size = 0
        visited[idx] = 1
        stack = [idx]
        while stack:
            current = stack.pop()
            size += 1
            y, x = divmod(current, width)
            for yy in (y - 1, y, y + 1):
                if yy < 0 or yy >= height:
                    continue
                base = yy * width
                for xx in (x - 1, x, x + 1):
                    if xx < 0 or xx >= width:
                        continue
                    n_idx = base + xx
                    if active[n_idx] and fg[n_idx] and not visited[n_idx]:
                        visited[n_idx] = 1
                        stack.append(n_idx)
        largest_component = max(largest_component, size)
        if size < SMALL_COMPONENT_PX:
            small_components += 1

    return components, small_components, largest_component


def measure(active: bytearray, fg: bytearray, width: int, height: int) -> dict[str, float]:
    foreground_count = sum(fg)
    active_count = sum(1 for a, m in zip(active, fg) if a and m)
    eroded_count = interior_support(active, fg, width, height)
    components, small_components, largest_component = component_stats(active, fg, width, height)

    return {
        "active_fg_pct": 100.0 * active_count / foreground_count if foreground_count else 0.0,
        "interior_active_pct": 100.0 * eroded_count / active_count if active_count else 0.0,
        "components_per_10k_active": components / (active_count / 10000.0) if active_count else 0.0,
        "small_component_pct": 100.0 * small_components / components if components else 0.0,
        "largest_component_px": float(largest_component),
    }


def collect_samples() -> list[tuple[str, Path, Path]]:
    samples: list[tuple[str, Path, Path]] = []
    for sensor in SENSORS:
        sensor_root = IMG_ROOT / sensor / "roll/png"
        files = sorted(sensor_root.glob("*.png"))
        rng = random.Random(SEED + ord(sensor))
        chosen = sorted(rng.sample(files, min(PER_SENSOR, len(files))))
        for img_path in chosen:
            rel = img_path.relative_to(IMG_ROOT)
            sauvola_path = SAUVOLA_ROOT / rel
            if sauvola_path.exists():
                samples.append((sensor, img_path, sauvola_path))
    return samples


def summarize(values: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = [
        "active_fg_pct",
        "interior_active_pct",
        "components_per_10k_active",
        "small_component_pct",
        "largest_component_px",
    ]
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        series = [row[key] for row in values]
        summary[key] = {
            "mean": statistics.mean(series),
            "std": statistics.stdev(series) if len(series) > 1 else 0.0,
        }
    return summary


def write_outputs(rows: list[dict[str, object]]) -> dict[str, object]:
    fieldnames = [
        "sensor",
        "image",
        "method",
        "active_fg_pct",
        "interior_active_pct",
        "components_per_10k_active",
        "small_component_pct",
        "largest_component_px",
    ]
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})

    grouped: dict[str, list[dict[str, float]]] = {"Sauvola": [], "Canny": []}
    for row in rows:
        method = str(row["method"])
        grouped[method].append(
            {
                "active_fg_pct": float(row["active_fg_pct"]),
                "interior_active_pct": float(row["interior_active_pct"]),
                "components_per_10k_active": float(row["components_per_10k_active"]),
                "small_component_pct": float(row["small_component_pct"]),
                "largest_component_px": float(row["largest_component_px"]),
            }
        )

    payload: dict[str, object] = {
        "config": {
            "seed": SEED,
            "sensors": SENSORS,
            "per_sensor": PER_SENSOR,
            "num_images": len(rows) // 2,
            "foreground_threshold": FOREGROUND_THRESHOLD,
            "canny": {
                "low": CANNY_LOW,
                "high": CANNY_HIGH,
                "preprocess": "PIL autocontrast cutoff=0.2, GaussianBlur radius=1.0",
                "implementation": "Sobel gradients, non-maximum suppression, hysteresis",
            },
            "small_component_px": SMALL_COMPONENT_PX,
        },
        "summary": {method: summarize(values) for method, values in grouped.items()},
    }
    with OUT_JSON.open("w") as handle:
        json.dump(payload, handle, indent=2)
    return payload


def make_figure() -> None:
    rows = [
        ("SD302A B", "B/roll/png/00002302_B_roll_01.png"),
        ("SD302A D", "D/roll/png/00002302_D_roll_01.png"),
        ("SD302A G", "G/roll/png/00002302_G_roll_01.png"),
        ("SD302A H", "H/roll/png/00002302_H_roll_01.png"),
    ]
    headers = ["Real", "Sauvola ridge", "Canny edge"]
    tile = 170
    row_label_w = 92
    gap = 14
    header_h = 34
    bottom_pad = 12
    width = row_label_w + len(headers) * tile + (len(headers) + 1) * gap
    height = header_h + len(rows) * tile + (len(rows) + 1) * gap + bottom_pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    header_font = font(17, bold=True)
    row_font = font(14, bold=True)

    for col, title in enumerate(headers):
        x = row_label_w + gap + col * (tile + gap)
        centered_text(draw, (x, 0, tile, header_h), title, header_font)

    for row_idx, (label, rel) in enumerate(rows):
        img_path = IMG_ROOT / rel
        sauvola_path = SAUVOLA_ROOT / rel
        gray, sauvola = load_condition_pair(img_path, sauvola_path)
        canny = active_to_image(canny_edges(gray), gray.width, gray.height)
        bbox = foreground_bbox(gray)
        y = header_h + gap + row_idx * (tile + gap)
        centered_text(draw, (0, y, row_label_w, tile), label, row_font)
        imgs = [
            make_tile(gray, bbox, tile - 8, enhance=True),
            make_tile(sauvola, bbox, tile - 8, enhance=False),
            make_tile(canny, bbox, tile - 8, enhance=False),
        ]
        for col, tile_img in enumerate(imgs):
            x = row_label_w + gap + col * (tile + gap)
            canvas.paste(tile_img, (x + 4, y + 4))
            draw.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(210, 210, 210), width=1)

    canvas.save(OUT_FIG)


def main() -> None:
    metric_rows: list[dict[str, object]] = []
    samples = collect_samples()
    for sensor, img_path, sauvola_path in samples:
        gray, sauvola = load_condition_pair(img_path, sauvola_path)
        width, height = gray.size
        fg = foreground_mask(gray)
        sauvola_active = binary_active(sauvola)
        canny_active = canny_edges(gray)

        rel = str(img_path.relative_to(IMG_ROOT))
        for method, active in (("Sauvola", sauvola_active), ("Canny", canny_active)):
            row = {
                "sensor": sensor,
                "image": rel,
                "method": method,
            }
            row.update(measure(active, fg, width, height))
            metric_rows.append(row)

    payload = write_outputs(metric_rows)
    make_figure()

    print(f"Wrote {OUT_FIG}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
