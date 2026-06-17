from __future__ import annotations

import ast
import csv
import json
import math
import struct
import zipfile
from pathlib import Path


ROOT = Path("/home/nguyenthanhlam")
FIG_ROOT = ROOT / "Fingen/figures"

RANK = 8
ALPHA = 8
MID_CHANNELS = 256
FULL_STAGE2_PARAMS = 23_390_000
CONTROLNET_PARAMS = 7_040_000

CHECKPOINTS = {
    "DB1": ROOT / "checkpoints/fvc2004_db1a_stage2_unet_lora_r8_5k_20260615/step_0005000/unet_state.npz",
    "DB2": ROOT / "checkpoints/fvc2004_db2a_stage2_unet_lora_r8_5k_20260617/step_0005000/unet_state.npz",
    "DB3": ROOT / "checkpoints/fvc2004_db3a_stage2_unet_lora_r8_5k_20260617/step_0005000/unet_state.npz",
    "DB4": ROOT / "checkpoints/fvc2004_db4a_stage2_unet_lora_r8_5k_20260617/step_0005000/unet_state.npz",
}

OUT_CSV = FIG_ROOT / "ablation_lora_placement.csv"
OUT_JSON = FIG_ROOT / "ablation_lora_placement.json"


def lora_dense_params(in_features: int, out_features: int, rank: int = RANK) -> int:
    return rank * (in_features + out_features)


def self_attention_lora_params(channels: int, rank: int = RANK) -> int:
    qkv = lora_dense_params(channels, 3 * channels, rank)
    proj = lora_dense_params(channels, channels, rank)
    return qkv + proj


def candidate_rows() -> list[dict[str, object]]:
    mid_qkv = lora_dense_params(MID_CHANNELS, 3 * MID_CHANNELS)
    mid_proj = lora_dense_params(MID_CHANNELS, MID_CHANNELS)
    mid_both = mid_qkv + mid_proj

    # If encoder/decoder attention were enabled at ds={1,2,4}, the compact
    # UNet would contain 5 attention blocks at C=64, 5 at C=128, and 6 at C=256.
    all_attention = (
        5 * self_attention_lora_params(64)
        + 5 * self_attention_lora_params(128)
        + 6 * self_attention_lora_params(256)
    )

    return [
        {
            "placement": "No adapter",
            "trainable_params": 0,
            "relative_to_ours": 0.0,
            "decision": "lower bound; cannot adapt sensor appearance",
        },
        {
            "placement": "Stage-2 mid-attn qkv only",
            "trainable_params": mid_qkv,
            "relative_to_ours": mid_qkv / mid_both,
            "decision": "changes attention scores/values but leaves output remix fixed",
        },
        {
            "placement": "Stage-2 mid-attn proj_out only",
            "trainable_params": mid_proj,
            "relative_to_ours": mid_proj / mid_both,
            "decision": "remixes attention output but cannot change q/k/v content",
        },
        {
            "placement": "Stage-2 mid-attn qkv + proj_out",
            "trainable_params": mid_both,
            "relative_to_ours": 1.0,
            "decision": "selected; global style adapter with frozen ridge path",
        },
        {
            "placement": "ControlNet mid-attn qkv + proj_out",
            "trainable_params": mid_both,
            "relative_to_ours": 1.0,
            "decision": "rejected; edits the structural ridge-control path",
        },
        {
            "placement": "All Stage-2 self-attention blocks",
            "trainable_params": all_attention,
            "relative_to_ours": all_attention / mid_both,
            "decision": "rejected; 9.8x more adapter capacity for 80-image fitting",
        },
        {
            "placement": "Full ControlNet fine-tune",
            "trainable_params": CONTROLNET_PARAMS,
            "relative_to_ours": CONTROLNET_PARAMS / mid_both,
            "decision": "rejected; adapts structure rather than target appearance",
        },
        {
            "placement": "Full Stage-2 UNet fine-tune",
            "trainable_params": FULL_STAGE2_PARAMS,
            "relative_to_ours": FULL_STAGE2_PARAMS / mid_both,
            "decision": "rejected; too many parameters for small sensor adapters",
        },
    ]


def read_npy_float32(zf: zipfile.ZipFile, name: str) -> tuple[tuple[int, ...], tuple[float, ...], bool]:
    blob = zf.read(name)
    if not blob.startswith(b"\x93NUMPY"):
        raise ValueError(f"{name} is not an npy member")
    major = blob[6]
    if major == 1:
        header_len = struct.unpack("<H", blob[8:10])[0]
        offset = 10
    elif major == 2:
        header_len = struct.unpack("<I", blob[8:12])[0]
        offset = 12
    else:
        raise ValueError(f"Unsupported npy version {major} for {name}")

    header = ast.literal_eval(blob[offset : offset + header_len].decode("latin1"))
    offset += header_len
    shape = tuple(header["shape"])
    if header["descr"] != "<f4":
        raise ValueError(f"Expected float32 for {name}, got {header['descr']}")

    count = 1
    for dim in shape:
        count *= dim
    values = struct.unpack("<" + "f" * count, blob[offset : offset + 4 * count])
    return shape, values, bool(header["fortran_order"])


def matrix_value(values: tuple[float, ...], shape: tuple[int, ...], fortran_order: bool, i: int, j: int) -> float:
    rows, cols = shape
    if fortran_order:
        return values[i + rows * j]
    return values[i * cols + j]


def frobenius_norm(values: tuple[float, ...]) -> float:
    return math.sqrt(sum(value * value for value in values))


def lora_update_norm(
    down_shape: tuple[int, ...],
    down: tuple[float, ...],
    down_fortran: bool,
    up_shape: tuple[int, ...],
    up: tuple[float, ...],
    up_fortran: bool,
    scale: float,
) -> float:
    in_features, rank = down_shape
    rank2, out_features = up_shape
    if rank != rank2:
        raise ValueError(f"LoRA rank mismatch: {down_shape} vs {up_shape}")

    total = 0.0
    for i in range(in_features):
        for j in range(out_features):
            value = 0.0
            for k in range(rank):
                value += matrix_value(down, down_shape, down_fortran, i, k) * matrix_value(
                    up,
                    up_shape,
                    up_fortran,
                    k,
                    j,
                )
            value *= scale
            total += value * value
    return math.sqrt(total)


def checkpoint_update_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scale = ALPHA / RANK
    for db, path in CHECKPOINTS.items():
        with zipfile.ZipFile(path) as zf:
            for module in ("qkv", "proj_out"):
                prefix = f"ema_params/mid_attn/{module}"
                down_shape, down, down_fortran = read_npy_float32(zf, f"{prefix}/lora_down.npy")
                up_shape, up, up_fortran = read_npy_float32(zf, f"{prefix}/lora_up.npy")
                _kernel_shape, kernel, _kernel_fortran = read_npy_float32(zf, f"{prefix}/kernel.npy")
                update_norm = lora_update_norm(
                    down_shape,
                    down,
                    down_fortran,
                    up_shape,
                    up,
                    up_fortran,
                    scale,
                )
                kernel_norm = frobenius_norm(kernel)
                rows.append(
                    {
                        "dataset": db,
                        "module": module,
                        "trainable_params": len(down) + len(up),
                        "update_fro_norm": update_norm,
                        "kernel_fro_norm": kernel_norm,
                        "update_to_kernel_pct": 100.0 * update_norm / kernel_norm,
                    }
                )
    return rows


def summarize_updates(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for module in ("qkv", "proj_out"):
        selected = [row for row in rows if row["module"] == module]
        ratios = [float(row["update_to_kernel_pct"]) for row in selected]
        updates = [float(row["update_fro_norm"]) for row in selected]
        summary[module] = {
            "mean_update_fro_norm": sum(updates) / len(updates),
            "mean_update_to_kernel_pct": sum(ratios) / len(ratios),
            "min_update_to_kernel_pct": min(ratios),
            "max_update_to_kernel_pct": max(ratios),
        }
    return summary


def main() -> None:
    candidates = candidate_rows()
    updates = checkpoint_update_rows()
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["placement", "trainable_params", "relative_to_ours", "decision"],
        )
        writer.writeheader()
        writer.writerows(candidates)

    payload = {
        "config": {
            "rank": RANK,
            "alpha": ALPHA,
            "current_mid_channels": MID_CHANNELS,
            "current_lora_params": self_attention_lora_params(MID_CHANNELS),
            "checkpoint_step": 5000,
        },
        "candidates": candidates,
        "checkpoint_update_rows": updates,
        "checkpoint_update_summary": summarize_updates(updates),
    }
    with OUT_JSON.open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print(json.dumps(payload["checkpoint_update_summary"], indent=2))


if __name__ == "__main__":
    main()
