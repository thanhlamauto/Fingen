from __future__ import annotations

import csv
import json
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np


ROOT = Path("/home/nguyenthanhlam")
FIG_ROOT = ROOT / "Fingen/figures"

RANK_MAX = 8
ALPHA = 8
MID_CHANNELS = 256
MODULES = ("qkv", "proj_out")
THRESHOLDS = (0.90, 0.95, 0.99)

CHECKPOINTS = {
    "DB1": ROOT / "checkpoints/fvc2004_db1a_stage2_unet_lora_r8_5k_20260615/step_0005000/unet_state.npz",
    "DB2": ROOT / "checkpoints/fvc2004_db2a_stage2_unet_lora_r8_5k_20260617/step_0005000/unet_state.npz",
    "DB3": ROOT / "checkpoints/fvc2004_db3a_stage2_unet_lora_r8_5k_20260617/step_0005000/unet_state.npz",
    "DB4": ROOT / "checkpoints/fvc2004_db4a_stage2_unet_lora_r8_5k_20260617/step_0005000/unet_state.npz",
}

OUT_CSV = FIG_ROOT / "ablation_lora_rank_capacity.csv"
OUT_JSON = FIG_ROOT / "ablation_lora_rank_capacity.json"


def lora_params(rank: int) -> int:
    qkv = rank * (MID_CHANNELS + 3 * MID_CHANNELS)
    proj = rank * (MID_CHANNELS + MID_CHANNELS)
    return qkv + proj


def npy_from_zip(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as handle:
        return np.load(BytesIO(handle.read()))


def energy_rank(singular_values: np.ndarray, threshold: float) -> int:
    energy = np.square(singular_values)
    cumulative = np.cumsum(energy) / max(float(np.sum(energy)), 1e-12)
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def module_spectrum(zf: zipfile.ZipFile, module: str) -> dict[str, object]:
    prefix = f"ema_params/mid_attn/{module}"
    down = npy_from_zip(zf, f"{prefix}/lora_down.npy").astype(np.float64)
    up = npy_from_zip(zf, f"{prefix}/lora_up.npy").astype(np.float64)
    update = (ALPHA / RANK_MAX) * down @ up
    singular_values = np.linalg.svd(update, compute_uv=False)
    singular_values = singular_values[:RANK_MAX]
    energy = np.square(singular_values)
    cumulative = np.cumsum(energy) / max(float(np.sum(energy)), 1e-12)
    normalized_energy = energy / max(float(np.sum(energy)), 1e-12)
    return {
        "top_singular_values": [float(value) for value in singular_values],
        "energy": [float(value) for value in energy],
        "normalized_energy": [float(value) for value in normalized_energy],
        "cumulative_energy": [float(value) for value in cumulative],
        "k90": energy_rank(singular_values, 0.90),
        "k95": energy_rank(singular_values, 0.95),
        "k99": energy_rank(singular_values, 0.99),
        "effective_rank_entropy": float(np.exp(-np.sum((energy / np.sum(energy)) * np.log((energy / np.sum(energy)) + 1e-12)))),
    }


def combined_energy(modules: dict[str, dict[str, object]]) -> dict[str, object]:
    # Use total squared singular value energy across qkv and projection modules.
    # For a shared rank choice k, retain the first k singular modes in each module.
    total_energy = 0.0
    retained_by_k = []
    for k in range(1, RANK_MAX + 1):
        retained = 0.0
        for module_payload in modules.values():
            energy = np.asarray(module_payload["energy"], dtype=np.float64)
            total_energy += 0.0 if k > 1 else float(np.sum(energy))
            retained += float(np.sum(energy[:k]))
        retained_by_k.append(retained)

    ratios = np.asarray(retained_by_k, dtype=np.float64) / max(total_energy, 1e-12)
    return {
        "cumulative_energy": ratios.tolist(),
        "k90": int(np.searchsorted(ratios, 0.90, side="left") + 1),
        "k95": int(np.searchsorted(ratios, 0.95, side="left") + 1),
        "k99": int(np.searchsorted(ratios, 0.99, side="left") + 1),
        "energy_at_k4": float(ratios[3]),
        "energy_at_k6": float(ratios[5]),
        "energy_at_k8": float(ratios[7]),
    }


def main() -> None:
    rows: list[dict[str, object]] = []
    payload: dict[str, object] = {
        "config": {
            "pilot_rank": RANK_MAX,
            "alpha": ALPHA,
            "modules": MODULES,
            "params_per_rank": lora_params(1),
            "rank8_params": lora_params(RANK_MAX),
            "thresholds": THRESHOLDS,
            "rule": "k_tau is the smallest shared rank whose cumulative squared-singular-value energy reaches tau across qkv and proj_out updates.",
        },
        "sensors": {},
    }

    for db, path in CHECKPOINTS.items():
        with zipfile.ZipFile(path) as zf:
            modules = {module: module_spectrum(zf, module) for module in MODULES}
        combined = combined_energy(modules)
        row = {
            "sensor": db,
            "qkv_k95": modules["qkv"]["k95"],
            "proj_k95": modules["proj_out"]["k95"],
            "combined_k90": combined["k90"],
            "combined_k95": combined["k95"],
            "combined_k99": combined["k99"],
            "energy_at_k4": combined["energy_at_k4"],
            "energy_at_k6": combined["energy_at_k6"],
            "params_k95": lora_params(int(combined["k95"])),
            "params_k95_pct_rank8": lora_params(int(combined["k95"])) / lora_params(RANK_MAX),
        }
        rows.append(row)
        payload["sensors"][db] = {
            "checkpoint": str(path),
            "modules": modules,
            "combined": combined,
            "row": row,
        }

    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with OUT_JSON.open("w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
