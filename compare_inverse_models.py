#!/usr/bin/env python3
"""
Compare inverse-design models (MDN vs cGAN) using the forward surrogate.

Workflow:
  1) Generate + train forward surrogate + train MDN inverse via mmi_mzi_project.py
  2) Generate MDN candidates via `mmi_mzi_project.py inverse-design`
  3) Train cGAN via `cgan_inverse.py train` and sample via `cgan_inverse.py sample`
  4) Run this script to evaluate both sets of candidates using the forward surrogate.

Example:

  python compare_inverse_models.py \
    --run-dir runs/pilot_v1 \
    --mdn-csv runs/pilot_v1/reports/inverse_candidates.csv \
    --cgan-csv runs/pilot_v1/reports/cgan_candidates.csv \
    --target-er 20 \
    --target-bw 40 \
    --target-il 1.0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import mmi_mzi_project as proj


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_forward_model(run_dir: Path) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    ckpt_path = run_dir / "checkpoints" / "forward_best.pt"
    x_scaler_path = run_dir / "checkpoints" / "forward_x_scaler.json"
    y_scaler_path = run_dir / "checkpoints" / "forward_y_scaler.json"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing forward surrogate checkpoint: {ckpt_path}")
    if not x_scaler_path.exists() or not y_scaler_path.exists():
        raise FileNotFoundError("Missing forward scalers JSONs in checkpoints/")

    x_scaler = load_json(x_scaler_path)
    y_scaler = load_json(y_scaler_path)

    in_dim = len(x_scaler["mean"])
    out_dim = len(y_scaler["mean"])
    model = proj.MLP(in_dim=in_dim, out_dim=out_dim, hidden=(256, 256, 256), dropout=0.0)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model, x_scaler, y_scaler


def standardize_apply(X: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    mean = np.array(scaler["mean"], dtype=np.float64)
    std = np.array(scaler["std"], dtype=np.float64)
    # Note: the forward-model scalers already include numerical padding (std += 1e-12) at fit time.
    # Do NOT add epsilon again here, otherwise you double-pad and shift the normalization.
    return (X - mean) / std


def unstandardize_apply(Ys: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    mean = np.array(scaler["mean"], dtype=np.float64)
    std = np.array(scaler["std"], dtype=np.float64)
    return Ys * std + mean


def load_lambda_grid(run_dir: Path) -> np.ndarray:
    device_dir = run_dir / "data" / "device_long"
    parts = sorted(list(device_dir.glob("part-*.parquet")))
    if not parts:
        parts = sorted(list(device_dir.glob("part-*.csv.gz")))
    if not parts:
        raise FileNotFoundError(f"No device_long shards found under {device_dir}")
    # Read just one shard to infer wavelength grid
    if parts[0].suffix == ".parquet":
        df = pd.read_parquet(parts[0])
    else:
        df = pd.read_csv(parts[0])
    wl = np.sort(df["lambda_nm"].unique().astype(np.int32))
    return wl


def forward_spectrum_from_surrogate(
    model: torch.nn.Module,
    x_scaler: Dict[str, Any],
    y_scaler: Dict[str, Any],
    geom_row: Dict[str, float],
    lambda_nm: np.ndarray,
) -> List[np.ndarray]:
    """
    Use the forward surrogate to predict S(λ) for a single geometry across lambda_nm.
    Assumes zero Monte Carlo offsets (dW_nm = dGap_nm = 0).
    """
    feat_cols = [
        "W_mmi_um",
        "L_mmi_um",
        "gap_um",
        "W_io_um",
        "taper_len_um",
        "dW_nm",
        "dGap_nm",
        "lambda_nm",
    ]
    X_rows = []
    for lam in lambda_nm:
        X_rows.append(
            [
                geom_row["W_mmi_um"],
                geom_row["L_mmi_um"],
                geom_row["gap_um"],
                geom_row["W_io_um"],
                geom_row["taper_len_um"],
                0.0,
                0.0,
                float(lam),
            ]
        )
    X = np.asarray(X_rows, dtype=np.float64)
    Xs = standardize_apply(X, x_scaler)
    with torch.no_grad():
        X_tensor = torch.from_numpy(Xs.astype(np.float32))
        Y_pred_s = model(X_tensor).numpy()
    Y_pred = unstandardize_apply(Y_pred_s, y_scaler)

    # Map to S matrices
    S_list: List[np.ndarray] = []
    for row in Y_pred:
        S = np.zeros((2, 2), dtype=np.complex128)
        S[0, 0] = row[0] + 1j * row[1]  # S_out1_in1
        S[1, 0] = row[2] + 1j * row[3]  # S_out2_in1
        S[0, 1] = row[4] + 1j * row[5]  # S_out1_in2
        S[1, 1] = row[6] + 1j * row[7]  # S_out2_in2
        S_list.append(S)
    return S_list


def evaluate_candidate(
    model: torch.nn.Module,
    x_scaler: Dict[str, Any],
    y_scaler: Dict[str, Any],
    lambda_nm: np.ndarray,
    geom_row: Dict[str, float],
    target_er: float,
    target_bw: float,
    target_il: float,
) -> Dict[str, float]:
    """
    Evaluate one geometry via forward surrogate + MZI metrics, then compute
    how well it meets the target ER/BW/IL.
    """
    S_list = forward_spectrum_from_surrogate(model, x_scaler, y_scaler, geom_row, lambda_nm)
    metrics = proj._mzi_metrics_from_coupler_spectrum(
        wl_nm=lambda_nm.astype(np.float64),
        S_list=S_list,
        er_threshold_db=20.0,
    )

    # Define success criteria with respect to user targets.
    ER1 = metrics["ER1_min_dB"]
    ER2 = metrics["ER2_min_dB"]
    BW1 = metrics["ER1_bw_nm"]
    BW2 = metrics["ER2_bw_nm"]
    IL1 = metrics["IL1_mean_dB"]
    IL2 = metrics["IL2_mean_dB"]

    success = (
        ER1 >= target_er
        and ER2 >= target_er
        and BW1 >= target_bw
        and BW2 >= target_bw
        and IL1 <= target_il
        and IL2 <= target_il
    )

    return {
        "ER1_min_dB": ER1,
        "ER2_min_dB": ER2,
        "ER1_bw_nm": BW1,
        "ER2_bw_nm": BW2,
        "IL1_mean_dB": IL1,
        "IL2_mean_dB": IL2,
        "success": float(success),
        "ER_shortfall_dB": max(0.0, target_er - min(ER1, ER2)),
        "BW_shortfall_nm": max(0.0, target_bw - min(BW1, BW2)),
        "IL_excess_dB": max(0.0, max(IL1, IL2) - target_il),
    }


@dataclass
class CompareConfig:
    run_dir: Path
    mdn_csv: Path
    cgan_csv: Path
    target_er: float
    target_bw: float
    target_il: float
    max_samples: int | None = None


def compare_models(cfg: CompareConfig) -> None:
    model, x_scaler, y_scaler = load_forward_model(cfg.run_dir)
    lambda_nm = load_lambda_grid(cfg.run_dir)

    def load_geom_csv(path: Path) -> List[Dict[str, float]]:
        df = pd.read_csv(path)
        rows = []
        for _, r in df.iterrows():
            rows.append(
                {
                    "W_mmi_um": float(r["W_mmi_um"]),
                    "L_mmi_um": float(r["L_mmi_um"]),
                    "gap_um": float(r["gap_um"]),
                    "W_io_um": float(r["W_io_um"]),
                    "taper_len_um": float(r["taper_len_um"]),
                }
            )
        if cfg.max_samples is not None and len(rows) > cfg.max_samples:
            rows = rows[: cfg.max_samples]
        return rows

    mdn_rows = load_geom_csv(cfg.mdn_csv)
    cgan_rows = load_geom_csv(cfg.cgan_csv)

    print(f"Evaluating MDN candidates: {len(mdn_rows)}")
    mdn_metrics = [
        evaluate_candidate(model, x_scaler, y_scaler, lambda_nm, g, cfg.target_er, cfg.target_bw, cfg.target_il)
        for g in mdn_rows
    ]

    print(f"Evaluating cGAN candidates: {len(cgan_rows)}")
    cgan_metrics = [
        evaluate_candidate(model, x_scaler, y_scaler, lambda_nm, g, cfg.target_er, cfg.target_bw, cfg.target_il)
        for g in cgan_rows
    ]

    def summarize(name: str, metrics: List[Dict[str, float]]) -> None:
        if not metrics:
            print(f"{name}: no samples")
            return
        succ = np.array([m["success"] for m in metrics])
        er_short = np.array([m["ER_shortfall_dB"] for m in metrics])
        bw_short = np.array([m["BW_shortfall_nm"] for m in metrics])
        il_excess = np.array([m["IL_excess_dB"] for m in metrics])

        print(f"\n{name} summary:")
        print(f"  Success rate (all targets met): {succ.mean():.3f}")
        print(f"  Mean ER shortfall (dB):         {er_short.mean():.3f}")
        print(f"  Mean BW shortfall (nm):         {bw_short.mean():.3f}")
        print(f"  Mean IL excess (dB):            {il_excess.mean():.3f}")

    summarize("MDN", mdn_metrics)
    summarize("cGAN", cgan_metrics)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="compare_inverse_models",
        description="Compare MDN vs cGAN inverse models using the forward surrogate.",
    )
    ap.add_argument("--run-dir", required=True, help="Run directory (with data/ and checkpoints/ from mmi_mzi_project.py)")
    ap.add_argument("--mdn-csv", required=True, help="MDN candidates CSV (inverse_candidates.csv)")
    ap.add_argument("--cgan-csv", required=True, help="cGAN candidates CSV (cgan_candidates.csv)")
    ap.add_argument("--target-er", type=float, required=True, help="Target ER (dB)")
    ap.add_argument("--target-bw", type=float, required=True, help="Target BW (nm)")
    ap.add_argument("--target-il", type=float, required=True, help="Target mean IL (dB)")
    ap.add_argument("--max-samples", type=int, default=None, help="Optional: cap number of samples from each CSV")

    args = ap.parse_args()
    cfg = CompareConfig(
        run_dir=Path(args.run_dir),
        mdn_csv=Path(args.mdn_csv),
        cgan_csv=Path(args.cgan_csv),
        target_er=args.target_er,
        target_bw=args.target_bw,
        target_il=args.target_il,
        max_samples=args.max_samples,
    )
    compare_models(cfg)


if __name__ == "__main__":
    main()

