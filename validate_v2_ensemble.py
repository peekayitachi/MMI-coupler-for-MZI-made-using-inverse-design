#!/usr/bin/env python3
"""
v2 Ensemble Candidate Validation
==================================

Evaluates ensemble candidates using trained forward surrogate model.
Computes circuit metrics (ER, BW, IL) and identifies high-performing geometries.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import math
import json
import logging

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

from mmi_mzi_project import (
    MLP, _standardize_apply, load_json, save_json, ensure_dir
)

# Setup logging
def setup_logger(log_file):
    logger = logging.getLogger("validate_v2")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)
    
    fh = logging.FileHandler(log_file, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    
    return logger


def compute_mzi_metrics(S_list: np.ndarray, wl_nm: np.ndarray, er_threshold_db: float = 20.0, n_phase: int = 512) -> dict:
    """
    Compute MZI circuit metrics from 2×2 S-parameter array.
    
    Args:
        S_list: (n_wavelengths, 2, 2) complex S-parameters
        wl_nm: (n_wavelengths,) wavelength array in nm
        er_threshold_db: ER threshold for bandwidth calculation
        n_phase: number of phase sweep points
    
    Returns:
        Dictionary with ER1, ER2, IL1, IL2, BW1, BW2 metrics
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False, dtype=np.float64)
    ejphi = np.exp(1j * phi).astype(np.complex128)

    ER1, IL1, ER2, IL2 = [], [], [], []

    for S in S_list:
        C = S.astype(np.complex128)
        a = C @ np.array([1.0, 0.0], dtype=np.complex128)
        
        b0 = C[0, 0] * a[0] + C[0, 1] * (a[1] * ejphi)
        b1 = C[1, 0] * a[0] + C[1, 1] * (a[1] * ejphi)
        
        P1 = np.abs(b0) ** 2
        P2 = np.abs(b1) ** 2

        P1_max = float(np.max(P1))
        P1_min = float(np.min(P1))
        P2_max = float(np.max(P2))
        P2_min = float(np.min(P2))

        ER1.append(10.0 * math.log10((P1_max + 1e-15) / (P1_min + 1e-15)))
        IL1.append(-10.0 * math.log10(P1_max + 1e-15))
        ER2.append(10.0 * math.log10((P2_max + 1e-15) / (P2_min + 1e-15)))
        IL2.append(-10.0 * math.log10(P2_max + 1e-15))

    ER1 = np.array(ER1, dtype=np.float64)
    IL1 = np.array(IL1, dtype=np.float64)
    ER2 = np.array(ER2, dtype=np.float64)
    IL2 = np.array(IL2, dtype=np.float64)

    def bandwidth_nm(mask: np.ndarray) -> float:
        idx = np.where(mask)[0]
        if idx.size == 0:
            return 0.0
        runs = []
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
            else:
                runs.append((start, prev))
                start = i
                prev = i
        runs.append((start, prev))
        widths = [float(wl_nm[b] - wl_nm[a]) for a, b in runs]
        return float(max(widths)) if widths else 0.0

    mask1 = ER1 >= er_threshold_db
    mask2 = ER2 >= er_threshold_db

    return {
        "ER1_min_dB": float(np.min(ER1)),
        "ER1_mean_dB": float(np.mean(ER1)),
        "ER1_bw_nm": bandwidth_nm(mask1),
        "IL1_mean_dB": float(np.mean(IL1)),
        "ER2_min_dB": float(np.min(ER2)),
        "ER2_mean_dB": float(np.mean(ER2)),
        "ER2_bw_nm": bandwidth_nm(mask2),
        "IL2_mean_dB": float(np.mean(IL2)),
    }


def main():
    run_dir = WORKSPACE / "runs" / "pilot_v2"
    reports_dir = run_dir / "reports"
    ensure_dir(reports_dir)
    
    logger = setup_logger(reports_dir / "v2_validation.log")
    
    logger.info("=" * 80)
    logger.info("v2 ENSEMBLE CANDIDATE VALIDATION")
    logger.info("=" * 80)
    
    # Load candidates
    logger.info("\n1. LOADING CANDIDATES...")
    candidates_csv = reports_dir / "v2_ensemble_candidates.csv"
    df_candidates = pd.read_csv(candidates_csv)
    logger.info(f"   Loaded {len(df_candidates)} candidates from {candidates_csv.name}")
    
    # Load forward model
    logger.info("\n2. LOADING FORWARD SURROGATE MODEL...")
    forward_ckpt = run_dir / "checkpoints" / "forward_best.pt"
    x_scaler = load_json(run_dir / "checkpoints" / "forward_x_scaler.json", {})
    y_scaler = load_json(run_dir / "checkpoints" / "forward_y_scaler.json", {})
    
    device = torch.device("cpu")
    model = MLP(in_dim=8, out_dim=8, hidden=(256, 256, 256), dropout=0.05)
    ckpt = torch.load(forward_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    logger.info(f"   Loaded model from {forward_ckpt.name}, epoch {ckpt.get('epoch', '?')}")
    
    # Prepare wavelength grid
    wl_start_nm = 1520
    wl_stop_nm = 1580
    wl_step_nm = 1
    wl_nm = np.arange(wl_start_nm, wl_stop_nm + 1, wl_step_nm)
    
    logger.info(f"   Wavelength grid: {wl_start_nm}-{wl_stop_nm} nm @ {wl_step_nm} nm steps ({len(wl_nm)} points)")
    
    # Evaluate candidates
    logger.info("\n3. EVALUATING CANDIDATES...")
    results = []
    
    for idx, row in df_candidates.iterrows():
        W_mmi = row["W_mmi_um"]
        L_mmi = row["L_mmi_um"]
        gap = row["gap_um"]
        W_io = row["W_io_um"]
        taper = row["taper_len_um"]
        
        # Prepare features: [W_mmi, L_mmi, gap, W_io, taper_len, dW_nm, dGap_nm, lambda_nm]
        S_list = []
        
        for wl_i in wl_nm:
            X = np.array([[W_mmi, L_mmi, gap, W_io, taper, 0.0, 0.0, wl_i]], dtype=np.float64)
            X_std = _standardize_apply(X, x_scaler)
            X_torch = torch.from_numpy(X_std.astype(np.float32))
            
            with torch.no_grad():
                Y_std = model(X_torch).numpy()
            
            # Unstandardize
            mean = np.array(y_scaler["mean"], dtype=np.float64)
            std = np.array(y_scaler["std"], dtype=np.float64)
            Y = Y_std * std + mean
            
            # Construct S-matrix
            S = np.array([
                [complex(Y[0, 0], Y[0, 1]), complex(Y[0, 4], Y[0, 5])],
                [complex(Y[0, 2], Y[0, 3]), complex(Y[0, 6], Y[0, 7])]
            ], dtype=np.complex128)
            S_list.append(S)
        
        S_arr = np.array(S_list)
        metrics = compute_mzi_metrics(S_arr, wl_nm, er_threshold_db=20.0)
        
        result = {
            "geom_id": idx,
            "W_mmi_um": W_mmi,
            "L_mmi_um": L_mmi,
            "gap_um": gap,
            "W_io_um": W_io,
            "taper_len_um": taper,
            **metrics
        }
        results.append(result)
        
        if (idx + 1) % 10 == 0:
            logger.info(f"   Evaluated {idx + 1}/{len(df_candidates)} candidates")
    
    df_results = pd.DataFrame(results)
    logger.info(f"   Evaluation complete: {len(df_results)} results")
    
    # Save detailed results
    results_csv = reports_dir / "v2_validation_results.csv"
    df_results.to_csv(results_csv, index=False)
    logger.info(f"   Saved detailed results to {results_csv.name}")
    
    # Identify high-performing candidates (above avg on majority of metrics)
    logger.info("\n4. IDENTIFYING HIGH-PERFORMING CANDIDATES...")
    
    target_er = 20.0
    target_bw = 40.0
    target_il = 1.0
    
    # Score candidates
    scores = []
    for idx, row in df_results.iterrows():
        # Metrics to consider: ER (min of both), BW (max of both), IL (mean of both)
        er_score = max(row["ER1_min_dB"], row["ER2_min_dB"]) / target_er  # >= 1 is good
        bw_score = max(row["ER1_bw_nm"], row["ER2_bw_nm"]) / target_bw    # >= 1 is good
        il_score = 1.0 - (min(row["IL1_mean_dB"], row["IL2_mean_dB"]) / target_il)  # lower is better
        
        # Combined score (geometric mean)
        score = (er_score * bw_score * il_score) ** (1/3)
        scores.append(score)
    
    df_results["combined_score"] = scores
    df_results_sorted = df_results.sort_values("combined_score", ascending=False)
    
    # Log top performers
    logger.info(f"\n   Target specs: ER>={target_er}dB, BW>={target_bw}nm, IL<={target_il}dB")
    logger.info(f"\n   Top 10 candidates (by combined score):")
    logger.info(f"   {'Rank':<5} {'Score':<8} {'ER1':<8} {'BW1':<8} {'IL1':<8} {'ER2':<8} {'BW2':<8} {'IL2':<8}")
    
    for rank, (idx, row) in enumerate(df_results_sorted.head(10).iterrows(), 1):
        logger.info(
            f"   {rank:<5d} {row['combined_score']:<8.3f} "
            f"{row['ER1_min_dB']:<8.1f} {row['ER1_bw_nm']:<8.1f} {row['IL1_mean_dB']:<8.2f} "
            f"{row['ER2_min_dB']:<8.1f} {row['ER2_bw_nm']:<8.1f} {row['IL2_mean_dB']:<8.2f}"
        )
    
    # Count candidates meeting criteria
    meets_er = (df_results["ER1_min_dB"] >= target_er) | (df_results["ER2_min_dB"] >= target_er)
    meets_bw = (df_results["ER1_bw_nm"] >= target_bw) | (df_results["ER2_bw_nm"] >= target_bw)
    meets_il = (df_results["IL1_mean_dB"] <= target_il) | (df_results["IL2_mean_dB"] <= target_il)
    
    meets_all = meets_er & meets_bw & meets_il
    
    logger.info(f"\n   Criteria analysis:")
    logger.info(f"     Candidates with ER >= {target_er}dB: {meets_er.sum()}/{len(df_results)} ({100*meets_er.sum()/len(df_results):.1f}%)")
    logger.info(f"     Candidates with BW >= {target_bw}nm: {meets_bw.sum()}/{len(df_results)} ({100*meets_bw.sum()/len(df_results):.1f}%)")
    logger.info(f"     Candidates with IL <= {target_il}dB: {meets_il.sum()}/{len(df_results)} ({100*meets_il.sum()/len(df_results):.1f}%)")
    logger.info(f"     Candidates meeting ALL criteria: {meets_all.sum()}/{len(df_results)} ({100*meets_all.sum()/len(df_results):.1f}%)")
    
    # Save top performers
    top_performers = df_results_sorted.head(20)
    top_csv = reports_dir / "v2_top_performers.csv"
    top_performers.to_csv(top_csv, index=False)
    logger.info(f"\n   Saved top 20 performers to {top_csv.name}")
    
    # Summary statistics
    logger.info("\n5. SUMMARY STATISTICS:")
    logger.info(f"   ER1_min_dB:   mean={df_results['ER1_min_dB'].mean():.2f}, max={df_results['ER1_min_dB'].max():.2f}")
    logger.info(f"   ER1_bw_nm:    mean={df_results['ER1_bw_nm'].mean():.2f}, max={df_results['ER1_bw_nm'].max():.2f}")
    logger.info(f"   IL1_mean_dB:  mean={df_results['IL1_mean_dB'].mean():.2f}, min={df_results['IL1_mean_dB'].min():.2f}")
    logger.info(f"   ER2_min_dB:   mean={df_results['ER2_min_dB'].mean():.2f}, max={df_results['ER2_min_dB'].max():.2f}")
    logger.info(f"   ER2_bw_nm:    mean={df_results['ER2_bw_nm'].mean():.2f}, max={df_results['ER2_bw_nm'].max():.2f}")
    logger.info(f"   IL2_mean_dB:  mean={df_results['IL2_mean_dB'].mean():.2f}, min={df_results['IL2_mean_dB'].min():.2f}")
    
    # Create summary JSON
    summary = {
        "validation_dataset": "v2_ensemble_candidates (100 diverse geometries)",
        "forward_model": "MLP surrogate trained on 2000 v2 geometries",
        "wavelength_range": f"{wl_start_nm}-{wl_stop_nm} nm",
        "target_specs": {
            "ER_dB": target_er,
            "BW_nm": target_bw,
            "IL_dB": target_il
        },
        "results_summary": {
            "total_candidates": len(df_results),
            "candidates_meeting_er": int(meets_er.sum()),
            "candidates_meeting_bw": int(meets_bw.sum()),
            "candidates_meeting_il": int(meets_il.sum()),
            "candidates_meeting_all": int(meets_all.sum()),
            "best_combined_score": float(df_results["combined_score"].max()),
            "avg_combined_score": float(df_results["combined_score"].mean()),
        },
        "metric_statistics": {
            "ER1_min_dB": {
                "mean": float(df_results["ER1_min_dB"].mean()),
                "max": float(df_results["ER1_min_dB"].max()),
                "min": float(df_results["ER1_min_dB"].min())
            },
            "ER1_bw_nm": {
                "mean": float(df_results["ER1_bw_nm"].mean()),
                "max": float(df_results["ER1_bw_nm"].max()),
                "min": float(df_results["ER1_bw_nm"].min())
            },
            "IL1_mean_dB": {
                "mean": float(df_results["IL1_mean_dB"].mean()),
                "min": float(df_results["IL1_mean_dB"].min()),
                "max": float(df_results["IL1_mean_dB"].max())
            }
        }
    }
    
    summary_json = reports_dir / "v2_validation_summary.json"
    save_json(summary_json, summary)
    logger.info(f"\n   Saved summary to {summary_json.name}")
    
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION COMPLETE")
    logger.info("=" * 80)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
