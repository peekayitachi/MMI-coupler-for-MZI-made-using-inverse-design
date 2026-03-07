#!/usr/bin/env python3
"""
v2 Inverse Models Comparison: MDN vs cGAN
==========================================

Trains cGAN v2 and generates candidate geometries from both MDN and cGAN.
Shows final comparison and outputs.
"""

import sys
from pathlib import Path
import subprocess
import logging
import json

WORKSPACE = Path(__file__).parent
sys.path.insert(0, str(WORKSPACE))

from mmi_mzi_project import setup_logging, ensure_dir, save_json


def run_cmd(cmd_args: list, description: str, logger: logging.Logger) -> bool:
    """Execute Python command and log result."""
    logger.info(f"\n[+] {description}")
    logger.info(f"    Command: {' '.join(cmd_args)}")
    
    result = subprocess.run(cmd_args, capture_output=True, text=True, cwd=WORKSPACE)
    
    if result.returncode == 0:
        logger.info(f"    [OK] Success")
        if result.stdout:
            # Show last few lines of output
            lines = result.stdout.strip().split('\n')[-5:]
            for line in lines:
                logger.info(f"      {line}")
        return True
    else:
        logger.error(f"    ✗ Failed (exit code {result.returncode})")
        if result.stderr:
            logger.error(f"    {result.stderr[:300]}")
        if result.stdout:
            logger.error(f"    stdout: {result.stdout[:300]}")
        return False


def main():
    """Main orchestrator."""
    run_dir = WORKSPACE / "runs" / "pilot_v2"
    reports_dir = run_dir / "reports"
    cgan_dir = run_dir / "cgan"
    
    ensure_dir(reports_dir)
    
    logger = logging.getLogger("v2_comparison")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(ch)
    
    fh = logging.FileHandler(reports_dir / "v2_models_training.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    
    logger.info("=" * 80)
    logger.info("v2 INVERSE MODELS: MDN + cGAN TRAINING & COMPARISON")
    logger.info("=" * 80)
    
    # =========================================================================
    # STEP 1: Train cGAN v2
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: TRAINING cGAN v2")
    logger.info("=" * 80)
    
    venv_python = WORKSPACE / ".venv" / "Scripts" / "python.exe"
    
    cgan_cmd = [
        str(venv_python),
        "cgan_inverse.py",
        "train",
        "--run-dir", str(run_dir),
        "--out-dir", str(cgan_dir),
        "--epochs", "300",
        "--batch-size", "256",
        "--noise-dim", "16"
    ]
    
    cgan_success = run_cmd(cgan_cmd, "Train cGAN on v2 dataset (300 epochs for better convergence)", logger)
    
    if not cgan_success:
        logger.error("cGAN training failed. Check errors above.")
        return False
    
    logger.info(f"\n[OK] cGAN v2 training complete. Checkpoints saved to: {cgan_dir}/")
    
    # =========================================================================
    # STEP 2: Generate MDN v2 candidates
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: GENERATING MDN v2 CANDIDATES")
    logger.info("=" * 80)
    
    mdn_candidates_csv = reports_dir / "mdn_candidates_v2.csv"
    
    mdn_cmd = [
        str(venv_python),
        "mmi_mzi_project.py",
        "inverse-design",
        "--run-dir", str(run_dir),
        "--target-er", "20",
        "--target-bw", "40",
        "--target-il", "1.0"
    ]
    
    mdn_success = run_cmd(mdn_cmd, "Generate MDN v2 inverse design candidates", logger)
    
    if mdn_success and (reports_dir / "inverse_candidates.csv").exists():
        # Copy to v2-specific name
        import shutil
        shutil.copy2(reports_dir / "inverse_candidates.csv", mdn_candidates_csv)
        logger.info(f"\n[OK] MDN v2 candidates saved to: {mdn_candidates_csv}")
    
    # =========================================================================
    # STEP 3: Generate cGAN v2 candidates
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: GENERATING cGAN v2 CANDIDATES (with bounds clipping)")
    logger.info("=" * 80)
    
    cgan_candidates_csv = reports_dir / "cgan_candidates_v2.csv"
    
    cgan_sample_cmd = [
        str(venv_python),
        "cgan_inverse.py",
        "sample",
        "--run-dir", str(run_dir),
        "--cgan-dir", str(cgan_dir),
        "--target-er", "20",
        "--target-bw", "40",
        "--target-il", "1.0",
        "--n-samples", "200",
        "--out-csv", str(cgan_candidates_csv)
    ]
    
    cgan_sample_success = run_cmd(cgan_sample_cmd, "Generate cGAN v2 candidate geometries (with bounds)", logger)
    
    if cgan_sample_success:
        logger.info(f"\n[OK] cGAN v2 candidates saved to: {cgan_candidates_csv}")
    
    # =========================================================================
    # STEP 4: Load and compare both models' outputs
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: COMPARISON - MDN vs cGAN v2 OUTPUTS")
    logger.info("=" * 80)
    
    try:
        import pandas as pd
        import numpy as np
        
        # Load candidates
        mdn_df = pd.read_csv(mdn_candidates_csv) if mdn_candidates_csv.exists() else None
        cgan_df = pd.read_csv(cgan_candidates_csv) if cgan_candidates_csv.exists() else None
        
        comparison = {
            "dataset": "pilot_v2 (2000 geometries, mixed synthetic/real)",
            "target_specs": {
                "ER_dB": 20,
                "BW_nm": 40,
                "IL_dB": 1.0
            }
        }
        
        # MDN comparison
        if mdn_df is not None:
            logger.info("\n📊 MDN v2 OUTPUTS:")
            logger.info(f"  Generated {len(mdn_df)} candidates")
            
            if len(mdn_df) > 0:
                logger.info("\n  Top 5 MDN candidates:")
                for idx, row in mdn_df.head(5).iterrows():
                    logger.info(
                        f"    #{idx+1:02d}: W_mmi={row.get('W_mmi_um', '?'):6.2f} µm, "
                        f"L_mmi={row.get('L_mmi_um', '?'):7.1f} µm, "
                        f"gap={row.get('gap_um', '?'):5.2f} µm, "
                        f"W_io={row.get('W_io_um', '?'):5.2f} µm, "
                        f"taper={row.get('taper_len_um', '?'):6.1f} µm"
                    )
            
            comparison["mdn"] = {
                "n_candidates": int(len(mdn_df)),
                "geometry_stats": {
                    "W_mmi_um": {
                        "mean": float(mdn_df["W_mmi_um"].mean()),
                        "std": float(mdn_df["W_mmi_um"].std()),
                        "min": float(mdn_df["W_mmi_um"].min()),
                        "max": float(mdn_df["W_mmi_um"].max()),
                    },
                    "L_mmi_um": {
                        "mean": float(mdn_df["L_mmi_um"].mean()),
                        "std": float(mdn_df["L_mmi_um"].std()),
                        "min": float(mdn_df["L_mmi_um"].min()),
                        "max": float(mdn_df["L_mmi_um"].max()),
                    },
                    "gap_um": {
                        "mean": float(mdn_df["gap_um"].mean()),
                        "std": float(mdn_df["gap_um"].std()),
                        "min": float(mdn_df["gap_um"].min()),
                        "max": float(mdn_df["gap_um"].max()),
                    },
                    "W_io_um": {
                        "mean": float(mdn_df["W_io_um"].mean()),
                        "std": float(mdn_df["W_io_um"].std()),
                        "min": float(mdn_df["W_io_um"].min()),
                        "max": float(mdn_df["W_io_um"].max()),
                    },
                    "taper_len_um": {
                        "mean": float(mdn_df["taper_len_um"].mean()),
                        "std": float(mdn_df["taper_len_um"].std()),
                        "min": float(mdn_df["taper_len_um"].min()),
                        "max": float(mdn_df["taper_len_um"].max()),
                    },
                }
            }
        else:
            logger.warning("  ✗ MDN candidates not found")
        
        # cGAN comparison
        if cgan_df is not None:
            logger.info("\n📊 cGAN v2 OUTPUTS:")
            logger.info(f"  Generated {len(cgan_df)} candidates")
            
            if len(cgan_df) > 0:
                logger.info("\n  First 5 cGAN samples:")
                for idx, row in cgan_df.head(5).iterrows():
                    logger.info(
                        f"    #{idx+1:02d}: W_mmi={row.get('W_mmi_um', '?'):6.2f} µm, "
                        f"L_mmi={row.get('L_mmi_um', '?'):7.1f} µm, "
                        f"gap={row.get('gap_um', '?'):5.2f} µm, "
                        f"W_io={row.get('W_io_um', '?'):5.2f} µm, "
                        f"taper={row.get('taper_len_um', '?'):6.1f} µm"
                    )
            
            comparison["cgan"] = {
                "n_samples": int(len(cgan_df)),
                "geometry_stats": {
                    "W_mmi_um": {
                        "mean": float(cgan_df["W_mmi_um"].mean()),
                        "std": float(cgan_df["W_mmi_um"].std()),
                        "min": float(cgan_df["W_mmi_um"].min()),
                        "max": float(cgan_df["W_mmi_um"].max()),
                    },
                    "L_mmi_um": {
                        "mean": float(cgan_df["L_mmi_um"].mean()),
                        "std": float(cgan_df["L_mmi_um"].std()),
                        "min": float(cgan_df["L_mmi_um"].min()),
                        "max": float(cgan_df["L_mmi_um"].max()),
                    },
                    "gap_um": {
                        "mean": float(cgan_df["gap_um"].mean()),
                        "std": float(cgan_df["gap_um"].std()),
                        "min": float(cgan_df["gap_um"].min()),
                        "max": float(cgan_df["gap_um"].max()),
                    },
                    "W_io_um": {
                        "mean": float(cgan_df["W_io_um"].mean()),
                        "std": float(cgan_df["W_io_um"].std()),
                        "min": float(cgan_df["W_io_um"].min()),
                        "max": float(cgan_df["W_io_um"].max()),
                    },
                    "taper_len_um": {
                        "mean": float(cgan_df["taper_len_um"].mean()),
                        "std": float(cgan_df["taper_len_um"].std()),
                        "min": float(cgan_df["taper_len_um"].min()),
                        "max": float(cgan_df["taper_len_um"].max()),
                    },
                }
            }
        else:
            logger.warning("  ✗ cGAN candidates not found")
        
        # Save comparison JSON
        save_json(reports_dir / "v2_models_comparison.json", comparison)
        logger.info(f"\n[OK] Detailed comparison saved to: {reports_dir / 'v2_models_comparison.json'}")
        
    except Exception as e:
        logger.error(f"Error during comparison: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE - v2 INVERSE MODELS (MDN + cGAN)")
    logger.info("=" * 80)
    
    logger.info("\n📦 OUTPUTS:")
    logger.info(f"  • cGAN checkpoints: {cgan_dir}/")
    logger.info(f"  • MDN candidates: {mdn_candidates_csv}")
    logger.info(f"  • cGAN candidates: {cgan_candidates_csv}")
    logger.info(f"  • Comparison JSON: {reports_dir / 'v2_models_comparison.json'}")
    logger.info(f"  • Training log: {reports_dir / 'v2_models_training.log'}")
    
    logger.info("\n[OK] Ready for next phase (validation with forward model)")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
