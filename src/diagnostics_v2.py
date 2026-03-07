#!/usr/bin/env python3
"""
Dataset Quality & Model Diagnostics for MMI-MZI Inverse Design v2.

Validates dataset quality against physics-grounded criteria and identifies
necessary improvements for the v2 models.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
import torch

# Plotting imports (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use("seaborn-v0_8-darkgrid")
    sns.set_palette("husl")
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__}")
print(f"GPU Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU Memory: {mem_gb:.1f} GB")
print(f"Device: {device}\n")


# ============================================================================
# SECTION 1: DATA QUALITY ASSESSMENT
# ============================================================================

print("=" * 80)
print("SECTION 1: DATA QUALITY ASSESSMENT")
print("=" * 80)

run_dir = Path("runs/pilot_v1")

# Load selected geometries
geom_df = pd.read_csv(run_dir / "data" / "selected_geometries.csv")
print(f"\n✓ Loaded {len(geom_df)} geometries")
print(f"  Geometry features: {list(geom_df.columns)}")

# Data directory paths
device_dir = run_dir / "data" / "device_long"
mzi_dir = run_dir / "data" / "mzi_metrics"

device_shards = sorted(list(device_dir.glob("part-*.parquet")))
mzi_shards = sorted(list(mzi_dir.glob("part-*.parquet")))

print(f"\n✓ Found {len(device_shards)} device shards")
print(f"✓ Found {len(mzi_shards)} MZI metrics shards")

# Load samples to inspect
if device_shards:
    sample_device = pd.read_parquet(device_shards[0])
    print(f"\n  Device shard shape: {sample_device.shape}")
    print(f"  Device columns (first 8): {list(sample_device.columns)[:8]}")

if mzi_shards:
    sample_mzi = pd.read_parquet(mzi_shards[0])
    print(f"\n  MZI shard shape: {sample_mzi.shape}")
    print(f"  MZI columns: {list(sample_mzi.columns)}")


# ============================================================================
# CRITERION 1: YIELD ASSESSMENT
# ============================================================================

print("\n" + "-" * 80)
print("CRITERION 1: Dataset Yield")
print("-" * 80)

# Estimate total rows
if device_shards:
    total_device_rows = len(device_shards) * len(sample_device)
else:
    total_device_rows = 0

unique_geoms = geom_df["geom_id"].nunique() if "geom_id" in geom_df.columns else len(geom_df)

targets = {"rows": 10000, "geoms": 2000}
achieved = {"rows": total_device_rows, "geoms": unique_geoms}

print(f"\nDevice rows:       {achieved['rows']:6,} / {targets['rows']:6,} required  → {'✓ PASS' if achieved['rows'] >= targets['rows'] else '✗ FAIL'}")
print(f"Unique geometries: {achieved['geoms']:6,} / {targets['geoms']:6,} required  → {'✓ PASS' if achieved['geoms'] >= targets['geoms'] else '✗ FAIL'}")

yield_pass = (achieved['rows'] >= targets['rows']) and (achieved['geoms'] >= targets['geoms'])


# ============================================================================
# CRITERION 2: PARAMETER COVERAGE
# ============================================================================

print("\n" + "-" * 80)
print("CRITERION 2: Parameter Coverage (target: ≥80% each)")
print("-" * 80)

param_ranges = {
    "W_mmi_um": (1.5, 15.0),
    "L_mmi_um": (15.0, 400.0),
    "gap_um": (0.10, 2.5),
    "W_io_um": (0.25, 0.70),
    "taper_len_um": (0.5, 100.0),
}

coverage = {}
coverage_pass = True

for param, (pmin, pmax) in param_ranges.items():
    if param in geom_df.columns:
        vals = geom_df[param]
        in_range = ((vals >= pmin) & (vals <= pmax)).sum()
        coverage_pct = (in_range / len(geom_df)) * 100
        coverage[param] = coverage_pct
        status = "✓ PASS" if coverage_pct >= 80 else "✗ FAIL"
        print(f"  {param:15s}: [{vals.min():7.2f}, {vals.max():7.2f}] → Coverage: {coverage_pct:5.1f}%  {status}")
        if coverage_pct < 80:
            coverage_pass = False
    else:
        print(f"  {param:15s}: NOT FOUND")
        coverage_pass = False


# ============================================================================
# CRITERION 3: QUALITY CONTROL (QC) PASS RATE
# ============================================================================

print("\n" + "-" * 80)
print("CRITERION 3: Quality Control Pass Rate (target: 20-70%)")
print("-" * 80)

# Check for QC indicator in geometry file
if "qc_pass" in geom_df.columns or "QC_pass" in geom_df.columns:
    qc_col = "qc_pass" if "qc_pass" in geom_df.columns else "QC_pass"
    qc_pass_count = (geom_df[qc_col] == True).sum() + (geom_df[qc_col] == 1).sum()
    qc_pass_pct = (qc_pass_count / len(geom_df)) * 100
    qc_in_range = 20 <= qc_pass_pct <= 70
    print(f"\n  QC Pass Rate: {qc_pass_pct:.1f}%  → {'✓ PASS' if qc_in_range else '✗ FAIL'}")
else:
    print(f"\n  ⚠ No QC column found (expected 'qc_pass' or 'QC_pass')")
    qc_in_range = None


# ============================================================================
# CRITERION 4: POWER CONSERVATION
# ============================================================================

print("\n" + "-" * 80)
print("CRITERION 4: Power Conservation (|Tsum-1| median < 1%)")
print("-" * 80)

# Load MZI metrics to check power conservation
if mzi_shards:
    all_mzi_dfs = []
    for shard in mzi_shards[:3]:  # Load first 3 shards for speed
        df = pd.read_parquet(shard)
        all_mzi_dfs.append(df)
    
    mzi_combined = pd.concat(all_mzi_dfs, ignore_index=True)
    
    # Look for transmission sum or power metrics
    power_cols = [c for c in mzi_combined.columns if 'power' in c.lower() or 'tsum' in c.lower() or 'sum' in c.lower()]
    
    if power_cols:
        for col in power_cols[:3]:  # Check first 3 power-like columns
            if 'sum' in col.lower() or 'power' in col.lower():
                tsum = mzi_combined[col]
                tsum_norm = np.abs(tsum - 1.0)
                median_error = np.median(tsum_norm) * 100
                print(f"\n  {col}:")
                print(f"    Median error: {median_error:.2f}%  → {'✓ PASS' if median_error < 1.0 else '✗ WARNING'}")
                print(f"    95th percentile error: {np.percentile(tsum_norm, 95) * 100:.2f}%")
    else:
        print(f"\n  ⚠ No power/transmission columns found in MZI metrics")
        print(f"  Available columns: {list(mzi_combined.columns)[:8]}")


# ============================================================================
# CRITERION 5: METRIC DIVERSITY
# ============================================================================

print("\n" + "-" * 80)
print("CRITERION 5: Metric Diversity")
print("-" * 80)

# Look for ER, IL, BW in MZI metrics
if mzi_shards:
    diversity_targets = {
        "ER": 5.0,  # dB
        "IL": 0.2,  # dB
        "BW": 10.0,  # nm
    }
    
    print("\n  Target IQR (inter-quartile range):")
    for metric, target_iqr in diversity_targets.items():
        matching_cols = [c for c in mzi_combined.columns if metric in c.upper()]
        if matching_cols:
            col = matching_cols[0]
            values = mzi_combined[col].dropna()
            iqr = np.percentile(values, 75) - np.percentile(values, 25)
            status = "✓ PASS" if iqr >= target_iqr else "✗ FAIL"
            print(f"    {metric}: IQR={iqr:.2f} (target ≥{target_iqr:.1f})  {status}")
        else:
            print(f"    {metric}: NOT FOUND")


# ============================================================================
# SUMMARY VERDICT
# ============================================================================

print("\n" + "=" * 80)
print("DATASET QUALITY VERDICT")
print("=" * 80)

checks = [
    (yield_pass, "Yield: ≥10k rows + ≥2k geoms"),
    (coverage_pass, "Parameter Coverage: ≥80% each"),
]

overall_pass = all(c[0] for c in checks if c[0] is not None)

print("\n  Status Summary:")
for check_result, label in checks:
    symbol = "✓" if check_result else "✗"
    print(f"    {symbol} {label}")

print("\n" + "-" * 80)
if overall_pass:
    print("VERDICT: ✓ Dataset quality is ACCEPTABLE")
    print("\nNext step: Proceed to model v2 improvements")
    print("  1. Train forward surrogate v2 (improved architecture)")
    print("  2. Design inverse model v2 (VAE-based)")
    print("  3. Enable GPU training")
    print("  4. Evaluate & compare vs v1")
else:
    print("VERDICT: ✗ Dataset quality issues detected")
    print("\nRequired actions:")
    print("  1. AUGMENT dataset using:")
    print("     - Latin Hypercube Sampling (LHS) for parameter coverage")
    print("     - Monte Carlo perturbations (±20% around existing geometries)")
    print("     - Target: 10k rows, 2k+ unique geometries")
    print("\n  2. After augmentation, re-run diagnostics")
    print("  3. Then proceed to model v2 training")

print("\n" + "=" * 80)


# ============================================================================
# SAVE DIAGNOSTIC REPORT
# ============================================================================

report = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "dataset_path": str(run_dir),
    "criteria_results": {
        "yield": {
            "pass": yield_pass,
            "device_rows": total_device_rows,
            "unique_geometries": unique_geoms,
            "targets": targets,
        },
        "coverage": {
            "pass": coverage_pass,
            "values": coverage,
        },
    },
    "verdict": "ACCEPTABLE" if overall_pass else "NEEDS AUGMENTATION",
    "next_steps": [
        "Train forward surrogate v2" if overall_pass else "Augment dataset",
        "Design & train inverse model v2",
        "Enable GPU training",
        "Evaluate on test set",
        "Deploy to Hugging Face",
    ],
}

report_path = run_dir / "diagnostics_v2_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n✓ Report saved to: {report_path}")
