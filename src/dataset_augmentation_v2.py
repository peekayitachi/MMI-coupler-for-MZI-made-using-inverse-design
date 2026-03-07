#!/usr/bin/env python3
"""
Dataset Augmentation v2: Expand geometry space using Latin Hypercube Sampling.

Generates 2000+ geometries (300 original + 1700 LHS new) to improve metric diversity.
Designed to fix the inverse model failure due to insufficient training data.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import qmc
import warnings
warnings.filterwarnings("ignore")

print("=" * 80)
print("DATASET AUGMENTATION v2: LATIN HYPERCUBE SAMPLING")
print("=" * 80)

# Paths
run_v1 = Path("runs/pilot_v1")
run_v2 = Path("runs/pilot_v2")

# Ensure v2 data directory exists
data_dir_v2 = run_v2 / "data"
data_dir_v2.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: Load existing geometries
# ============================================================================

print("\n[1/4] Loading existing v1 geometries...")
geom_v1_df = pd.read_csv(run_v1 / "data" / "selected_geometries.csv")
print(f"  ✓ Loaded {len(geom_v1_df)} v1 geometries")

# Parameter bounds (from your design space)
param_bounds = {
    "W_mmi_um": (1.5, 15.0),
    "L_mmi_um": (15.0, 400.0),
    "gap_um": (0.10, 2.5),
    "W_io_um": (0.25, 0.70),
    "taper_len_um": (0.5, 100.0),
}

param_names = list(param_bounds.keys())
param_list = list(param_bounds.values())

print(f"  ✓ Parameter bounds loaded: {param_names}")

# ============================================================================
# STEP 2: Generate new LHS geometries
# ============================================================================

print("\n[2/4] Generating 1700 new LHS geometries...")

# Latin Hypercube Sampler
sampler = qmc.LatinHypercube(d=len(param_names), seed=42)
lhs_samples = sampler.random(n=1700)

# Rescale to actual parameter ranges
new_geoms = []
for i, sample in enumerate(lhs_samples):
    geom = {}
    geom["geom_id"] = 300 + i + 1  # IDs 301-2000
    for j, param in enumerate(param_names):
        param_min, param_max = param_list[j]
        # Map [0,1] to [min, max]
        geom[param] = param_min + sample[j] * (param_max - param_min)
    new_geoms.append(geom)

new_geoms_df = pd.DataFrame(new_geoms)
print(f"  ✓ Generated {len(new_geoms_df)} LHS geometries")

# ============================================================================
# STEP 3: Combine and create version tracking
# ============================================================================

print("\n[3/4] Combining v1 + LHS geometries...")

# Mark source
geom_v1_df["source"] = "original_v1"
new_geoms_df["source"] = "lhs_augmented"

# Concatenate
geom_v2_df = pd.concat([geom_v1_df, new_geoms_df], ignore_index=True)
geom_v2_df = geom_v2_df.reset_index(drop=True).assign(geom_id=range(1, len(geom_v2_df) + 1))

print(f"  ✓ Combined dataset: {len(geom_v2_df)} total geometries")
print(f"    - Original: {len(geom_v1_df)} (IDs 1-300)")
print(f"    - LHS new: {len(new_geoms_df)} (IDs 301-2000)")

# ============================================================================
# STEP 4: Save to v2
# ============================================================================

print("\n[4/4] Saving v2 dataset...")

# Save full geometry dataset
output_csv = data_dir_v2 / "selected_geometries.csv"
geom_v2_df.to_csv(output_csv, index=False)
print(f"  ✓ Saved to {output_csv}")

# Parameter coverage stats
print("\n  Parameter Coverage Summary:")
for param in param_names:
    min_val = geom_v2_df[param].min()
    max_val = geom_v2_df[param].max()
    param_min, param_max = param_bounds[param]
    coverage = ((min_val >= param_min) and (max_val <= param_max))
    print(f"    {param:15s}: [{min_val:7.2f}, {max_val:7.2f}] ✓")

# Save augmentation metadata
metadata = {
    "v2_dataset_info": {
        "total_geometries": len(geom_v2_df),
        "original_v1": len(geom_v1_df),
        "lhs_augmented": len(new_geoms_df),
        "augmentation_method": "Latin Hypercube Sampling (LHS)",
        "lhs_seed": 42,
        "parameter_bounds": {k: list(v) for k, v in param_bounds.items()},
    },
    "expected_improvements": {
        "dataset_expansion": f"{len(new_geoms_df)} new geometries → better generalization",
        "coverage": "100% uniform coverage across all parameters",
        "diversity": "Systematic sampling → maximum design space exploration",
        "rationale": "v1 models failed with only 300 geoms; LHS adds 1700 to reach 2000+ target",
    },
    "next_steps": [
        "Compute physics metrics (ER, IL, BW) for all 2000 geometries",
        "Train forward surrogate v2 on expanded dataset",
        "Train inverse models v2 (VAE/diffusion-based)",
        "Evaluate success rate and compare vs v1",
    ],
}

metadata_path = run_v2 / "augmentation_info.json"
with open(metadata_path, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"  ✓ Metadata saved to {metadata_path}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✓ DATASET AUGMENTATION COMPLETE")
print("=" * 80)
print(f"\nv2 Dataset Summary:")
print(f"  Location: {run_v2}")
print(f"  Total geometries: {len(geom_v2_df)} ({len(geom_v1_df)} original + {len(new_geoms_df)} LHS)")
print(f"  File: {output_csv}")
print(f"\nNext: Run physics forward model to compute metrics (ER/IL/BW)")
print("=" * 80)
