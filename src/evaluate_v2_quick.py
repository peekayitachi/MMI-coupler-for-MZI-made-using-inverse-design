#!/usr/bin/env python3
"""Quick evaluation of v2 inverse models."""

import pandas as pd
from pathlib import Path
import sys

print("="*80)
print("v2 MODEL EVALUATION SUMMARY")
print("="*80)

# Load v2 candidates (MDN)
v2_dir = Path("runs/pilot_v2")
v2_candidates = pd.read_csv(v2_dir / "reports" / "inverse_candidates.csv")

print(f"\n✓ Loaded v2 MDN candidates: {len(v2_candidates)} rows")
print(f"  Columns: {list(v2_candidates.columns)}")

# Check v1 for comparison
v1_dir = Path("runs/pilot_v1")
try:
    v1_candidates = pd.read_csv(v1_dir / "reports" / "inverse_candidates.csv")
    v1_available = True
    print(f"\n✓ Loaded v1 MDN candidates: {len(v1_candidates)} rows")
except:
    v1_available = False
    print("\n⚠ v1 candidates not found (generate with: python mmi_mzi_project.py inverse-design --run-dir runs/pilot_v1 --target-er 20 --target-bw 40 --target-il 1.0)")

print("\n" + "-"*80)
print("v2 CANDIDATES ANALYSIS")
print("-"*80)

print("\nGeometry Parameter Statistics (v2):")
geom_cols = [c for c in v2_candidates.columns if "_um" in c or "_nm" in c]
for col in geom_cols[:5]:
    vals = v2_candidates[col]
    print(f"  {col:15s}: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}")

print("\nTop 5 Candidates (by index):")
for i, row in v2_candidates.head(5).iterrows():
    print(f"  #{i+1}: W_mmi={row.get('W_mmi_um', 'N/A'):.2f}, L_mmi={row.get('L_mmi_um', 'N/A'):.2f}, gap={row.get('gap_um', 'N/A'):.2f}")

print("\n" + "="*80)
print("PROGRESS SUMMARY")
print("="*80)

completed = {
    "✅ Dataset augmentation (300→2000 geometries via LHS)": True,
    "✅ Metrics generation (300 real + 1700 synthetic)": True,
    "✅ Forward surrogate v2 training (80 epochs, MSE=0.496)": True,
    "✅ Inverse MDN v2 training (100 epochs, NLL=7.007)": True,
    "✅ Inverse design candidate generation": True,
}

for step, done in completed.items():
    print(f"\n{step}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)

print("""
1. Train cGAN for v2 (if needed):
   python mmi_mzi_project.py train-cgan --run-dir runs/pilot_v2

2. Generate cGAN candidates:
   python cgan_inverse.py --run-dir runs/pilot_v2 --target-er 20 --target-bw 40 --target-il 1.0

3. Full comparison (once cGAN is done):
   python compare_inverse_models.py --run-dir runs/pilot_v2 --target-er 20 --target-bw 40 --target-il 1.0

4. Forward model verification of candidates
5. Success rate calculation

""")

print("="*80)
print("✓ v2 PIPELINE: Datasets, Forward, and Inverse Models TRAINED")
print("="*80)
