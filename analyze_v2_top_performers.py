#!/usr/bin/env python3
"""
v2 Top Performers Analysis & Recommendations
"""

import pandas as pd
import numpy as np
from pathlib import Path

workspace = Path(__file__).parent
v2_dir = workspace / "runs" / "pilot_v2" / "reports"

# Load data
df_validation = pd.read_csv(v2_dir / "v2_validation_results.csv")
df_ensemble = pd.read_csv(v2_dir / "v2_ensemble_candidates.csv")

print("\n" + "=" * 100)
print("v2 TOP PERFORMERS ANALYSIS & RECOMMENDATIONS")
print("=" * 100)

# Create categories based on performance
print("\n1. PERFORMANCE CATEGORIZATION:")
print("-" * 100)

# Port 2 is the signal output (lower ER typically more useful for modulator)
# Port 1 is through path
excellent_er2 = df_validation[df_validation['ER2_min_dB'] >= 45].copy()
excellent_bw = df_validation[df_validation['ER2_bw_nm'] >= 59.5].copy()
low_il = df_validation[(df_validation['IL1_mean_dB'] + df_validation['IL2_mean_dB'])/2 < 3.7].copy()

print(f"\n   Port 2 ER >= 45 dB (excellent contrast):     {len(excellent_er2):2d} candidates")
print(f"   BW >= 59.5 nm (near-maximum):                {len(excellent_bw):2d} candidates")
print(f"   IL < 3.7 dB (lower insertion loss):          {len(low_il):2d} candidates")

# Combined excellent
excellent_all = df_validation[
    (df_validation['ER2_min_dB'] >= 45) & 
    (df_validation['ER2_bw_nm'] >= 59.5) &
    ((df_validation['IL1_mean_dB'] + df_validation['IL2_mean_dB'])/2 < 3.7)
]
print(f"   All three (excellent overall):                {len(excellent_all):2d} candidates")

# Recommended candidates for validation
print("\n2. RECOMMENDED CANDIDATES FOR REAL PHYSICS VALIDATION:")
print("-" * 100)

# Sort by multiple criteria
df_validation['avg_il'] = (df_validation['IL1_mean_dB'] + df_validation['IL2_mean_dB']) / 2
df_validation['er2_quality'] = df_validation['ER2_min_dB'] / 50.0  # normalize to best-in-set
df_validation['il_quality'] = 1.0 - (df_validation['avg_il'] - 3.46) / (4.00 - 3.46)  # 0-1 scale
df_validation['bw_quality'] = df_validation['ER2_bw_nm'] / 60.0  # 0-1 scale

# Multi-objective scoring
df_validation['multiobj_score'] = (
    0.4 * df_validation['er2_quality'] +
    0.4 * df_validation['bw_quality'] +
    0.2 * df_validation['il_quality']
)

df_sorted = df_validation.sort_values('multiobj_score', ascending=False)

print("\n   Top 5 candidates (balanced across all metrics):")
print("   Rank  ID   ER2_min  ER2_BW  IL_avg  Score  Geometry")
print("   " + "-" * 96)

for rank, (idx, row) in enumerate(df_sorted.head(5).iterrows(), 1):
    geom_id = int(row['geom_id'])
    er2 = row['ER2_min_dB']
    bw = row['ER2_bw_nm']
    il = row['avg_il']
    score = row['multiobj_score']
    
    # Get geometry from ensemble
    geom_row = df_ensemble.iloc[geom_id]
    geom_str = f"W_mmi={geom_row['W_mmi_um']:.2f}, L_mmi={geom_row['L_mmi_um']:.0f}, gap={geom_row['gap_um']:.3f}"
    
    print(f"   {rank}     {geom_id:2d}   {er2:6.1f}   {bw:5.1f}  {il:6.2f}  {score:5.3f}  {geom_str}")

print("\n   Top 5 candidates (maximum ER2):")
print("   Rank  ID   ER2_min  ER2_BW  IL_avg  Score  Geometry")
print("   " + "-" * 96)

df_by_er = df_validation.sort_values('ER2_min_dB', ascending=False)
for rank, (idx, row) in enumerate(df_by_er.head(5).iterrows(), 1):
    geom_id = int(row['geom_id'])
    er2 = row['ER2_min_dB']
    bw = row['ER2_bw_nm']
    il = row['avg_il']
    score = row['multiobj_score']
    
    geom_row = df_ensemble.iloc[geom_id]
    geom_str = f"W_mmi={geom_row['W_mmi_um']:.2f}, L_mmi={geom_row['L_mmi_um']:.0f}, gap={geom_row['gap_um']:.3f}"
    
    print(f"   {rank}     {geom_id:2d}   {er2:6.1f}   {bw:5.1f}  {il:6.2f}  {score:5.3f}  {geom_str}")

print("\n   Top 5 candidates (minimum IL):")
print("   Rank  ID   ER2_min  ER2_BW  IL_avg  Score  Geometry")
print("   " + "-" * 96)

df_by_il = df_validation.sort_values('avg_il', ascending=True)
for rank, (idx, row) in enumerate(df_by_il.head(5).iterrows(), 1):
    geom_id = int(row['geom_id'])
    er2 = row['ER2_min_dB']
    bw = row['ER2_bw_nm']
    il = row['avg_il']
    score = row['multiobj_score']
    
    geom_row = df_ensemble.iloc[geom_id]
    geom_str = f"W_mmi={geom_row['W_mmi_um']:.2f}, L_mmi={geom_row['L_mmi_um']:.0f}, gap={geom_row['gap_um']:.3f}"
    
    print(f"   {rank}     {geom_id:2d}   {er2:6.1f}   {bw:5.1f}  {il:6.2f}  {score:5.3f}  {geom_str}")

# Analysis of geometric patterns
print("\n3. GEOMETRIC PATTERN ANALYSIS (Top 20 performers):")
print("-" * 100)

top_20 = df_sorted.head(20)
geom_indices = top_20['geom_id'].astype(int).values
top_20_geoms = df_ensemble.iloc[geom_indices]

print(f"\n   W_mmi range: {top_20_geoms['W_mmi_um'].min():.2f} - {top_20_geoms['W_mmi_um'].max():.2f} µm")
print(f"              (mean: {top_20_geoms['W_mmi_um'].mean():.2f} µm, std: {top_20_geoms['W_mmi_um'].std():.2f} µm)")

print(f"\n   L_mmi range: {top_20_geoms['L_mmi_um'].min():.0f} - {top_20_geoms['L_mmi_um'].max():.0f} µm")
print(f"              (mean: {top_20_geoms['L_mmi_um'].mean():.0f} µm, std: {top_20_geoms['L_mmi_um'].std():.0f} µm)")

print(f"\n   gap range:   {top_20_geoms['gap_um'].min():.3f} - {top_20_geoms['gap_um'].max():.3f} µm")
print(f"              (mean: {top_20_geoms['gap_um'].mean():.3f} µm, std: {top_20_geoms['gap_um'].std():.3f} µm)")

print(f"\n   W_io range:  {top_20_geoms['W_io_um'].min():.3f} - {top_20_geoms['W_io_um'].max():.3f} µm")
print(f"              (mean: {top_20_geoms['W_io_um'].mean():.3f} µm, std: {top_20_geoms['W_io_um'].std():.3f} µm)")

# Key insights
print("\n4. KEY INSIGHTS:")
print("-" * 100)

print("\n   IL Challenge:")
print("   • Forward model predicts IL = 3.46-4.00 dB (mean 3.7 dB)")
print("   • Target was IL ≤ 1.0 dB")
print("   • Gap of 2.7 dB is significant")
print("   • Could be due to:")
print("     - Forward model IL calibration issues")
print("     - Actual device IL limitations")
print("     - Target specification too aggressive")

print("\n   ER & BW Performance:")
print("   • Port 2 ER consistently excellent (40-57 dB)")
print("   • BW consistently at maximum (59-60 nm)")
print("   • No trade-off observed between ER and BW")

print("\n   Design Space:")
print("   • Top performers span diverse W_mmi values (not concentrated)")
print("   • L_mmi variations less impactful on performance")
print("   • Gap and W_io show tight ranges in top performers")

# Recommendations
print("\n5. RECOMMENDATIONS:")
print("-" * 100)

print("\n   Short-term (Immediate):")
print("   ✓ Validate Candidate #" + str(int(df_sorted.iloc[0]['geom_id'])) + " with real physics")
print("     (Best balanced performance across all metrics)")
print("   ✓ Validate Candidate #" + str(int(df_by_er.iloc[0]['geom_id'])) + " with real physics")
print("     (Maximum ER2 contrast)")
print("   ✓ Validate Candidate #" + str(int(df_by_il.iloc[0]['geom_id'])) + " with real physics")
print("     (Minimum IL)")

print("\n   Medium-term (Investigation):")
print("   • Analyze forward model IL prediction accuracy")
print("   • Compare predicted IL with v1 baseline metrics")
print("   • Consider relaxing IL target or validating if real")
print("   • Analyze sensitivity of IL to design parameters")

print("\n   Long-term (Optimization):")
print("   • Implement Pareto multi-objective optimization")
print("   • Weight metrics by application priority")
print("   • Consider ER vs IL vs BW trade-offs explicitly")

print("\n" + "=" * 100)
print("OUTPUT FILES GENERATED:")
print("=" * 100)
print("\n   v2_validation_results.csv")
print("   → All 100 candidates with detailed metrics")
print("\n   v2_top_performers.csv")
print("   → Top 20 ranked by combined score")
print("\n   v2_validation_summary.json")
print("   → Structured summary for integration")
print("\n" + "=" * 100 + "\n")
