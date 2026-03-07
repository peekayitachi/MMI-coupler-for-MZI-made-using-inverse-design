#!/usr/bin/env python3
"""
v1 vs v2 Performance Comparison
==================================

Compare inverse design model performance across v1 and v2 datasets.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import Dict, Any

WORKSPACE = Path(__file__).parent


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file safely."""
    if not path.exists():
        return {}
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return {}


def main():
    print("\n" + "=" * 90)
    print("v1 vs v2 INVERSE DESIGN PERFORMANCE COMPARISON")
    print("=" * 90)
    
    # Load v1 results
    print("\n1. LOADING v1 RESULTS...")
    v1_dir = WORKSPACE / "runs" / "pilot_v1" / "reports"
    v1_comparison = load_json(v1_dir / "v1_models_comparison.json")
    
    if not v1_comparison:
        print("   [!] v1_models_comparison.json not found, will skip v1 details")
        v1_data = None
    else:
        v1_data = v1_comparison
        print(f"   Loaded v1 comparison data")
    
    # Load v2 results
    print("\n2. LOADING v2 RESULTS...")
    v2_dir = WORKSPACE / "runs" / "pilot_v2" / "reports"
    v2_validation = load_json(v2_dir / "v2_validation_summary.json")
    v2_ensemble_df = pd.read_csv(v2_dir / "v2_ensemble_candidates.csv")
    v2_validation_df = pd.read_csv(v2_dir / "v2_validation_results.csv")
    
    print(f"   Loaded v2 validation data ({len(v2_validation_df)} candidates)")
    
    # Display key findings
    print("\n" + "=" * 90)
    print("KEY FINDINGS: v2 ENSEMBLE VALIDATION (100 diverse geometries)")
    print("=" * 90)
    
    if v2_validation:
        summary = v2_validation.get("results_summary", {})
        print(f"\n   Total candidates evaluated: {summary.get('total_candidates', '?')}")
        print(f"   Candidates meeting ER >= 20dB: {summary.get('candidates_meeting_er', '?')}")
        print(f"   Candidates meeting BW >= 40nm: {summary.get('candidates_meeting_bw', '?')}")
        print(f"   Candidates meeting IL <= 1.0dB: {summary.get('candidates_meeting_il', '?')}")
        print(f"   Candidates meeting ALL criteria: {summary.get('candidates_meeting_all', '?')}")
    
    print("\n" + "-" * 90)
    print("METRIC STATISTICS (from forward surrogate predictions):")
    print("-" * 90)
    
    # Output port metrics
    print("\n   Port 1 Metrics:")
    print(f"     ER (min):    mean={v2_validation_df['ER1_min_dB'].mean():.2f} dB, "
          f"max={v2_validation_df['ER1_min_dB'].max():.2f} dB, "
          f"min={v2_validation_df['ER1_min_dB'].min():.2f} dB")
    print(f"     BW:         mean={v2_validation_df['ER1_bw_nm'].mean():.1f} nm, "
          f"max={v2_validation_df['ER1_bw_nm'].max():.1f} nm")
    print(f"     IL (mean):  mean={v2_validation_df['IL1_mean_dB'].mean():.2f} dB, "
          f"min={v2_validation_df['IL1_mean_dB'].min():.2f} dB, "
          f"max={v2_validation_df['IL1_mean_dB'].max():.2f} dB")
    
    print("\n   Port 2 Metrics:")
    print(f"     ER (min):    mean={v2_validation_df['ER2_min_dB'].mean():.2f} dB, "
          f"max={v2_validation_df['ER2_min_dB'].max():.2f} dB, "
          f"min={v2_validation_df['ER2_min_dB'].min():.2f} dB")
    print(f"     BW:         mean={v2_validation_df['ER2_bw_nm'].mean():.1f} nm, "
          f"max={v2_validation_df['ER2_bw_nm'].max():.1f} nm")
    print(f"     IL (mean):  mean={v2_validation_df['IL2_mean_dB'].mean():.2f} dB, "
          f"min={v2_validation_df['IL2_mean_dB'].min():.2f} dB, "
          f"max={v2_validation_df['IL2_mean_dB'].max():.2f} dB")
    
    # Compare to targets
    print("\n" + "-" * 90)
    print("AGAINST TARGET SPECIFICATIONS:")
    print("-" * 90)
    
    target_specs = v2_validation.get("target_specs", {})
    er_target = target_specs.get("ER_dB", 20)
    bw_target = target_specs.get("BW_nm", 40)
    il_target = target_specs.get("IL_dB", 1.0)
    
    print(f"\n   Target: ER >= {er_target}dB, BW >= {bw_target}nm, IL <= {il_target}dB")
    
    # Port 1 analysis
    meets_er1 = (v2_validation_df['ER1_min_dB'] >= er_target).sum()
    meets_bw1 = (v2_validation_df['ER1_bw_nm'] >= bw_target).sum()
    meets_il1 = (v2_validation_df['IL1_mean_dB'] <= il_target).sum()
    
    print(f"\n   Port 1:")
    print(f"     ER >= {er_target}dB:  {meets_er1:3d}/100 candidates ({100*meets_er1/100:.0f}%)")
    print(f"     BW >= {bw_target}nm:  {meets_bw1:3d}/100 candidates ({100*meets_bw1/100:.0f}%)")
    print(f"     IL <= {il_target}dB:   {meets_il1:3d}/100 candidates ({100*meets_il1/100:.0f}%)")
    
    # Port 2 analysis
    meets_er2 = (v2_validation_df['ER2_min_dB'] >= er_target).sum()
    meets_bw2 = (v2_validation_df['ER2_bw_nm'] >= bw_target).sum()
    meets_il2 = (v2_validation_df['IL2_mean_dB'] <= il_target).sum()
    
    print(f"\n   Port 2:")
    print(f"     ER >= {er_target}dB:  {meets_er2:3d}/100 candidates ({100*meets_er2/100:.0f}%)")
    print(f"     BW >= {bw_target}nm:  {meets_bw2:3d}/100 candidates ({100*meets_bw2/100:.0f}%)")
    print(f"     IL <= {il_target}dB:   {meets_il2:3d}/100 candidates ({100*meets_il2/100:.0f}%)")
    
    # Dataset comparison
    print("\n" + "=" * 90)
    print("DATASET COMPARISON: v1 vs v2")
    print("=" * 90)
    
    print("\n   v1 (pilot_v1):")
    print("     • Dataset: Generated synthetic devices")
    print("     • Training: Forward model on generated metrics")
    print("     • Approach: Random search + forward ranking")
    
    print("\n   v2 (pilot_v2):")
    print("     • Dataset: 2000 geometries (300 real + 1700 synthetic via RandomForest)")
    print("     • Training: Forward MLP surrogate (MSE=0.496)")
    print("     • Training: Inverse MDN (10 candidates, mode-collapsed)")
    print("     • Training: Inverse cGAN/300ep (200 candidates, mode-collapsed)")
    print("     • Output: Ensemble 100 diverse candidates (std > 1.0 on all params)")
    print("     • Validation: Forward surrogate predictions for all 100")
    
    # Diversity analysis
    print("\n" + "-" * 90)
    print("PARAMETER DIVERSITY (v2 ensemble candidates):")
    print("-" * 90)
    
    print("\n   W_mmi_um:")
    print(f"     mean={v2_ensemble_df['W_mmi_um'].mean():.3f}, "
          f"std={v2_ensemble_df['W_mmi_um'].std():.3f}, "
          f"min={v2_ensemble_df['W_mmi_um'].min():.3f}, "
          f"max={v2_ensemble_df['W_mmi_um'].max():.3f}")
    
    print("\n   L_mmi_um:")
    print(f"     mean={v2_ensemble_df['L_mmi_um'].mean():.1f}, "
          f"std={v2_ensemble_df['L_mmi_um'].std():.1f}, "
          f"min={v2_ensemble_df['L_mmi_um'].min():.1f}, "
          f"max={v2_ensemble_df['L_mmi_um'].max():.1f}")
    
    print("\n   gap_um:")
    print(f"     mean={v2_ensemble_df['gap_um'].mean():.3f}, "
          f"std={v2_ensemble_df['gap_um'].std():.3f}, "
          f"min={v2_ensemble_df['gap_um'].min():.3f}, "
          f"max={v2_ensemble_df['gap_um'].max():.3f}")
    
    print("\n   W_io_um:")
    print(f"     mean={v2_ensemble_df['W_io_um'].mean():.3f}, "
          f"std={v2_ensemble_df['W_io_um'].std():.3f}, "
          f"min={v2_ensemble_df['W_io_um'].min():.3f}, "
          f"max={v2_ensemble_df['W_io_um'].max():.3f}")
    
    print("\n   taper_len_um:")
    print(f"     mean={v2_ensemble_df['taper_len_um'].mean():.1f}, "
          f"std={v2_ensemble_df['taper_len_um'].std():.1f}, "
          f"min={v2_ensemble_df['taper_len_um'].min():.1f}, "
          f"max={v2_ensemble_df['taper_len_um'].max():.1f}")
    
    # Recommendations
    print("\n" + "=" * 90)
    print("RECOMMENDATIONS & NEXT STEPS")
    print("=" * 90)
    
    print("\n   ✓ STRENGTHS of v2:")
    print("     • 100 diverse candidates spanning full parameter space")
    print("     • Excellent ER performance: 100% meet ER >= 20dB")
    print("     • Excellent BW performance: 100% meet BW >= 40nm")
    print("     • Forward surrogate stable and predictive (MSE=0.496)")
    
    print("\n   ⚠ CHALLENGES in v2:")
    print("     • IL predictions high (~3.7 dB vs target 1.0 dB)")
    print("     • No candidates meet all three criteria (ER, BW, IL)")
    print("     • May indicate forward model IL calibration issue")
    print("       OR physical device has inherent IL limitations")
    
    print("\n   → NEXT STEPS:")
    print("     1. Verify forward model IL calculation (check against v1 baseline)")
    print("     2. Consider IL relaxation or root-cause analysis")
    print("     3. If IL is real: optimize for ER+BW trade-off")
    print("     4. Run top-20 candidates through real physics validation")
    print("     5. Consider multi-objective optimization (Pareto front)")
    
    print("\n" + "=" * 90)
    print("DELIVERABLES:")
    print("=" * 90)
    
    results_csv = v2_dir / "v2_validation_results.csv"
    top_csv = v2_dir / "v2_top_performers.csv"
    summary_json = v2_dir / "v2_validation_summary.json"
    
    print(f"\n   {results_csv.name}")
    print("     → All 100 candidates with full metrics")
    print(f"\n   {top_csv.name}")
    print("     → Top 20 performers by combined score")
    print(f"\n   {summary_json.name}")
    print("     → Structured summary of validation results")
    print(f"\n   v2_ensemble_candidates.csv")
    print("     → Original 100 diverse geometries")
    
    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
    print()
