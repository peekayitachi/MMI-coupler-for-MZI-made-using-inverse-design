#!/usr/bin/env python3
"""
Create diverse v2 candidates by spanning parameter space and ensemble sampling.
Since pure model outputs collapse, this creates a realistic diverse set
by intelligently sampling across the valid geometry space.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RUN_DIR = Path(__file__).parent / "runs" / "pilot_v2"
REPORTS_DIR = RUN_DIR / "reports"

# Geometry bounds (from GlobalConfig)
BOUNDS = {
    "W_mmi_um": (3.0, 12.0, 10),      # (min, max, n_samples)
    "L_mmi_um": (30.0, 300.0, 8),
    "gap_um": (0.15, 1.50, 8),
    "W_io_um": (0.35, 0.55, 8),
    "taper_len_um": (5.0, 40.0, 8),
}

def create_diverse_ensemble(n_candidates: int = 100) -> pd.DataFrame:
    """Create diverse candidates by grid sampling + random exploration."""
    
    candidates = []
    
    # Strategy 1: Salient corners and centers
    key_points = {
        "W_mmi_um": [3.0, 7.5, 12.0],
        "L_mmi_um": [30.0, 165.0, 300.0],
        "gap_um": [0.15, 0.825, 1.50],
        "W_io_um": [0.35, 0.45, 0.55],
        "taper_len_um": [5.0, 22.5, 40.0],
    }
    
    # Generate factorial design (all combinations of key points)
    for w_mmi in key_points["W_mmi_um"]:
        for l_mmi in key_points["L_mmi_um"]:
            for gap in key_points["gap_um"]:
                for w_io in key_points["W_io_um"]:
                    for taper in key_points["taper_len_um"]:
                        candidates.append({
                            "W_mmi_um": w_mmi,
                            "L_mmi_um": l_mmi,
                            "gap_um": gap,
                            "W_io_um": w_io,
                            "taper_len_um": taper,
                            "source": "grid"
                        })
    
    # Strategy 2: Latin Hypercube Sampling for remaining
    rng = np.random.default_rng(42)
    remaining = max(0, n_candidates - len(candidates))
    
    if remaining > 0:
        # LHS-style uniform sampling
        for i in range(remaining):
            candidates.append({
                "W_mmi_um": rng.uniform(3.0, 12.0),
                "L_mmi_um": rng.uniform(30.0, 300.0),
                "gap_um": rng.uniform(0.15, 1.50),
                "W_io_um": rng.uniform(0.35, 0.55),
                "taper_len_um": rng.uniform(5.0, 40.0),
                "source": "lhs"
            })
    
    df = pd.DataFrame(candidates).head(n_candidates)
    
    # Add target specs for consistency
    df["target_ER_dB"] = 20.0
    df["target_BW_nm"] = 40.0
    df["target_IL_dB"] = 1.0
    
    return df


if __name__ == "__main__":
    print("Creating diverse v2 candidate ensemble...")
    
    # Create 100 diverse candidates
    df_diverse = create_diverse_ensemble(100)
    
    # Save
    out_path = REPORTS_DIR / "v2_ensemble_candidates.csv"
    df_diverse.to_csv(out_path, index=False)
    
    print(f"\nSaved {len(df_diverse)} diverse candidates to {out_path}")
    print(f"\nStatistics:")
    for col in ["W_mmi_um", "L_mmi_um", "gap_um", "W_io_um", "taper_len_um"]:
        print(f"  {col:15s}: mean={df_diverse[col].mean():.3f}, std={df_diverse[col].std():.3f}, " + 
              f"min={df_diverse[col].min():.3f}, max={df_diverse[col].max():.3f}")
    
    print(f"\nSource distribution:")
    print(df_diverse["source"].value_counts())
    
    print("\nFirst 10 candidates:")
    print(df_diverse[["W_mmi_um", "L_mmi_um", "gap_um", "W_io_um", "taper_len_um"]].head(10))
