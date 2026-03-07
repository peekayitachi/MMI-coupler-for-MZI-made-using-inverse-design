#!/usr/bin/env python3
"""
Copy and prepare v2 dataset metrics from v1 where applicable.

For the 300 original geometries: copy their v1 metrics to v2
For the 1700 LHS geometries: ready for physics simulation
"""

import shutil
import json
import pandas as pd
from pathlib import Path
import numpy as np

print("=" * 80)
print("v2 METRICS PREPARATION: COPY v1 DATA + PREP FOR NEW GEOMETRIES")
print("=" * 80)

run_v1 = Path("runs/pilot_v1")
run_v2 = Path("runs/pilot_v2")

# Load both datasets
geom_v1 = pd.read_csv(run_v1 / "data" / "selected_geometries.csv")
geom_v2 = pd.read_csv(run_v2 / "data" / "selected_geometries.csv")

print(f"\nv1 geometries: {len(geom_v1)}")
print(f"v2 geometries: {len(geom_v2)} (first 300 are original, 301-2000 are LHS new)")

# ============================================================================
# STEP 1: Copy device_long metrics for original 300 geometries
# ============================================================================

print("\n[1/2] Copying device_long metrics for original 300 geometries...")

device_v1_dir = run_v1 / "data" / "device_long"
device_v2_dir = run_v2 / "data" / "device_long"

device_shards = list(device_v1_dir.glob("part-*.parquet"))
print(f"  Found {len(device_shards)} device shards in v1")

if device_shards:
    # Copy all shards
    for shard in device_shards:
        dest = device_v2_dir / shard.name
        shutil.copy2(shard, dest)
    print(f"  ✓ Copied {len(device_shards)} device shards to v2")
    
    # Load first shard to check structure
    sample = pd.read_parquet(device_shards[0])
    print(f"  ✓ Device metrics shape per shard: {sample.shape}")
else:
    print(f"  ⚠ No device shards found in v1")

# ============================================================================
# STEP 2: Copy mzi_metrics for original 300 geometries
# ============================================================================

print("\n[2/2] Copying mzi_metrics for original 300 geometries...")

mzi_v1_dir = run_v1 / "data" / "mzi_metrics"
mzi_v2_dir = run_v2 / "data" / "mzi_metrics"

mzi_shards = list(mzi_v1_dir.glob("part-*.parquet"))
print(f"  Found {len(mzi_shards)} MZI metric shards in v1")

if mzi_shards:
    # Copy all shards
    for shard in mzi_shards:
        dest = mzi_v2_dir / shard.name
        shutil.copy2(shard, dest)
    print(f"  ✓ Copied {len(mzi_shards)} MZI metric shards to v2")
    
    # Load first shard to check structure
    sample_mzi = pd.read_parquet(mzi_shards[0])
    print(f"  ✓ MZI metrics shape per shard: {sample_mzi.shape}")
else:
    print(f"  ⚠ No MZI shards found in v1")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✓ v2 METRICS PREPARATION COMPLETE")
print("=" * 80)

status = {
    "v1_metrics_copied": {
        "device_shards": len(device_shards),
        "mzi_shards": len(mzi_shards),
        "coverage": "Original 300 geometries",
    },
    "new_lhs_geometries": {
        "count": 1700,
        "status": "Ready for physics simulation",
        "next_step": "Run: python mmi_mzi_project.py generate --stage pilot --run-name pilot_v2",
    },
    "v2_dataset_ready": {
        "total_geometries": len(geom_v2),
        "with_metrics": 300,
        "pending_metrics": 1700,
    },
}

print("\nStatus Summary:")
print(f"  ✓ v2 Dataset: {len(geom_v2)} geometries")
print(f"  ✓ Metrics copied from v1: {len(device_shards)} device + {len(mzi_shards)} MZI shards")
print(f"  ⏳ Pending metrics for 1700 LHS geometries")
print(f"\nNext Step:")
print(f"  Run physics simulation for new geometries:")
print(f"  python mmi_mzi_project.py generate --stage pilot --run-name pilot_v2 --yes")
print("=" * 80)
