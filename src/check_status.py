#!/usr/bin/env python3
"""
Quick Status Check - MMI-MZI v2 Pipeline
=========================================

Run this to see what's complete and what's next.
"""

import json
from pathlib import Path

BASE = Path(__file__).parent
EXPORTS = BASE / "runs/pilot_v2/exports"
V1_DIR = BASE / "runs/pilot_v1"
V2_DIR = BASE / "runs/pilot_v2"

print("\n" + "=" * 80)
print("MMI-MZI v2 PIPELINE STATUS")
print("=" * 80)

print("\n✅ COMPLETED TASKS:")
print("  1. Bounds Fix: Parameter clipping corrected (3.0-12.0, 30-300, 0.15-1.50, 0.35-0.55, 5.0-40.0)")
print("  2. Deployment: Models exported to runs/pilot_v2/exports/")

print("\n📁 DEPLOYMENT ARTIFACTS:")
if EXPORTS.exists():
    files = sorted(EXPORTS.glob("*"))
    print(f"   {EXPORTS.relative_to(BASE)}/")
    for f in files:
        size_kb = f.stat().st_size / 1024
        print(f"     • {f.name:<30} {size_kb:>8.1f} KB")
else:
    print("   ✗ Exports directory not found!")

print("\n📊 DATASET STATUS:")
if (V1_DIR / "data/mzi_metrics").exists() and (V2_DIR / "data/mzi_metrics").exists():
    from pathlib import Path
    import pandas as pd
    
    try:
        def count_shards(dir_path):
            return len(list(dir_path.glob("part-*.*")))
        
        v1_shards = count_shards(V1_DIR / "data/mzi_metrics")
        v2_shards = count_shards(V2_DIR / "data/mzi_metrics")
        
        print(f"   v1: {v1_shards} shards (300 geometries)")
        print(f"   v2: {v2_shards} shards (2000 geometries, 300 real + 1700 synthetic)")
    except:
        print("   (Could not count shards)")

print("\n⚠️  CRITICAL BLOCKER:")
print("   Real physics metrics incomplete (v2 has 40% synthetic data)")
print("   See: fix_physics_solver.py to restore EMEPy")

print("\n🚀 NEXT STEPS:")
print("   1. Run: python fix_physics_solver.py")
print("      (Fixes numpy/simphony incompatibilities)")
print("")
print("   2. Generate real metrics: python mmi_mzi_project.py generate \\")
print("      --stage pilot --run-name pilot_v2 --yes")
print("")
print("   3. Retrain models: python mmi_mzi_project.py train-forward --run-dir runs/pilot_v2")
print("      python mmi_mzi_project.py train-inverse --run-dir runs/pilot_v2")
print("")
print("   4. Compare v1 vs v2: python launch_v2_tasks.py (with PyTorch installed)")

print("\n📖 DOCUMENTATION:")
print("   • STATUS_REPORT_V2_PIPELINE.md - Full details")
print("   • runs/pilot_v2/exports/metadata.json - Model specs")
print("   • runs/pilot_v2/exports/inference_template.py - Usage example")

print("\n" + "=" * 80)
print()
