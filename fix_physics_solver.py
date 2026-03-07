#!/usr/bin/env python3
"""
EMEPy Physics Solver Recovery Script
=====================================

Attempts to fix the numpy/simphony/emepy version incompatibilities.
Run this to restore real FDTD physics simulation capability.

Steps:
  1. Clean uninstall of problematic packages
  2. Fresh reinstall of compatible stack
  3. Verification test
"""

import sys
import subprocess
from pathlib import Path


def run_cmd(cmd: str, description: str) -> bool:
    """Run shell command and return success status."""
    print(f"\n[→] {description}")
    print(f"    Command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"    ✓ Success")
        return True
    else:
        print(f"    ✗ Failed")
        if result.stderr:
            print(f"    Error: {result.stderr[:200]}")
        return False


def main():
    print("=" * 80)
    print("EMEPy Physics Solver Recovery")
    print("=" * 80)
    
    venv_python = Path(".venv/Scripts/python.exe").resolve()
    
    if not venv_python.exists():
        print("\n⚠️  Virtual environment not found at .venv/Scripts/python.exe")
        print("   This script should be run from the workspace root directory.")
        return False
    
    print(f"Using Python: {venv_python}")
    
    # Option A: Clean reinstall
    print("\n" + "=" * 80)
    print("OPTION A: Clean Reinstall (Recommended)")
    print("=" * 80)
    
    pip_cmd = f'"{venv_python}" -m pip'
    
    # Step 1: Uninstall problematic packages
    print("\nStep 1: Uninstalling old packages...")
    packages_to_remove = ["emepy", "simphony", "simphony-core", "EMpy", "EMpy_gpu"]
    for pkg in packages_to_remove:
        cmd = f'{pip_cmd} uninstall {pkg} -y --quiet'
        run_cmd(cmd, f"Uninstall {pkg}")
    
    # Step 2: Upgrade pip
    print("\nStep 2: Upgrading pip...")
    run_cmd(f'{pip_cmd} install --upgrade --quiet pip setuptools wheel', "Upgrade pip")
    
    # Step 3: Install fresh EMEPy (will resolve transitive deps)
    print("\nStep 3: Installing EMEPy with compatible dependencies...")
    success_a = run_cmd(
        f'{pip_cmd} install --upgrade --no-cache-dir emepy',
        "Fresh EMEPy install"
    )
    
    # Step 4: Test import
    print("\nStep 4: Testing EMEPy import...")
    test_cmd = f'"{venv_python}" -c "from emepy.eme import EME, Layer; print(\'✓ EMEPy import OK\')"'
    success_test_a = run_cmd(test_cmd, "Test EMEPy import")
    
    if success_a and success_test_a:
        print("\n" + "=" * 80)
        print("✓ SUCCESS - EMEPy physics solver restored!")
        print("=" * 80)
        print("\nNext: Generate real physics metrics:")
        print("  python mmi_mzi_project.py generate \\")
        print("    --stage pilot --run-name pilot_v2 --yes")
        return True
    
    # Option B: Fallback - downgrade numpy
    print("\n" + "=" * 80)
    print("OPTION B: Fallback - Downgrade numpy (if Option A failed)")
    print("=" * 80)
    
    print("\nDowngrading numpy to 1.23 (still has Tester class)...")
    success_b = run_cmd(
        f'{pip_cmd} install "numpy<1.24" --quiet',
        "Downgrade numpy to <1.24"
    )
    
    # Try reinstalling EMEPy with older numpy
    if success_b:
        print("\nReinstalling EMEPy with older numpy...")
        run_cmd(
            f'{pip_cmd} install --upgrade --no-cache-dir --force-reinstall emepy',
            "Reinstall EMEPy"
        )
        
        success_test_b = run_cmd(test_cmd, "Test EMEPy import (Option B)")
        
        if success_test_b:
            print("\n" + "=" * 80)
            print("✓ SUCCESS - EMEPy working with numpy 1.23!")
            print("=" * 80)
            return True
    
    # Option C: Manual specification (last resort)
    print("\n" + "=" * 80)
    print("OPTION C: Manual Package Specification (Last Resort)")
    print("=" * 80)
    print("If neither option worked, try:")
    print(f"  {pip_cmd} install numpy==1.23.5 scipy>=1.0 pytest-runner")
    print(f"  {pip_cmd} install emepy --no-deps")
    print(f"  {pip_cmd} install simphony")
    print("\nThen: python -c \"from emepy.eme import EME; print('✓ EMEPy OK')\"")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print("Option A (clean install): " + ("✓ Worked" if success_test_a else "✗ Failed"))
    print("Option B (numpy downgrade): " + ("✓ Worked" if 'success_test_b' in locals() and success_test_b else "✗ Failed or skipped"))
    print("\nRecommendation:")
    if success_test_a or (isinstance(success_test_b, bool) and success_test_b):
        print("  ✓ Physics solver is ready! Run:")
        print("    python mmi_mzi_project.py generate --stage pilot --run-name pilot_v2 --yes")
    else:
        print("  ✗ Try Option C or consult EMEPy documentation")
    
    return success_a or (isinstance(success_test_b, bool) and success_test_b)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
