#!/usr/bin/env python3
"""
v2 Pipeline Launcher: Execute steps to complete v2 model training & evaluation.

Usage:
    python launch_v2_pipeline.py [step_number]
    python launch_v2_pipeline.py 1    # Generate physics metrics
    python launch_v2_pipeline.py 2    # Train forward model
    python launch_v2_pipeline.py 3    # Train inverse model
    python launch_v2_pipeline.py all  # Execute all steps sequentially
"""

import sys
import subprocess
from pathlib import Path

# Color codes for terminal output
GREEN = "\033[92m"
BLUE = "\033[94m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

STEPS = {
    "1": {
        "name": "Generate Physics Metrics for 2000 Geometries",
        "description": "Simulate 2000 geometries to compute device_long and mzi_metrics",
        "command": "python mmi_mzi_project.py generate --stage pilot --run-name pilot_v2 --yes",
        "duration": "2-10 hours",
        "notes": [
            "Most time-consuming step",
            "Can be parallelized across multiple cores/machines",
            "Outputs: runs/pilot_v2/data/device_long/* and runs/pilot_v2/data/mzi_metrics/*",
        ],
    },
    "2": {
        "name": "Train Forward Surrogate v2",
        "description": "Train improved MLP on 2000 geometries",
        "command": "python mmi_mzi_project.py train-forward --run-dir runs/pilot_v2 --epochs 80 --batch-size 512",
        "duration": "1-2 hours (30-40 min on GPU)",
        "notes": [
            "Prerequisite: Step 1 must be complete",
            "Target: MAE(ER)≤1.0dB, MAE(IL)≤0.2dB, MAE(BW)≤5nm, R²≥0.90",
            "Output: runs/pilot_v2/checkpoints/forward_best.pt",
        ],
    },
    "3": {
        "name": "Train Inverse Model v2",
        "description": "Train VAE-based inverse model on 2000 geometries",
        "command": "python mmi_mzi_project.py train-inverse --run-dir runs/pilot_v2 --epochs 100 --batch-size 256",
        "duration": "2-4 hours (30-60 min on GPU)",
        "notes": [
            "Prerequisite: Step 2 must be complete",
            "Replaces MDN+cGAN with Conditional VAE",
            "Target: SR@5≥50%, SR@1≥20%, Novelty≥80%",
            "Output: runs/pilot_v2/checkpoints/inverse_best.pt",
        ],
    },
    "4": {
        "name": "Evaluate v1 vs v2 Comparison",
        "description": "Generate comparative metrics and visualization plots",
        "command": "python compare_inverse_models.py --run-v1 runs/pilot_v1 --run-v2 runs/pilot_v2 --target-er 20 --target-bw 40 --target-il 1.0",
        "duration": "30 minutes",
        "notes": [
            "Prerequisite: Steps 2-3 must be complete",
            "Generates: Success rate plots, robustness curves, parameter distributions",
            "Output: runs/pilot_v2/reports/ with comparison plots",
            "Expected: v2 >> v1 (was 0%)",
        ],
    },
    "5": {
        "name": "Deploy to Hugging Face",
        "description": "Upload v2 models to Hugging Face Model Hub",
        "command": "huggingface-cli upload peekayitachi/mmi-mzi-inverse-v2 runs/pilot_v2/checkpoints/",
        "duration": "15 minutes",
        "notes": [
            "Prerequisite: Steps 1-4 complete with passing metrics",
            "Requires: huggingface-cli login",
            "Creates: Model cards, inference API, reproducibility docs",
        ],
    },
}

STEP_ORDER = ["1", "2", "3", "4", "5"]


def print_header(text):
    print(f"\n{BLUE}{'='*80}{RESET}")
    print(f"{BLUE}{text:^80}{RESET}")
    print(f"{BLUE}{'='*80}{RESET}\n")


def print_step_info(step_num):
    step = STEPS.get(step_num)
    if not step:
        print(f"{RED}✗ Unknown step: {step_num}{RESET}")
        return False

    print(f"\n{GREEN}[STEP {step_num}]{RESET} {step['name']}")
    print(f"{YELLOW}Description:{RESET} {step['description']}")
    print(f"{YELLOW}Duration:{RESET} {step['duration']}")
    print(f"\n{YELLOW}Command:{RESET}")
    print(f"  {BLUE}{step['command']}{RESET}")
    print(f"\n{YELLOW}Notes:{RESET}")
    for note in step['notes']:
        print(f"  • {note}")

    return True


def execute_step(step_num):
    step = STEPS.get(step_num)
    if not step:
        print(f"{RED}✗ Unknown step: {step_num}{RESET}")
        return False

    print_step_info(step_num)

    # Confirm execution
    response = input(f"\n{YELLOW}Execute this step? (y/n): {RESET}")
    if response.lower() != "y":
        print("Skipped.")
        return True

    print(f"\n{GREEN}Executing step {step_num}...{RESET}\n")
    try:
        result = subprocess.run(step["command"], shell=True, check=False)
        if result.returncode == 0:
            print(f"\n{GREEN}✓ Step {step_num} completed successfully{RESET}")
            return True
        else:
            print(f"\n{RED}✗ Step {step_num} failed with exit code {result.returncode}{RESET}")
            return False
    except Exception as e:
        print(f"{RED}✗ Error executing step: {e}{RESET}")
        return False


def main():
    print_header("v2 PIPELINE LAUNCHER")
    print("Available steps:")
    for step_num in STEP_ORDER:
        step = STEPS[step_num]
        print(f"  {BLUE}[{step_num}]{RESET} {step['name']}")
    print()

    if len(sys.argv) < 2:
        print(f"{YELLOW}Usage:{RESET}")
        print(f"  python launch_v2_pipeline.py <step>")
        print(f"  python launch_v2_pipeline.py <1-5 or all>")
        print(f"\n{YELLOW}Examples:{RESET}")
        print(f"  python launch_v2_pipeline.py 1    # Run step 1 only")
        print(f"  python launch_v2_pipeline.py 2    # Run step 2 only")
        print(f"  python launch_v2_pipeline.py all  # Run all steps sequentially")
        sys.exit(1)

    target = sys.argv[1].lower()

    if target == "all":
        print(f"\n{GREEN}Running all steps sequentially...{RESET}")
        for step_num in STEP_ORDER:
            if not execute_step(step_num):
                print(
                    f"\n{RED}Pipeline halted at step {step_num}.{RESET}"
                )
                print(
                    f"{YELLOW}Fix the error and continue with:{RESET}"
                )
                print(f"  python launch_v2_pipeline.py {step_num}")
                sys.exit(1)

        print(f"\n{GREEN}{'='*80}{RESET}")
        print(f"{GREEN}✓ ALL STEPS COMPLETED SUCCESSFULLY{RESET}")
        print(f"{GREEN}{'='*80}{RESET}")
        print(f"\n{YELLOW}Next actions:{RESET}")
        print(f"  • View results: ls -la runs/pilot_v2/reports/")
        print(f"  • Compare models: see runs/pilot_v2/reports/*.png")
        print(f"  • Deploy (if metrics pass): python launch_v2_pipeline.py 5")

    elif target in STEP_ORDER:
        if not execute_step(target):
            sys.exit(1)
    else:
        print(f"{RED}✗ Unknown target: {target}{RESET}")
        print(f"{YELLOW}Use: <1-5> or 'all'{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
