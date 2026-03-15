# Setup, Execution & Deployment Guide

## Environment Setup

### Quick Start (Recommended: Conda)

```bash
# 1. Install environment
conda env create -f environment.yml
conda activate mmi-mzi

# 2. Verify installation
python -c "import torch; import numpy as np; print('Ready!')"

# 3. Optional: Development install
pip install -e .
```

### Alternative: Requirements.txt

If conda is unavailable:
```bash
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate (Windows)
pip install -r requirements.txt
```

### Python Version Requirements
- **Minimum:** Python 3.10
- **Recommended:** Python 3.11 or 3.12
- **Check:** `python --version`

---

## Core Execution Commands

### Pipeline Execution Flow

**1. Geometry Generation**
```bash
cd c:\Users\peeka\Desktop\MMI-MZI-INVERSE-DESIGN
python src/mmi_mzi_project.py generate \
    --stage pilot \
    --run-name pilot_v2 \
    --yes
```
*Outputs: `runs/pilot_v2/data/selected_geometries.csv`*

---

**2. Physics Metrics Computation**
```bash
python src/mmi_mzi_project.py compute-metrics \
    --run-dir runs/pilot_v2 \
    --num-wavelengths 11 \
    --yes
```
*Outputs: `runs/pilot_v2/data/device_long/` and `runs/pilot_v2/data/mzi_metrics/`*

---

**3. Forward Model Training**
```bash
python src/mmi_mzi_project.py train-forward \
    --run-dir runs/pilot_v2 \
    --epochs 80 \
    --batch-size 32 \
    --learning-rate 0.001
```
*Outputs: `runs/pilot_v2/checkpoints/forward_best.pt`*

---

**4. Inverse Model Training**
```bash
python src/mmi_mzi_project.py train-inverse \
    --run-dir runs/pilot_v2 \
    --model-type mdn \
    --epochs 100 \
    --batch-size 32
```
*Outputs: `runs/pilot_v2/checkpoints/inverse_best.pt`*

*Alternative (cGAN):*
```bash
python src/mmi_mzi_project.py train-inverse \
    --run-dir runs/pilot_v2 \
    --model-type cgan \
    --epochs 200 \
    --generator-lr 0.0002 \
    --discriminator-lr 0.0002
```

---

**5. Inverse Design & Evaluation**
```bash
python src/mmi_mzi_project.py inverse-design \
    --run-dir runs/pilot_v2 \
    --target-er 20.0 \
    --target-bw 40.0 \
    --target-il 1.0 \
    --num-candidates 100 \
    --yes
```
*Outputs: `runs/pilot_v2/reports/inverse_design_*.json`*

---

**6. Model Export (Deployment)**
```bash
python src/mmi_mzi_project.py export \
    --run-dir runs/pilot_v2 \
    --fx-trace \
    --yes
```
*Outputs: `runs/pilot_v2/exports/`*

---

## Model Usage Examples

### Using Exported Forward Model

```python
import torch
import json
import numpy as np

# Load model and scalers
forward_model = torch.jit.load("runs/pilot_v2/exports/forward_best.pt")

with open("runs/pilot_v2/exports/forward_x_scaler.json") as f:
    x_scaler = json.load(f)
with open("runs/pilot_v2/exports/forward_y_scaler.json") as f:
    y_scaler = json.load(f)

# Define geometry
geometry = np.array([
    5.0,      # W_mmi [μm]
    100.0,    # L_mmi [μm]
    0.5,      # gap [μm]
    0.45,     # W_io [μm]
    20.0,     # taper_len [μm]
    0.0,      # dW_nm (fabrication error)
    0.0,      # dGap_nm (fabrication error)
    1550.0    # lambda_nm (wavelength)
], dtype=np.float32)

# Standardize input
x_std = (geometry - x_scaler["mean"]) / np.sqrt(x_scaler["var"])

# Predict
with torch.no_grad():
    y_std = forward_model(torch.from_numpy(x_std).unsqueeze(0))

# Inverse transform
y_std_np = y_std.numpy()[0]
y_raw = y_std_np * np.sqrt(y_scaler["var"]) + y_scaler["mean"]

# Extract S-parameters
s11, s12, s21, s22 = y_raw[0], y_raw[1], y_raw[2], y_raw[3]
s11_imag, s12_imag, s21_imag, s22_imag = y_raw[4:8]

print(f"S11: {s11} + {s11_imag}i")
print(f"S12: {s12} + {s12_imag}i")
```

---

### Using Exported Inverse Model (MDN)

```python
import torch
import json
import numpy as np

# Load model and scalers
inverse_model = torch.jit.load("runs/pilot_v2/exports/inverse_best.pt")

with open("runs/pilot_v2/exports/inverse_x_scaler.json") as f:
    x_scaler = json.load(f)
with open("runs/pilot_v2/exports/inverse_y_scaler.json") as f:
    y_scaler = json.load(f)

# Define target MZI specs
target_specs = np.array([
    20.0,   # ER [dB]
    1.0,    # IL [dB]
    40.0    # BW [nm]
], dtype=np.float32)

# Standardize input (MZI metrics)
x_std = (target_specs - x_scaler["mean"]) / np.sqrt(x_scaler["var"])

# Sample geometry distribution
with torch.no_grad():
    # MDN output: [mean_1, ..., mean_5, logvar_1, ..., logvar_5]
    params = inverse_model(torch.from_numpy(x_std).unsqueeze(0))[0]

means = params[:5].numpy()
logvars = params[5:].numpy()
stds = np.exp(0.5 * logvars)

# Sample candidates
num_samples = 100
samples = np.random.randn(num_samples, 5) * stds + means

# Clip to valid bounds
bounds = {
    0: (3.0, 12.0),     # W_mmi
    1: (30.0, 300.0),   # L_mmi
    2: (0.15, 1.50),    # gap
    3: (0.35, 0.55),    # W_io
    4: (5.0, 40.0)      # taper_len
}

for i, (low, high) in bounds.items():
    samples[:, i] = np.clip(samples[:, i], low, high)

# Inverse transform to raw geometry space
geometries = samples * np.sqrt(y_scaler["var"]) + y_scaler["mean"]

print(f"Generated {geometries.shape[0]} candidate geometries")
print(f"Sample geometry: {geometries[0]}")
```

---

## Analysis & Diagnosis Scripts

### Check Dataset Quality

```bash
python src/diagnostics_v2.py \
    --run-dir runs/pilot_v2 \
    --output-report reports/diagnostics.json
```
*Generates:*
- `reports/diagnostics.json` — Metric distribution stats
- `reports/metric_histograms.pdf` — Visual distributions
- Console output — Quick summary

---

### Compare v1 vs v2 Performance

```bash
python src/compare_v1_v2_performance.py \
    --run-v1 runs/pilot_v1 \
    --run-v2 runs/pilot_v2 \
    --output-dir runs/comparison
```
*Outputs: Side-by-side metric comparisons*

---

### Validate Inverse Model

```bash
python src/validate_v2_ensemble.py \
    --run-dir runs/pilot_v2 \
    --num-trials 1000 \
    --tolerance-er 1.0 \
    --tolerance-il 0.5 \
    --tolerance-bw 5.0
```
*Outputs: Success rate statistics*

---

## Configuration Files

### config.json

Experiment-level settings:

```json
{
  "project_name": "mmi_mzi_inverse_design",
  "stage": "pilot",
  "device_layer_thickness_nm": 220,
  "cladding_material": "SiO2",
  "wavelength_range": {
    "min_nm": 1500,
    "max_nm": 1600,
    "num_points": 11
  },
  "geometry_bounds": {
    "W_mmi": [3.0, 12.0],
    "L_mmi": [30.0, 300.0],
    "gap": [0.15, 1.50],
    "W_io": [0.35, 0.55],
    "taper_len": [5.0, 40.0]
  }
}
```

---

### stage.json

Pipeline execution state:

```json
{
  "current_stage": "inverse-design",
  "completed_stages": [
    "generate",
    "compute-metrics",
    "train-forward",
    "train-inverse"
  ],
  "timestamp": "2026-03-15T10:30:00Z",
  "metrics": {
    "num_geometries": 2000,
    "num_metrics": 3,
    "forward_model_mse": 0.496
  }
}
```

---

## Troubleshooting

### Issue: "Module not found" errors

```bash
# Verify environment
conda list | grep torch
pip show torch

# Reinstall if needed
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

---

### Issue: Out of memory during training

```bash
# Reduce batch size
python src/mmi_mzi_project.py train-forward \
    --run-dir runs/pilot_v2 \
    --batch-size 8  # reduced from 32

# Or reduce model size in code
```

---

### Issue: Physics computation hangs

```bash
# Check if running as background process
ps aux | grep mmi_mzi_project.py

# Run with explicit logging
python src/mmi_mzi_project.py compute-metrics \
    --run-dir runs/pilot_v2 \
    --verbose \
    --timeout 3600
```

---

## Monitoring & Logging

### Real-time Training Logs

```bash
# Terminal 1: Monitor training
tail -f runs/pilot_v2/logs/training.log

# Terminal 2: Run training
python src/mmi_mzi_project.py train-inverse \
    --run-dir runs/pilot_v2 \
    --epochs 100
```

---

### Tensorboard Visualization (If enabled)

```bash
tensorboard --logdir runs/pilot_v2/
# Navigate to http://localhost:6006
```

---

## Performance Benchmarks

### Expected Execution Times (Intel i7, GPU-enabled)

| Stage | Dataset Size | Time | Notes |
|-------|--------------|------|-------|
| Geometry generation | — | ~5 sec | Single-threaded |
| Physics computation | 2,000 geoms | ~30 min | Parallelizable |
| Forward training | 2,000→22k samples | ~2 min | Epochs=80 |
| Inverse training | 2,000→2k samples | ~5 min | Epochs=100, MDN |
| Inverse design (100 queries) | — | ~10 sec | Batch sampling |
| Export & packaging | — | ~1 min | FX-trace overhead |

---

## Version Control & Deployment

### Git Workflow

```bash
# Add all changes
git add runs/pilot_v2/exports/* docs/

# Commit
git commit -m "v2 pipeline: models trained, exported, documented"

# Push
git push origin main
```

### Export Package Contents

```
runs/pilot_v2/exports/
├── forward_best.pt              # Model weights + architecture
├── inverse_best.pt
├── forward_x_scaler.json        # Input standardizer
├── forward_y_scaler.json        # Output standardizer
├── inverse_x_scaler.json
├── inverse_y_scaler.json
├── metadata.json                # Model hyperparameters
└── inference_template.py        # Import and use example
```

All files can be committed to version control (no secrets, portable).

---

*Last updated: March 2026*
