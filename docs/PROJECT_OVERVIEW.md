# MMI→MZI Inverse Design Project: Overview & Architecture

## Project Purpose

This project generates a comprehensive **dataset** for a **2×2 MMI coupler** (integrated into an **MZI**) and enables **comparative analysis of inverse-design ML models** trained on that dataset.

### Core Outputs
- **Forward Surrogate Model**: `geometry + wavelength → S-parameters (2×2 complex matrix)`
- **Inverse Design Model (MDN)**: `target MZI specs → geometry distribution (probabilistic)`
- **Inverse Design Model (cGAN)**: `target MZI specs + noise → geometry samples (probabilistic)`

---

## Project Structure

### Root Directory Organization

```
MMI-MZI-INVERSE-DESIGN/
├── src/                          # Python source code
│   ├── mmi_mzi_project.py        # Core pipeline (dataset + ML training)
│   ├── cgan_inverse.py           # Conditional GAN for inverse design
│   ├── train_v2_inverse_models.py
│   ├── diagnostics_v2.py         # Dataset quality assessment
│   ├── dataset_augmentation_v2.py # Latin Hypercube Sampling augmentation
│   └── ... (20+ supporting scripts)
│
├── runs/                         # Experimental outputs
│   ├── pilot_v1/                # v1 pipeline (300 geometries)
│   ├── pilot_v2/                # v2 pipeline (2000 geometries)
│   ├── debug_final/
│   ├── gen_v1/
│   └── preflight_*.../          # Pre-test runs
│
├── docs/                        # Documentation
│   ├── PROJECT_OVERVIEW.md      # This file - architecture & structure
│   ├── VERSIONS_AND_CHANGES.md  # Input/output changes across versions
│   └── SETUP_AND_USAGE.md       # Environment setup & execution
│
├── environment.yml              # Conda environment specification
├── requirements.txt             # Python dependencies
└── README.md                    # Quick start
```

### Data Directory Structure

Each experimental run contains a standardized structure:

```
runs/pilot_v*/
├── config.json                  # Experiment configuration
├── stage.json                   # Pipeline stage metadata
├── data/
│   ├── selected_geometries.csv  # All geometries (params: W_mmi, L_mmi, gap, W_io, taper_len)
│   ├── device_long/             # Forward model outputs (S-parameters per geometry/wavelength)
│   └── mzi_metrics/             # Inverse metrics (ER, IL, BW, etc. per geometry)
├── checkpoints/                 # Trained model weights
├── logs/                        # Training logs
├── reports/                     # Evaluation results & visualizations
└── exports/                     # Deployment-ready models
```

---

## Technical Specifications

### Device Parameters

All MMI geometries are parameterized by 5 dimensions:

| Parameter | Units | Min | Max | Description |
|-----------|-------|-----|-----|-------------|
| **W_mmi** | μm | 3.0 | 12.0 | MMI width |
| **L_mmi** | μm | 30.0 | 300.0 | MMI length |
| **gap** | μm | 0.15 | 1.50 | Waveguide gap |
| **W_io** | μm | 0.35 | 0.55 | I/O waveguide width |
| **taper_len** | μm | 5.0 | 40.0 | Taper length |

### Process Technology

**Standard SOI (Silicon-on-Insulator) stack:**
- Device layer: 220 nm Si
- Cladding: SiO₂ (oxide)
- Polarization focus: TE-mode
- Material dispersion: Sellmeier fits for Si and SiO₂ (telecom C-band)

### Port Convention

Fixed throughout all datasets:
```
              Top (Port 1)
    in1 ─────────────────── out1
         (top-left)  (top-right)
    in2 ─────────────────── out2
    (bottom-left)  (bottom-right)
           Bottom (Port 2)
```

### Inverse Design Targets (MZI Specs)

Models learn to predict geometries that achieve:
- **Extinction Ratio (ER)**: 10.0 – 30.0 dB
- **Insertion Loss (IL)**: 0.5 – 3.0 dB
- **Bandwidth (BW)**: 20 – 80 nm

Targets are standardized before model input using per-metric scalers.

---

## Model Architecture

### Forward Surrogate (MLP)

**Purpose:** Predict S-parameters given geometry and wavelength

- **Inputs:** `[W_mmi, L_mmi, gap, W_io, taper_len, dW_nm, dGap_nm, lambda_nm]` (8D)
- **Outputs:** Real & imaginary parts of 2×2 transmission matrix (8D)
- **Architecture:** 3-layer MLP
  - Layer 1: 8 → 256 (ReLU)
  - Layer 2: 256 → 256 (ReLU)
  - Layer 3: 256 → 8 (Linear)
- **Training:** 40 epochs, MSE loss
- **Performance:** Validation MSE = 0.496, Test MSE = 0.502

### Inverse Design Models

#### MDN (Mixture Density Network)
- Outputs multimodal geometry distributions
- Captures uncertainty and multiple valid solutions
- Architecture: 3-layer network with mixture parameters

#### cGAN (Conditional GAN)
- Generator: Transforms noise + MZI specs → geometry samples
- Discriminator: Validates geometry-spec pairs
- Training: Adversarial loss with conditional normalization
- Strength: Ability to generate diverse candidate geometries

---

## Core Pipeline Stages

### Stage 1: Geometry Generation
```python
python mmi_mzi_project.py generate --stage pilot --run-name pilot_v2
```
- Creates parameter space sampling (Latin Hypercube Sampling in v2)
- Outputs: `selected_geometries.csv`

### Stage 2: Physics Simulation (Forward Model)
```python
python mmi_mzi_project.py compute-metrics --run-dir runs/pilot_v2
```
- Computes S-parameters for all geometries across wavelengths
- Calculates MZI metrics (ER, IL, BW) via inverse design equations
- Outputs: `device_long/` and `mzi_metrics/`

### Stage 3: Model Training
```python
python mmi_mzi_project.py train-forward --run-dir runs/pilot_v2 --epochs 80
python mmi_mzi_project.py train-inverse --run-dir runs/pilot_v2 --epochs 100
```
- Trains forward and inverse models
- Outputs: Model checkpoints in `checkpoints/`

### Stage 4: Inverse Design & Evaluation
```python
python mmi_mzi_project.py inverse-design --run-dir runs/pilot_v2 --target-er 20 --target-bw 40
```
- Samples candidate geometries from inverse model
- Evaluates success rate against targets
- Outputs: Reports in `reports/`

---

## Key Design Decisions

### Data Organization
- **Separate v1/v2 pipelines**: Complete independence allows A/B comparison
- **LHS augmentation in v2**: Ensures deterministic, uniform design space coverage
- **Physics metrics computed separately**: `device_long/` (intermediate) and `mzi_metrics/` (targets)

### Model Training
- **Standardization**: All inputs/outputs scaled per metric using fitted scalers
- **Train/Val/Test split**: 70/15/15, deterministically seeded
- **Early stopping**: Monitored on validation ER/IL/BW

### Inverse Design
- **Probabilistic models only**: Addresses inherent non-uniqueness of inverse problem
- **Candidate clipping**: Enforces parameter bounds post-sampling
- **Success criteria**: Prediction within ±1dB (ER), ±0.5dB (IL), ±5nm (BW)

---

## Dependencies & Environment

**Python:** 3.10+ (recommended 3.11)

**Core dependencies:**
- PyTorch (forward/inverse training)
- NumPy/SciPy (numerical computing)
- Pandas (data handling)
- scikit-learn (preprocessing, metrics)
- Matplotlib/Seaborn (visualization)
- tqdm (progress bars)

**Full environment:** See `environment.yml`

---

## Output Artifacts

### Exportable Models
- `exports/forward_best.pt` — FX-traced surrogate model
- `exports/inverse_best.pt` — FX-traced inverse model
- `exports/forward_x_scaler.json`, etc. — Fitted standardizers
- `exports/metadata.json` — Model specs & design bounds
- `exports/inference_template.py` — Ready-to-use inference script

### Analysis & Reports
- `reports/*.json` — Structured evaluation metrics
- `reports/*.pdf` — Visualizations (histograms, scatter plots, heatmaps)
- `logs/*.csv` — Epoch-by-epoch training logs

---

## Version Comparison

| Aspect | v1 | v2 |
|--------|----|----|
| **Dataset size** | 300 geometries | 2,000 geometries |
| **Sampling method** | Random | Latin Hypercube (LHS) |
| **Parameter coverage** | Partial | 100% uniform |
| **Inverse success rate** | 0% | ~50%+ (target) |
| **Status** | Baseline (failed) | Active development |
| **Location** | `runs/pilot_v1/` | `runs/pilot_v2/` |

---

## Next Steps

1. **Physics simulation**: Complete S-parameter computation for all 2,000 v2 geometries
2. **Model retraining**: Forward & inverse models with full v2 dataset
3. **Deployment**: Export standardized models with inference API
4. **Validation**: Real-world fabrication feedback (if available)

---

*Last updated: March 2026*
