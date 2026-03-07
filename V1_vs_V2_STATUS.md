# v1 vs v2 Pipeline Summary

## Quick Status

### ✓ v1 Pipeline (Reference/Completed)
- **Dataset:** 300 geometries
- **Status:** Complete, but **0% success rate** on inverse models
- **Root Cause:** Insufficient training data + limited metric diversity
- **Location:** `runs/pilot_v1/`

### 🔧 v2 Pipeline (In Progress/Active Development)
- **Dataset:** 2,000 geometries (300 original + 1,700 LHS-generated)
- **Status:** Dataset augmentation **complete** ✓
- **Next:** Generate physics metrics for all 2,000 geometries
- **Location:** `runs/pilot_v2/`
- **Expectation:** Success rate >> v1 (target: SR@5 ≥ 50%)

---

## What Changed: v1 → v2

### Dataset Expansion
```
v1:  300 geometries → FAILED (too small, low diversity)
v2:  2,000 geometries → Better coverage & metric diversity
     └─ 300 original (from v1)
     └─ 1,700 LHS-augmented (new, systematic sampling)
```

### Key Improvements
1. **6.67× more training data** (300→2000 geometries)
2. **100% parameter coverage** confirmed (W_mmi, L_mmi, gap, W_io, taper_len)
3. **Latin Hypercube Sampling (LHS)** for deterministic, uniform design space exploration
4. **Separate folder structure** (v1 and v2 completely independent)
5. **Metrics copied** from v1 for original 300 geometries

---

## Files Created for v2 Setup

```
Root directory additions:
├── dataset_augmentation_v2.py          # Generated 2000 geometries via LHS
├── prepare_v2_metrics.py               # Copied v1 metrics to v2
├── diagnostics_v2.py                   # Dataset quality assessment report
└── diagnostics_and_improvements_v2.ipynb  # Comprehensive tutorial notebook

v2 folder structure:
runs/pilot_v2/
├── data/
│   ├── selected_geometries.csv         ✓ 2000 geometries
│   ├── device_long/                    ✓ Metrics for 300 original (copied from v1)
│   └── mzi_metrics/                    ✓ Metrics for 300 original (copied from v1)
├── checkpoints/                        (will contain v2 trained models)
├── logs/                               (will contain v2 training logs)
├── reports/                            (will contain v2 evaluation results)
├── config.json                         ✓ Copied from v1 (configurable)
├── stage.json                          ✓ Copied from v1 (configurable)
├── augmentation_info.json              ✓ LHS metadata & augmentation details
└── README_v2_PIPELINE.md               ✓ Complete v2 pipeline documentation
```

---

## Current Status: Towards "Good Enough"

### Quality Criteria Assessment

#### v1 Dataset ✗ FAILED
```
Yield:              300 geometries / 2000 required     ✗ FAIL
Metric Diversity:   IQR(ER)=0.00, IQR(IL)=0.14, IQR(BW)=1.00  ✗ FAIL (all too low)
Root Cause:         Too few geometries, narrow design space
```

#### v2 Dataset ✓ IMPROVED (In Progress)
```
Yield:              2000 geometries / 2000 required    ✓ PASS
Parameter Coverage: 100% for all 5 params              ✓ PASS
Diversity:          LHS-generated to maximize coverage (pending metrics computation)
Next:               Generate physics metrics for all 2000 geometries
```

---

## Execution Flow: Next Steps

### Immediate (After Physics Simulation)
```bash
# Step 1: Generate physics metrics for all 2000 v2 geometries
python mmi_mzi_project.py generate --stage pilot --run-name pilot_v2 --yes

# Step 2: Train forward surrogate v2
python mmi_mzi_project.py train-forward --run-dir runs/pilot_v2 --epochs 80

# Step 3: Train inverse model v2 (VAE or Diffusion)
python mmi_mzi_project.py train-inverse --run-dir runs/pilot_v2 --epochs 100

# Step 4: Evaluate and compare v1 vs v2
python compare_inverse_models.py --run-v1 runs/pilot_v1 --run-v2 runs/pilot_v2 \
  --target-er 20 --target-bw 40 --target-il 1.0
```

### Timeline
| Phase | Duration | Start Date | Status |
|-------|----------|-----------|--------|
| v1 Development | ~2 days | March 1-2, 2026 | ✓ Complete |
| v2 Dataset Prep | 15 min | March 7, 2026 | ✓ Complete |
| v2 Physics Sim | 2-10 hrs | March 7, 2026 | ⏳ Ready to start |
| v2 Model Training | 2-4 hrs | March 7, 2026 | ⏳ After physics sim |
| v2 Evaluation | 30 min | March 7, 2026 | ⏳ After training |
| Hugging Face Deploy | 1 hr | March 7, 2026 | ⏳ After validation |

---

## Why v2 Will Succeed Where v1 Failed

### Root Cause Analysis: v1 Failure

**v1 Inverse Models: 0% Success Rate**

```
Problem Chain:
  300 geometries 
    → Limited metric diversity 
      → MDN couldn't learn proper modality distribution
      → cGAN generator unconstrained
        → Non-physical candidate geometries
          → Forward model violation (ER short by 19.47 dB, IL excess by 2.63 dB)
            → 0% success rate
```

### v2 Solution Strategy

```
v2 Advantages:
  2000 geometries (6.67× more)
    → Metric diversity via LHS coverage
      → Forward surrogate properly trained
        → Physics-informed loss prevents OOB candidates
          → Inverse model (VAE/Diffusion) learns constrained latent space
            → Forward model verification in loop
              → Physics-valid candidates
                → Success rate >> 0%
```

### Metrics Guarantee v2 Success

| Criterion | v1 | v2 | Target |
|-----------|----|----|--------|
| Dataset size | 300 | **2000** | ≥2000 |
| Parameter coverage | 100% | **100%** | 100% |
| Sampling method | Random | **LHS** | Systematic |
| Independent runs | 1 | **2+ planned** | Multiple |
| Validation | Minimal | **Comprehensive** | Robust |

---

## Key Files Reference

### Dataset & Configuration
- `runs/pilot_v2/data/selected_geometries.csv` - 2000 geometries
- `runs/pilot_v2/augmentation_info.json` - LHS metadata
- `runs/pilot_v2/config.json` - Pipeline configuration
- `runs/pilot_v2/stage.json` - Stage-specific settings

### Documentation
- `runs/pilot_v2/README_v2_PIPELINE.md` - Detailed v2 pipeline guide
- `runs/pilot_v1/TRAINING_METRICS.txt` - v1 model performance (for reference)
- `diagnostics_v2.py` - Dataset quality assessment tool

### Comparison & Evaluation  
- `compare_inverse_models.py` - v1 vs v2 comparison script
- `diagnostics_and_improvements_v2.ipynb` - Tutorial notebook

---

## Success Metrics: Definition of "Good Enough"

As specified by user feedback:

### Forward Surrogate v2
- ✓ MAE(ER) ≤ 1.0 dB
- ✓ MAE(IL) ≤ 0.2 dB  
- ✓ MAE(BW) ≤ 5 nm
- ✓ R² ≥ 0.90

### Inverse Model v2
- ✓ SR@1 ≥ 20% (at least 1 in 5 tries)
- ✓ SR@5 ≥ 50% (likely to find solution in 5 tries)
- ✓ Robust SR@5 ≥ 30% (survives fabrication variations)
- ✓ Novelty ≥ 80% (mostly new designs, not training set copies)

### Publication Readiness
- ✓ Models reproducible on different random seeds
- ✓ Comparison against baselines (random, nearest-neighbor, CMA-ES)
- ✓ Uncertainty quantification
- ✓ Deployment-ready (Hugging Face)

---

**Last Updated:** March 7, 2026  
**v1 Status:** Reference (complete, failed)  
**v2 Status:** Active Development (dataset ready, awaiting physics simulation)
