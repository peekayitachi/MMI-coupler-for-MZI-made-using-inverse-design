# MMI→MZI Inverse Design Project (dataset + comparative inverse ML)

This project generates a **dataset** for a **2×2 MMI coupler** (used inside an **MZI**) and supports **comparative analysis of inverse-design ML models** trained on that dataset.

It includes:

- a **forward surrogate**: `geometry + wavelength → complex S(λ)`
- an **inverse design model (MDN)** (probabilistic): `target MZI specs → geometry distribution`
- an **inverse design model (cGAN)** (probabilistic): `target MZI specs + noise → geometry samples`

The core dataset + baseline ML pipeline is intentionally packed into **one Python file**:

- `mmi_mzi_project.py` — dataset generation + evaluation + ML training + inverse sampling

The cGAN inverse model is in a separate file:

- `cgan_inverse.py` — train a conditional GAN on `mzi_metrics` and sample candidate geometries

---

## What “publishable defaults” means here

Out of the box, the script assumes a defensible “standard SOI” stack:

- **SOI device layer thickness:** 220 nm  
- **Cladding:** SiO₂ (oxide clad)  
- **Polarization:** TE-focused (filters port modes by TE fraction)

Material dispersion is handled by simple Sellmeier fits for Si and SiO₂ (telecom band).

Port convention is fixed and used everywhere:

- `in1` = **top-left** waveguide
- `in2` = **bottom-left** waveguide
- `out1` = **top-right**
- `out2` = **bottom-right**

Because eigenmode solvers naturally return **supermodes**, the code includes a **deterministic port mapping step** (even/odd parity detection + sign disambiguation) so you don’t end up with a dataset full of arbitrary phase / port flips.

---

## Environment setup

### Recommended Python version

Use **Python 3.10 or 3.11**.

### Option A (recommended): conda via `environment.yml`

This repo includes `environment.yml` for a reproducible install:

```bash
conda env create -f environment.yml
conda activate mmi-mzi
```

If you edit dependencies later:

```bash
conda env update -n mmi-mzi -f environment.yml
conda activate mmi-mzi
```

### Option B: venv + pip (simple)

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

---

## How to run

### 1) Preflight (checks your install + assumptions)
```bash
python mmi_mzi_project.py preflight
```

This prints:
- whether `emepy` is installed
- whether Parquet is enabled (`pyarrow`)
- the key physics assumptions you should confirm before running a paper dataset

---

### 2) Dataset generation

The generator supports **stages** so you can start small and scale safely:

- `debug`  → tiny dataset (fast sanity check)
- `pilot`  → moderate dataset (good for developing the ML pipeline)
- `paper`  → higher-fidelity sweep (more publishable)

#### Dry-run mode (no emepy required)
This makes synthetic data so you can test training and plotting immediately:

```bash
python mmi_mzi_project.py generate --stage debug --run-name debug_synth --dry-run --yes
```

#### Real physics mode (EMEPy)
```bash
python mmi_mzi_project.py generate --stage pilot --run-name pilot_v1 --yes
```

**Fail-safe / resume behavior:**
- The script writes **sharded outputs** as it runs.
- It maintains a `manifest.json` so re-running the same run directory skips completed `(geom_id, mc_id)` cases.
- Failures (solver errors, QC failures) are recorded in `failures.jsonl`.

---

### 3) Evaluate dataset quality

```bash
python mmi_mzi_project.py evaluate --run-dir runs/pilot_v1
```

This generates plots into:

- `runs/pilot_v1/reports/`

Including:
- parameter histograms
- geometry coverage scatter plot
- split ratio vs throughput at λ0
- MZI metric histograms
- bin count table to confirm stratification

---

### 4) Train the forward surrogate

```bash
python mmi_mzi_project.py train-forward --run-dir runs/pilot_v1 --epochs 40
```

Forward model learns:

**Inputs**
- `W_mmi_um, L_mmi_um, gap_um, W_io_um, taper_len_um`
- `dW_nm, dGap_nm`
- `lambda_nm`

**Outputs**
- `S_out1_in1, S_out2_in1, S_out1_in2, S_out2_in2` (complex stored as real/imag)

Saved artifacts:
- `runs/.../checkpoints/forward_best.pt`
- feature/target scalers in JSON

---

### 5) Train the inverse design model

```bash
python mmi_mzi_project.py train-inverse --run-dir runs/pilot_v1 --epochs 80 --K 8
```

Inverse MDN learns a distribution:

**Input features**
- `ER_bw_nm, ER_min_dB, IL_mean_dB` (for both outputs)

**Output distribution**
- `W_mmi_um, L_mmi_um, gap_um, W_io_um, taper_len_um`

Saved artifacts:
- `runs/.../checkpoints/inverse_best.pt`
- scalers in JSON

---

### 6) Sample candidate geometries (inverse design)

```bash
python mmi_mzi_project.py inverse-design \
  --run-dir runs/pilot_v1 \
  --target-er 20 \
  --target-bw 40 \
  --target-il 1.0 \
  --n-samples 512 \
  --top-k 20
```

This writes:
- `runs/.../reports/inverse_candidates.csv`

These candidates are **not automatically physics-verified**. The next step (which you can add) is:
- re-simulate candidates in EMEPy, or
- run them through the trained forward surrogate as a filter.

---

## End-to-end convenience command

If you want to run the full pipeline (generate → evaluate → train → inverse sample) with one command:

```bash
python mmi_mzi_project.py full-run --stage debug --run-name debug_v1 --dry-run --yes
```

For a real run (requires EMEPy working in your environment):

```bash
python mmi_mzi_project.py full-run --stage pilot --run-name pilot_v1 --yes
```

---

## cGAN inverse model (comparative baseline)

The MDN inverse model is built into `mmi_mzi_project.py`. For a second probabilistic inverse model,
this repo also provides a **conditional GAN** (`cgan_inverse.py`) that learns:

- **Condition**: MZI metrics (6D)
  - `ER1_bw_nm, ER1_min_dB, IL1_mean_dB, ER2_bw_nm, ER2_min_dB, IL2_mean_dB`
- **Output**: geometry (5D)
  - `W_mmi_um, L_mmi_um, gap_um, W_io_um, taper_len_um`

### Train cGAN

First generate a run so you have `data/mzi_metrics/`:

```bash
python mmi_mzi_project.py generate --stage pilot --run-name pilot_v1 --yes
```

Then train the cGAN (stores checkpoints and scalers under `--out-dir`):

```bash
python cgan_inverse.py train \
  --run-dir runs/pilot_v1 \
  --out-dir runs/pilot_v1/cgan \
  --epochs 200 \
  --batch-size 512
```

Outputs (in `runs/pilot_v1/cgan/`):
- `G_final.pt`, `D_final.pt`
- `cond_scaler.json`, `geom_scaler.json`
- `train_log.txt`

### Sample candidate geometries from cGAN

```bash
python cgan_inverse.py sample \
  --run-dir runs/pilot_v1 \
  --cgan-dir runs/pilot_v1/cgan \
  --target-er 20 \
  --target-bw 40 \
  --target-il 1.0 \
  --n-samples 512 \
  --out-csv runs/pilot_v1/reports/cgan_candidates.csv
```

This writes a CSV of sampled geometries conditioned on your target specs.

---

## Comparing MDN vs cGAN (inverse design quality)

To compare the MDN inverse model and the cGAN inverse model **using the same evaluator**, this repo provides:

- `compare_inverse_models.py` — evaluates candidate geometries with the **forward surrogate** and reports:
  - success rate (fraction of samples meeting all targets),
  - mean ER shortfall, BW shortfall, and IL excess relative to your target spec.

### Prerequisites

For a given run directory `runs/<run_name>/` you should have:

- Forward surrogate trained:
  - `mmi_mzi_project.py train-forward --run-dir runs/<run_name>`
- MDN inverse trained and sampled:
  - `mmi_mzi_project.py train-inverse --run-dir runs/<run_name>`
  - `mmi_mzi_project.py inverse-design --run-dir runs/<run_name> --target-er ... --target-bw ... --target-il ...`
- cGAN inverse trained and sampled:
  - `cgan_inverse.py train --run-dir runs/<run_name> --out-dir runs/<run_name>/cgan ...`
  - `cgan_inverse.py sample --run-dir runs/<run_name> --cgan-dir runs/<run_name>/cgan ...`

This should produce:

- `runs/<run_name>/reports/inverse_candidates.csv` (MDN candidates)
- `runs/<run_name>/reports/cgan_candidates.csv` (cGAN candidates)

### Run comparison

```bash
python compare_inverse_models.py \
  --run-dir runs/pilot_v1 \
  --mdn-csv runs/pilot_v1/reports/inverse_candidates.csv \
  --cgan-csv runs/pilot_v1/reports/cgan_candidates.csv \
  --target-er 20 \
  --target-bw 40 \
  --target-il 1.0
```

This:

- Uses the **forward surrogate** to predict full S(λ) spectra for each candidate geometry.
- Recomputes MZI metrics via the same phase-sweep method used in dataset generation.
- Reports, separately for MDN and cGAN:
  - **Success rate**: fraction of samples where both outputs meet ER ≥ target, BW ≥ target, IL ≤ target.
  - **Mean ER shortfall** (how many dB below target, if any).
  - **Mean BW shortfall** (how many nm below target, if any).
  - **Mean IL excess** (how many dB above target, if any).

This gives you a **quantitative, apples-to-apples comparison** of inverse models on the same dataset and evaluator.

---

## Output file layout

Inside your run directory:

```
runs/<run_name>/
  config.json
  stage.json
  logs/run.log

  data/
    selected_geometries.csv
    manifest.json
    failures.jsonl
    device_long/
      part-00000.parquet  (or .csv.gz if pyarrow missing)
      ...
    mzi_metrics/
      part-00000.parquet
      ...

  checkpoints/
    forward_best.pt
    forward_x_scaler.json
    forward_y_scaler.json
    inverse_best.pt
    inverse_x_scaler.json
    inverse_y_scaler.json

  reports/
    hist_*.png
    scatter_Wmmi_Lmmi.png
    lambda0_split_vs_tau.png
    lambda0_bin_counts.csv
    summary.json
    inverse_candidates.csv
    cgan_candidates.csv
```

---

## Things you should decide (before you generate a *paper* dataset)

1) **Foundry stack:** is cladding actually oxide, or air?
2) **Monte Carlo:** do you want gap variation in Stage 1, or only in Stage 2?
3) **Quality threshold:** what throughput (`tau`) counts as “low-quality” for your paper?

If any of these differ, change the defaults near the top of `mmi_mzi_project.py`.

---

## Notes on Simphony usage

EMEPy internally builds a Simphony-compatible model object when you call `EME.propagate()`.
This script focuses on extracting a clean **2×2 transmission matrix** and computing MZI metrics via
a direct phase-sweep method (fast, robust, and paper-friendly). If you want, you can extend the
MZI step to build the full circuit through Simphony explicitly — the device model produced by EMEPy
is already designed for that workflow.

---

If you want, tell me:
- your target fabrication limits (min width, min gap)
- whether your top cladding is oxide or air
- what “acceptable loss” means for you (e.g., IL < 1 dB)

…and I’ll tune the ranges/QC defaults so the dataset is both **high quality** and **not biased**.
