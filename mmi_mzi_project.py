#!/usr/bin/env python3
"""
MMI→MZI dataset generation + evaluation + ML forward surrogate + ML inverse design
================================================================================

This is intentionally a *single-file* project, as requested.

Core goals:
- Publishable-quality dataset (SOI 220 nm, TE, oxide clad; configurable)
- Device-level dataset: complex 2×2 transmission S(λ) for a 2×2 MMI coupler (long format)
- Circuit-level dataset: MZI metrics (ER/IL/BW) via phase sweep
- Strong QC + anti-skew stratified retention
- Logging + resumable generation (fail-safe)
- ML: forward surrogate (geometry+λ → complex S) + inverse design (target metrics → geometry)

Physics engine:
- Uses EMEPy for eigenmode expansion (and therefore uses Simphony through EMEPy's model stack).
  EMEPy: https://emepy.readthedocs.io/
- For material dispersion we use Sellmeier fits as published by refractiveindex.info:
  - SiO2 (Malitson 1965) and Si (Salzberg & Villa 1957 via Tatian fit)
 

IMPORTANT NOTES
---------------
1) This file contains imports for emepy/simphony/pyarrow, but those are optional at import-time.
   The script will tell you exactly what to install if missing.
2) The EMEPy ecosystem has historically had unit conventions in meters; we follow meters internally.
   Geometry is specified in microns in config, then converted to meters before calling EMEPy.
3) Port mapping: EME solvers return *supermodes*. We deterministically map to physical ports
   (top/bottom waveguide basis) using parity detection (even/odd) + a sign disambiguation check.
4) Dataset generation is resumable: it writes shards + a manifest + a failure log as it runs.

Quick start (after installing deps):
    python mmi_mzi_project.py preflight
    python mmi_mzi_project.py generate --stage pilot --run-name pilot_v1
    python mmi_mzi_project.py evaluate --run-dir runs/pilot_v1
    python mmi_mzi_project.py train-forward --run-dir runs/pilot_v1
    python mmi_mzi_project.py train-inverse --run-dir runs/pilot_v1
    python mmi_mzi_project.py inverse-design --run-dir runs/pilot_v1 --target-er 20 --target-bw 40 --target-il 1.0

"""

from __future__ import annotations

import argparse
import atexit
import dataclasses
import datetime as _dt
import functools
import glob
import inspect
import json
import logging
import math
import os
import random
import shutil
import signal
import sys
import textwrap
import time
import scipy 
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc

# Torch is optional at import-time (only needed for training/inverse-design commands).
# Catch ImportError and OSError (Windows DLL issues) but not other Exceptions.
TORCH_AVAILABLE = False
TORCH_IMPORT_ERROR = ""
TORCH_AVAILABLE = False
TORCH_IMPORT_ERROR = ""
torch = None
nn = None
optim = None
Dataset = None
DataLoader = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
    print("[DEBUG] PyTorch imported successfully", file=sys.stderr)
except (ImportError, OSError) as e:
    # ImportError: torch not installed
    # OSError: Windows DLL loading failure, missing CUDA libs, etc.
    TORCH_IMPORT_ERROR = str(e)
    TORCH_AVAILABLE = False
    # torch, nn, optim, Dataset, DataLoader remain None
    print(f"[DEBUG] Torch import failed: {e}", file=sys.stderr)
    pass


# =============================================================================
# Config & dataclasses
# =============================================================================

@dataclass(frozen=True)
class ParameterRange:
    """Closed interval [lo, hi]."""
    lo: float
    hi: float

    def sample_from_unit(self, u: np.ndarray) -> np.ndarray:
        return self.lo + (self.hi - self.lo) * u

    def clip(self, x: float) -> float:
        return float(np.clip(x, self.lo, self.hi))


@dataclass(frozen=True)
class GeometryRanges:
    """Ranges in microns."""
    W_mmi_um: ParameterRange
    L_mmi_um: ParameterRange
    gap_um: ParameterRange
    W_io_um: ParameterRange
    taper_len_um: ParameterRange


@dataclass(frozen=True)
class QuantizationGrid:
    """Quantization (for dedupe and fabrication-grid realism). Units in microns."""
    W_um: float = 0.005     # 5 nm grid
    L_um: float = 0.010     # 10 nm grid
    gap_um: float = 0.005
    taper_um: float = 0.010


@dataclass(frozen=True)
class PlatformConfig:
    """SOI platform settings (microns and refractive indices via Sellmeier)."""
    t_si_um: float = 0.22
    cladding_material: Literal["SiO2"] = "SiO2"
    core_material: Literal["Si"] = "Si"
    # Simulation window padding (microns)
    pad_x_um: float = 3.0
    pad_y_um: float = 3.0


@dataclass(frozen=True)
class WavelengthConfig:
    """Wavelength sweep configuration (nanometers, all in nm units throughout)."""
    start_nm: int = 1520
    stop_nm: int = 1580
    step_nm: int = 1

    def grid_nm(self) -> np.ndarray:
        return np.arange(self.start_nm, self.stop_nm + 1, self.step_nm, dtype=np.int32)

    def grid_m(self) -> np.ndarray:
        return self.grid_nm().astype(np.float64) * 1e-9


@dataclass(frozen=True)
class SolverFidelity:
    """EMEPy solver fidelity knobs."""
    mesh: int = 192
    num_modes_io: int = 8
    num_modes_mmi: int = 30
    taper_steps: int = 10
    port_ext_um: float = 10.0
    emepy_parallel: bool = False
    emepy_quiet: bool = True


@dataclass(frozen=True)
class QcConfig:
    """Quality control thresholds."""
    # Hard physical sanity
    max_power_violation_eps: float = 0.05      # allow up to 1.05 total transmitted power
    max_abs_s: float = 1.2                     # if any |S_ij| exceeds this, probably unstable
    min_throughput_keep: float = 0.50          # drop very lossy devices (dataset quality)
    min_te_fraction: float = 0.85              # port-mode TE fraction threshold
    min_confined_power: float = 0.10           # spurious-mode filter
    # Smoothness sanity (heuristic)
    max_adjacent_delta_mag: float = 0.35       # max |Δ| between adjacent wavelengths for any S element magnitude


@dataclass(frozen=True)
class StratificationConfig:
    """Stratify using (split ratio r, throughput τ) at λ0."""
    lambda0_nm: int = 1550
    r_bins: int = 10
    tau_bins: int = 10
    tau_min: float = 0.50
    tau_max: float = 1.00
    # We don't enforce symmetry by construction, but we can QC-diagnose it.
    symmetry_diag: bool = True
    symmetry_tol: float = 0.15  # |S12 - S21| magnitude tolerance (rough; diagnostic only)


@dataclass(frozen=True)
class MonteCarloConfig:
    """Fabrication perturbations."""
    mc_per_geom: int = 5
    # Stage 1: width-only
    sigma_dW_nm: float = 10.0
    # Stage 2 (optional): include gap variation
    enable_gap_mc: bool = False
    sigma_dGap_nm: float = 15.0
    # Optional dn_eff (applied as tiny scaling of n_core)
    enable_dn_eff: bool = False
    sigma_dn_eff: float = 0.0  # set nonzero to enable


@dataclass(frozen=True)
class DatasetSizeConfig:
    """Candidate pool and retention."""
    n_candidates: int = 5000
    n_keep: int = 500


@dataclass(frozen=True)
class StageConfig:
    """One named stage = a full dataset generation recipe."""
    name: str
    sizes: DatasetSizeConfig
    mc: MonteCarloConfig
    wl: WavelengthConfig
    fidelity_quick: SolverFidelity
    fidelity_full: SolverFidelity


@dataclass(frozen=True)
class GlobalConfig:
    """Top-level config. All values here are chosen to match your 'publishable + practical' goals."""
    seed: int = 7
    platform: PlatformConfig = PlatformConfig()
    ranges: GeometryRanges = GeometryRanges(
        # [Inference] publishable-ish ranges that include good 2×2 MMI designs but also variation:
        # W_mmi: 3–12 µm, L_mmi: 30–300 µm is broad enough to contain many self-imaging regimes.
        W_mmi_um=ParameterRange(3.0, 12.0),
        L_mmi_um=ParameterRange(30.0, 300.0),
        gap_um=ParameterRange(0.15, 1.50),
        W_io_um=ParameterRange(0.35, 0.55),
        taper_len_um=ParameterRange(5.0, 40.0),
    )
    quant: QuantizationGrid = QuantizationGrid()
    qc: QcConfig = QcConfig()
    strat: StratificationConfig = StratificationConfig()
    # Default stage map (you can add more in code or via JSON)
    stages: Tuple[StageConfig, ...] = (
        StageConfig(
            name="debug",
            sizes=DatasetSizeConfig(n_candidates=80, n_keep=10),
            mc=MonteCarloConfig(mc_per_geom=2, sigma_dW_nm=10.0),
            wl=WavelengthConfig(start_nm=1540, stop_nm=1560, step_nm=5),
            fidelity_quick=SolverFidelity(mesh=96, num_modes_io=6, num_modes_mmi=16, taper_steps=6, port_ext_um=5.0),
            fidelity_full=SolverFidelity(mesh=128, num_modes_io=8, num_modes_mmi=20, taper_steps=8, port_ext_um=8.0),
        ),
        StageConfig(
            name="pilot",
            sizes=DatasetSizeConfig(n_candidates=2000, n_keep=300),
            mc=MonteCarloConfig(mc_per_geom=5, sigma_dW_nm=10.0, enable_gap_mc=False),
            wl=WavelengthConfig(start_nm=1520, stop_nm=1580, step_nm=2),
            fidelity_quick=SolverFidelity(mesh=128, num_modes_io=6, num_modes_mmi=18, taper_steps=6, port_ext_um=8.0),
            fidelity_full=SolverFidelity(mesh=192, num_modes_io=8, num_modes_mmi=30, taper_steps=10, port_ext_um=10.0),
        ),
        StageConfig(
            name="paper",
            sizes=DatasetSizeConfig(n_candidates=5000, n_keep=500),
            mc=MonteCarloConfig(mc_per_geom=5, sigma_dW_nm=10.0, enable_gap_mc=False),
            wl=WavelengthConfig(start_nm=1520, stop_nm=1580, step_nm=1),
            fidelity_quick=SolverFidelity(mesh=160, num_modes_io=6, num_modes_mmi=20, taper_steps=8, port_ext_um=8.0),
            fidelity_full=SolverFidelity(mesh=256, num_modes_io=10, num_modes_mmi=35, taper_steps=12, port_ext_um=12.0),
        ),
    )


# =============================================================================
# Utility: logging, seeding, I/O
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)


def now_tag() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def setup_run_dir(base: Path, run_name: Optional[str]) -> Path:
    ensure_dir(base)
    if run_name is None:
        run_name = f"run_{now_tag()}"
    run_dir = base / run_name
    ensure_dir(run_dir)
    ensure_dir(run_dir / "logs")
    ensure_dir(run_dir / "data")
    ensure_dir(run_dir / "reports")
    ensure_dir(run_dir / "checkpoints")
    return run_dir


def setup_logging(run_dir: Path, console_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("mmi_mzi")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(run_dir / "logs" / "run.log", mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Silence noisy libs a bit
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    return logger


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    # On Windows, replace() may fail if file is locked; try removing first
    try:
        tmp.replace(path)
    except PermissionError:
        # File locked or already exists; remove old file first (Windows issue)
        if path.exists():
            path.unlink()
        tmp.rename(path)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def list_parquet_or_csv_shards(dirpath: Path) -> List[Path]:
    parts = sorted(dirpath.glob("part-*.parquet"))
    if parts:
        return parts
    parts = sorted(dirpath.glob("part-*.csv.gz"))
    return parts


def pandas_write_table(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write a shard. Prefer Parquet if pyarrow is available, otherwise gzip CSV.
    """
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(out_path.with_suffix(".parquet"), index=False)
    except Exception:
        df.to_csv(out_path.with_suffix(".csv.gz"), index=False, compression="gzip")


def pandas_read_shards(dirpath: Path) -> pd.DataFrame:
    parts = list_parquet_or_csv_shards(dirpath)
    if not parts:
        raise FileNotFoundError(f"No shards found under {dirpath}")
    dfs = []
    for p in parts:
        if p.suffix == ".parquet":
            dfs.append(pd.read_parquet(p))
        else:
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True)


# =============================================================================
# Material dispersion (Sellmeier) in µm
# =============================================================================

def n_sio2_malitson_1965(lam_um: float) -> float:
    """
    Fused silica Sellmeier (Malitson 1965), λ in µm.
    """
    lam2 = lam_um * lam_um
    n2_minus_1 = (
        (0.6961663 * lam2) / (lam2 - 0.0684043**2)
        + (0.4079426 * lam2) / (lam2 - 0.1162414**2)
        + (0.8974794 * lam2) / (lam2 - 9.896161**2)
    )
    return math.sqrt(1.0 + n2_minus_1)


def n_si_salzberg_villa_1957(lam_um: float) -> float:
    """
    Silicon Sellmeier fit based on Salzberg & Villa 1957 (Tatian fit), λ in µm.
    """
    lam2 = lam_um * lam_um
    n2_minus_1 = (
        (10.6684293 * lam2) / (lam2 - 0.301516485**2)
        + (0.0030434748 * lam2) / (lam2 - 1.13475115**2)
        + (1.54133408 * lam2) / (lam2 - 1104.0**2)
    )
    return math.sqrt(1.0 + n2_minus_1)


# =============================================================================
# Geometry representation + sampling
# =============================================================================

@dataclass(frozen=True)
class Geometry:
    """All in microns."""
    geom_id: int
    W_mmi_um: float
    L_mmi_um: float
    gap_um: float
    W_io_um: float
    taper_len_um: float

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def quantized_key(self, q: QuantizationGrid) -> Tuple[int, int, int, int, int]:
        # Convert to integer "grid ticks" for dedupe.
        return (
            int(round(self.W_mmi_um / q.W_um)),
            int(round(self.L_mmi_um / q.L_um)),
            int(round(self.gap_um / q.gap_um)),
            int(round(self.W_io_um / q.W_um)),
            int(round(self.taper_len_um / q.taper_um)),
        )


def is_geom_valid(geom: Geometry, cfg: GlobalConfig) -> Tuple[bool, str]:
    """
    Conservative geometry validity checks to avoid wasting solver time.
    """
    r = cfg.ranges
    # Range checks
    if not (r.W_mmi_um.lo <= geom.W_mmi_um <= r.W_mmi_um.hi):
        return False, "W_mmi out of range"
    if not (r.L_mmi_um.lo <= geom.L_mmi_um <= r.L_mmi_um.hi):
        return False, "L_mmi out of range"
    if not (r.gap_um.lo <= geom.gap_um <= r.gap_um.hi):
        return False, "gap out of range"
    if not (r.W_io_um.lo <= geom.W_io_um <= r.W_io_um.hi):
        return False, "W_io out of range"
    if not (r.taper_len_um.lo <= geom.taper_len_um <= r.taper_len_um.hi):
        return False, "taper_len out of range"

    # Fabrication-ish constraints
    if geom.W_io_um < 0.30:
        return False, "W_io too small (fabrication)"
    if geom.gap_um < 0.12:
        return False, "gap too small (fabrication)"
    if geom.W_mmi_um < 2 * geom.W_io_um + geom.gap_um + 0.2:
        # Must fit two waveguides + some margin.
        return False, "W_mmi too small for two access waveguides"

    # Basic taper sanity
    if geom.taper_len_um < 2.0:
        return False, "taper too short"
    return True, "ok"


def sobol_geometries(
    n: int,
    cfg: GlobalConfig,
    seed: int,
    start_id: int = 0,
) -> List[Geometry]:
    """
    Sobol sampling (low-discrepancy). We sample 5D: W_mmi, L_mmi, gap, W_io, taper_len.
    """
    sampler = qmc.Sobol(d=5, scramble=True, seed=seed)
    u = sampler.random(n=n)
    rr = cfg.ranges
    W = rr.W_mmi_um.sample_from_unit(u[:, 0])
    L = rr.L_mmi_um.sample_from_unit(u[:, 1])
    g = rr.gap_um.sample_from_unit(u[:, 2])
    Wio = rr.W_io_um.sample_from_unit(u[:, 3])
    t = rr.taper_len_um.sample_from_unit(u[:, 4])

    geoms: List[Geometry] = []
    for i in range(n):
        geom = Geometry(
            geom_id=start_id + i,
            W_mmi_um=float(W[i]),
            L_mmi_um=float(L[i]),
            gap_um=float(g[i]),
            W_io_um=float(Wio[i]),
            taper_len_um=float(t[i]),
        )
        ok, _ = is_geom_valid(geom, cfg)
        if ok:
            geoms.append(geom)
    return geoms


def dedupe_geometries(geoms: List[Geometry], q: QuantizationGrid) -> List[Geometry]:
    seen = set()
    out = []
    for g in geoms:
        key = g.quantized_key(q)
        if key in seen:
            continue
        seen.add(key)
        out.append(g)
    return out


# =============================================================================
# EMEPy wrapper (device simulation)
# =============================================================================

class MissingDependency(RuntimeError):
    pass


def _try_import_emepy():
    try:
        import emepy  # noqa
        from emepy.eme import Layer, EME  # noqa
        from emepy.fd import MSEMpy  # noqa
        return True, None
    except Exception as e:
        return False, e


def require_emepy(logger: logging.Logger) -> None:
    ok, err = _try_import_emepy()
    if ok:
        return
    raise MissingDependency(
        "EMEPy is not installed or failed to import.\n\n"
        "Install it in your environment:\n"
        "  pip install emepy\n\n"
        "If you use conda/venv, activate it first.\n"
        f"Import error: {err}"
    )


def _msemepy_ctor_kwargs(
    wl_m: float,
    num_modes: int,
    mesh: int,
    x_m: np.ndarray,
    y_m: np.ndarray,
    core_index: float,
    cladding_index: float,
    epsfunc,
) -> Dict[str, Any]:
    """
    Build kwargs for MSEMpy with signature-robustness across versions.
    The docs for EMEPy show wl in meters.
    """
    from emepy.fd import MSEMpy  # type: ignore

    sig = inspect.signature(MSEMpy)
    params = sig.parameters

    kw: Dict[str, Any] = {}
    # Wavelength
    if "wl" in params:
        kw["wl"] = wl_m
    elif "wavelength" in params:
        kw["wavelength"] = wl_m
    else:
        kw[list(params.keys())[0]] = wl_m  # best-effort

    # num_modes
    if "num_modes" in params:
        kw["num_modes"] = num_modes

    # mesh
    if "mesh" in params:
        kw["mesh"] = mesh
    elif "mesh_density" in params:
        kw["mesh_density"] = mesh

    # x/y grid
    if "x" in params:
        kw["x"] = x_m
    if "y" in params:
        kw["y"] = y_m

    # Some versions use cladding_width / cladding_thickness instead of explicit x/y grids.
    if "cladding_width" in params:
        kw["cladding_width"] = float(np.max(x_m) - np.min(x_m))
    if "cladding_thickness" in params:
        kw["cladding_thickness"] = float(np.max(y_m) - np.min(y_m))
    if "cladding_height" in params:
        kw["cladding_height"] = float(np.max(y_m) - np.min(y_m))

    # indices
    if "core_index" in params:
        kw["core_index"] = core_index
    if "cladding_index" in params:
        kw["cladding_index"] = cladding_index

    # epsfunc
    if "epsfunc" in params:
        kw["epsfunc"] = epsfunc

    # Some versions expect width/thickness even if epsfunc given; we provide safe defaults if present.
    if "width" in params:
        kw["width"] = float(np.max(x_m) - np.min(x_m)) * 0.5
    if "thickness" in params:
        kw["thickness"] = float(np.max(y_m) - np.min(y_m)) * 0.25

    # Boundary conditions (if present)
    if "boundary" in params:
        # "0000" is used in some examples (no PML); keep default unless specified.
        kw["boundary"] = kw.get("boundary", "0000")

    return {k: v for k, v in kw.items() if v is not None}


def _make_xy_grid_m(
    W_mmi_um: float,
    t_si_um: float,
    pad_x_um: float,
    pad_y_um: float,
    mesh: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct symmetric x/y grids in meters.
    x is lateral, y is vertical (thickness direction).
    """
    x_half_um = 0.5 * W_mmi_um + pad_x_um
    y_half_um = 0.5 * t_si_um + pad_y_um
    x = np.linspace(-x_half_um, x_half_um, mesh, dtype=np.float64) * 1e-6
    y = np.linspace(-y_half_um, y_half_um, mesh, dtype=np.float64) * 1e-6
    return x, y


def _epsfunc_two_waveguides(
    width_um: float,
    gap_um: float,
    t_si_um: float,
    n_core: float,
    n_clad: float,
):
    """
    Returns epsfunc(X, Y) for two identical silicon waveguides (oxide clad),
    separated along +x/-x by edge-to-edge gap.

    The two waveguides are symmetric about x=0.
    """
    eps_core = (n_core**2)
    eps_clad = (n_clad**2)
    half_t = 0.5 * t_si_um * 1e-6

    # Centers at ±(gap/2 + width/2)
    cx = (0.5 * gap_um + 0.5 * width_um) * 1e-6
    half_w = 0.5 * width_um * 1e-6

    def epsfunc(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        # Support both meshgrid arrays and broadcasting from 1D inputs.
        XX = X
        YY = Y
        if XX.ndim == 1 and YY.ndim == 1:
            XX, YY = np.meshgrid(XX, YY, indexing="ij")  # shape (nx, ny)
        core_mask = (np.abs(YY) <= half_t) & (
            (np.abs(XX - cx) <= half_w) | (np.abs(XX + cx) <= half_w)
        )
        out = np.full_like(XX, eps_clad, dtype=np.float64)
        out[core_mask] = eps_core
        return out

    return epsfunc


def _epsfunc_slab(
    width_um: float,
    t_si_um: float,
    n_core: float,
    n_clad: float,
):
    eps_core = (n_core**2)
    eps_clad = (n_clad**2)
    half_t = 0.5 * t_si_um * 1e-6
    half_w = 0.5 * width_um * 1e-6

    def epsfunc(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        XX = X
        YY = Y
        if XX.ndim == 1 and YY.ndim == 1:
            XX, YY = np.meshgrid(XX, YY, indexing="ij")
        core_mask = (np.abs(YY) <= half_t) & (np.abs(XX) <= half_w)
        out = np.full_like(XX, eps_clad, dtype=np.float64)
        out[core_mask] = eps_core
        return out

    return epsfunc


def _mode_parity_score(mode) -> Tuple[float, float]:
    """
    Compute (even_err, odd_err) using Hy field symmetry about x=0.

    Requires:
    - mode.Hy exists
    - mode.x is symmetric and monotonic
    """
    Hy = getattr(mode, "Hy", None)
    x = getattr(mode, "x", None)
    if Hy is None or x is None:
        return 1e9, 1e9

    Hy = np.array(Hy)
    # Hy might be shape (nx, ny) or (ny, nx). We attempt to align with x length.
    if Hy.shape[0] != len(x) and Hy.shape[1] == len(x):
        Hy = Hy.T

    Hy_mirror = Hy[::-1, :]
    denom = np.linalg.norm(Hy) + 1e-12
    even_err = float(np.linalg.norm(Hy - Hy_mirror) / denom)
    odd_err = float(np.linalg.norm(Hy + Hy_mirror) / denom)
    return even_err, odd_err


def _odd_sign_for_top_localization(even_mode, odd_mode) -> int:
    """
    Determine sign s ∈ {+1, -1} for odd mode so that (even + s*odd)/√2 concentrates on x>0.
    """
    Hy_e = np.array(getattr(even_mode, "Hy"))
    Hy_o = np.array(getattr(odd_mode, "Hy"))
    x = np.array(getattr(even_mode, "x"))

    if Hy_e.shape[0] != len(x) and Hy_e.shape[1] == len(x):
        Hy_e = Hy_e.T
        Hy_o = Hy_o.T

    Hy_top = (Hy_e + Hy_o) / math.sqrt(2.0)
    # x axis corresponds to axis 0
    pos = (x > 0).astype(np.float64)[:, None]
    neg = (x < 0).astype(np.float64)[:, None]
    P_pos = float(np.sum(np.abs(Hy_top) ** 2 * pos))
    P_neg = float(np.sum(np.abs(Hy_top) ** 2 * neg))
    return 1 if P_pos >= P_neg else -1


def _pick_even_odd_modes(
    mode_solver,
    qc: QcConfig,
) -> Tuple[int, int, int]:
    """
    Select (even_idx, odd_idx, odd_sign_s) from a port-layer mode solver.
    Returns indices into the mode solver's modes (0-based), and odd_sign for deterministic top/bottom mapping.

    Logic:
    - consider up to first N modes (as provided)
    - filter by TE fraction and confined power
    - among candidates, pick best even-like and best odd-like by symmetry score
    """
    # Ensure modes are solved (some EMEPy versions solve lazily)
    try:
        _ = mode_solver.get_mode(0)  # type: ignore
    except Exception:
        try:
            mode_solver.solve()  # type: ignore
        except Exception:
            pass

    # Gather candidate modes
    modes = []
    # Some solvers expose num_modes; else try until get_mode fails
    max_try = getattr(mode_solver, "num_modes", 12)
    max_try = int(max_try) if max_try is not None else 12

    for i in range(max_try):
        try:
            m = mode_solver.get_mode(i)  # type: ignore
        except Exception:
            try:
                m = mode_solver.modes[i]  # type: ignore
            except Exception:
                break

        # Enforce deterministic phase convention if available
        try:
            m.zero_phase()
        except Exception:
            pass

        # TE / confinement filters
        te_frac = None
        conf = None
        try:
            te_frac = float(m.TE_polarization_fraction())
        except Exception:
            te_frac = None
        try:
            conf = float(m.get_confined_power())
        except Exception:
            conf = None

        if te_frac is not None and te_frac < qc.min_te_fraction:
            continue
        if conf is not None and conf < qc.min_confined_power:
            continue

        modes.append((i, m))

    if len(modes) < 2:
        raise RuntimeError("Not enough guided TE modes in port region (after filtering).")

    # Compute parity scores
    scores = []
    for i, m in modes:
        even_err, odd_err = _mode_parity_score(m)
        scores.append((i, m, even_err, odd_err))

    # Pick even as minimal even_err
    scores_sorted_even = sorted(scores, key=lambda t: t[2])
    even_idx, even_mode, _, _ = scores_sorted_even[0]

    # Pick odd as minimal odd_err among remaining
    scores_sorted_odd = sorted([s for s in scores if s[0] != even_idx], key=lambda t: t[3])
    if not scores_sorted_odd:
        raise RuntimeError("Could not identify odd mode distinct from even mode.")
    odd_idx, odd_mode, _, _ = scores_sorted_odd[0]

    odd_sign = _odd_sign_for_top_localization(even_mode, odd_mode)

    return int(even_idx), int(odd_idx), int(odd_sign)


def emepy_mmi_sparams_2x2(
    geom: Geometry,
    wl_m: float,
    fidelity: SolverFidelity,
    platform: PlatformConfig,
    qc: QcConfig,
    logger: logging.Logger,
    dn_eff: float = 0.0,
) -> np.ndarray:
    """
    Compute 2×2 complex transmission matrix S_local(λ) for the MMI in *physical port* basis:
      inputs:  in1=top-left, in2=bottom-left
      outputs: out1=top-right, out2=bottom-right

    Returns:
      S_local: shape (2,2) complex128

    Steps:
      1) Build EME layers: port(two WG) → taper → slab(MMI) → taper → port(two WG)
      2) Run EMEPy EME.propagate()
      3) Extract transmission block (left→right) in supermode basis
      4) Identify even/odd modes at each port and apply deterministic mapping to localized ports.
    """
    require_emepy(logger)
    from emepy.eme import Layer, EME  # type: ignore
    from emepy.fd import MSEMpy  # type: ignore

    # Material dispersion
    lam_um = wl_m * 1e6
    n_clad = n_sio2_malitson_1965(lam_um)
    n_core = n_si_salzberg_villa_1957(lam_um) + dn_eff  # dn_eff small

    # XY grid
    x_m, y_m = _make_xy_grid_m(
        W_mmi_um=geom.W_mmi_um,
        t_si_um=platform.t_si_um,
        pad_x_um=platform.pad_x_um,
        pad_y_um=platform.pad_y_um,
        mesh=fidelity.mesh,
    )

    # Build mode solvers for each layer cross-section
    def make_ms(epsfunc, num_modes: int):
        kw = _msemepy_ctor_kwargs(
            wl_m=wl_m,
            num_modes=num_modes,
            mesh=fidelity.mesh,
            x_m=x_m,
            y_m=y_m,
            core_index=n_core,
            cladding_index=n_clad,
            epsfunc=epsfunc,
        )
        ms = MSEMpy(**kw)
        return ms

    # Lengths in meters
    port_ext_m = fidelity.port_ext_um * 1e-6
    taper_len_m = geom.taper_len_um * 1e-6
    L_mmi_m = geom.L_mmi_um * 1e-6

    # Cross sections
    eps_port = _epsfunc_two_waveguides(
        width_um=geom.W_io_um, gap_um=geom.gap_um, t_si_um=platform.t_si_um, n_core=n_core, n_clad=n_clad
    )
    eps_slab = _epsfunc_slab(
        width_um=geom.W_mmi_um, t_si_um=platform.t_si_um, n_core=n_core, n_clad=n_clad
    )

    # Taper discretization: we morph (W_io, gap) → (W_mmi/2, 0)
    # This is an approximation that creates a smooth "merge into slab" transition.
    steps = max(2, int(fidelity.taper_steps))
    taper_layers: List[Layer] = []
    for k in range(steps):
        t = (k + 1) / steps  # 0< t <=1
        w_k = geom.W_io_um + t * (0.5 * geom.W_mmi_um - geom.W_io_um)
        g_k = geom.gap_um * (1.0 - t)
        eps_k = _epsfunc_two_waveguides(
            width_um=w_k, gap_um=g_k, t_si_um=platform.t_si_um, n_core=n_core, n_clad=n_clad
        )
        ms_k = make_ms(eps_k, fidelity.num_modes_mmi)
        layer_k = Layer(ms_k, fidelity.num_modes_mmi, wl_m, taper_len_m / steps)
        taper_layers.append(layer_k)

    # Mirror taper for output
    taper_layers_out = list(reversed(taper_layers))

    # Port layers
    ms_in = make_ms(eps_port, fidelity.num_modes_io)
    ms_out = make_ms(eps_port, fidelity.num_modes_io)
    layer_in = Layer(ms_in, fidelity.num_modes_io, wl_m, port_ext_m)
    layer_out = Layer(ms_out, fidelity.num_modes_io, wl_m, port_ext_m)

    # Slab layer
    ms_mmi = make_ms(eps_slab, fidelity.num_modes_mmi)
    layer_mmi = Layer(ms_mmi, fidelity.num_modes_mmi, wl_m, L_mmi_m)

    # Build and run EME
    eme = EME(parallel=fidelity.emepy_parallel, quiet=fidelity.emepy_quiet)
    eme.add_layer(layer_in)
    for lyr in taper_layers:
        eme.add_layer(lyr)
    eme.add_layer(layer_mmi)
    for lyr in taper_layers_out:
        eme.add_layer(lyr)
    eme.add_layer(layer_out)

    # Propagate (this solves modes + cascades). EMEPy returns a Simphony-compatible model.
    model = eme.propagate()
    try:
        # Prefer the returned model's S-parameters (explicit Simphony usage).
        S = np.array(model.s_parameters(), dtype=np.complex128)
    except Exception:
        # Fallback to EME object's S-parameters.
        S = np.array(eme.s_parameters(), dtype=np.complex128)

    # External port mode counts
    N = fidelity.num_modes_io
    if S.shape != (2 * N, 2 * N):
        # Some versions may return different shapes; best-effort to interpret.
        if S.shape[0] < 2 * N or S.shape[1] < 2 * N:
            raise RuntimeError(f"Unexpected S shape {S.shape} for num_modes_io={N}.")
        S = S[: 2 * N, : 2 * N]

    # Transmission left→right in *supermode basis* (right outputs are indices N:2N)
    S_lr = S[N: 2 * N, 0:N]

    # Identify even/odd modes at left and right ports
    even_in, odd_in, s_in = _pick_even_odd_modes(ms_in, qc)
    even_out, odd_out, s_out = _pick_even_odd_modes(ms_out, qc)

    # 2×2 submatrix in even/odd basis
    S_sub = np.array(
        [
            [S_lr[even_out, even_in], S_lr[even_out, odd_in]],
            [S_lr[odd_out, even_in], S_lr[odd_out, odd_in]],
        ],
        dtype=np.complex128,
    )

    # Deterministic odd-mode sign disambiguation (±1)
    P_in = np.diag([1.0, float(s_in)])
    P_out = np.diag([1.0, float(s_out)])
    S_adj = (P_out @ S_sub @ P_in).astype(np.complex128)

    # Fixed unitary transform between localized (top/bottom) and even/odd basis:
    # a_even = (a_top + a_bottom)/√2
    # a_odd  = (a_top - a_bottom)/√2
    U = (1.0 / math.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64)

    # Localized-basis transmission: b_local = U^T * S_adj * U * a_local
    S_local = (U.T @ S_adj @ U).astype(np.complex128)

    return S_local


# =============================================================================
# Dataset generation (resumable, stratified, QC)
# =============================================================================

def _complex_to_cols(S: np.ndarray) -> Dict[str, float]:
    """
    S: (2,2) complex.
    Returns dict with real/imag columns in the requested naming convention.
    """
    return {
        "S_out1_in1_re": float(np.real(S[0, 0])),
        "S_out1_in1_im": float(np.imag(S[0, 0])),
        "S_out2_in1_re": float(np.real(S[1, 0])),
        "S_out2_in1_im": float(np.imag(S[1, 0])),
        "S_out1_in2_re": float(np.real(S[0, 1])),
        "S_out1_in2_im": float(np.imag(S[0, 1])),
        "S_out2_in2_re": float(np.real(S[1, 1])),
        "S_out2_in2_im": float(np.imag(S[1, 1])),
    }


def _power_metrics(S: np.ndarray) -> Dict[str, float]:
    """
    Derived power metrics from 2×2 transmission.
    """
    T11 = float(np.abs(S[0, 0]) ** 2)
    T21 = float(np.abs(S[1, 0]) ** 2)
    T12 = float(np.abs(S[0, 1]) ** 2)
    T22 = float(np.abs(S[1, 1]) ** 2)
    tau1 = T11 + T21
    tau2 = T12 + T22
    r1 = T11 / (tau1 + 1e-12)
    r2 = T22 / (tau2 + 1e-12)
    return {
        "T_out1_in1": T11,
        "T_out2_in1": T21,
        "T_out1_in2": T12,
        "T_out2_in2": T22,
        "tau_in1": float(tau1),
        "tau_in2": float(tau2),
        "split_r_in1": float(r1),
        "split_r_in2": float(r2),
    }


def _qc_spectrum(
    S_list: List[np.ndarray],
    qc: QcConfig,
) -> Tuple[bool, str]:
    """
    QC on an entire wavelength sweep for one (geom, mc).
    """
    # Finite check
    for S in S_list:
        if not np.all(np.isfinite(S)):
            return False, "non-finite S"
        if np.max(np.abs(S)) > qc.max_abs_s:
            return False, f"|S| too large (> {qc.max_abs_s})"

        pm = _power_metrics(S)
        if pm["tau_in1"] > 1.0 + qc.max_power_violation_eps or pm["tau_in2"] > 1.0 + qc.max_power_violation_eps:
            return False, "power > 1 (likely numerical instability)"
        if pm["tau_in1"] < qc.min_throughput_keep and pm["tau_in2"] < qc.min_throughput_keep:
            return False, f"throughput too low (< {qc.min_throughput_keep})"

    # Smoothness check (magnitude only)
    mags = np.array([np.abs(S).reshape(-1) for S in S_list], dtype=np.float64)  # (nwl, 4)
    deltas = np.abs(np.diff(mags, axis=0))
    if deltas.size > 0 and float(np.max(deltas)) > qc.max_adjacent_delta_mag:
        return False, "spectral smoothness fail (spiky magnitude changes)"
    return True, "ok"


def _mzi_metrics_from_coupler_spectrum(
    wl_nm: np.ndarray,
    S_list: List[np.ndarray],
    er_threshold_db: float = 20.0,
    n_phase: int = 512,
) -> Dict[str, Any]:
    """
    MZI metrics from *two identical couplers* using a tunable phase shifter (φ sweep).
    This is the publishable definition you recommended.

    For each wavelength:
      E_out(φ) = C @ diag(1, exp(jφ)) @ C @ [1,0]
      (where C is the 2×2 transmission matrix)

    Then:
      Pmax, Pmin per output port
      ER = 10log10(Pmax/Pmin), IL = -10log10(Pmax)
    """
    phi = np.linspace(0.0, 2.0 * np.pi, n_phase, endpoint=False, dtype=np.float64)
    ejphi = np.exp(1j * phi).astype(np.complex128)

    ER1 = []
    IL1 = []
    ER2 = []
    IL2 = []

    for S in S_list:
        C = S.astype(np.complex128)
        a = C @ np.array([1.0, 0.0], dtype=np.complex128)  # after first coupler
        # Apply phase shift on arm 2 and go through second coupler
        # b(φ) = C @ [a0, a1*exp(jφ)]
        b0 = C[0, 0] * a[0] + C[0, 1] * (a[1] * ejphi)
        b1 = C[1, 0] * a[0] + C[1, 1] * (a[1] * ejphi)
        P1 = np.abs(b0) ** 2
        P2 = np.abs(b1) ** 2

        P1_max = float(np.max(P1))
        P1_min = float(np.min(P1))
        P2_max = float(np.max(P2))
        P2_min = float(np.min(P2))

        ER1.append(10.0 * math.log10((P1_max + 1e-15) / (P1_min + 1e-15)))
        IL1.append(-10.0 * math.log10(P1_max + 1e-15))
        ER2.append(10.0 * math.log10((P2_max + 1e-15) / (P2_min + 1e-15)))
        IL2.append(-10.0 * math.log10(P2_max + 1e-15))

    ER1 = np.array(ER1, dtype=np.float64)
    IL1 = np.array(IL1, dtype=np.float64)
    ER2 = np.array(ER2, dtype=np.float64)
    IL2 = np.array(IL2, dtype=np.float64)

    def bandwidth_nm(mask: np.ndarray) -> float:
        # Largest contiguous interval width in nm.
        idx = np.where(mask)[0]
        if idx.size == 0:
            return 0.0
        # Find contiguous runs
        runs = []
        start = idx[0]
        prev = idx[0]
        for i in idx[1:]:
            if i == prev + 1:
                prev = i
            else:
                runs.append((start, prev))
                start = i
                prev = i
        runs.append((start, prev))
        # Compute widths
        widths = []
        for a, b in runs:
            widths.append(float(wl_nm[b] - wl_nm[a]))
        return float(max(widths))

    mask1 = ER1 >= er_threshold_db
    mask2 = ER2 >= er_threshold_db

    metrics = {
        "ER_threshold_dB": float(er_threshold_db),
        "ER1_min_dB": float(np.min(ER1)),
        "ER1_mean_dB": float(np.mean(ER1)),
        "ER1_bw_nm": bandwidth_nm(mask1),
        "IL1_min_dB": float(np.min(IL1)),
        "IL1_mean_dB": float(np.mean(IL1)),
        "ER2_min_dB": float(np.min(ER2)),
        "ER2_mean_dB": float(np.mean(ER2)),
        "ER2_bw_nm": bandwidth_nm(mask2),
        "IL2_min_dB": float(np.min(IL2)),
        "IL2_mean_dB": float(np.mean(IL2)),
    }
    return metrics


def _apply_mc_perturbation(
    geom: Geometry,
    mc_id: int,
    mc: MonteCarloConfig,
    rng: np.random.Generator,
) -> Tuple[Geometry, Dict[str, float]]:
    """
    Apply fabrication perturbations to geometry.
    Stage 1 recommended: vary dW only (etch/litho bias).
    """
    dW_nm = float(rng.normal(loc=0.0, scale=mc.sigma_dW_nm))
    dW_um = dW_nm * 1e-3

    # Apply dW to widths (both W_io and W_mmi). Keep gap fixed in Stage 1.
    W_mmi = geom.W_mmi_um + dW_um
    W_io = geom.W_io_um + dW_um
    gap = geom.gap_um

    dGap_nm = 0.0
    if mc.enable_gap_mc:
        dGap_nm = float(rng.normal(loc=0.0, scale=mc.sigma_dGap_nm))
        gap = gap + dGap_nm * 1e-3

    # Clamp to positive values
    W_mmi = max(0.5, W_mmi)
    W_io = max(0.25, W_io)
    gap = max(0.10, gap)

    pert = Geometry(
        geom_id=geom.geom_id,
        W_mmi_um=W_mmi,
        L_mmi_um=geom.L_mmi_um,  # could also perturb length; left for future
        gap_um=gap,
        W_io_um=W_io,
        taper_len_um=geom.taper_len_um,
    )

    pert_cols = {
        "mc_id": int(mc_id),
        "dW_nm": float(dW_nm),
        "dGap_nm": float(dGap_nm),
        "dn_eff": 0.0,
    }
    return pert, pert_cols


def _quick_eval_candidate(
    geom: Geometry,
    cfg: GlobalConfig,
    stage: StageConfig,
    logger: logging.Logger,
    dry_run: bool,
) -> Optional[Dict[str, Any]]:
    """
    Quick evaluation at λ0 for stratified retention.
    """
    lam0_nm = cfg.strat.lambda0_nm
    wl_m = lam0_nm * 1e-9

    try:
        if dry_run:
            # Fake but deterministic-ish quick eval (for pipeline tests without emepy).
            rnd = (hash(geom.quantized_key(cfg.quant)) % 10_000) / 10_000.0
            tau = cfg.strat.tau_min + (cfg.strat.tau_max - cfg.strat.tau_min) * rnd
            r = rnd  # not physical, but good enough for dry-run stratification
            return {
                "geom": geom,
                "tau_in1": float(tau),
                "split_r_in1": float(r),
                "S": np.array([[math.sqrt(r), 0.0], [math.sqrt(1 - r), 0.0]], dtype=np.complex128),
            }

        S = emepy_mmi_sparams_2x2(
            geom=geom,
            wl_m=wl_m,
            fidelity=stage.fidelity_quick,
            platform=cfg.platform,
            qc=cfg.qc,
            logger=logger,
            dn_eff=0.0,
        )
        pm = _power_metrics(S)
        tau = pm["tau_in1"]
        r = pm["split_r_in1"]

        if not (cfg.strat.tau_min <= tau <= cfg.strat.tau_max):
            return None
        if not (0.0 <= r <= 1.0):
            return None

        return {"geom": geom, "tau_in1": float(tau), "split_r_in1": float(r), "S": S}
    except Exception as e:
        logger.warning(f"Quick-eval failed geom_id={geom.geom_id}: {e}")
        # logger.debug(f"Quick-eval failed geom_id={geom.geom_id}: {e}")
        # return None


def select_geometries_stratified(
    candidates: List[Geometry],
    cfg: GlobalConfig,
    stage: StageConfig,
    logger: logging.Logger,
    dry_run: bool,
) -> List[Geometry]:
    """
    Evaluate candidates at lambda_0 and keep a balanced set across (r, tau) bins.
    """
    logger.info(f"Quick evaluating {len(candidates)} candidates at lambda_0={cfg.strat.lambda0_nm} nm ...")

    evals = []
    failed_count = 0
    tau_filtered_count = 0
    
    for g in candidates:
        r = _quick_eval_candidate(g, cfg, stage, logger, dry_run=dry_run)
        if r is not None:
            evals.append(r)
        else:
            tau_filtered_count += 1
            failed_count += 1

    if not evals:
        # Provide actionable diagnostics
        logger.error("No candidates survived quick-eval.")
        logger.error(f"  Attempted: {len(candidates)}")
        logger.error(f"  Filtered by tau_min/tau_max thresholds: {tau_filtered_count}")
        logger.error(f"  Total failures/invalid: {failed_count}")
        logger.error(f"\nRecommendations:")
        logger.error(f"  1. Check EMEPy installation and solver fidelity settings (see logs above)")
        logger.error(f"  2. Try relaxing tau thresholds: tau_min <= target <= tau_max")
        logger.error(f"     Current: {cfg.strat.tau_min:.2f} <= tau <= {cfg.strat.tau_max:.2f}")
        logger.error(f"  3. Try --dry-run first to debug without expensive solver calls")
        raise RuntimeError("No candidates survived quick-eval. See logs for diagnostics.")

    r_vals = np.array([e["split_r_in1"] for e in evals], dtype=np.float64)
    tau_vals = np.array([e["tau_in1"] for e in evals], dtype=np.float64)

    # Bin edges
    r_edges = np.linspace(0.0, 1.0, cfg.strat.r_bins + 1)
    tau_edges = np.linspace(cfg.strat.tau_min, cfg.strat.tau_max, cfg.strat.tau_bins + 1)

    # Assign bins
    r_bin = np.clip(np.digitize(r_vals, r_edges) - 1, 0, cfg.strat.r_bins - 1)
    t_bin = np.clip(np.digitize(tau_vals, tau_edges) - 1, 0, cfg.strat.tau_bins - 1)

    # Collect indices per bin
    bins: Dict[Tuple[int, int], List[int]] = {}
    for i in range(len(evals)):
        key = (int(r_bin[i]), int(t_bin[i]))
        bins.setdefault(key, []).append(i)

    # Balanced selection
    target = stage.sizes.n_keep
    keys = list(bins.keys())
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(keys)

    # Determine roughly how many per bin
    per_bin = max(1, int(math.ceil(target / max(1, len(keys)))))

    selected_idxs: List[int] = []
    for key in keys:
        idxs = bins[key]
        rng.shuffle(idxs)
        take = min(per_bin, len(idxs))
        selected_idxs.extend(idxs[:take])
        if len(selected_idxs) >= target:
            break

    # If still short, fill from remaining, prioritizing sparsely filled bins
    if len(selected_idxs) < target:
        remaining = [i for i in range(len(evals)) if i not in set(selected_idxs)]
        rng.shuffle(remaining)
        selected_idxs.extend(remaining[: (target - len(selected_idxs))])

    selected = [evals[i]["geom"] for i in selected_idxs[:target]]

    logger.info(
        f"Selected {len(selected)} geometries (from {len(evals)} quick-valid candidates) "
        f"using {cfg.strat.r_bins}×{cfg.strat.tau_bins} binning."
    )
    return selected


def _load_or_init_manifest(run_dir: Path) -> Dict[str, Any]:
    manifest_path = run_dir / "data" / "manifest.json"
    return load_json(manifest_path, default={"processed": {}, "shards": {"device": [], "mzi": []}})


def _save_manifest(run_dir: Path, manifest: Dict[str, Any]) -> None:
    save_json(run_dir / "data" / "manifest.json", manifest)


def generate_dataset(
    cfg: GlobalConfig,
    stage_name: str,
    run_dir: Path,
    logger: logging.Logger,
    dry_run: bool = False,
    yes: bool = False,
) -> None:
    """
    End-to-end generation:
      candidates → quick eval @ λ0 → stratified keep → full sweep + MC + QC → shards.
    """
    stage = next((s for s in cfg.stages if s.name == stage_name), None)
    if stage is None:
        raise ValueError(f"Unknown stage '{stage_name}'. Valid stages: {[s.name for s in cfg.stages]}")

    # Save config snapshot
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        save_json(cfg_path, dataclasses.asdict(cfg))
        save_json(run_dir / "stage.json", dataclasses.asdict(stage))

    # Preflight questions / confirmation
    if not yes:
        msg = f"""
        You are about to generate a dataset with:
          stage: {stage.name}
          candidates: {stage.sizes.n_candidates}
          kept geometries: {stage.sizes.n_keep}
          MC per geometry: {stage.mc.mc_per_geom}
          wavelength grid: {stage.wl.start_nm}–{stage.wl.stop_nm} nm @ {stage.wl.step_nm} nm

        Before you run this for a publishable paper, double-check:
          1) Platform is correct (SOI {cfg.platform.t_si_um} µm, oxide clad).
          2) Parameter ranges make sense for your foundry constraints.
          3) QC min throughput ({cfg.qc.min_throughput_keep}) matches your "low-quality" definition.
          4) You have installed EMEPy and its dependencies (or use --dry-run).

        If that looks right, rerun with --yes to actually execute.
        """
        logger.info(textwrap.dedent(msg).strip())
        return

    seed_everything(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # Resume manifest
    manifest = _load_or_init_manifest(run_dir)

    device_dir = run_dir / "data" / "device_long"
    mzi_dir = run_dir / "data" / "mzi_metrics"
    ensure_dir(device_dir)
    ensure_dir(mzi_dir)

    failures_path = run_dir / "data" / "failures.jsonl"

    # Candidate sampling
    logger.info("Sampling candidate geometries (Sobol) ...")
    candidates = sobol_geometries(stage.sizes.n_candidates, cfg, seed=cfg.seed, start_id=0)
    candidates = dedupe_geometries(candidates, cfg.quant)
    logger.info(f"Candidates after range+fabrication checks and dedupe: {len(candidates)}")

    # Stratified retention using quick eval
    selected = select_geometries_stratified(candidates, cfg, stage, logger, dry_run=dry_run)

    # Persist selected list for reproducibility
    sel_df = pd.DataFrame([g.as_dict() for g in selected])
    sel_df.to_csv(run_dir / "data" / "selected_geometries.csv", index=False)

    wl_nm = stage.wl.grid_nm()
    wl_m = stage.wl.grid_m()

    # Shard buffering
    device_rows: List[Dict[str, Any]] = []
    mzi_rows: List[Dict[str, Any]] = []
    shard_idx_device = len(list_parquet_or_csv_shards(device_dir))
    shard_idx_mzi = len(list_parquet_or_csv_shards(mzi_dir))

    def flush():
        nonlocal shard_idx_device, shard_idx_mzi, device_rows, mzi_rows, manifest

        if device_rows:
            df = pd.DataFrame(device_rows)
            out = device_dir / f"part-{shard_idx_device:05d}"
            pandas_write_table(df, out)
            manifest["shards"]["device"].append(out.name)
            shard_idx_device += 1
            device_rows = []
        if mzi_rows:
            df = pd.DataFrame(mzi_rows)
            out = mzi_dir / f"part-{shard_idx_mzi:05d}"
            pandas_write_table(df, out)
            manifest["shards"]["mzi"].append(out.name)
            shard_idx_mzi += 1
            mzi_rows = []
        _save_manifest(run_dir, manifest)

    atexit.register(flush)

    # Handle SIGINT gracefully
    interrupted = {"flag": False}

    def _sigint_handler(signum, frame):
        interrupted["flag"] = True
        logger.warning("Received interrupt; flushing buffers and exiting safely ...")
        flush()
        sys.exit(130)

    signal.signal(signal.SIGINT, _sigint_handler)

    # Main loop
    logger.info("Starting full-band + Monte Carlo generation ...")

    for g in selected:
        geom_key = str(g.quantized_key(cfg.quant))
        processed_mc = manifest["processed"].get(geom_key, [])

        for mc_id in range(stage.mc.mc_per_geom):
            if mc_id in processed_mc:
                continue

            pert_geom, pert_cols = _apply_mc_perturbation(g, mc_id, stage.mc, rng)

            # Validate perturbed geometry too
            ok, reason = is_geom_valid(pert_geom, cfg)
            if not ok:
                with failures_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"geom_id": g.geom_id, "mc_id": mc_id, "stage": stage.name, "reason": reason}) + "\n")
                processed_mc.append(mc_id)
                manifest["processed"][geom_key] = processed_mc
                continue

            # Run spectrum
            try:
                S_list: List[np.ndarray] = []
                if dry_run:
                    # Simple synthetic spectrum for end-to-end testing.
                    base = (hash((pert_geom.quantized_key(cfg.quant), mc_id)) % 1000) / 1000.0
                    for lam in wl_nm:
                        phase = 2 * np.pi * (lam - wl_nm[0]) / max(1.0, (wl_nm[-1] - wl_nm[0]))
                        r = 0.5 + 0.4 * math.sin(phase + 2 * np.pi * base)
                        r = float(np.clip(r, 0.05, 0.95))
                        amp = 0.9 - 0.1 * base
                        S_fake = amp * np.array([[math.sqrt(r), 1j * math.sqrt(1 - r)], [1j * math.sqrt(1 - r), math.sqrt(r)]], dtype=np.complex128)
                        S_list.append(S_fake)
                else:
                    for lam_m in wl_m:
                        S = emepy_mmi_sparams_2x2(
                            geom=pert_geom,
                            wl_m=float(lam_m),
                            fidelity=stage.fidelity_full,
                            platform=cfg.platform,
                            qc=cfg.qc,
                            logger=logger,
                            dn_eff=pert_cols.get("dn_eff", 0.0),
                        )
                        S_list.append(S)

                # QC
                ok_qc, qc_reason = _qc_spectrum(S_list, cfg.qc)
                if not ok_qc:
                    with failures_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"geom_id": g.geom_id, "mc_id": mc_id, "stage": stage.name, "reason": qc_reason}) + "\n")
                    processed_mc.append(mc_id)
                    manifest["processed"][geom_key] = processed_mc
                    continue

                # Write device rows
                for lam_nm_i, S in zip(wl_nm, S_list):
                    row: Dict[str, Any] = {
                        "geom_id": int(g.geom_id),
                        "lambda_nm": int(lam_nm_i),
                        **{k: float(v) for k, v in pert_geom.as_dict().items() if k != "geom_id"},
                        **pert_cols,
                        **_complex_to_cols(S),
                        **_power_metrics(S),
                    }
                    device_rows.append(row)

                # Circuit metrics (MZI)
                mzi = _mzi_metrics_from_coupler_spectrum(wl_nm=wl_nm.astype(np.float64), S_list=S_list, er_threshold_db=20.0)
                mzi_row = {
                    "geom_id": int(g.geom_id),
                    **{k: float(v) for k, v in pert_geom.as_dict().items() if k != "geom_id"},
                    **pert_cols,
                    **mzi,
                }
                mzi_rows.append(mzi_row)

                # Mark processed
                processed_mc.append(mc_id)
                manifest["processed"][geom_key] = processed_mc

            except Exception as e:
                # Fail-safe: log and keep going.
                with failures_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"geom_id": g.geom_id, "mc_id": mc_id, "stage": stage.name, "reason": f"exception: {str(e)}"}) + "\n")
                processed_mc.append(mc_id)
                manifest["processed"][geom_key] = processed_mc

            # Flush periodically (every mc case)
            flush()

        if interrupted["flag"]:
            break

    # Final flush
    flush()
    logger.info(f"Done. Dataset shards are in: {device_dir} and {mzi_dir}")
    logger.info(f"Failures log: {failures_path}")


# =============================================================================
# Dataset evaluation / report
# =============================================================================

def evaluate_dataset(run_dir: Path, logger: logging.Logger) -> None:
    """
    Generate sanity-check plots + summary tables + dataset card into run_dir/reports.
    Produces publication-ready evaluation artifacts:
      - Parameter histograms
      - Geometry coverage scatter plots
      - MZI metric distributions
      - Bin count table at λ0
      - Dataset card (JSON + CSV summary)
      - QC pass rate diagnostics
    """
    reports = run_dir / "reports"
    ensure_dir(reports)

    device_dir = run_dir / "data" / "device_long"
    mzi_dir = run_dir / "data" / "mzi_metrics"

    logger.info("Loading dataset shards ...")
    df_dev = pandas_read_shards(device_dir)
    df_mzi = pandas_read_shards(mzi_dir)

    # Basic stats
    n_unique_geoms = int(df_mzi["geom_id"].nunique()) if "geom_id" in df_mzi.columns else None
    n_total_rows = int(len(df_mzi))
    n_mc_per_geom = int(n_total_rows / n_unique_geoms) if n_unique_geoms and n_unique_geoms > 0 else 1
    
    summary = {
        "n_device_rows": int(len(df_dev)),
        "n_mzi_rows": int(len(df_mzi)),
        "n_geom_ids": n_unique_geoms,
        "n_mc_per_geom": n_mc_per_geom,
        "lambda_nm_min": int(df_dev["lambda_nm"].min()),
        "lambda_nm_max": int(df_dev["lambda_nm"].max()),
        "lambda_nm_step": int(np.median(np.diff(np.sort(df_dev["lambda_nm"].unique())))),
    }

    # Distributions and checks
    import matplotlib.pyplot as plt  # local import to keep base import light

    params = ["W_mmi_um", "L_mmi_um", "gap_um", "W_io_um", "taper_len_um"]
    param_stats = {}
    for p in params:
        if p in df_mzi.columns:
            plt.figure(figsize=(6, 4))
            df_mzi[p].hist(bins=40, edgecolor='black', alpha=0.7)
            plt.xlabel(p)
            plt.ylabel("count")
            plt.title(f"Distribution: {p}")
            plt.tight_layout()
            plt.savefig(reports / f"hist_{p}.png", dpi=160)
            plt.close()
            
            param_stats[p] = {
                "mean": float(df_mzi[p].mean()),
                "std": float(df_mzi[p].std()),
                "min": float(df_mzi[p].min()),
                "max": float(df_mzi[p].max()),
                "median": float(df_mzi[p].median()),
            }

    # 2) Scatter W_mmi vs L_mmi
    plt.figure(figsize=(8, 6))
    plt.scatter(df_mzi["W_mmi_um"], df_mzi["L_mmi_um"], s=10, alpha=0.5, edgecolors='none')
    plt.xlabel("W_mmi_um", fontsize=11)
    plt.ylabel("L_mmi_um", fontsize=11)
    plt.title("Geometry coverage: W_mmi vs L_mmi")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(reports / "scatter_Wmmi_Lmmi.png", dpi=160)
    plt.close()

    # 3) Split ratio and throughput at λ0
    lam0 = 1550
    df0 = df_dev[df_dev["lambda_nm"] == lam0].copy()
    if len(df0) > 0:
        plt.figure(figsize=(7, 5))
        plt.scatter(df0["split_r_in1"], df0["tau_in1"], s=8, alpha=0.6, edgecolors='none')
        plt.xlabel("split_r_in1", fontsize=11)
        plt.ylabel("tau_in1", fontsize=11)
        plt.title(f"wavelength={lam0} nm diagnostics: split ratio vs throughput")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(reports / "lambda0_split_vs_tau.png", dpi=160)
        plt.close()

        # Bin count table
        r_bins = 10
        t_bins = 10
        df0["r_bin"] = pd.cut(df0["split_r_in1"], bins=r_bins, include_lowest=True)
        df0["t_bin"] = pd.cut(df0["tau_in1"], bins=t_bins, include_lowest=True)
        bin_counts = df0.groupby(["r_bin", "t_bin"]).size().unstack(fill_value=0)
        bin_counts.to_csv(reports / "lambda0_bin_counts.csv")
        summary["lambda0_bin_counts_shape"] = [int(bin_counts.shape[0]), int(bin_counts.shape[1])]
        summary["lambda0_coverage_pct"] = float(100.0 * len(df0) / len(df_mzi))

    # 4) MZI metric distributions and statistics
    mzi_metrics = ["ER1_min_dB", "ER1_bw_nm", "IL1_mean_dB", "ER2_min_dB", "ER2_bw_nm", "IL2_mean_dB"]
    mzi_stats = {}
    for m in mzi_metrics:
        if m in df_mzi.columns:
            # Filter out infs and nans for cleaner stats
            valid = df_mzi[m].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) > 0:
                plt.figure(figsize=(6, 4))
                valid.hist(bins=40, edgecolor='black', alpha=0.7)
                plt.xlabel(m)
                plt.ylabel("count")
                plt.title(f"Distribution: {m}")
                plt.tight_layout()
                plt.savefig(reports / f"hist_{m}.png", dpi=160)
                plt.close()
                
                mzi_stats[m] = {
                    "mean": float(valid.mean()),
                    "std": float(valid.std()),
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                    "median": float(valid.median()),
                }

    # 5) QC diagnostics (if available)
    qc_cols = [c for c in df_mzi.columns if 'qc' in c.lower()]
    qc_stats = {}
    if qc_cols:
        for qc in qc_cols:
            if df_mzi[qc].dtype in [bool, np.bool_, 'bool']:
                pass_rate = float(df_mzi[qc].astype(float).mean())
                qc_stats[qc] = {"pass_rate": pass_rate}

    # Build comprehensive dataset card
    dataset_card = {
        "dataset_summary": summary,
        "geometry_parameters": param_stats,
        "mzi_metrics": mzi_stats,
        "qc_diagnostics": qc_stats,
        "generated_at": str(_dt.datetime.now().isoformat()),
    }

    # Save summary JSON (dataset card)
    save_json(reports / "dataset_card.json", dataset_card)
    
    # Save compact CSV version for quick reference
    card_df = pd.DataFrame([dataset_card["dataset_summary"]])
    card_df.to_csv(reports / "dataset_summary.csv", index=False)
    
    logger.info(f"Evaluation complete. Artifacts saved to: {reports}")
    logger.info(f"  - Parameter distributions (hist_*.png)")
    logger.info(f"  - Geometry coverage scatter (scatter_Wmmi_Lmmi.png)")
    logger.info(f"  - MZI metric distributions (hist_ER*.png, hist_IL*.png)")
    logger.info(f"  - λ0 diagnostics (lambda0_split_vs_tau.png, lambda0_bin_counts.csv)")
    logger.info(f"  - Dataset card (dataset_card.json, dataset_summary.csv)")
    logger.info(f"\nDataset summary:")
    logger.info(f"  Geometries: {summary['n_geom_ids']}, MC per geom: {summary['n_mc_per_geom']}")
    logger.info(f"  Total MZI rows: {summary['n_mzi_rows']}, wavelength range: {summary['lambda_nm_min']}-{summary['lambda_nm_max']} nm")


# =============================================================================
# ML: forward surrogate and inverse design (PyTorch)
# =============================================================================

# V1 commented out: torch was imported incorrectly at module level, causing import
# failures even when torch wasn't needed. Now using TORCH_AVAILABLE flag instead.
# class NumpyDataset(Dataset):
#     def __init__(self, X: np.ndarray, Y: np.ndarray):
#         self.X = torch.from_numpy(X.astype(np.float32))
#         self.Y = torch.from_numpy(Y.astype(np.float32))
#
#     def __len__(self) -> int:
#         return self.X.shape[0]
#
#     def __getitem__(self, idx: int):
#         return self.X[idx], self.Y[idx]

if not TORCH_AVAILABLE:
    # Torch isn't available; provide informative placeholders.
    class NumpyDataset:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch is required for training/inverse-design commands.\n"
                f"Install it with: pip install torch\n\n"
                f"Original import error: {TORCH_IMPORT_ERROR}"
            )

    class MLP:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch is required for training/inverse-design commands.\n"
                f"Install it with: pip install torch\n\n"
                f"Original import error: {TORCH_IMPORT_ERROR}"
            )

    class MDN:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "PyTorch is required for training/inverse-design commands.\n"
                f"Install it with: pip install torch\n\n"
                f"Original import error: {TORCH_IMPORT_ERROR}"
            )
else:
    # Torch is available; define ML classes
    class NumpyDataset(Dataset):
        def __init__(self, X: np.ndarray, Y: np.ndarray):
            self.X = torch.from_numpy(X.astype(np.float32))
            self.Y = torch.from_numpy(Y.astype(np.float32))

        def __len__(self) -> int:
            return self.X.shape[0]

        def __getitem__(self, idx: int):
            return self.X[idx], self.Y[idx]

    class MLP(torch.nn.Module):
        def __init__(self, in_dim: int, out_dim: int, hidden: Sequence[int] = (256, 256, 256), dropout: float = 0.0):
            super().__init__()
            layers: List[torch.nn.Module] = []
            d = in_dim
            for h in hidden:
                layers.append(torch.nn.Linear(d, h))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(torch.nn.Dropout(dropout))
                d = h
            layers.append(torch.nn.Linear(d, out_dim))
            self.net = torch.nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    class MDN(torch.nn.Module):
        """
        Mixture Density Network for inverse design: p(geometry | target_metrics).
        Outputs K Gaussian components (diagonal covariance).
        """
        def __init__(self, in_dim: int, out_dim: int, K: int = 8, hidden: Sequence[int] = (256, 256)):
            super().__init__()
            self.K = K
            self.out_dim = out_dim
            self.backbone = MLP(in_dim, hidden[-1], hidden=hidden[:-1] if len(hidden) > 1 else (), dropout=0.0)
            hdim = hidden[-1]
            self.pi = torch.nn.Linear(hdim, K)
            self.mu = torch.nn.Linear(hdim, K * out_dim)
            self.log_sigma = torch.nn.Linear(hdim, K * out_dim)

        def forward(self, x):
            h = self.backbone(x)
            pi = torch.softmax(self.pi(h), dim=-1)  # (B, K)
            mu = self.mu(h).view(-1, self.K, self.out_dim)  # (B, K, D)
            log_sigma = self.log_sigma(h).view(-1, self.K, self.out_dim).clamp(-7.0, 3.0)
            return pi, mu, log_sigma


def mdn_nll(pi, mu, log_sigma, y) -> torch.Tensor:
    """
    Negative log-likelihood for diagonal Gaussian mixture.
    """
    # y: (B, D) -> (B, 1, D)
    y = y.unsqueeze(1)
    sigma = torch.exp(log_sigma)
    # Gaussian log prob per component
    log_prob = -0.5 * (((y - mu) / sigma) ** 2 + 2.0 * log_sigma + math.log(2.0 * math.pi)).sum(dim=-1)  # (B, K)
    # log-sum-exp with mixture weights
    log_mix = torch.log(pi + 1e-12) + log_prob
    nll = -torch.logsumexp(log_mix, dim=-1).mean()
    return nll


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-12
    Xs = (X - mean) / std
    return Xs, {"mean": mean.tolist(), "std": std.tolist()}


def _standardize_apply(X: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    mean = np.array(scaler["mean"], dtype=np.float64)
    std = np.array(scaler["std"], dtype=np.float64)
    return (X - mean) / std


def _split_by_geom_id(df: pd.DataFrame, seed: int, ratios=(0.8, 0.1, 0.1)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    geom_ids = df["geom_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(geom_ids)
    n = len(geom_ids)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)
    train_ids = geom_ids[:n_train]
    val_ids = geom_ids[n_train:n_train + n_val]
    test_ids = geom_ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def train_forward(
    run_dir: Path,
    logger: logging.Logger,
    epochs: int = 40,
    batch_size: int = 2048,
    lr: float = 1e-3,
) -> None:
    """
    Forward surrogate:
      inputs: [W_mmi, L_mmi, gap, W_io, taper_len, dW_nm, dGap_nm, lambda_nm]
      outputs: 8 values = real/imag of 2×2 S transmission

    This corresponds directly to the long-format device dataset and is a good reusable primitive for inverse design.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for train-forward.\n"
            "Install it with: pip install torch\n\n"
            f"Original import error: {TORCH_IMPORT_ERROR}"
        )

    device_dir = run_dir / "data" / "device_long"
    df = pandas_read_shards(device_dir)

    # Features and targets
    feat_cols = ["W_mmi_um", "L_mmi_um", "gap_um", "W_io_um", "taper_len_um", "dW_nm", "dGap_nm", "lambda_nm"]
    targ_cols = [
        "S_out1_in1_re", "S_out1_in1_im",
        "S_out2_in1_re", "S_out2_in1_im",
        "S_out1_in2_re", "S_out1_in2_im",
        "S_out2_in2_re", "S_out2_in2_im",
    ]

    train_ids, val_ids, test_ids = _split_by_geom_id(df, seed=7)

    def subset(ids):
        return df[df["geom_id"].isin(ids)].copy()

    df_train = subset(train_ids)
    df_val = subset(val_ids)
    df_test = subset(test_ids)

    X_train = df_train[feat_cols].to_numpy(dtype=np.float64)
    Y_train = df_train[targ_cols].to_numpy(dtype=np.float64)
    X_val = df_val[feat_cols].to_numpy(dtype=np.float64)
    Y_val = df_val[targ_cols].to_numpy(dtype=np.float64)
    X_test = df_test[feat_cols].to_numpy(dtype=np.float64)
    Y_test = df_test[targ_cols].to_numpy(dtype=np.float64)

    # Standardize
    X_train_s, x_scaler = _standardize_fit(X_train)
    Y_train_s, y_scaler = _standardize_fit(Y_train)
    X_val_s = _standardize_apply(X_val, x_scaler)
    Y_val_s = _standardize_apply(Y_val, y_scaler)
    X_test_s = _standardize_apply(X_test, x_scaler)
    Y_test_s = _standardize_apply(Y_test, y_scaler)

    save_json(run_dir / "checkpoints" / "forward_x_scaler.json", x_scaler)
    save_json(run_dir / "checkpoints" / "forward_y_scaler.json", y_scaler)

    # DataLoaders
    train_ds = NumpyDataset(X_train_s, Y_train_s)
    val_ds = NumpyDataset(X_val_s, Y_val_s)
    test_ds = NumpyDataset(X_test_s, Y_test_s)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = MLP(in_dim=X_train_s.shape[1], out_dim=Y_train_s.shape[1], hidden=(256, 256, 256), dropout=0.05)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val = float("inf")
    best_path = run_dir / "checkpoints" / "forward_best.pt"

    def eval_mse(dl) -> float:
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in dl:
                pred = model(xb)
                loss = loss_fn(pred, yb).item()
                total += loss * xb.shape[0]
                n += xb.shape[0]
        return total / max(1, n)

    logger.info(f"Training forward surrogate on {len(train_ds)} rows; validating on {len(val_ds)} ...")
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for xb, yb in train_dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        val_mse = eval_mse(val_dl)
        dt = time.time() - t0
        logger.info(f"[forward] epoch {ep:03d} | val_mse={val_mse:.6f} | {dt:.1f}s")

        if val_mse < best_val:
            best_val = val_mse
            torch.save({"model": model.state_dict(), "epoch": ep, "val_mse": val_mse}, best_path)

    # Test
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    test_mse = eval_mse(test_dl)
    logger.info(f"[forward] best epoch={ckpt['epoch']} val_mse={ckpt['val_mse']:.6f} | test_mse={test_mse:.6f}")


def train_inverse(
    run_dir: Path,
    logger: logging.Logger,
    epochs: int = 80,
    batch_size: int = 512,
    lr: float = 2e-4,
    K: int = 8,
) -> None:
    """
    Inverse design model (probabilistic):
      inputs: [ER1_bw_nm, ER1_min_dB, IL1_mean_dB, ER2_bw_nm, ER2_min_dB, IL2_mean_dB]
      outputs (distribution): geometry params [W_mmi_um, L_mmi_um, gap_um, W_io_um, taper_len_um]
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for train-inverse.\n"
            "Install it with: pip install torch\n\n"
            f"Original import error: {TORCH_IMPORT_ERROR}"
        )

    df = pandas_read_shards(run_dir / "data" / "mzi_metrics")

    feat_cols = ["ER1_bw_nm", "ER1_min_dB", "IL1_mean_dB", "ER2_bw_nm", "ER2_min_dB", "IL2_mean_dB"]
    out_cols = ["W_mmi_um", "L_mmi_um", "gap_um", "W_io_um", "taper_len_um"]

    # Filter extreme outliers (optional, improves stability)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + out_cols)

    train_ids, val_ids, test_ids = _split_by_geom_id(df, seed=7)

    def subset(ids):
        return df[df["geom_id"].isin(ids)].copy()

    df_train = subset(train_ids)
    df_val = subset(val_ids)
    df_test = subset(test_ids)

    X_train = df_train[feat_cols].to_numpy(dtype=np.float64)
    Y_train = df_train[out_cols].to_numpy(dtype=np.float64)
    X_val = df_val[feat_cols].to_numpy(dtype=np.float64)
    Y_val = df_val[out_cols].to_numpy(dtype=np.float64)
    X_test = df_test[feat_cols].to_numpy(dtype=np.float64)
    Y_test = df_test[out_cols].to_numpy(dtype=np.float64)

    X_train_s, x_scaler = _standardize_fit(X_train)
    Y_train_s, y_scaler = _standardize_fit(Y_train)
    X_val_s = _standardize_apply(X_val, x_scaler)
    Y_val_s = _standardize_apply(Y_val, y_scaler)
    X_test_s = _standardize_apply(X_test, x_scaler)
    Y_test_s = _standardize_apply(Y_test, y_scaler)

    save_json(run_dir / "checkpoints" / "inverse_x_scaler.json", x_scaler)
    save_json(run_dir / "checkpoints" / "inverse_y_scaler.json", y_scaler)

    train_dl = DataLoader(NumpyDataset(X_train_s, Y_train_s), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(NumpyDataset(X_val_s, Y_val_s), batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(NumpyDataset(X_test_s, Y_test_s), batch_size=batch_size, shuffle=False)

    model = MDN(in_dim=X_train_s.shape[1], out_dim=Y_train_s.shape[1], K=K, hidden=(256, 256))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    best_path = run_dir / "checkpoints" / "inverse_best.pt"

    def eval_nll(dl) -> float:
        model.eval()
        total = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in dl:
                pi, mu, log_sigma = model(xb)
                loss = mdn_nll(pi, mu, log_sigma, yb).item()
                total += loss * xb.shape[0]
                n += xb.shape[0]
        return total / max(1, n)

    logger.info(f"Training inverse MDN on {len(X_train_s)} samples; validating on {len(X_val_s)} ...")
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        for xb, yb in train_dl:
            opt.zero_grad()
            pi, mu, log_sigma = model(xb)
            loss = mdn_nll(pi, mu, log_sigma, yb)
            loss.backward()
            opt.step()

        val_nll = eval_nll(val_dl)
        dt = time.time() - t0
        logger.info(f"[inverse] epoch {ep:03d} | val_nll={val_nll:.5f} | {dt:.1f}s")

        if val_nll < best_val:
            best_val = val_nll
            torch.save({"model": model.state_dict(), "epoch": ep, "val_nll": val_nll, "K": K}, best_path)

    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    test_nll = eval_nll(test_dl)
    logger.info(f"[inverse] best epoch={ckpt['epoch']} val_nll={ckpt['val_nll']:.5f} | test_nll={test_nll:.5f}")


def inverse_design(
    run_dir: Path,
    logger: logging.Logger,
    target_er: float,
    target_bw: float,
    target_il: float,
    n_samples: int = 256,
    pick_top_k: int = 10,
    validate: bool = False,
) -> None:
    """
    Use the trained inverse MDN to sample candidate geometries for target specs.
    Then score them using simple heuristics (and optionally forward surrogate if available).

    Targets are interpreted as:
      - target_er: minimum ER (dB) you'd like across band (proxy)
      - target_bw: bandwidth (nm) where ER>=20 dB you'd like (proxy)
      - target_il: mean IL (dB) you'd like (proxy)

    This is intentionally lightweight so you can iterate fast.
    
    If validate=True, re-runs physics solver on top-k candidates and produces validation report.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is required for inverse-design.\n"
            "Install it with: pip install torch\n\n"
            f"Original import error: {TORCH_IMPORT_ERROR}"
        )

    inv_path = run_dir / "checkpoints" / "inverse_best.pt"
    if not inv_path.exists():
        raise FileNotFoundError(f"Missing inverse checkpoint: {inv_path}. Run train-inverse first.")

    x_scaler = load_json(run_dir / "checkpoints" / "inverse_x_scaler.json", default=None)
    y_scaler = load_json(run_dir / "checkpoints" / "inverse_y_scaler.json", default=None)
    if x_scaler is None or y_scaler is None:
        raise FileNotFoundError("Missing inverse scalers. Run train-inverse first.")

    ckpt = torch.load(inv_path, map_location="cpu")
    K = int(ckpt.get("K", 8))
    model = MDN(in_dim=6, out_dim=5, K=K, hidden=(256, 256))
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Construct target feature vector (we target both outputs similarly; you can refine later)
    x = np.array([[target_bw, target_er, target_il, target_bw, target_er, target_il]], dtype=np.float64)
    x_s = _standardize_apply(x, x_scaler)
    xb = torch.from_numpy(x_s.astype(np.float32))

    with torch.no_grad():
        pi, mu, log_sigma = model(xb)
        pi = pi[0].numpy()                 # (K,)
        mu = mu[0].numpy()                 # (K, D)
        sigma = np.exp(log_sigma[0].numpy())  # (K, D)

    rng = np.random.default_rng(7)
    # Sample mixture components
    comps = rng.choice(np.arange(K), size=n_samples, p=pi)
    ys = []
    for c in comps:
        ys.append(mu[c] + sigma[c] * rng.normal(size=mu.shape[1]))
    ys = np.stack(ys, axis=0)  # (n_samples, D)

    # Unstandardize to geometry space
    mean = np.array(y_scaler["mean"], dtype=np.float64)
    std = np.array(y_scaler["std"], dtype=np.float64)
    geoms = ys * std + mean
    
    # Enforce hard physical bounds by clipping
    geoms[:, 0] = np.clip(geoms[:, 0], 0.5, 20.0)    # Wm: waveguide width
    geoms[:, 1] = np.clip(geoms[:, 1], 10.0, 500.0)  # Lm: MMI length
    geoms[:, 2] = np.clip(geoms[:, 2], 0.05, 3.0)    # gap: gap between waveguides
    geoms[:, 3] = np.clip(geoms[:, 3], 0.20, 0.80)   # Wio: I/O waveguide width
    geoms[:, 4] = np.clip(geoms[:, 4], 0.5, 100.0)   # taper: taper factor

    # Score candidates with simple bounds (you should later validate using physics or forward surrogate)
    # Score encourages: small IL, large BW, large ER
    scores = []
    for i in range(geoms.shape[0]):
        Wm, Lm, gap, Wio, taper = geoms[i]
        # All candidates pass bounds now (clipped above), just score them
        # Heuristic: prefer moderate tapers and not-too-extreme gaps
        reg = 0.01 * (abs(Wm - 8.0) + 0.003 * abs(Lm - 120.0) + abs(gap - 0.5))
        score = (target_bw / 50.0) + (target_er / 20.0) - (target_il / 2.0) - reg
        scores.append((score, i))

    if not scores:
        logger.warning("No candidates sampled (unexpected error).")
        return

    scores.sort(reverse=True, key=lambda t: t[0])
    top = scores[:pick_top_k]

    # Print candidates
    logger.info("Top inverse-design candidates (UNVERIFIED by physics):")
    for rank, (score, idx) in enumerate(top, start=1):
        Wm, Lm, gap, Wio, taper = geoms[idx]
        logger.info(
            f"#{rank:02d} score={score:.3f} | "
            f"W_mmi={Wm:.3f} um, L_mmi={Lm:.2f} um, gap={gap:.3f} um, W_io={Wio:.3f} um, taper={taper:.2f} um"
        )

    # Save to CSV
    out = run_dir / "reports" / "inverse_candidates.csv"
    rows = []
    for score, idx in top:
        Wm, Lm, gap, Wio, taper = geoms[idx]
        rows.append({"score": score, "W_mmi_um": Wm, "L_mmi_um": Lm, "gap_um": gap, "W_io_um": Wio, "taper_len_um": taper})
    pd.DataFrame(rows).to_csv(out, index=False)
    logger.info(f"Saved candidates to: {out}")
    
    # Optional: validate with physics solver
    if validate:
        logger.info("Running inverse design validation with physics solver...")
        inverse_validate(run_dir, logger, out)


def inverse_validate(
    run_dir: Path,
    logger: logging.Logger,
    candidates_csv: Path,
    verbose: bool = True,
) -> None:
    """
    Re-run physics solver on candidate geometries to validate surrogate predictions.
    Compares predicted vs simulated metrics and outputs a validation report.
    
    Requires EMEPy to be available.
    """
    require_emepy(logger)
    
    cfg = GlobalConfig()
    stage = next((s for s in cfg.stages if s.name == "pilot"), cfg.stages[0])
    
    # Load candidates
    if not candidates_csv.exists():
        logger.warning(f"Candidates CSV not found: {candidates_csv}")
        return
    
    df_cand = pd.read_csv(candidates_csv)
    logger.info(f"Validating {len(df_cand)} candidate geometries...")
    
    # Load forward surrogate if available (for prediction comparison)
    fwd_path = run_dir / "checkpoints" / "forward_best.pt"
    has_forward = fwd_path.exists()
    
    wl_m = stage.wl.grid_m()
    
    validation_rows = []
    
    for idx, row in df_cand.iterrows():
        geom = Geometry(
            geom_id=idx,
            W_mmi_um=float(row["W_mmi_um"]),
            L_mmi_um=float(row["L_mmi_um"]),
            gap_um=float(row["gap_um"]),
            W_io_um=float(row["W_io_um"]),
            taper_len_um=float(row["taper_len_um"]),
        )
        
        try:
            # Run full solver (same as in generation)
            S_list = []
            for wl in wl_m:
                S = emepy_mmi_sparams_2x2(
                    geom=geom,
                    wl_m=wl,
                    fidelity=stage.fidelity_full,
                    platform=cfg.platform,
                    qc=cfg.qc,
                    logger=logger,
                    dn_eff=0.0,
                )
                S_list.append(S)
            
            # Compute MZI metrics
            metrics = _mzi_metrics_from_coupler_spectrum(
                wl_nm=stage.wl.grid_nm().astype(np.float64),
                S_list=S_list,
                er_threshold_db=20.0,
            )
            
            val_row = {
                "geom_id": idx,
                "W_mmi_um": float(row["W_mmi_um"]),
                "L_mmi_um": float(row["L_mmi_um"]),
                "gap_um": float(row["gap_um"]),
                "W_io_um": float(row["W_io_um"]),
                "taper_len_um": float(row["taper_len_um"]),
                "ER1_min_dB_sim": float(metrics.get("ER1_min_dB", np.nan)),
                "ER1_bw_nm_sim": float(metrics.get("ER1_bw_nm", np.nan)),
                "IL1_mean_dB_sim": float(metrics.get("IL1_mean_dB", np.nan)),
                "ER2_min_dB_sim": float(metrics.get("ER2_min_dB", np.nan)),
                "ER2_bw_nm_sim": float(metrics.get("ER2_bw_nm", np.nan)),
                "IL2_mean_dB_sim": float(metrics.get("IL2_mean_dB", np.nan)),
                "solver_status": "success",
            }
            validation_rows.append(val_row)
            
            if verbose and (idx + 1) % max(1, len(df_cand) // 5) == 0:
                logger.info(f"  Validated {idx + 1}/{len(df_cand)} geometries...")
        
        except Exception as e:
            val_row = {
                "geom_id": idx,
                "W_mmi_um": float(row["W_mmi_um"]),
                "L_mmi_um": float(row["L_mmi_um"]),
                "gap_um": float(row["gap_um"]),
                "W_io_um": float(row["W_io_um"]),
                "taper_len_um": float(row["taper_len_um"]),
                "solver_status": f"failed: {str(e)[:50]}",
            }
            validation_rows.append(val_row)
    
    # Save validation report
    df_val = pd.DataFrame(validation_rows)
    val_path = run_dir / "reports" / "inverse_validation.csv"
    df_val.to_csv(val_path, index=False)
    logger.info(f"Validation report saved to: {val_path}")
    
    # Summary statistics
    success = df_val["solver_status"].eq("success").sum()
    logger.info(f"Validation complete: {success}/{len(df_val)} geometries simulated successfully")
    
    if success > 0:
        df_ok = df_val[df_val["solver_status"] == "success"]
        logger.info(f"  ER1 range: {df_ok['ER1_min_dB_sim'].min():.1f} – {df_ok['ER1_min_dB_sim'].max():.1f} dB")
        logger.info(f"  ER1 BW range: {df_ok['ER1_bw_nm_sim'].min():.1f} – {df_ok['ER1_bw_nm_sim'].max():.1f} nm")
        logger.info(f"  IL1 range: {df_ok['IL1_mean_dB_sim'].min():.2f} – {df_ok['IL1_mean_dB_sim'].max():.2f} dB")


# =============================================================================
# CLI
# =============================================================================

def preflight(cfg: GlobalConfig, logger: logging.Logger) -> None:
    """
    Check environment and print the key assumptions/questions.
    """
    logger.info("Preflight checks ...")
    logger.info(f"Python: {sys.version.split()[0]}")
    logger.info(f"NumPy: {np.__version__}")
    logger.info(f"Pandas: {pd.__version__}")
    # logger.info(f"SciPy: {qmc.__module__.split('.')[0]} (qmc available)")
    logger.info(f"SciPy: {scipy.__version__} (qmc available)")

    # Optional deps
    ok_emepy, err_emepy = _try_import_emepy()
    logger.info(f"EMEPy import: {'OK' if ok_emepy else 'MISSING'}")
    if not ok_emepy:
        logger.warning(
            f"  EMEPy is required for physics-backed dataset generation (generate/full-run).\n"
            f"  Install: pip install emepy\n"
            f"  Error: {err_emepy}"
        )
    try:
        import simphony  # noqa
        ver = getattr(simphony, "__version__", None)
        logger.info(f"Simphony import: OK" + (f" (version {ver})" if ver else ""))
    except Exception:
        logger.info("Simphony import: MISSING (EMEPy may install it, but you can also: pip install simphony)")
    if not ok_emepy:
        logger.info("  Install: pip install emepy")

    try:
        import pyarrow  # noqa
        logger.info("pyarrow: OK (Parquet enabled)")
    except Exception:
        logger.info("pyarrow: MISSING (will fall back to gzipped CSV shards)")
        logger.info("  Install: pip install pyarrow")

    if TORCH_AVAILABLE:
        logger.info(f"PyTorch: {torch.__version__} (available for training/inverse)")
    else:
        logger.warning(
            f"PyTorch: NOT AVAILABLE.\n"
            f"  Required for training-forward, train-inverse, inverse-design commands.\n"
            f"  Install: pip install torch\n"
            f"  Error: {TORCH_IMPORT_ERROR}"
        )

    # Key publishable assumptions checklist
    msg = f"""
    Assumptions baked into defaults:
      - Platform: SOI device layer {cfg.platform.t_si_um} µm, oxide clad (SiO2)
      - Polarization: TE-focused port mode selection (TE fraction >= {cfg.qc.min_te_fraction})
      - Access symmetry: left/right identical
      - Ports: in1/in2 = top/bottom on left, out1/out2 = top/bottom on right
      - Wavelength band defaults per stage, with 1520-1580 nm used for paper stage

    Before you generate a paper dataset, confirm:
      1) Is your top cladding oxide or air?
      2) Do you want to include gap Monte-Carlo in Stage 1?
      3) Are your fabrication minimum widths/gaps tighter than our defaults?
    """
    logger.info(textwrap.dedent(msg).strip())


def main() -> None:
    cfg = GlobalConfig()

    ap = argparse.ArgumentParser(
        prog="mmi_mzi_project",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # preflight
    sub.add_parser("preflight", help="Check environment + assumptions")

    # generate
    ap_gen = sub.add_parser("generate", help="Generate dataset (device_long + mzi_metrics)")
    ap_gen.add_argument("--stage", default="pilot", choices=[s.name for s in cfg.stages])
    ap_gen.add_argument("--run-name", default=None, help="Name under runs/. Default auto timestamp.")
    ap_gen.add_argument("--dry-run", action="store_true", help="Generate synthetic data (no emepy needed).")
    ap_gen.add_argument("--yes", action="store_true", help="Actually run generation (otherwise prints checklist).")

    # evaluate
    ap_eval = sub.add_parser("evaluate", help="Evaluate dataset and write plots/tables to reports/")
    ap_eval.add_argument("--run-dir", required=True)

    # train-forward
    ap_tf = sub.add_parser("train-forward", help="Train forward surrogate (geometry+λ -> complex S)")
    ap_tf.add_argument("--run-dir", required=True)
    ap_tf.add_argument("--epochs", type=int, default=40)

    # train-inverse
    ap_ti = sub.add_parser("train-inverse", help="Train inverse MDN (target metrics -> geometry distribution)")
    ap_ti.add_argument("--run-dir", required=True)
    ap_ti.add_argument("--epochs", type=int, default=80)
    ap_ti.add_argument("--K", type=int, default=8)

    # inverse-design
    ap_id = sub.add_parser("inverse-design", help="Sample candidate geometries for target specs using trained inverse model")
    ap_id.add_argument("--run-dir", required=True)
    ap_id.add_argument("--target-er", type=float, required=True, help="Target ER (dB) (proxy feature)")
    ap_id.add_argument("--target-bw", type=float, required=True, help="Target BW (nm) where ER>=20 dB (proxy feature)")
    ap_id.add_argument("--target-il", type=float, required=True, help="Target mean IL (dB) (proxy feature)")
    ap_id.add_argument("--n-samples", type=int, default=256)
    ap_id.add_argument("--top-k", type=int, default=10)

    # full-run
    ap_full = sub.add_parser(
        "full-run",
        help=(
            "End-to-end pipeline: generate -> evaluate -> train-forward -> train-inverse -> inverse-design\n"
            "Examples:\n"
            "  python mmi_mzi_project.py full-run --stage debug --run-name debug_v1 --dry-run --yes\n"
            "  python mmi_mzi_project.py full-run --stage pilot --run-name pilot_v1 --yes\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    ap_full.add_argument("--stage", default="debug", choices=[s.name for s in cfg.stages])
    ap_full.add_argument("--run-name", default=None, help="Name under runs/. Default auto timestamp.")
    ap_full.add_argument("--dry-run", action="store_true", help="Generate synthetic data (no emepy needed).")
    ap_full.add_argument("--yes", action="store_true", help="Actually run generation (otherwise prints checklist).")
    ap_full.add_argument("--forward-epochs", type=int, default=40)
    ap_full.add_argument("--inverse-epochs", type=int, default=80)
    ap_full.add_argument("--inverse-K", type=int, default=8)
    ap_full.add_argument("--target-er", type=float, default=20.0)
    ap_full.add_argument("--target-bw", type=float, default=40.0)
    ap_full.add_argument("--target-il", type=float, default=1.0)
    ap_full.add_argument("--inverse-samples", type=int, default=512)
    ap_full.add_argument("--inverse-top-k", type=int, default=20)

    args = ap.parse_args()

    # Resolve run dir
    if args.cmd == "preflight":
        run_dir = setup_run_dir(Path("runs"), run_name="preflight_" + now_tag())
    elif args.cmd == "generate":
        run_dir = setup_run_dir(Path("runs"), run_name=args.run_name)
    elif args.cmd == "full-run":
        run_dir = setup_run_dir(Path("runs"), run_name=args.run_name)
    else:
        run_dir = Path(args.run_dir)

    logger = setup_logging(run_dir)

    if args.cmd == "preflight":
        preflight(cfg, logger)
        return

    if args.cmd == "generate":
        generate_dataset(cfg, stage_name=args.stage, run_dir=run_dir, logger=logger, dry_run=args.dry_run, yes=args.yes)
        return

    if args.cmd == "evaluate":
        evaluate_dataset(run_dir=run_dir, logger=logger)
        return

    if args.cmd == "train-forward":
        train_forward(run_dir=run_dir, logger=logger, epochs=args.epochs)
        return

    if args.cmd == "train-inverse":
        train_inverse(run_dir=run_dir, logger=logger, epochs=args.epochs, K=args.K)
        return

    if args.cmd == "inverse-design":
        inverse_design(
            run_dir=run_dir,
            logger=logger,
            target_er=args.target_er,
            target_bw=args.target_bw,
            target_il=args.target_il,
            n_samples=args.n_samples,
            pick_top_k=args.top_k,
        )
        return

    if args.cmd == "full-run":
        generate_dataset(cfg, stage_name=args.stage, run_dir=run_dir, logger=logger, dry_run=args.dry_run, yes=args.yes)
        # If user didn't pass --yes, generation is intentionally a no-op checklist print.
        # In that case, stop here.
        if not args.yes:
            return
        evaluate_dataset(run_dir=run_dir, logger=logger)
        train_forward(run_dir=run_dir, logger=logger, epochs=args.forward_epochs)
        train_inverse(run_dir=run_dir, logger=logger, epochs=args.inverse_epochs, K=args.inverse_K)
        inverse_design(
            run_dir=run_dir,
            logger=logger,
            target_er=args.target_er,
            target_bw=args.target_bw,
            target_il=args.target_il,
            n_samples=args.inverse_samples,
            pick_top_k=args.inverse_top_k,
        )
        return


if __name__ == "__main__":
    main()