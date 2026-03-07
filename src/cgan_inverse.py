#!/usr/bin/env python3
"""
Conditional GAN for inverse design: MZI metrics -> MMI geometry
================================================================

This script is intentionally separate from mmi_mzi_project.py so that:
- The existing MDN-based inverse model remains untouched.
- You can experiment with a cGAN-style inverse model on the SAME dataset.

It expects a completed run directory produced by mmi_mzi_project.py, i.e.:
  runs/<run_name>/
    data/mzi_metrics/part-*.parquet  (or .csv.gz)

Two main commands:

1) Train cGAN:

   python cgan_inverse.py train \
     --run-dir runs/pilot_v1 \
     --out-dir runs/pilot_v1/cgan \
     --epochs 200 \
     --batch-size 512

2) Sample candidate geometries for target specs:

   python cgan_inverse.py sample \
     --run-dir runs/pilot_v1 \
     --cgan-dir runs/pilot_v1/cgan \
     --target-er 20 \
     --target-bw 40 \
     --target-il 1.0 \
     --n-samples 512 \
     --out-csv runs/pilot_v1/reports/cgan_candidates.csv
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Small utilities
# =============================================================================


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def list_shards(dirpath: Path) -> Sequence[Path]:
    parts = sorted(dirpath.glob("part-*.parquet"))
    if parts:
        return parts
    parts = sorted(dirpath.glob("part-*.csv.gz"))
    if parts:
        return parts
    raise FileNotFoundError(f"No shards found under {dirpath}")


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-12
    Xs = (X - mean) / std
    return Xs, {"mean": mean.tolist(), "std": std.tolist()}


def standardize_apply(X: np.ndarray, scaler: Dict[str, Any]) -> np.ndarray:
    mean = np.array(scaler["mean"], dtype=np.float64)
    std = np.array(scaler["std"], dtype=np.float64)
    return (X - mean) / std


def set_seed(seed: int = 7) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# =============================================================================
# Dataset
# =============================================================================


COND_COLS = [
    "ER1_bw_nm",
    "ER1_min_dB",
    "IL1_mean_dB",
    "ER2_bw_nm",
    "ER2_min_dB",
    "IL2_mean_dB",
]

GEOM_COLS = [
    "W_mmi_um",
    "L_mmi_um",
    "gap_um",
    "W_io_um",
    "taper_len_um",
]


class MziCganDataset(Dataset):
    def __init__(self, cond: np.ndarray, geom: np.ndarray):
        assert cond.shape[0] == geom.shape[0]
        self.cond = torch.from_numpy(cond.astype(np.float32))
        self.geom = torch.from_numpy(geom.astype(np.float32))

    def __len__(self) -> int:
        return self.cond.shape[0]

    def __getitem__(self, idx: int):
        return self.cond[idx], self.geom[idx]


def load_mzi_dataset(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = run_dir / "data" / "mzi_metrics"
    shards = list_shards(data_dir)
    dfs = []
    for p in shards:
        if p.suffix == ".parquet":
            dfs.append(pd.read_parquet(p))
        else:
            dfs.append(pd.read_csv(p))
    df = pd.concat(dfs, ignore_index=True)

    # Drop obvious NaNs/Infs
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=COND_COLS + GEOM_COLS)

    cond = df[COND_COLS].to_numpy(dtype=np.float64)
    geom = df[GEOM_COLS].to_numpy(dtype=np.float64)
    return cond, geom


# =============================================================================
# Models
# =============================================================================


class Generator(nn.Module):
    def __init__(self, noise_dim: int, cond_dim: int, out_dim: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        d = noise_dim + cond_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.LeakyReLU(0.2))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, cond], dim=-1)
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, geom_dim: int, cond_dim: int, hidden: Sequence[int] = (256, 256)):
        super().__init__()
        layers: list[nn.Module] = []
        d = geom_dim + cond_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.LeakyReLU(0.2))
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, geom: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = torch.cat([geom, cond], dim=-1)
        return self.net(x)


# =============================================================================
# Training
# =============================================================================


@dataclass
class TrainConfig:
    run_dir: Path
    out_dir: Path
    epochs: int = 200
    batch_size: int = 512
    noise_dim: int = 16
    lr_g: float = 2e-4
    lr_d: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    seed: int = 7
    d_steps: int = 1
    g_steps: int = 1


def train_cgan(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.out_dir / "train_log.txt"

    def log(msg: str) -> None:
        print(msg)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    cond_np, geom_np = load_mzi_dataset(cfg.run_dir)

    # Standardize cond and geom separately; store scalers.
    cond_s, cond_scaler = standardize_fit(cond_np)
    geom_s, geom_scaler = standardize_fit(geom_np)
    save_json(cfg.out_dir / "cond_scaler.json", cond_scaler)
    save_json(cfg.out_dir / "geom_scaler.json", geom_scaler)

    dataset = MziCganDataset(cond_s, geom_s)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator(noise_dim=cfg.noise_dim, cond_dim=cond_s.shape[1], out_dim=geom_s.shape[1]).to(device)
    D = Discriminator(geom_dim=geom_s.shape[1], cond_dim=cond_s.shape[1]).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2))
    opt_D = torch.optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2))

    bce = nn.BCEWithLogitsLoss()

    real_label = 1.0
    fake_label = 0.0

    log(f"Starting cGAN training on {len(dataset)} samples, device={device} ...")

    for epoch in range(1, cfg.epochs + 1):
        G.train()
        D.train()
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        n_batches = 0

        for cond_batch, geom_batch in loader:
            cond_batch = cond_batch.to(device)
            geom_batch = geom_batch.to(device)
            bsize = geom_batch.size(0)

            # -----------------
            #  Train Discriminator
            # -----------------
            for _ in range(cfg.d_steps):
                opt_D.zero_grad(set_to_none=True)

                # Real
                logits_real = D(geom_batch, cond_batch).view(-1)
                labels_real = torch.full_like(logits_real, real_label, device=device)
                loss_real = bce(logits_real, labels_real)

                # Fake
                z = torch.randn(bsize, cfg.noise_dim, device=device)
                fake_geom = G(z, cond_batch).detach()
                logits_fake = D(fake_geom, cond_batch).view(-1)
                labels_fake = torch.full_like(logits_fake, fake_label, device=device)
                loss_fake = bce(logits_fake, labels_fake)

                loss_D = loss_real + loss_fake
                loss_D.backward()
                opt_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            for _ in range(cfg.g_steps):
                opt_G.zero_grad(set_to_none=True)
                z = torch.randn(bsize, cfg.noise_dim, device=device)
                fake_geom = G(z, cond_batch)
                logits = D(fake_geom, cond_batch).view(-1)
                labels = torch.full_like(logits, real_label, device=device)
                loss_G = bce(logits, labels)
                loss_G.backward()
                opt_G.step()

            epoch_d_loss += float(loss_D.item())
            epoch_g_loss += float(loss_G.item())
            n_batches += 1

        epoch_d_loss /= max(1, n_batches)
        epoch_g_loss /= max(1, n_batches)
        log(f"Epoch {epoch:03d} | D_loss={epoch_d_loss:.4f} | G_loss={epoch_g_loss:.4f}")

        # Save checkpoints periodically
        if epoch % 20 == 0 or epoch == cfg.epochs:
            torch.save(G.state_dict(), cfg.out_dir / f"G_epoch{epoch:03d}.pt")
            torch.save(D.state_dict(), cfg.out_dir / f"D_epoch{epoch:03d}.pt")

    # Final best checkpoints
    torch.save(G.state_dict(), cfg.out_dir / "G_final.pt")
    torch.save(D.state_dict(), cfg.out_dir / "D_final.pt")
    log("Training complete. Saved G_final.pt and D_final.pt.")


# =============================================================================
# Sampling
# =============================================================================


def build_condition_vector(
    target_er: float,
    target_bw: float,
    target_il: float,
) -> np.ndarray:
    """
    Build a 6D condition vector from scalar targets.
    We mirror between outputs 1 and 2 by using same targets twice.
    """
    return np.array(
        [
            target_bw,
            target_er,
            target_il,
            target_bw,
            target_er,
            target_il,
        ],
        dtype=np.float64,
    )


def sample_cgan(
    run_dir: Path,
    cgan_dir: Path,
    target_er: float,
    target_bw: float,
    target_il: float,
    n_samples: int,
    out_csv: Path,
    noise_dim: int = 16,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cond_scaler = load_json(cgan_dir / "cond_scaler.json")
    geom_scaler = load_json(cgan_dir / "geom_scaler.json")

    cond_vec = build_condition_vector(target_er=target_er, target_bw=target_bw, target_il=target_il)
    cond_vec = cond_vec[None, :]  # (1, 6)
    cond_s = standardize_apply(cond_vec, cond_scaler)

    cond_dim = cond_s.shape[1]
    geom_dim = len(geom_scaler["mean"])

    G = Generator(noise_dim=noise_dim, cond_dim=cond_dim, out_dim=geom_dim)
    ckpt_path = cgan_dir / "G_final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing generator checkpoint: {ckpt_path}")
    G.load_state_dict(torch.load(ckpt_path, map_location=device))
    G.to(device)
    G.eval()

    cond_tensor = torch.from_numpy(cond_s.astype(np.float32)).to(device)
    cond_tensor = cond_tensor.repeat(n_samples, 1)

    with torch.no_grad():
        z = torch.randn(n_samples, noise_dim, device=device)
        geom_samp_s = G(z, cond_tensor).cpu().numpy()

    # Un-standardize to geometry space
    mean = np.array(geom_scaler["mean"], dtype=np.float64)
    std = np.array(geom_scaler["std"], dtype=np.float64)
    geom_samp = geom_samp_s * std + mean

    # Soft clipping using tanh-like function to preserve diversity
    # This allows slight overshoot but gradually pushes back towards bounds
    # Column order: [W_mmi_um, L_mmi_um, gap_um, W_io_um, taper_len_um]
    
    bounds = np.array([
        [3.0, 12.0],      # W_mmi
        [30.0, 300.0],    # L_mmi
        [0.15, 1.50],     # gap
        [0.35, 0.55],     # W_io
        [5.0, 40.0]       # taper
    ])
    
    for col_idx in range(5):
        lo, hi = bounds[col_idx]
        mid = (lo + hi) / 2.0
        scale = (hi - lo) / 2.0
        # Soft clip: normalize to [-1, 1], tanh, then scale back
        norm = (geom_samp[:, col_idx] - mid) / scale
        clipped = mid + scale * np.tanh(norm)
        geom_samp[:, col_idx] = clipped

    rows = []
    for i in range(n_samples):
        Wm, Lm, gap, Wio, taper = geom_samp[i]
        rows.append(
            {
                "W_mmi_um": float(Wm),
                "L_mmi_um": float(Lm),
                "gap_um": float(gap),
                "W_io_um": float(Wio),
                "taper_len_um": float(taper),
                "target_ER_dB": float(target_er),
                "target_BW_nm": float(target_bw),
                "target_IL_dB": float(target_il),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {n_samples} cGAN samples to {out_csv}")


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="cgan_inverse",
        description="Conditional GAN for inverse design: MZI metrics -> MMI geometry",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train
    ap_train = sub.add_parser("train", help="Train cGAN on mzi_metrics from a run directory")
    ap_train.add_argument("--run-dir", required=True, help="Path to run directory (from mmi_mzi_project.py)")
    ap_train.add_argument(
        "--out-dir",
        required=True,
        help="Directory to store cGAN checkpoints and scalers (e.g., runs/<run_name>/cgan)",
    )
    ap_train.add_argument("--epochs", type=int, default=200)
    ap_train.add_argument("--batch-size", type=int, default=512)
    ap_train.add_argument("--noise-dim", type=int, default=16)
    ap_train.add_argument("--lr-g", type=float, default=2e-4)
    ap_train.add_argument("--lr-d", type=float, default=2e-4)

    # sample
    ap_samp = sub.add_parser("sample", help="Sample geometries from a trained cGAN for target specs")
    ap_samp.add_argument("--run-dir", required=True, help="Path to run directory (unused except for consistency)")
    ap_samp.add_argument(
        "--cgan-dir",
        required=True,
        help="Directory with G_final.pt and scaler JSONs (typically same as --out-dir for train)",
    )
    ap_samp.add_argument("--target-er", type=float, required=True, help="Target ER (dB)")
    ap_samp.add_argument("--target-bw", type=float, required=True, help="Target BW (nm)")
    ap_samp.add_argument("--target-il", type=float, required=True, help="Target mean IL (dB)")
    ap_samp.add_argument("--n-samples", type=int, default=512)
    ap_samp.add_argument("--out-csv", required=True, help="Path to write sampled geometries CSV")
    ap_samp.add_argument("--noise-dim", type=int, default=16)

    args = ap.parse_args()

    if args.cmd == "train":
        cfg = TrainConfig(
            run_dir=Path(args.run_dir),
            out_dir=Path(args.out_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            noise_dim=args.noise_dim,
            lr_g=args.lr_g,
            lr_d=args.lr_d,
        )
        train_cgan(cfg)
        return

    if args.cmd == "sample":
        sample_cgan(
            run_dir=Path(args.run_dir),
            cgan_dir=Path(args.cgan_dir),
            target_er=args.target_er,
            target_bw=args.target_bw,
            target_il=args.target_il,
            n_samples=args.n_samples,
            out_csv=Path(args.out_csv),
            noise_dim=args.noise_dim,
        )
        return


if __name__ == "__main__":
    main()

