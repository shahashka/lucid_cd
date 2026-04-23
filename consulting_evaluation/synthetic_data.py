"""
Synthetic gene-expression data generator for consulting_evaluation.

Experimental design (mirrors the RPE1 radiation study):
  - 6 treatment groups: 1 control + 5 dose rates
  - 9 weeks, 2 measurements per week  ->  n = 108 samples
  - p genes organized in M correlated modules (latent factors)
  - M_signal modules have treatment- and time-dependent trajectories
  - Remaining M - M_signal modules are noise

Generation model:
  X = F @ U.T + noise
  F in R^{n x M}  -- latent factor scores
  U in R^{p x M}  -- sparse loading matrix (K genes per factor)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


# Dose rates matching the real experiment (Gy)
DOSE_RATES = [0.0, 0.004, 0.04, 0.4, 4.0, 8.0]
DOSE_LABELS = ["ctrl", "dA", "dB", "dC", "dD", "dE"]


@dataclass
class SyntheticConfig:
    n_groups: int = 6         # treatment groups (1 control + 5 doses)
    n_weeks: int = 9
    n_per_week: int = 1000       # measurements per week per group
    p: int = 1000             # number of genes
    M: int = 20               # total latent factor[[s
    M_signal: int = 4         # factors with treatment/time signal
    K: int = 20               # genes per factor (sparsity)
    noise_std: float = 0.5    # observation noise
    signal_scale: float = 50.0 # amplitude of treatment effect
    seed: int = 42


def _build_loading_matrix(p: int, M: int, K: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build sparse loading matrix U in R^{p x M}.
    Each factor loads on exactly K genes; gene sets are non-overlapping
    for the first floor(p/K) factors, then randomly assigned for the rest.
    """
    U = np.zeros((p, M))
    # Non-overlapping blocks for first min(M, p//K) factors
    n_blocks = min(M, p // K)
    for m in range(n_blocks):
        start = m * K
        loadings = rng.standard_normal(K)
        # Normalize so each column has unit L2 norm
        loadings /= np.linalg.norm(loadings)
        U[start : start + K, m] = loadings # L1 norm = K (number of nonzero elements)
    # Any remaining factors get random K-sparse support
    for m in range(n_blocks, M):
        genes = rng.choice(p, size=K, replace=False)
        loadings = rng.standard_normal(K)
        loadings /= np.linalg.norm(loadings)
        U[genes, m] = loadings
    return U


def _treatment_time_curve(
    dose: float,
    time: np.ndarray,
    signal_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return a time-dependent effect for one treatment group on one signal factor.
    Each signal factor gets a random dose-specific amplitude and temporal shape.
    """
    # Amplitude proportional to log(dose+1): control gets zero effect
    amplitude = signal_scale * np.log1p(dose) * rng.uniform(0.5, 1.5)
    # Acute early peak (exponential decay from week 1)
    peak = np.exp(-(time - 1) / rng.uniform(1.5, 3.0))
    # Sustained dose-dependent steady state (sigmoid plateau)
    steady = 0.4 * (1.0 / (1.0 + np.exp(-(time - 4))))
    return amplitude * (peak + steady)


def generate(cfg: SyntheticConfig = SyntheticConfig()):
    """
    Generate synthetic longitudinal gene-expression data.

    Returns
    -------
    X : np.ndarray, shape (n, p)
        Normalized gene-expression matrix (log2-fold-change scale).
    meta : pd.DataFrame, shape (n, 4)
        Sample metadata with columns: sample_id, group, dose, week.
    U_true : np.ndarray, shape (p, M)
        Ground-truth sparse loading matrix.
    signal_factors : list[int]
        Indices of the M_signal treatment-responsive factors.
    gene_sets : dict[int, list[int]]
        Mapping from factor index to list of gene indices with nonzero loadings.
    """
    rng = np.random.default_rng(cfg.seed)

    # --- Sample metadata ---
    weeks = np.arange(1, cfg.n_weeks + 1)
    records = []
    for g_idx, (label, dose) in enumerate(zip(DOSE_LABELS, DOSE_RATES)):
        for w in weeks:
            for rep in range(cfg.n_per_week):
                records.append(
                    {
                        "sample_id": f"{label}_w{w:02d}_r{rep}",
                        "group": label,
                        "dose": dose,
                        "week": w,
                        "group_idx": g_idx,
                    }
                )
    meta = pd.DataFrame(records).reset_index(drop=True)
    n = len(meta)  # should be 108

    # --- Loading matrix ---
    U_true = _build_loading_matrix(cfg.p, cfg.M, cfg.K, rng)

    # Gene sets per factor
    gene_sets = {
        m: list(np.nonzero(U_true[:, m])[0]) for m in range(cfg.M)
    }

    # --- Choose which factors carry signal ---
    signal_factors = list(range(cfg.M_signal))

    # --- Latent factor scores F in R^{n x M} ---
    F = rng.standard_normal((n, cfg.M))  # baseline variation for all factors

    # For signal factors, add treatment + time effect
    # Pre-compute random per-factor, per-group curves (use a fresh rng draw per combo)
    for m in signal_factors:
        for g_idx, dose in enumerate(DOSE_RATES):
            mask = meta["group_idx"].values == g_idx
            times = meta.loc[mask, "week"].values.astype(float)
            curve = _treatment_time_curve(dose, times, cfg.signal_scale, rng)
            F[mask, m] += curve

    # --- Observation model ---
    X = F @ U_true.T + rng.normal(scale=cfg.noise_std, size=(n, cfg.p))

    # Drop the helper column before returning
    meta = meta.drop(columns=["group_idx"])

    return X, meta, U_true, signal_factors, gene_sets


def make_response_matrix(meta: pd.DataFrame) -> dict:
    """
    Build response variables Y from metadata.

    Returns a dict with:
      'dose'   : scalar dose vector d, shape (n,)
      'onehot' : one-hot treatment matrix Z, shape (n, 6)
      'time'   : week vector, shape (n,)
      'dose_x_time' : element-wise dose * time, shape (n,)  [interaction term]
    """
    dose = meta["dose"].values
    time = meta["week"].values.astype(float)

    groups = DOSE_LABELS
    onehot = np.zeros((len(meta), len(groups)))
    for j, g in enumerate(groups):
        onehot[:, j] = (meta["group"] == g).astype(float)

    return {
        "dose": dose,
        "onehot": onehot,
        "time": time,
        "dose_x_time": dose * time,
    }


if __name__ == "__main__":
    cfg = SyntheticConfig()
    X, meta, U_true, signal_factors, gene_sets = generate(cfg)
    Y = make_response_matrix(meta)

    print(f"X shape         : {X.shape}  (n={X.shape[0]}, p={X.shape[1]})")
    print(f"n >> p?         : {X.shape[0]} samples vs {X.shape[1]} genes  ->  p >> n: {X.shape[1] > X.shape[0]}")
    print(f"U_true shape    : {U_true.shape}")
    print(f"Signal factors  : {signal_factors}")
    print(f"Genes per factor: {cfg.K}  (non-overlapping for first {cfg.p // cfg.K} factors)")
    print(f"\nSample metadata (first 6 rows):")
    print(meta.head(6).to_string(index=False))
    print(f"\nGroup counts:")
    print(meta.groupby("group")["sample_id"].count().to_string())
    print(f"\ndose response matrix keys: {list(Y.keys())}")
    print(f"  dose shape     : {Y['dose'].shape}")
    print(f"  onehot shape   : {Y['onehot'].shape}")
    print(f"  time shape     : {Y['time'].shape}")

    # Quick sanity check: signal factors should have higher between-group variance
    group_vars = []
    for m in range(cfg.M):
        scores = X @ U_true[:, m]
        group_means = [scores[meta["group"] == g].mean() for g in DOSE_LABELS]
        group_vars.append(np.var(group_means))
    print(f"\nBetween-group variance of factor scores:")
    for m, v in enumerate(group_vars):
        tag = " <- signal" if m in signal_factors else ""
        print(f"  Factor {m:02d}: {v:.4f}{tag}")
