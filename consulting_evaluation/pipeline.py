"""
Two-stage pipeline for gene module discovery and treatment association.

Stage 1 — Sparse PCA (sparse_pca):
  Solves: max_U tr(U^T X^T X U)  s.t.  U^T U = I_M,  ||u_m||_0 <= K
  Uses sklearn SparsePCA (L1 penalty). Evaluation compares U_est vs U_true
  via gene-set recovery (Jaccard, Spearman) and subspace alignment.

Stage 2 — Downstream modeling (stage2):
  2a. ANCOVA F-test: which latent factors are treatment-associated?
      Full:    T_m ~ treatment_onehot + time
      Reduced: T_m ~ time
      BH-corrects p-values across all M factors.
  2b. Temporal trajectories: per-group linear slopes in time for
      significant factors.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats
from scipy.linalg import subspace_angles
from scipy.stats import f as f_dist
from scipy.stats import spearmanr
from sklearn.decomposition import SparsePCA


DOSE_LABELS = ["ctrl", "dA", "dB", "dC", "dD", "dE"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    M: int = 20          # number of factors to extract
    K: int = 20          # target sparsity: nonzeros per loading vector
    max_iter: int = 500  # max iterations for SparsePCA
    alpha: float = 1.0   # L1 penalty for SparsePCA
    fdr: float = 0.05    # FDR threshold for treatment association test
    seed: int = 0


# ---------------------------------------------------------------------------
# Stage 1: Sparse PCA
# ---------------------------------------------------------------------------

def fit_sparse_pca(X: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """
    Fit sklearn SparsePCA on centred X.

    Returns U_est : (p, M) loading matrix with unit-norm columns.
    """
    model = SparsePCA(
        n_components=cfg.M,
        alpha=cfg.alpha,
        max_iter=cfg.max_iter,
        random_state=cfg.seed,
        n_jobs=-1,
    )
    model.fit(X)
    U_raw = model.components_.T          # (p, M)
    norms = np.linalg.norm(U_raw, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return U_raw / norms


# ---------------------------------------------------------------------------
# Stage 1 evaluation (requires U_true)
# ---------------------------------------------------------------------------

def _match_factors(U_true: np.ndarray, U_est: np.ndarray):
    """Match each true factor to the most aligned estimated factor by cosine similarity."""
    sim = np.abs(U_true.T @ U_est)
    return sim.argmax(axis=1).tolist()


def gene_set_recovery(
    U_true: np.ndarray,
    U_est: np.ndarray,
    K: int,
    factors,
) -> dict:
    """
    For each factor in `factors`, compute Jaccard and Spearman between
    true and estimated gene sets / loading vectors.
    """
    matches = _match_factors(U_true[:, factors], U_est)
    results = {}

    for i, true_m in enumerate(factors):
        est_m      = matches[i]
        true_genes = set(np.nonzero(U_true[:, true_m])[0])
        est_genes  = set(np.argpartition(np.abs(U_est[:, est_m]), -K)[-K:])

        tp        = len(true_genes & est_genes)
        precision = tp / len(est_genes)  if est_genes  else 0.0
        recall    = tp / len(true_genes) if true_genes else 0.0
        jaccard   = tp / len(true_genes | est_genes) if (true_genes | est_genes) else 0.0
        rho, _    = spearmanr(np.abs(U_true[:, true_m]), np.abs(U_est[:, est_m]))

        results[true_m] = {
            "matched_est_factor": est_m,
            "precision":          precision,
            "recall":             recall,
            "jaccard":            jaccard,
            "spearman_rho":       float(rho),
        }

    summary = {
        "mean_precision":    np.mean([v["precision"]    for v in results.values()]),
        "mean_recall":       np.mean([v["recall"]       for v in results.values()]),
        "mean_jaccard":      np.mean([v["jaccard"]      for v in results.values()]),
        "mean_spearman_rho": np.mean([v["spearman_rho"] for v in results.values()]),
    }
    return {"per_factor": results, "summary": summary}


def subspace_recovery(U_true: np.ndarray, U_est: np.ndarray) -> dict:
    """Principal angles between column spaces of U_true and U_est (degrees)."""
    angles_deg = np.degrees(subspace_angles(U_true, U_est))
    return {
        "principal_angles_deg": angles_deg.tolist(),
        "mean_angle_deg":       float(angles_deg.mean()),
        "max_angle_deg":        float(angles_deg.max()),
    }


# ---------------------------------------------------------------------------
# Stage 2a: Treatment association (ANCOVA F-test + BH correction)
# ---------------------------------------------------------------------------

def _ols_rss(X_design: np.ndarray, y: np.ndarray) -> float:
    """Residual sum of squares for OLS of y on X_design."""
    coef, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    resid = y - X_design @ coef
    return float(resid @ resid)


def _build_designs(meta: pd.DataFrame):
    """
    Full model:    [intercept | 5 treatment dummies | time]
    Reduced model: [intercept | time]
    ctrl is the baseline group (dropped to avoid multicollinearity).
    """
    n         = len(meta)
    intercept = np.ones((n, 1))
    time      = meta["week"].values.astype(float).reshape(-1, 1)
    time      = (time - time.mean()) / time.std()

    dummies = np.zeros((n, 5))
    for j, g in enumerate(DOSE_LABELS[1:]):
        dummies[:, j] = (meta["group"] == g).astype(float)

    full    = np.hstack([intercept, dummies, time])
    reduced = np.hstack([intercept, time])
    return full, reduced


def test_treatment_association(
    T: np.ndarray,
    meta: pd.DataFrame,
    fdr: float = 0.05,
) -> pd.DataFrame:
    """
    F-test per factor: does treatment explain T_m beyond time alone?
    Returns DataFrame sorted by p-value with BH-adjusted significance.
    """
    full, reduced = _build_designs(meta)
    n      = len(meta)
    df_num = full.shape[1] - reduced.shape[1]   # = 5 treatment dummies
    df_den = n - full.shape[1]

    rows = []
    for m in range(T.shape[1]):
        y      = T[:, m]
        f_stat = (((_ols_rss(reduced, y) - _ols_rss(full, y)) / df_num)
                  / (_ols_rss(full, y) / df_den))
        rows.append({
            "factor":  m,
            "f_stat":  f_stat,
            "p_value": float(1 - f_dist.cdf(f_stat, df_num, df_den)),
        })

    df = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)

    m_total      = len(df)
    df["p_adj"]  = (df["p_value"] * m_total / (np.arange(m_total) + 1)).clip(upper=1.0)
    df["p_adj"]  = df["p_adj"][::-1].cummin()[::-1]   # BH monotonicity
    df["significant"] = df["p_adj"] < fdr
    return df


# ---------------------------------------------------------------------------
# Stage 2b: Temporal trajectories
# ---------------------------------------------------------------------------

def fit_trajectories(T: np.ndarray, meta: pd.DataFrame, factors: list) -> pd.DataFrame:
    """
    Per-group linear regression of factor scores on time.
    Returns slope and intercept per (factor, group).
    """
    rows  = []
    weeks = meta["week"].values.astype(float)

    for m in factors:
        scores = T[:, m]
        for g in DOSE_LABELS:
            mask = meta["group"].values == g
            slope, intercept, _, _, _ = stats.linregress(weeks[mask], scores[mask])
            rows.append({"factor": m, "group": g,
                         "intercept": float(intercept), "slope": float(slope)})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detection evaluation (requires ground truth signal_factors)
# ---------------------------------------------------------------------------

def evaluate_detection(assoc_df: pd.DataFrame, signal_factors: list) -> dict:
    """Precision, recall, F1 of detecting true signal factors."""
    detected = set(assoc_df.loc[assoc_df["significant"], "factor"].tolist())
    true_set = set(signal_factors)
    tp = len(detected & true_set)
    fp = len(detected - true_set)
    fn = len(true_set - detected)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {
        "detected_factors": sorted(detected),
        "true_factors":     sorted(true_set),
        "true_positives":   tp,
        "false_positives":  fp,
        "false_negatives":  fn,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    X: np.ndarray,
    meta: pd.DataFrame,
    cfg: PipelineConfig,
    U_true: np.ndarray = None,
    signal_factors: list = None,
) -> dict:
    """
    Run Stage 1 (sparse PCA) then Stage 2 (ANCOVA + trajectories).

    If U_true and signal_factors are provided, also evaluates Stage 1
    gene-set recovery and Stage 2 detection accuracy.
    """
    X_c   = X - X.mean(axis=0)

    # Stage 1
    print("Stage 1: fitting sparse PCA ...")
    U_est = fit_sparse_pca(X_c, cfg)
    T     = X_c @ U_est

    # Stage 2
    print("Stage 2: treatment association...")
    assoc_df    = test_treatment_association(T, meta, fdr=cfg.fdr)
    sig_factors = assoc_df.loc[assoc_df["significant"], "factor"].tolist()
    traj_df     = fit_trajectories(T, meta, sig_factors)

    result = {
        "U_est":        U_est,
        "T":            T,
        "association":  assoc_df,
        "trajectories": traj_df,
    }

    # Optional evaluation against ground truth
    if U_true is not None and signal_factors is not None:
        all_factors  = list(range(U_true.shape[1]))
        result["stage1_eval"] = {
            "gene_set_recovery_signal": gene_set_recovery(U_true, U_est, cfg.K, signal_factors),
            "gene_set_recovery_all":    gene_set_recovery(U_true, U_est, cfg.K, all_factors),
            "subspace_recovery":        subspace_recovery(U_true[:, signal_factors], U_est),
        }
        result["stage2_eval"] = evaluate_detection(assoc_df, signal_factors)

    return result


# ---------------------------------------------------------------------------
# Main: run full pipeline on synthetic data
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from synthetic_data import SyntheticConfig, generate

    for signal_scale in [2.0, 10.0, 100.0]:
        print(f"\n{'#'*60}")
        print(f"signal_scale = {signal_scale}")
        print(f"{'#'*60}")

        cfg_data = SyntheticConfig(signal_scale=signal_scale)
        X, meta, U_true, signal_factors, _ = generate(cfg_data)
        cfg = PipelineConfig(M=cfg_data.M, K=cfg_data.K)

        res = run_pipeline(X, meta, cfg, U_true=U_true, signal_factors=signal_factors)

        # Stage 1 summary
        s1 = res["stage1_eval"]
        s  = s1["gene_set_recovery_signal"]["summary"]
        print(f"\n--- Stage 1: Gene Set Recovery (signal factors) ---")
        print(f"  Jaccard      : {s['mean_jaccard']:.3f}")
        print(f"  Spearman rho : {s['mean_spearman_rho']:.3f}")
        sr = s1["subspace_recovery"]
        print(f"  Mean subspace angle : {sr['mean_angle_deg']:.1f} deg")

        # Stage 2 summary
        d = res["stage2_eval"]
        print(f"\n--- Stage 2: Treatment Detection ---")
        print(f"  True signal factors : {d['true_factors']}")
        print(f"  Detected factors    : {d['detected_factors']}")
        print(f"  Precision : {d['precision']:.3f}  Recall : {d['recall']:.3f}  F1 : {d['f1']:.3f}")

        # Trajectories
        if not res["trajectories"].empty:
            print(f"\n--- Stage 2b: Temporal Slopes (per week) ---")
            pivot = res["trajectories"].pivot(index="group", columns="factor", values="slope")
            print(pivot.to_string(float_format=lambda x: f"{x:+.3f}"))
