"""Run pipeline.py on real RPE1 ionizing-radiation gene expression data."""

import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
sys.path.insert(0, "..")

from real_data import load_data_lfc, load_data_tpm
from pipeline import PipelineConfig, run_pipeline
from pathway_enrichment import pathway_enrichment
from global_variables import _fs_axis, _fs_title, _DEFAULT_CMAPS


def plot_factor_enrichment(pe_results, output_path="factor_pathway_enrichment.png"):
    """Horizontal bar plot of top-10 pathways per factor (-log10 p-value).

    pe_results : dict[int, pd.DataFrame]  factor index -> pathway_enrichment() output
    """
    factors = [m for m, pe in pe_results.items() if not pe.empty]
    if not factors:
        print("No significant pathways to plot.")
        return

    n = len(factors)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5 * n))
    if n == 1:
        axes = [axes]

    color = plt.get_cmap(_DEFAULT_CMAPS[0])(0.6)
    for ax, m in zip(axes, factors):
        pe = pe_results[m].head(10).copy()
        pe["-log10p"] = -np.log10(pe["p_value"])
        labels = [f"{row['name']}\n{row['native']}" for _, row in pe.iterrows()]
        ax.barh(range(len(pe)), pe["-log10p"].values[::-1], color=color, edgecolor="white")
        ax.set_yticks(range(len(pe)))
        ax.set_yticklabels(labels[::-1], fontsize=_fs_axis - 4)
        ax.set_xlabel(r"$-\log_{10}(p)$", fontsize=_fs_axis)
        ax.set_title(f"Factor {m} — top pathways", fontsize=_fs_title, fontweight="bold")
        ax.tick_params(axis="x", labelsize=_fs_axis - 2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


X, meta, gene_cols = load_data_lfc()
print(f"Loaded X: {X.shape}, groups: {sorted(meta['group'].unique())}, weeks: {sorted(meta['week'].unique())}")

cfg = PipelineConfig(M=20, K=20, alpha=5.0, max_iter=500)
res = run_pipeline(X, meta, cfg)

print("\n--- Stage 2: Treatment Association ---")
print(res["association"].to_string(index=False))

sig_factors = res["association"].loc[res["association"]["significant"], "factor"].tolist()
print(f"\nSignificant factors (FDR<{cfg.fdr}): {sig_factors}")

# Map each significant factor to its top-K genes by absolute loading
factor_genes = {}
if sig_factors:
    print(f"\n--- Top-{cfg.K} genes per significant factor ---")
    U_est = res["U_est"]  # (n_genes, M)
    gene_arr = np.array(gene_cols)
    for m in sig_factors:
        top_idx = np.argpartition(np.abs(U_est[:, m]), -cfg.K)[-cfg.K:]
        top_idx = top_idx[np.argsort(np.abs(U_est[top_idx, m]))[::-1]]
        factor_genes[m] = gene_arr[top_idx].tolist()
        print(f"  Factor {m}: {factor_genes[m]}")

    with open("factor_genes.pkl", "wb") as f:
        pickle.dump(factor_genes, f)
    print(f"\nSaved factor→genes to factor_genes.pkl")

    print(f"\n--- Pathway Enrichment (top 10 per factor) ---")
    pe_results = {}
    _,_, background_genes = load_data_tpm()
    for m, genes in factor_genes.items():
        print(f"\nFactor {m}: {genes}")
        pe = pathway_enrichment(genes, background_genes, pathways=None)
        pe_results[m] = pe
        if pe.empty:
            print("  No significant pathways found.")
        else:
            print(pe[["native", "name", "p_value", "source"]].head(10).to_string(index=False))

    plot_factor_enrichment(pe_results)

if not res["trajectories"].empty:
    print("\n--- Stage 2b: Temporal Slopes (per week) ---")
    pivot = res["trajectories"].pivot(index="group", columns="factor", values="slope")
    print(pivot.to_string(float_format=lambda x: f"{x:+.4f}"))
