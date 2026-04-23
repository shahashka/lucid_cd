"""
Nested CV evaluation of feature selection methods (multitask only).

Outer loop: Leave-one-group-out where each group = one (dose_rate, week) combination.
Inner loop: GridSearchCV for hyperparameter tuning of MultiTaskElasticNet.
  - Unsupervised methods: fixed gene set, tune model hyperparameters.
  - Supervised methods: stability selection across inner folds, then tune.
"""
from sklearn.model_selection import RepeatedKFold, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.base import clone
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from itertools import combinations
from collections import Counter
import argparse
import os
import pickle
import networkx as nx
from joblib import Parallel, delayed

from feature_selection import (
    load_data,
    random as random_features,
    variance_thresholding,
    causal_features,
    ai_features,
    random_forest_features,
    deg_features,
    top_multiple_correlation,
    CAUSAL_DOSE_RATES,
    GRAPHS,
)

SINGLETASK_MODEL = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=200000, tol=1e-3)


def parse_arguments(args=None):
    parser = argparse.ArgumentParser(
        description="Nested CV evaluation of feature selection methods."
    )
    parser.add_argument("--normalize", "-n", default=False, action="store_true")
    parser.add_argument("--prune_log2fold", "-p", default=False, action="store_true")
    parser.add_argument(
        "--short_version", "-sv", default=False, action="store_true",
        help="Skip supervised feature selection methods",
    )
    parser.add_argument("--output_dir", "-o", default="./nested_cv_results")
    parser.add_argument("--n_jobs", "-j", type=int, default=1,
                        help="Number of parallel jobs for feature set evaluation")
    parser.add_argument("--replot", action="store_true",
                        help="Skip CV; regenerate plots from precomputed CSV")
    return parser.parse_args(args)


def build_feature_sets(X, y_dr_reg, y_w):
    """Build unsupervised / external feature sets."""
    feature_sets = {}
    feature_sets["all_genes"] = X.columns.tolist()
    feature_sets["random_10"] = random_features(X, size=10)
    feature_sets["random_100"] = random_features(X, size=100)
    feature_sets["random"] = random_features(X)
    feature_sets["random_5000"] = random_features(X, size=5000)
    feature_sets["variance"] = variance_thresholding(X)
    
    # Causal features per dose rate
    for dr in CAUSAL_DOSE_RATES:
        feature_sets[f"causal_{dr}"] = causal_features(dr, tf=False)
        feature_sets[f"causal_tf_{dr}"] = causal_features(dr, tf=True)
    feature_sets["causal_dose_rate"] = causal_features("all_doses_dose_rate", tf=False)
    feature_sets["causal_week"] = causal_features("all_doses_week", tf=False)

    # Full causal graph gene sets
    for name, gexf_path in GRAPHS.items():
        G = nx.read_gexf(gexf_path)
        feature_sets[f"causal_full_{name}"] = [n for n in G.nodes() if n not in {"radiation", "dose_rate", "week"}]

    # AI-derived features
    for name in ["kosmos", "chatgpt"]:
        feature_sets[f"ai_{name}"] = ai_features(name)

    # Differentially expressed genes per dose rate
    deg_by_dose = deg_features()
    for dr, genes in deg_by_dose.items():
        feature_sets[f"deg_{dr}"] = genes

    # Invariant subgraph genes (shared across all dose-rate causal graphs)
    invariant_path = os.path.join("structure_analysis", "invariant_genes.pkl")
    if os.path.exists(invariant_path):
        with open(invariant_path, "rb") as f:
            feature_sets["invariant_subgraph_genes"] = pickle.load(f)
    
    # House keeping genes (hub nodes/sinks in invariant subgraph)
    feature_sets['housekeeping_genes'] = housekeeping_invariant_features()

    # Perfect bootstrap edge genes (100% bootstrap frequency, per dose)
    for dr in CAUSAL_DOSE_RATES:
        perfect_path = os.path.join("structure_analysis", f"perfect_genes_{dr}.pkl")
        if os.path.exists(perfect_path):
            with open(perfect_path, "rb") as f:
                feature_sets[f"perfect_edge_genes_{dr}"] = pickle.load(f)

    return feature_sets


def housekeeping_invariant_features():
    """Return invariant genes that are also housekeeping genes."""
    import json
    with open("structure_analysis/invariant_genes.pkl", "rb") as f:
        invariant_genes = set(pickle.load(f))
    hk_path = "data/prior_knowledge/HSIAO_HOUSEKEEPING_GENES.v2026.1.Hs.json"
    with open(hk_path, "r") as f:
        hk_data = json.load(f)
    hk_genes = set(hk_data["HSIAO_HOUSEKEEPING_GENES"]["geneSymbols"])
    return sorted(invariant_genes & hk_genes)

def get_param_grid():
    """Return hyperparameter grid for MultiTaskElasticNet GridSearchCV."""
    return {"alpha": [0.01, 0.1, 1], "l1_ratio": [0.1, 0.5, 0.9]}


def stability_select(X_train, y_train, selector_fn, n_splits=4, threshold=0.75):
    """
    Run selector_fn on each inner fold, return genes selected in
    >= threshold fraction of folds.
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    gene_counts = Counter()
    for train_idx, _ in kfold.split(X_train):
        X_inner = X_train.iloc[train_idx]
        y_inner = y_train[train_idx]
        selected = selector_fn(X_inner, y_inner)
        gene_counts.update(selected)

    min_count = int(np.ceil(threshold * n_splits))
    stable_genes = [g for g, c in gene_counts.items() if c >= min_count]
    # Filter to genes actually in X_train
    stable_genes = [g for g in stable_genes if g in X_train.columns]
    return stable_genes


def run_supervised_feature_selection(X, y_combined, supervised_selectors, args):
    """Run RepeatedKFold to learn stable gene sets from supervised selectors.

    For each selector, runs stability_select in each outer fold, then aggregates
    genes that appear in >=75% of folds.

    Returns:
        dict mapping selector name -> list of stable genes
    """
    # Compute cumulative dose target
    #y_cumulative_dose = y_combined[:, 0] * y_combined[:, 1] * 168

    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    n_folds = rkf.get_n_splits(X)
    print(f"\n=== Supervised feature selection: {n_folds} folds (5-fold x 5 repeats) ===")

    # Collect genes per selector per fold
    fold_genes = {name: [] for name in supervised_selectors}

    for fold_i, (train_idx, _) in enumerate(rkf.split(X)):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y_combined[train_idx]

        if args.normalize:
            scaler = StandardScaler()
            X_train = pd.DataFrame(
                scaler.fit_transform(X_train), columns=X_train.columns,
                index=X_train.index,
            )

        for name, selector_fn in supervised_selectors.items():
            genes = stability_select(X_train, y_train, selector_fn)
            fold_genes[name].append(genes)
            if fold_i % 5 == 0:
                print(f"  Fold {fold_i} — {name}: {len(genes)} stable genes")

    # Aggregate across folds: keep genes in >=75% of folds
    out_dir = args.output_dir
    result = {}
    for name, gene_lists in fold_genes.items():
        gene_counts = Counter(g for gl in gene_lists for g in gl)
        n = len(gene_lists)
        gene_stability = sorted(gene_counts.items(), key=lambda x: -x[1])
        stable_df = pd.DataFrame(gene_stability, columns=["gene", "fold_count"])
        stable_df["fold_fraction"] = stable_df["fold_count"] / n

        stable_path = f"{out_dir}/stable_features_{name}.csv"
        stable_df.to_csv(stable_path, index=False)

        stable_genes = stable_df.loc[
            stable_df["fold_fraction"] >= 0.50, "gene"
        ].tolist()
        result[name] = stable_genes

        print(f"\n{name}: {len(gene_counts)} unique genes across {n} folds, "
              f"{len(stable_genes)} stable (>=50% folds)")
        print(f"  Saved to {stable_path}")

    return result


def evaluate_phenotype_unsupervised(X_train, phenotypes_train, X_test,
                                     phenotypes_test, genes, normalize):
    """Per-phenotype ElasticNet: fit on train, score on test.
    Returns dict with per-phenotype R², combined (mean) R², and n_genes,
    or None if no genes match.
    """
    genes_filtered = [g for g in genes if g in X_train.columns]
    if not genes_filtered:
        return None
    X_tr = X_train[genes_filtered].copy()
    X_te = X_test[genes_filtered].copy()

    if normalize:
        scaler = StandardScaler()
        X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns, index=X_tr.index)
        X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns, index=X_te.index)

    param_grid = get_param_grid()
    phenotype_cols = list(phenotypes_train.columns)
    scores = {}
    for pc in phenotype_cols:
        y_tr = phenotypes_train[pc].values
        y_te = phenotypes_test[pc].values
        cv = KFold(n_splits=min(4, X_tr.shape[0]), shuffle=True, random_state=42)
        gs = GridSearchCV(clone(SINGLETASK_MODEL), param_grid, cv=cv, scoring="r2")
        gs.fit(X_tr, y_tr)
        y_pred = gs.best_estimator_.predict(X_te)
        scores[pc] = r2_score(y_te, y_pred)

    scores["combined"] = float(np.mean([scores[pc] for pc in phenotype_cols]))
    scores["n_genes"] = len(genes_filtered)
    return scores


def plot_phenotype_results(df, output_dir, prefix="nested_cv_phenotype"):
    """One bar plot per phenotype target showing R² by feature set."""
    phenotype_labels = [l for l in df["label"].unique() if l != "combined"]

    for phenotype in phenotype_labels:
        sub = df[df["label"] == phenotype].copy()
        sub = sub.sort_values("mean", ascending=False)

        n_genes_map = sub.set_index("feature_set")["n_genes"]
        labels = [f"{fs} ({int(n_genes_map[fs])})" for fs in sub["feature_set"]]

        fig, ax = plt.subplots(figsize=(15, 6))
        x = np.arange(len(sub))
        ax.bar(x, sub["mean"].values, color="tab:blue", alpha=0.7,
               yerr=sub["std"].values, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("R²")
        ax.set_title(f"{prefix} — {phenotype} R² by feature set (mean ± std)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefix}_{phenotype}.png", bbox_inches="tight")
        plt.close()


def plot_nested_cv_results(df, output_dir, prefix="nested_cv"):
    """Bar plots and Pareto front from nested CV results.
    df has columns: label, feature_set, mean, std, n_genes.
    """
    pivot_mean = df.pivot(index="feature_set", columns="label", values="mean")
    pivot_std = df.pivot(index="feature_set", columns="label", values="std")

    # Sort by combined R² if available
    if "combined" in pivot_mean.columns:
        pivot_mean = pivot_mean.sort_values("combined", ascending=False)
        pivot_std = pivot_std.loc[pivot_mean.index]
    elif {"week", "dose_rate"}.issubset(pivot_mean.columns):
        dist = np.sqrt(pivot_mean["week"] ** 2 + pivot_mean["dose_rate"] ** 2)
        pivot_mean = pivot_mean.loc[dist.sort_values(ascending=False).index]
        pivot_std = pivot_std.loc[pivot_mean.index]

    # Add n_genes to feature set labels
    n_genes_map = df[df.label == df.label.iloc[0]].set_index("feature_set")["n_genes"]
    new_index = [f"{fs} ({int(n_genes_map[fs])})" for fs in pivot_mean.index]
    pivot_mean.index = new_index
    pivot_std.index = new_index

    # --- Bar plot: per-target R² side by side ---
    target_cols = [c for c in pivot_mean.columns if c != "combined"]
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(pivot_mean))
    width = 0.8 / max(len(target_cols), 1)
    for i, label in enumerate(target_cols):
        ax.bar(x + i * width, pivot_mean[label], width,
               yerr=pivot_std[label], capsize=3, label=label)
    ax.set_xticks(x + width * len(target_cols) / 2)
    ax.set_xticklabels(pivot_mean.index, rotation=45, ha="right")
    ax.set_ylabel("R²")
    ax.set_title(f"{prefix} — per-target R² by feature set (mean ± std)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{prefix}_barplot.png")
    plt.close()

    # --- Combined R² bar plot ---
    if "combined" in pivot_mean.columns:
        sort_idx = pivot_mean["combined"].sort_values(ascending=False).index

        fig, ax = plt.subplots(figsize=(15, 6))
        x = np.arange(len(sort_idx))
        ax.bar(x, pivot_mean.loc[sort_idx, "combined"], color="tab:green", alpha=0.7,
               yerr=pivot_std.loc[sort_idx, "combined"], capsize=3)
        all_genes_key = [k for k in sort_idx if k.startswith("all_genes")]
        if all_genes_key:
            baseline = pivot_mean.loc[all_genes_key[0], "combined"]
            ax.axhline(baseline, linestyle="--", color="gray", label="all_genes baseline")
            ax.legend()
        ax.set_xticks(x)
        ax.set_xticklabels(sort_idx, rotation=45, ha="right")
        ax.set_ylabel("Combined R²")
        ax.set_title(f"{prefix} — combined R² by feature set (mean ± std)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefix}_combined_r2_barplot.png")
        plt.close()

    # --- Pareto front scatter ---
    if {"week", "dose_rate"}.issubset(pivot_mean.columns):
        # Compute Pareto ranks (maximising both objectives)
        week_vals = pivot_mean["week"].values
        dr_vals = pivot_mean["dose_rate"].values
        ranks = np.zeros(len(pivot_mean), dtype=int)
        remaining = set(range(len(pivot_mean)))
        rank = 0
        while remaining:
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i != j and week_vals[j] >= week_vals[i] and dr_vals[j] >= dr_vals[i] \
                            and (week_vals[j] > week_vals[i] or dr_vals[j] > dr_vals[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            for i in front:
                ranks[i] = rank
                remaining.discard(i)
            rank += 1
        max_rank = ranks.max()
        alphas = [1.0 - 0.7 * (ranks[i] / max(max_rank, 1)) for i in range(len(ranks))]

        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = plt.get_cmap("viridis")
        n_points = len(pivot_mean)
        colors = [cmap(i / max(n_points - 1, 1)) for i in range(n_points)]
        markers = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

        for idx, (feat, row) in enumerate(pivot_mean.iterrows()):
            ax.scatter(
                row["week"], row["dose_rate"],
                marker=markers[idx % len(markers)],
                color=colors[idx], s=36, zorder=3,
                alpha=alphas[idx],
                label=feat,
            )
            ellipse = Ellipse(
                (row["week"], row["dose_rate"]),
                width=2 * pivot_std.loc[feat, "week"],
                height=2 * pivot_std.loc[feat, "dose_rate"],
                facecolor=colors[idx], alpha=0.2 * alphas[idx],
                edgecolor=colors[idx], linewidth=1,
            )
            ax.add_patch(ellipse)

        ax.set_xlabel("Week R²")
        ax.set_ylabel("Dose rate R²")
        ax.set_title(f"{prefix} — Pareto front")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{prefix}_pareto.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    args = parse_arguments()
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.replot:
        # csv_path = f"{out_dir}/repeated_kfold_results.csv"
        # df = pd.read_csv(csv_path)
        # print(f"Loaded {len(df)} rows from {csv_path}")
        # plot_nested_cv_results(df, out_dir, prefix="repeated_kfold")
        csv_path = f"{out_dir}/repeated_kfold_phenotype_results.csv"
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        plot_nested_cv_results(df, out_dir, prefix="repeated_kfold_phenotype")
        plot_phenotype_results(df, out_dir, prefix="repeated_kfold_phenotype")
        print(f"Plots saved to {out_dir}/")
        raise SystemExit(0)

    X, _, y_dr_reg, y_w, phenotypes = load_data(args)
    y_dr_reg_flat = np.ravel(y_dr_reg)
    y_w_flat = np.ravel(y_w)

    # Build unsupervised feature sets
    feature_sets = build_feature_sets(X, y_dr_reg_flat, y_w_flat)
    print(f"Unsupervised feature sets: {len(feature_sets)}")

    # --- Phase 1: Supervised feature selection (learn stable genes, save to CSV) ---
    if not args.short_version:
        supervised_selectors = {
            "rf": lambda X_t, y_t: random_forest_features(X_t, y_t),
            "multiple_correlation_joint": lambda X_t, y_t: top_multiple_correlation(X_t, y_t)
        }
        y_combined = np.column_stack([y_dr_reg_flat, y_w_flat])
        supervised_gene_sets = run_supervised_feature_selection(
            X, y_combined, supervised_selectors, args,
        )
        # Add supervised gene lists as fixed feature sets
        for name, genes in supervised_gene_sets.items():
            if genes:
                feature_sets[name] = genes
                print(f"Added supervised feature set '{name}': {len(genes)} genes")

    # --- Phase 2: Phenotype evaluation (all feature sets treated uniformly) ---
    phenotype_cols = list(phenotypes.columns)
    target_names = phenotype_cols
    all_feat_names = list(feature_sets.keys())

    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    n_outer = rkf.get_n_splits(X)
    print(f"\n=== Phenotype evaluation: {n_outer} folds (5-fold x 5 repeats) ===")

    fold_scores = {fn: [] for fn in all_feat_names}

    for fold_i, (train_idx, test_idx) in enumerate(rkf.split(X)):
        print(f"\n--- Fold {fold_i} (train={len(train_idx)}, test={len(test_idx)}) ---")

        X_train = X.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        phenotypes_train = phenotypes.iloc[train_idx].reset_index(drop=True)
        phenotypes_test = phenotypes.iloc[test_idx].reset_index(drop=True)

        results = Parallel(n_jobs=args.n_jobs)(
            delayed(evaluate_phenotype_unsupervised)(
                X_train, phenotypes_train, X_test, phenotypes_test,
                genes, args.normalize,
            )
            for genes in feature_sets.values()
        )
        for feat_name, scores in zip(feature_sets.keys(), results):
            if scores is not None:
                fold_scores[feat_name].append(scores)
                print(f"  {feat_name}: "
                      + " ".join(f"{t}={s:.3f}" for t, s in scores.items() if t not in ("n_genes",))
                      + f" (n={scores['n_genes']})")

    # Aggregate across folds: mean +/- std
    rows = []
    for feat_name in all_feat_names:
        scores = fold_scores[feat_name]
        if not scores:
            continue
        n_genes_median = int(np.median([s["n_genes"] for s in scores]))
        for target in target_names + ["combined"]:
            vals = [s[target] for s in scores]
            rows.append({
                "label": target,
                "feature_set": feat_name,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "n_genes": n_genes_median,
            })
        print(f"\n{feat_name} (n={n_genes_median}): "
              + " ".join(f"{t}={np.mean([s[t] for s in scores]):.3f}±{np.std([s[t] for s in scores]):.3f}"
                         for t in target_names + ["combined"]))

    # Save and plot
    df = pd.DataFrame(rows)
    csv_path = f"{out_dir}/repeated_kfold_phenotype_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(df)} rows to {csv_path}")

    if not df.empty:
        plot_nested_cv_results(df, out_dir, prefix="repeated_kfold_phenotype")
        plot_phenotype_results(df, out_dir, prefix="repeated_kfold_phenotype")
        print(f"Plots saved to {out_dir}/")
