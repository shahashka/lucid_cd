# To quantify separability, train classifiers to predict dose rate 
# and week using different feature selections
from cProfile import label
from tkinter import YES
from typing import Any
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso, ElasticNet
from sklearn.feature_selection import VarianceThreshold, RFECV, RFE
from sklearn.base import clone

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from itertools import combinations
import networkx as nx
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.linear_model import MultiTaskElasticNet

import argparse
import sys

def parse_arguments(args=None):
    """
    Creates the argument parser and defines command-line arguments.
    """
    parser = argparse.ArgumentParser(description="A simple command-line utility.")
    parser.add_argument("--only_landscape", "-ol", default=False, action="store_true", help="Only run the causal landscape code")
    parser.add_argument("--normalize", "-n", default=False, action="store_true", help="Normalize the log2fold dataset")
    parser.add_argument("--prune_log2fold", "-p", default=False, action="store_true", help="Prune the log2fold dataset")
    parser.add_argument("--run_increasing_genes", "-rig", default=False, action="store_true", help="Run run_increasing_num_genes and save accuracies to CSV")
    parser.add_argument("--short_version", "-sv", default=False, action="store_true", help="Skip the recursive methods to save time")

    return parser.parse_args(args) # If args is None, it uses sys.argv

with open("/homes/shahashka/lucid_cd/data/gene_groups.pkl", "rb") as f:
    CAUSAL_TFS, CAUSAL_NEIGHBORHOODS, KOSMOS, CHATGPT, BNL = pickle.load(f)
MODELS = {
    "logistic_regression": LogisticRegression(penalty="l1", solver="saga", max_iter=30000),
    "svc_rbf": SVC(kernel="rbf", max_iter=30000),
    "svc_linear": LinearSVC(penalty="l1", max_iter=30000),
    "elastic_multitask": MultiTaskElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=30000)
}
PHENOTYPE_MODEL = { 'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=30000), "elastic_multitask" : MultiTaskElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=30000)}
CAUSAL_DOSE_RATES = ["F", "G", "H", "I", "J"]
DOSE_RATES = {"control": 0.0, "F": 0.004, "G": 0.04, "H": 0.4, "I": 4.0, "J": 8.0}
DOSE_RATES_ACTUAL = {"control": 0.0, "F": 0.4, "G": 0.4, "H": 0.4, "I": 4.0, "J": 8.0}

GRAPHS = {"invariant": "/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs3/dag_gnn_combined.gexf",
          "F":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseF.gexf",
          "G":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseG.gexf",
          "H":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseH.gexf",
          "I":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseI.gexf",
          "J":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseJ.gexf" 
          }
TOP_PERFORMERS = ["ai_kosmos", "causal_H", "causal_I", "causal_intersections_H_I"]

# Helper to get children / successors in a graph-agnostic way
def _children(g, node):
    if hasattr(g, "successors"):
        return list(g.successors(node))
    if hasattr(g, "neighbors"):
        return list(g.neighbors(node))
    return []

# Helper to get k-hop descendants
def _k_hop_neighbors(g, start, k):
    g = g.to_undirected()
    visited = {start}
    current = {start}
    for _ in range(k):
        nxt = set()
        for n in current:
            for child in _children(g, n):
                if child not in visited:
                    nxt.add(child)
        visited |= nxt
        current = nxt
    return current - {start}

def create_k_hop_neighbors(feature_selection_method):
    doses = feature_selection_method.split("_")[1:]
    graphs = [nx.read_gexf(GRAPHS[dose]) for dose in doses]
     # Get all k-hop neighbors of the 'radiation' node across graphs
    candidates = []
    max_k=20
    for k in range(max_k): 
        k_candidates = set()   # store all possible candidates for this hop
        for g in graphs:
            # Get k hop neighbors of the 'radiation' node
            # Add neighbors to candidates 
            neighbors = _k_hop_neighbors(g, 'radiation', k)
            if len(k_candidates) == 0:
                k_candidates = neighbors
            else:
                k_candidates = set.intersection(k_candidates, neighbors)
        hop_list = list(k_candidates)
        np.random.shuffle(hop_list)
        candidates.append(hop_list)
    return candidates

# Sample a set of genes from the data based on the feature selection method
def sample_features(num_genes, feature_selection_method, X, y, y_name, cached_data=dict()):
    rng = np.random.default_rng(42)
    if feature_selection_method == "random":
        selected = random(X, size=min(num_genes, X.shape[1]))
    elif feature_selection_method == "variance":
        selected = variance_thresholding(X)
        n = min(num_genes, len(selected))
        selected = list(rng.choice(selected, size=n, replace=False))
    elif feature_selection_method == "recursive":
        if y is None:
            raise ValueError(f"{feature_selection_method} requires y for selection")
        if (feature_selection_method, y_name) in cached_data:
            ranking = cached_data[(feature_selection_method, y_name)]
        else:
            ranking = recursive_feature_ranking(X, y, MODELS["svc_linear"]) 
            cached_data[(feature_selection_method, y_name)] = ranking
        selected = ranking[:num_genes]
    elif feature_selection_method == "sparse":
        if y is None:
            raise ValueError(f"{feature_selection_method} requires y for selection")
        selected = sparse_features(X, y, n_features=num_genes)
    elif feature_selection_method == "ai_kosmos":
        selected = ai_features("kosmos")
        if len(selected) > num_genes:
            selected = list(rng.choice(selected, size=num_genes, replace=False))
    elif feature_selection_method == "ai_chatgpt":
        selected = ai_features("chatgpt")
        if len(selected) > num_genes:
            selected = list(rng.choice(selected, size=num_genes, replace=False))
    elif feature_selection_method.startswith("causal"):
        if (feature_selection_method, y_name) in cached_data:
            candidates = cached_data[(feature_selection_method, y_name)]
        else:
            candidates = create_k_hop_neighbors(feature_selection_method)
            cached_data[(feature_selection_method, y_name)] = candidates
        # Add genes to selected until we have num_genes so that low hop genes are selected first
        selected = set()
        for hop in candidates:
            for gene in hop:
                selected.add(gene)
                if len(selected) >= num_genes:
                    break
            if len(selected) >= num_genes:
                break
        selected = list(selected)
    else:
        raise ValueError(f"Unknown feature selection method: {feature_selection_method}")
    return selected, cached_data

def model_fit(X,y, model, holdout, genes=None):
    X_eval, y_eval = holdout
    if genes is not None:
        genes_filtered = list(set(genes).intersection(set(X.columns)))
        X = X[genes_filtered]
        X_eval = X_eval[genes_filtered]
    n_splits = 4
    if model==MODELS["elastic_multitask"]: # only model that requires one hot y's
        # One-hot encode targets so each class is a separate task
        enc = OneHotEncoder()
        y = enc.fit_transform(y.reshape(-1, 1)).toarray()
        y_eval = enc.transform(y_eval.reshape(-1, 1)).toarray()

    if model == PHENOTYPE_MODEL['elastic']:
        kfold = KFold(n_splits=4, random_state=42, shuffle=True)
    else:
        kfold = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    scores = []

    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        X_train = X.loc[train_index]
        y_train = y[train_index]
        X_test = X.loc[test_index]
        y_test = y[test_index]
        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))
    mean, std = np.mean(scores), np.std(scores)
    score = model.score(X_eval, y_eval)
    return mean, std, score

def load_data(args):
    log2fold_df = pd.read_csv(f"/homes/shahashka/lucid_cd/data/rpe1_experiment2/rpe1_9week_study_experiment2_diffexp_deseq_vs_control_all_dG_W2_adjust.txt", sep="\t")
    if args.prune_log2fold:
        print("Pruning log2fold data")
        log2fold_df = log2fold_df.loc[log2fold_df['padj'] < 0.05] # This is new, I think I should filter by p value. However this means there are no genes that DE across all dose rates
        log2fold_df = log2fold_df.loc[abs(log2fold_df['log2FoldChange']) > 1]
    log2fold_df = log2fold_df.groupby(["Dose", "Week", "Gene"]).mean(numeric_only=True)
    log2fold_df = log2fold_df["log2FoldChange"].unstack(level='Gene')
    log2fold_df = log2fold_df.reset_index()
    log2fold_df = log2fold_df.rename(columns={"Dose":"dose_rate", "Week": "week"})
    
    log2fold_df['dose_rate'] = [DOSE_RATES_ACTUAL[d[1]] for d in log2fold_df['dose_rate']]
    log2fold_df['week'] = [float(w[1]) for w in log2fold_df['week']]
    if args.prune_log2fold:
        log2fold_df_na = log2fold_df.fillna(0) # if we drop na after pruning, we are left with no genes that overlap conditions
    else:
        log2fold_df_na = log2fold_df.dropna(axis=1)
    
    labels_dr = OrdinalEncoder().fit_transform(log2fold_df_na[['dose_rate']])
    labels_w = log2fold_df_na['week']
    
    X = log2fold_df_na.drop(columns=["dose_rate", "week"])
    
    # ============================================================
    # A) Build Y: morphology diff-from-control (indexed by dose_label, week_num)
    # ============================================================
    DATA_SUMMARY=f"/homes/shahashka/lucid_cd/data/rpe1_experiment2/cell_painting_summary_statistics.csv"
    summary_df = pd.read_csv(DATA_SUMMARY)
    summary_df["radiation_label"] = summary_df["radiation_label"].astype(str)

    colmap = {
        "area": "area_mean",
        "perimeter": "perimeter_mean",
        "mean_intensity": "mean_intensity_mean",
        "eccentricity": "eccentricity_mean",
        "solidity": "solidity_mean",
        "glcm_contrast": "glcm_contrast_mean",
        "glcm_correlation": "glcm_correlation_mean",
        "glcm_energy": "glcm_energy_mean",
        "glcm_homogeneity": "glcm_homogeneity_mean",
    }
    Y_cols = [f"{k}_diff" for k in colmap.keys()]

    control_df = summary_df[summary_df["radiation_label"].str.lower().eq("control")]
    ctrl_pw = control_df.set_index("week_num")[list(colmap.values())].sort_index()
    dose_keep = ["dF", "dG", "dH", "dI", "dJ"]
    treated_df = summary_df[summary_df["radiation_label"].isin(dose_keep)].copy()

    def row_diff(row):
        wk = int(row["week_num"])
        c = ctrl_pw.loc[wk]
        if isinstance(c, pd.DataFrame):
            c = c.iloc[0]
        return {f"{k}_diff": float(row[v]) - float(c[v]) for k,v in colmap.items()}

    morph = pd.concat(
        [
            treated_df[["radiation_label","week_num"]].rename(columns={"radiation_label":"dose_label"}),
            treated_df.apply(row_diff, axis=1).apply(pd.Series),
        ],
        axis=1,
    )
    morph["dose_label"] = morph["dose_label"].astype(str)
    morph.set_index(["dose_label","week_num"], inplace=True)
    phenotypes = morph[Y_cols].sort_index()

    phenotypes = phenotypes.reset_index()
    phenotypes = phenotypes.drop(columns=["week_num", "dose_label"])
    return X, labels_dr, labels_w, phenotypes

def random(X, size=1000):
    return np.random.choice(X.columns, size=size, replace=False).tolist()

def _remove_labels(X):
    if "radiation" in X.columns:
        X = X.drop(columns=["radiation"])
    if "dose_rate" in X.columns:
        X = X.drop(columns=["dose_rate"])
    if "week" in X.columns:
        X = X.drop(columns=["week"])
    return X

def variance_thresholding(X, threshold=0.05):
    X = _remove_labels(X)
    sel = VarianceThreshold(threshold=threshold)
    sel.fit(X)
    return X.columns[sel.get_support()].tolist()
    
def causal_features(dose_rate, tf=False):
    if tf:
        genes = CAUSAL_TFS[dose_rate]
    else:
        genes = CAUSAL_NEIGHBORHOODS[dose_rate]

    return genes
    
def causal_features_intersections(dose_rate_set, tf=False):
    genes = [set(causal_features(dose_rate, tf)) for dose_rate in dose_rate_set]
    genes = set.intersection(*genes)
    return genes

def ai_features(name):
    if name == "kosmos":
        return KOSMOS
    if name == "chatgpt":
        return CHATGPT

def recursive_features(X, y, model, min_features=1000):
    X = _remove_labels(X)
    selector = RFECV(model, step=100, cv=4, min_features_to_select=min_features)
    print(X.shape, y.shape)
    selector = selector.fit(X, y)
    genes = X.columns
    return genes[selector.support_]


def recursive_feature_ranking(X, y, model):
    """Return list of gene names in selection order (first = most important)."""
    X = _remove_labels(X)
    selector = RFE(clone(model), n_features_to_select=1, step=50)
    selector = selector.fit(X, y)
    # ranking_: 1 = selected first, higher = eliminated later
    order = np.argsort(selector.ranking_)
    return X.columns[order].tolist()

def sparse_features(X, y, n_features=1000):
    """
    Fit a Lasso model and return the names of the top `n_features`
    genes by absolute coefficient magnitude.
    """
    #model = Lasso(max_iter=10000)
    model=ElasticNet()
    X = _remove_labels(X)
    model.fit(X, y)

    coefs = np.abs(model.coef_)
    # Indices sorted from largest to smallest coefficient magnitude
    sorted_idx = np.argsort(coefs)[::-1]

    # Keep only non-zero coefficients
    sorted_idx = [i for i in sorted_idx if coefs[i] > 0]

    if not sorted_idx:
        return []

    k = min(n_features, len(sorted_idx))
    top_idx = sorted_idx[:k]
    return X.columns[top_idx].tolist()

# def compute_row(args):
#     node, scope_name, genes, label_name, y, base_model, X = args
#     model = base_model
#     mean, std = model_fit(X, y, model, genes)
#     # mean=0
#     # std = 0
#     if scope_name.endswith('hop'):
#         rows = []
#         for neighbor_node in genes:
#             # print(neighbor_node, len(genes))
#             rows.append(
#             {
#                 "node": neighbor_node,
#                 "scope": scope_name,  # self / children / k_hop
#                 "k": int(scope_name.split("_")[0]) if "hop" in scope_name else 0,
#                 "label": label_name,  # week / dose_rate
#                 "n_genes": len(genes),
#                 "mean": float(mean),
#                 "std": float(std),
#             } )
#         return rows
#     else:
#         return [{
#             "node": node,
#             "scope": scope_name,  # self / children / k_hop
#             "k": int(scope_name.split("_")[0]) if "hop" in scope_name else 0,
#             "label": label_name,  # week / dose_rate
#             "n_genes": len(genes),
#             "mean": float(mean),
#             "std": float(std),
#         }]
                        
# def analyze_causal_landscape(graph, name,args, max_hop=3, model_name="svc_linear"):
#     """
#     Explore how predictive power changes as we move outwards in the causal graph.
#     For each node, we:
#       1) Fit classifiers using only that node as a feature.
#       2) Fit classifiers using its direct children as features.
#       3) Fit classifiers using k-hop descendants for k = 1..max_hop.

#     For each of the above, we evaluate both week and dose-rate prediction.
#     Results are returned as a DataFrame and also written to
#     'causal_landscape_scores.csv'.
#     """

#     t_start = time.perf_counter()
#     print(f"[analyze_causal_landscape] Start for '{name}' with model '{model_name}'")

#     # Load expression data and labels
#     X, y_dr, y_w, _ = load_data(args)
#     if args.normalize:
#         print("Normalizing data, causal landscape.")
#         X_norm = StandardScaler().fit_transform(X) # Normalize
#         X = pd.DataFrame(data=X_norm, columns=X.columns)
        
#     print(
#         f"[analyze_causal_landscape] Data loaded for '{name}' "
#         f"(X shape={X.shape}) at {time.perf_counter() - t_start:.1f}s"
#     )
#     y_dr_flat = np.ravel(y_dr)
#     y_w_flat = np.ravel(y_w)
#     labels = {"dose_rate": y_dr_flat, "week": y_w_flat}

#     if model_name not in MODELS:
#         raise ValueError(f"Unknown model_name '{model_name}'. Available: {list(MODELS.keys())}")
#     base_model = MODELS[model_name]

#     valid_genes = set(X.columns)
#     rows = []

#     nodes = list(graph.nodes())
#     total_nodes = len(nodes)
#     print(
#         f"[analyze_causal_landscape] Beginning per-node analysis for '{name}' "
#         f"over {total_nodes} nodes at {time.perf_counter() - t_start:.1f}s"
#     )

#     for i, node in enumerate(nodes, start=1):
#         if (node not in valid_genes):
#             if (node not in ['radiation', 'dose_rate', 'week']):
#                 continue

#         if i == 1 or i % 10 == 0 or i == total_nodes:
#             print(
#                 f"[analyze_causal_landscape] Processed {i}/{total_nodes} nodes "
#                 f"for '{name}' at {time.perf_counter() - t_start:.1f}s"
#             )

#         # 1) Self only
#         if node not in ["radiation", "dose_rate", "week"]:
#             feature_sets = {
#                 "self": [node],
#             }

#         # 2) Direct children
#         children = [c for c in _children(graph, node) if c in valid_genes]
#         if children:
#             feature_sets["children"] = children

#         # 3) k-hop descendants (k = 1..max_hop), only for key driver nodes
#         if node in ["radiation", "dose_rate", "week"]:
#             # print('test1')
#             for k in range(1, max_hop + 1):
#                 k_hop = [g for g in _k_hop_neighbors(graph, node, k) if g in valid_genes]
#                 if k_hop:
#                     # print('test2', len(k_hop))
#                     feature_sets[f"{k}_hop"] = k_hop

#         tasks = [
#             (node, scope_name, genes, label_name, y, base_model, X)
#             for scope_name, genes in feature_sets.items()
#             for label_name, y in labels.items()
#         ]
        
#         # parallel execution
#         with ProcessPoolExecutor() as executor:
#             # submit all tasks
#             futures = [executor.submit(compute_row, t) for t in tasks]
#             for f in as_completed(futures):
#                 rows.append(f.result())

#     rows = [r for row in rows for r in row] # flatten
#     df = pd.DataFrame(rows)
#     print(
#         f"[analyze_causal_landscape] Finished model evaluations for '{name}' "
#         f"at {time.perf_counter() - t_start:.1f}s (rows={len(df)})"
#     )
#     df.to_csv(f"causal_landscape_scores_{name}.csv", index=False)
#     print(
#         f"[analyze_causal_landscape] Wrote CSV 'causal_landscape_scores_{name}.csv' "
#         f"at {time.perf_counter() - t_start:.1f}s"
#     )
#     return df

def plot_scores(all_scores):
    """
    For each model, create a barplot that compares week and dose-rate
    prediction accuracy for each feature-selection method. Also draw
    dashed horizontal lines for the all_genes baseline accuracies.
    """
    for model_name, model_scores in all_scores.items():
        rows = []
        for label_name, label_scores in model_scores.items():
            for feat_name, stats in label_scores.items():
                rows.append(
                    {
                        "feature": feat_name,
                        "label": label_name,
                        "score": stats["score"],
                        "mean": stats["mean"],
                        "std": stats.get("std", 0.0),
                        "n_genes": stats.get("n_genes", None),
                    }
                )

        df = pd.DataFrame(rows)

        # Ensure both labels are present
        if not {"week", "dose_rate"}.issubset(set(df["label"].unique())):
            continue

        # Pivot so we have columns for week and dose_rate per feature set
        pivot = df.pivot(index="feature", columns="label", values="mean")
        pivot_std = df.pivot(index="feature", columns="label", values="std")
        pivot_score = df.pivot(index="feature", columns="label", values="score")

        # Order methods by "closeness" to Pareto front:
        # larger average of (week, dose_rate) first
        if {"week", "dose_rate"}.issubset(pivot.columns):
            pivot = pivot.assign(_score=(pivot["week"] + pivot["dose_rate"]) / 2.0)
            pivot = pivot.sort_values(by="_score", ascending=False).drop(columns="_score")

        # Map feature -> n_genes (take one entry per feature)
        n_genes_map = (
            df.drop_duplicates(subset=["feature"])
            .set_index("feature")["n_genes"]
            .to_dict()
        )

        # Get all_genes baselines
        baseline_week = model_scores["week"]["all_genes"]["mean"]
        baseline_dr = model_scores["dose_rate"]["all_genes"]["mean"]

        # Sort features for a consistent x-axis (optionally put all_genes first)
        pivot = pivot.sort_index()
        if "all_genes" in pivot.index:
            pivot = pivot.reindex(
                ["all_genes"]
                + [f for f in pivot.index if f != "all_genes"]
            )

        # --- Bar plot: week vs dose_rate by feature set ---
        ax = pivot.plot(
            kind="bar",
            figsize=(15, 6),
            rot=45,
            ylabel="Accuracy",
            title=f"{model_name} performance by feature set",
        )

        # Dashed lines for all_genes
        ax.axhline(
            baseline_week,
            linestyle="--",
            color="tab:orange",
            label="all_genes week baseline",
        )
        ax.axhline(
            baseline_dr,
            linestyle="--",
            color="tab:blue",
            label="all_genes dose_rate baseline",
        )

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(f"{model_name}_feature_selection_barplot_norm.png")
        plt.close()
        

        # --- Scatter plot: Pareto-style week vs dose_rate ---
        fig, (ax1, ax2) = plt.subplots(2,figsize=(14, 12))

        # Color each feature set differently using a continuous viridis colormap
        cmap = plt.get_cmap("viridis")
        n_points = len(pivot.index)
        colors = [cmap(i / max(n_points - 1, 1)) for i in range(n_points)]

        # Use a variety of markers to visually separate methods
        marker_cycle = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "*"]

        # Draw shaded ellipses first (error on each axis), then points on top
        for idx, (feature_name, row) in enumerate(pivot.iterrows()):
            color = colors[idx]
            x, y = row["week"], row["dose_rate"]
            std_week = pivot_std.loc[feature_name, "week"] if "week" in pivot_std.columns else 0.0
            std_dr = pivot_std.loc[feature_name, "dose_rate"] if "dose_rate" in pivot_std.columns else 0.0
            # Ellipse widths = ±1 std on each axis (width/height in data coords)
            w = 2 * float(std_week)
            h = 2 * float(std_dr)
            if w > 0 or h > 0:
                ell = Ellipse((x, y), width=max(w, 1e-6), height=max(h, 1e-6), facecolor=color, edgecolor="none", alpha=0.1)
                ax1.add_patch(ell)

        for idx, ((feature_name, row), color) in enumerate(zip(pivot.iterrows(), colors)):
            n_genes = n_genes_map.get(feature_name)
            if n_genes is not None:
                label = f"{feature_name} ({int(n_genes)})"
            else:
                label = feature_name

            # Distinct marker per point, cycling through marker list
            marker = marker_cycle[idx % len(marker_cycle)]
            size = 60 if feature_name == "all_genes" else 50

            ax1.scatter(
                row["week"],
                row["dose_rate"],
                s=size,
                color=color,
                marker=marker,
                label=label,
            )
            row_holdout = pivot_score[feature_name]
            ax2.scatter(
                row_holdout["week"],
                row_holdout["dose_rate"],
                s=size,
                color=color,
                marker=marker,
                label=label,
            )


        ax2.set_xlabel("Week accuracy")
        ax1.set_ylabel("Dose rate accuracy")
        ax1.set_title(f"{model_name} Pareto front (feature sets)")

        # Put legend outside the scatter plot
        ax1.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0.0,
            fontsize=8,
        )


        plt.tight_layout()
        plt.savefig(f"{model_name}_pareto_front.png", bbox_inches="tight")
        plt.close()

GENE_RANGE = [10, 20, 30, 50, 100, 200, 400, 600, 800]
#GENE_RANGE = [100, 200, 300, 400, 500, 600, 700]
INCREASING_NUM_GENES_METHODS = [
    "random",
    "variance",
    "recursive",
    "sparse",
    "ai_kosmos",
    "ai_chatgpt",
    "causal_H",
    "causal_I",
    "causal_intersections_H_I",
]


def run_increasing_num_genes(X, labels, phenotypes, holdout):
    """
    For each number of genes in GENE_RANGE and each feature selection method,
    fit all models on both labels (dose_rate, week) and save accuracies.
    """
    rows = []
    cached_data = dict()
    for num_genes in GENE_RANGE:
        for method in INCREASING_NUM_GENES_METHODS:
            for model_name, model in MODELS.items():
                for label_name, y in labels.items():
                    try:
                        genes, cached_data = sample_features(num_genes, method, X, y, label_name, cached_data)
                    except Exception as e:
                        print(f"[run_increasing_num_genes] {method} n={num_genes}: {e}")
                        continue
                    mean_acc, std_acc, score = model_fit(X, y, clone(model), genes=genes, holdout=(holdout["X"], holdout["y"][label_name]))
                    rows.append({
                        "num_genes": num_genes,
                        "feature_selection": method,
                        "model": model_name,
                        "label": label_name,
                        "accuracy_mean": float(mean_acc),
                        "accuracy_std": float(std_acc),
                        "score": float(score),
                        "n_genes_used": len(genes),
                    })
                    print(
                        f"num_genes={num_genes} | {method} | {model_name} | {label_name}: "
                        f"train_accuracy={mean_acc:.3f} ± {std_acc:.3f} (n_genes={len(genes)}) | holdout_accuracy={score:.3f}"
                    )
            for p_label in list(phenotypes.columns):
                y_p = phenotypes[p_label].to_numpy().ravel()
                y_p_holdout = holdout['phenotype'][p_label].to_numpy().ravel()
                mean_acc, std_acc, score = model_fit(X, y_p, clone(PHENOTYPE_MODEL["elastic"]), genes=genes, holdout=(holdout["X"], y_p_holdout))
                rows.append({
                    "num_genes": num_genes,
                    "feature_selection": method,
                    "model": "elastic",
                    "label": p_label,
                    "accuracy_mean": float(mean_acc),
                    "accuracy_std": float(std_acc),
                    "score": float(score),
                    "n_genes_used": len(genes),
                })
                print(
                    f"num_genes={num_genes} | {method} | elastic | {p_label}: "
                    f"accuracy={mean_acc:.3f} ± {std_acc:.3f} (n_genes={len(genes)}) | holdout_accuracy={score:.3f}"
                )

    df = pd.DataFrame(rows)
    out_csv = "increasing_num_genes_accuracy.csv"
    df.to_csv(out_csv, index=False)
    print(f"[run_increasing_num_genes] Saved {len(df)} rows to {out_csv}")
    return df


def plot_increasing_num_genes(df, out_prefix="increasing_num_genes"):
    """
    For each (model, label), create a plot: x = num_genes, y = accuracy_mean,
    with error bars from accuracy_std. One line per feature_selection method.
    """
    if df is None or df.empty:
        return
    models = df["model"].unique().tolist()
    labels = df["label"].unique().tolist()
    methods = df["feature_selection"].unique().tolist()
    num_genes = sorted(df["num_genes"].unique())

    for model_name in models:
        for label_name in labels:
            sub = df[(df["model"] == model_name) & (df["label"] == label_name)]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for method in methods:
                m = sub[sub["feature_selection"] == method]
                if m.empty:
                    continue
                m = m.sort_values("num_genes")
                x = m["num_genes"].values
                y = m["score"].values
                ax.plot(
                    x, y,
                    label=method, marker="o", markersize=4,
                )
            ax.set_xlabel("Number of genes")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Model: {model_name} — Label: {label_name}")
            ax.legend(loc="best", fontsize=8)
            ax.set_xticks(num_genes)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            plt.tight_layout()
            safe_name = f"{out_prefix}_{model_name}_{label_name}_holdout.png"
            plt.savefig(safe_name, bbox_inches="tight")
            plt.close()
            print(f"[plot_increasing_num_genes] Saved {safe_name}")

def multi_task_elastic_net(X, y_dr, y_w, holdout = None, out_path: str = "multi_task_elastic_net_coefs.pkl"):
    """
    Fit MultiTaskElasticNet models for dose-rate and week (time),
    and save:
      - coefficient arrays (one row per class, one column per feature),
      - feature name order,
      - class order as seen by the encoders.
    """
    # Ensure y are 1D arrays
    y_dr_arr = np.ravel(y_dr)
    y_w_arr = np.ravel(y_w)

    # One-hot encode targets so each class is a separate task
    enc_dr = OneHotEncoder()
    y_dr_one_hot = enc_dr.fit_transform(y_dr_arr.reshape(-1, 1)).toarray()

    enc_w = OneHotEncoder()
    y_w_one_hot = enc_w.fit_transform(y_w_arr.reshape(-1, 1)).toarray()

    # Dose-rate multitask model
    model_dr = clone(PHENOTYPE_MODEL["elastic_multitask"])
    model_dr.fit(X, y_dr_one_hot)

    # Week/time multitask model
    model_w = clone(PHENOTYPE_MODEL["elastic_multitask"])
    model_w.fit(X, y_w_one_hot)
    if holdout:
        print(model_dr.score(holdout["X"], enc_dr.transform(holdout["y"]["dose_rate"].reshape(-1, 1)).toarray()))
        print(model_w.score(holdout["X"], enc_w.transform(holdout["y"]["week"].reshape(-1, 1)).toarray()))

    # Package everything with explicit ordering information
    result = {
        "features": list(X.columns),
        "dose_rate": {
            "classes": enc_dr.categories_[0].tolist(),
            "coef": model_dr.coef_.tolist(),
            "intercept": model_dr.intercept_.tolist(),
        },
        "week": {
            "classes": enc_w.categories_[0].tolist(),
            "coef": model_w.coef_.tolist(),
            "intercept": model_w.intercept_.tolist(),
        },
    }

    with open(out_path, "wb") as f:
        pickle.dump(result, f)

    return result

def split_holdout(X,y_dr, y_w, phenotypes):
    print("Splitting into a holdout set")
    holdout_inds = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    train_inds = list(set(np.arange(X.shape[0])) - set(holdout_inds))

    # Safety: splits must be disjoint (in original index space)
    assert np.intersect1d(train_inds, holdout_inds).size == 0

    # Safety: holdout must include every class for both labels
    assert set(np.unique(y_dr)).issubset(set(np.unique(y_dr[holdout_inds])))
    assert set(np.unique(y_w)).issubset(set(np.unique(y_w[holdout_inds])))

    X_holdout, y_dr_holdout, y_w_holdout, phenotypes_holdout = (
        X.loc[holdout_inds],
        y_dr[holdout_inds],
        y_w[holdout_inds],
        phenotypes.loc[holdout_inds],
    )
    X, y_dr, y_w, phenotypes = (
        X.loc[train_inds],
        y_dr[train_inds],
        y_w[train_inds],
        phenotypes.loc[train_inds],
    )
    return (
        X.reset_index(drop=True),
        y_dr,
        y_w,
        phenotypes.reset_index(drop=True),
        X_holdout.reset_index(drop=True),
        y_dr_holdout,
        y_w_holdout,
        phenotypes_holdout.reset_index(drop=True),
    )

if __name__ == "__main__":
    args = parse_arguments()
    X, y_dr, y_w, phenotypes = load_data(args)
    if args.normalize:
        print("Normalizing data")
        X_norm = StandardScaler().fit_transform(X) # Normalize
        X = pd.DataFrame(data=X_norm, columns=X.columns)
    X,y_dr, y_w, phenotypes, X_holdout, y_dr_holdout, y_w_holdout, phenotypes_holdout = split_holdout(X,y_dr, y_w, phenotypes)
    if args.only_landscape:
        print("Only running causal landscape")
    # Flatten encoded dose-rate labels for downstream use
    y_dr_flat = np.ravel(y_dr)
    y_w_flat = np.ravel(y_w)

    # Define labels (tasks)
    labels = {
            "dose_rate": y_dr_flat,
            "week": y_w_flat    }
    labels_holdout = {
        "dose_rate": np.ravel(y_dr_holdout),
        "week": np.ravel(y_w_holdout)    }
    
    # Collect all feature sets produced by the various algorithms
    feature_sets = {}
    
    # 0) All genes
    feature_sets["all_genes"] = X.columns.tolist()
    
    # 1) Random features
    feature_sets["random"] = random(X)

    # 2) Variance thresholding
    feature_sets["variance"] = variance_thresholding(X)

    # 3) Recursive feature elimination 
    if not args.short_version:
        feature_sets["recursive_dose_rate"] = list(
            recursive_features(X, y_dr_flat, MODELS["svc_linear"])
        )
        
        feature_sets["recursive_week"] = list(
            recursive_features(X, y_w, MODELS["svc_linear"])
        )

    # 4) Causal features for different dose rates
    for dr in CAUSAL_DOSE_RATES:
        feature_sets[f"causal_{dr}"] = causal_features(dr, tf=False)
        feature_sets[f"causal_tf_{dr}"] = causal_features(dr, tf=True)
    feature_sets[f"causal_dose_rate"] = causal_features("all_doses_dose_rate", tf=False)
    feature_sets[f"causal_week"] = causal_features("all_doses_week", tf=False)


    # 5) Causal feature intersections for different dose-rate sets
    dose_rate_sets = []
    for r in range(1, len(CAUSAL_DOSE_RATES) + 1):
        dose_rate_sets.extend(combinations(CAUSAL_DOSE_RATES, r))

    for dr_set in dose_rate_sets:
        key_base = "_".join(dr_set)
        tf_set = list(
            causal_features_intersections(dr_set, tf=True) 
        )
        neighborhood_set = list(
            causal_features_intersections(dr_set, tf=False)
        )
        if len(dr_set) > 1:
            if len(neighborhood_set) > 0:
                feature_sets[f"causal_intersections_{key_base}"] = neighborhood_set
            if len(tf_set) > 0:
                feature_sets[f"causal_intersections_tf_{key_base}"] = tf_set

    # 6) AI-derived features with different names
    for name in ["kosmos", "chatgpt"]:
        feature_sets[f"ai_{name}"] = ai_features(name)
        
    # 7) Sparse features
    if not args.only_landscape:
        feature_sets['elastic_week'] = sparse_features(X, y_w_flat)
        feature_sets['elastic_dose_rate'] = sparse_features(X, y_dr_flat)
    
    if not args.short_version:
        TOP_GENES = set.intersection(*[set(feature_sets[top]) for top in TOP_PERFORMERS])
        feature_sets["top_genes"] = TOP_GENES
        for top in TOP_PERFORMERS:
            genes_filtered = list(set(feature_sets[top]).intersection(set(X.columns)))
            feature_sets[f"{top}_recursive_dose_rate"] = recursive_features(X[genes_filtered], y_dr_flat, MODELS['svc_linear'], min_features=10)
            feature_sets[f"{top}_recursive_week"] = recursive_features(X[genes_filtered], y_w_flat, MODELS['svc_linear'], min_features=10)

    if not args.only_landscape:
        #Run each model, each label, on each feature set and store mean/std scores
        all_scores = {}
        for model_name, model in MODELS.items():
            all_scores[model_name] = {}
            for label_name, y in labels.items():
                all_scores[model_name][label_name] = {}
                for feat_name, genes in feature_sets.items():
                    # Determine how many genes are actually used
                    if genes is None:
                        genes_filtered = X.columns
                    else:
                        genes_filtered = list(set(genes).intersection(set(X.columns)))
                    mean, std, score = model_fit(X, y, clone(model), genes=genes,holdout=(X_holdout, labels_holdout[label_name]))
                    all_scores[model_name][label_name][feat_name] = {
                        "mean": float(mean),
                        "std": float(std),
                        "score": float(score),
                        "n_genes": int(len(genes_filtered)),
                    }
                    print(
                        f"{model_name} | {label_name} | {len(genes_filtered)} genes | {feat_name}:  "
                        f"mean={mean:.3f}, std={std:.3f}, score={score:.3f}"
                    )

        # Optionally persist results for later analysis
        if args.normalize:
            with open("feature_selection_scores_norm.pkl", "wb") as f:
                pickle.dump(all_scores, f)
            with open("feature_norm.pkl", "wb") as f:
                pickle.dump(feature_sets, f)
        else:
            with open("feature_selection_scores.pkl", "wb") as f:
                pickle.dump(all_scores, f)
            with open("feature.pkl", "wb") as f:
                pickle.dump(feature_sets, f)

        # Generate plots for each model
        plot_scores(all_scores)

    if args.run_increasing_genes:
        print("Running increasing number of genes...")
        df_rig = run_increasing_num_genes(X,labels,phenotypes, holdout={"X":X_holdout, "y": labels_holdout, "phenotype":phenotypes_holdout})
        plot_increasing_num_genes(df_rig)
        
    for top in TOP_PERFORMERS:
        print(top)
        genes_filtered = list(set(feature_sets[top]).intersection(set(X.columns)))
        multi_task_elastic_net(X[genes_filtered],y_dr,y_w, out_path=f"multi_task_elastic_net_coefs_{top}.pkl", holdout={"X":X_holdout[genes_filtered], "y": labels_holdout})
    # Analyze causal graphs
    # for name, file in GRAPHS.items():
    #     print(name)
    #     G = nx.read_gexf(file)
    #     if name == 'invariant':
    #         G = nx.subgraph(G, CAUSAL_NEIGHBORHOODS["all_doses_dose_rate"])
    #     else:
    #         G = nx.subgraph(G, CAUSAL_NEIGHBORHOODS[name])
    #     analyze_causal_landscape(G, name, model_name='svc_linear')
    # G = nx.read_gexf(GRAPHS["H"])
    # # G = nx.subgraph(G, CAUSAL_NEIGHBORHOODS["H"])
    # analyze_causal_landscape(G, "H", args=args, model_name='svc_linear')