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
from sklearn.metrics import accuracy_score, r2_score
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
import os
import sys

def parse_arguments(args=None):
    """
    Creates the argument parser and defines command-line arguments.
    """
    parser = argparse.ArgumentParser(description="A simple command-line utility.")
    parser.add_argument("--plot_only", "-po", default=False, action="store_true", help="Only run plotting from saved files")
    parser.add_argument("--normalize", "-n", default=False, action="store_true", help="Normalize the log2fold dataset")
    parser.add_argument("--prune_log2fold", "-p", default=False, action="store_true", help="Prune the log2fold dataset")
    parser.add_argument("--run_pareto", "-rp", default=False, action="store_true", help="Run prediction for dose_rate + week independently")
    parser.add_argument("--run_increasing_genes", "-rig", default=False, action="store_true", help="Run run_increasing_num_genes and save accuracies to CSV")
    parser.add_argument("--short_version", "-sv", default=False, action="store_true", help="Skip the recursive methods to save time")
    parser.add_argument("--run_multitask_combined", "-rmc", default=False, action="store_true", help="Run multitask elastic net predicting dose_rate + week simultaneously")
    parser.add_argument("--output_dir", "-o", default="./features", help="Output directory for results and plots (default: ./features)")
    parser.add_argument("--use_tpm", "-tpm", default=False, action="store_true", help="Use TPM data instead of log2fold data")

    return parser.parse_args(args) # If args is None, it uses sys.argv

with open("/homes/shahashka/lucid_cd/data/gene_groups.pkl", "rb") as f:
    CAUSAL_TFS, CAUSAL_NEIGHBORHOODS, KOSMOS, CHATGPT, BNL = pickle.load(f)
MODELS = {
    "logistic_regression": LogisticRegression(penalty="l1", solver="saga", max_iter=30000),
    # "svc_rbf": SVC(kernel="rbf", max_iter=30000),
    "svc_linear": LinearSVC(penalty="l1", max_iter=30000),
    "logistic_regression_elastic": LogisticRegression(penalty="elasticnet", l1_ratio=0.5,solver="saga", max_iter=30000),
    "elastic_multitask": MultiTaskElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=30000)
}
PHENOTYPE_MODEL = { 'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=30000), "elastic_multitask" : MultiTaskElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=30000)}
CAUSAL_DOSE_RATES = ["F", "G", "H", "I", "J"]
DOSE_RATES = {"control": 0.0, "F": 0.004, "G": 0.04, "H": 0.4, "I": 4.0, "J": 8.0}
DOSE_RATES_ACTUAL = {"control": 0.0, "F": 0.4, "G": 0.4, "H": 0.4, "I": 4.0, "J": 8.0}
DOSE_RATES_REGRESSION = {"F": 0.38, "G": 0.28, "H": 0.55, "I": 6.66, "J": 12.11, "shared":0.0, "control":0.0}

GRAPHS = {"invariant": "/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs3/dag_gnn_combined.gexf",
          "F":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseF.gexf",
          "G":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseG.gexf",
          "H":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseH.gexf",
          "I":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseI.gexf",
          "J":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseJ.gexf" 
          }
TOP_PERFORMERS = ["causal_full_G", "causal_full_I", "causal_dose_rate", "multiple_correlation_joint"]
# ["ai_kosmos", "ai_chatgpt", "causal_H", "causal_I", "causal_intersections_H_I",
#                   "causal_intersections_H_J", "causal_J" ]
                #   "causal_tf_J", "causal_tf_H", "causal_tf_I"]

def deg_features():
    """Load DEGs per dose rate, filtered by padj < 0.05 and |log2FoldChange| > 1."""
    deg_df = pd.read_csv(
        "/homes/shahashka/lucid_cd/data/rpe1_experiment2/"
        "rpe1_9week_study_experiment2_diffexp_deseq_vs_control_all_dG_W2_adjust.txt",
        sep="\t",
    )
    deg_df = deg_df.loc[deg_df["padj"] < 0.05]
    deg_df = deg_df.loc[abs(deg_df["log2FoldChange"]) > 1]
    genes_by_dose = {}
    for dr in CAUSAL_DOSE_RATES:
        dose_df = deg_df.loc[deg_df["Dose"] == f"d{dr}"]
        genes_by_dose[dr] = list(set(dose_df["Gene"]))
    return genes_by_dose

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
def sample_features(num_genes, feature_selection_method, X, y, y_name,cached_data=dict()):
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

def model_fit(X, y, model, holdout, model_name, genes=None, feature_selector=None, normalize=False):
    X_eval, y_eval = holdout
    genes_used = list(X.columns)

    # For unsupervised/external gene lists, filter once up front
    if genes is not None and feature_selector is None:
        genes_used = list(set(genes).intersection(set(X.columns)))
        X = X[genes_used]
        X_eval = X_eval[genes_used]

    # Normalize before feature selection so selectors see scaled data
    if normalize:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        X_eval = pd.DataFrame(scaler.transform(X_eval), columns=X_eval.columns, index=X_eval.index)

    if feature_selector is not None:
        genes_used = feature_selector(X, y)
        genes_used = list(set(genes_used).intersection(set(X.columns)))
        if not genes_used:
            print(f"Warning: feature selector returned no genes, setting scores to 0")
            return 0.0, 0.0, 0.0, model, []
        X_eval = X_eval[genes_used]

    n_splits = 4
    encode = model_name == 'elastic_multitask'
    if encode: # only model that requires one hot y's
        enc = OneHotEncoder()
        y_enc = enc.fit_transform(y.reshape(-1, 1)).toarray()
        y_eval_enc = enc.transform(y_eval.reshape(-1, 1)).toarray()

    if model_name == "elastic":
        kfold = KFold(n_splits=4, random_state=42, shuffle=True)
    else:
        kfold = StratifiedKFold(n_splits=n_splits, random_state=42, shuffle=True)
    scores = []

    for i, (train_index, test_index) in enumerate(kfold.split(X, y)):
        X_train = X.loc[train_index]
        y_train = y_enc[train_index] if encode else y[train_index]
        X_test = X.loc[test_index]
        y_test = y_enc[test_index] if encode else y[test_index]

        # Normalize per fold: fit on train only, transform both
        if normalize:
            fold_scaler = StandardScaler()
            X_train = pd.DataFrame(fold_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(fold_scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        # Supervised feature selection: select on training fold only
        if feature_selector is not None:
            y_train_raw = y[train_index]
            fold_genes = feature_selector(X_train, y_train_raw)
            fold_genes = list(set(fold_genes).intersection(set(X_train.columns)))
            if not fold_genes:
                scores.append(0.0)
                continue
            X_train = X_train[fold_genes]
            X_test = X_test[fold_genes]

        model.fit(X_train, y_train)
        scores.append(model.score(X_test, y_test))

    # Holdout: refit on full training data with selected features
    if feature_selector is not None:
        X_final = X[genes_used]
    else:
        X_final = X
    y_final = y_enc if encode else y
    model.fit(X_final, y_final)

    mean, std = np.mean(scores), np.std(scores)
    score = model.score(X_eval, y_eval_enc) if encode else model.score(X_eval, y_eval)
    fitted = (model, enc) if encode else model
    return mean, std, score, fitted, genes_used

def model_fit_multitask(X, y_multi, model, holdout, target_names,
                        genes=None, feature_selector=None, normalize=False):
    """
    Fit a multi-output regression model predicting y_multi (n_samples, n_targets).
    Returns per-target R² for both CV and holdout.
    """
    X_eval, y_eval = holdout
    genes_used = list(X.columns)

    if genes is not None and feature_selector is None:
        genes_used = list(set(genes).intersection(set(X.columns)))
        X = X[genes_used]
        X_eval = X_eval[genes_used]

    if normalize:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        X_eval = pd.DataFrame(scaler.transform(X_eval), columns=X_eval.columns, index=X_eval.index)

    if feature_selector is not None:
        genes_used = feature_selector(X, y_multi)
        genes_used = list(set(genes_used).intersection(set(X.columns)))
        if not genes_used:
            n_targets = y_multi.shape[1]
            zeros = [0.0] * (n_targets + 1)
            print(f"Warning: feature selector returned no genes, setting scores to 0")
            return zeros, zeros, zeros, model, []
        X_eval = X_eval[genes_used]

    n_targets = y_multi.shape[1]
    kfold = KFold(n_splits=4, random_state=42, shuffle=True)
    fold_scores = []  # list of per-target R² arrays

    for train_index, test_index in kfold.split(X):
        X_train = X.loc[train_index]
        y_train = y_multi[train_index]
        X_test = X.loc[test_index]
        y_test = y_multi[test_index]

        if normalize:
            fold_scaler = StandardScaler()
            X_train = pd.DataFrame(fold_scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
            X_test = pd.DataFrame(fold_scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        if feature_selector is not None:
            fold_genes = feature_selector(X_train, y_train)
            fold_genes = list(set(fold_genes).intersection(set(X_train.columns)))
            if not fold_genes:
                fold_scores.append([0.0] * (n_targets + 1))
                continue
            X_train = X_train[fold_genes]
            X_test = X_test[fold_genes]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        per_target = [r2_score(y_test[:, j], y_pred[:, j]) for j in range(n_targets)]
        per_target.append(r2_score(y_test, y_pred, multioutput='variance_weighted'))
        fold_scores.append(per_target)

    # Holdout: refit on full training data
    X_final = X[genes_used] if feature_selector is not None else X
    model.fit(X_final, y_multi)
    y_pred_eval = model.predict(X_eval)
    holdout_per_target = [r2_score(y_eval[:, j], y_pred_eval[:, j]) for j in range(n_targets)]
    holdout_per_target.append(r2_score(y_eval, y_pred_eval, multioutput='variance_weighted'))

    cv_means = np.mean(fold_scores, axis=0)
    cv_stds = np.std(fold_scores, axis=0)
    return cv_means, cv_stds, holdout_per_target, model, genes_used

def load_data_tpm():
    df = pd.read_csv("/homes/shahashka/lucid_cd/data/rpe1_experiment2/cd_tpm_matrix_combined_dose_rate.csv", header=0)
    dose_rate = np.array([0.4 if (d==0.004) or (d==0.04) else d for d in df['dose_rate'] ])
    labels_dr = OrdinalEncoder().fit_transform(dose_rate.reshape(-1,1))
    labels_w = df['week']

    # Build labels_dr_regression: map raw dose_rate values to letters via DOSE_RATES, then to regression values
    value_to_letter = {v: k for k, v in DOSE_RATES.items()}
    labels_dr_regression = np.array([DOSE_RATES_REGRESSION[value_to_letter[d]] for d in df['dose_rate']])
    
    X = df.drop(columns=["dose_rate", "week"])

    # Build phenotypes (morphology diff-from-control)
    DATA_SUMMARY = f"/homes/shahashka/lucid_cd/data/rpe1_experiment2/cell_painting_summary_statistics.csv"
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
        return {f"{k}_diff": float(row[v]) - float(c[v]) for k, v in colmap.items()}

    morph = pd.concat(
        [
            treated_df[["radiation_label", "week_num"]].rename(columns={"radiation_label": "dose_label"}),
            treated_df.apply(row_diff, axis=1).apply(pd.Series),
        ],
        axis=1,
    )
    morph["dose_label"] = morph["dose_label"].astype(str)
    morph.set_index(["dose_label", "week_num"], inplace=True)
    phenotypes = morph[Y_cols].sort_index()
    phenotypes = phenotypes.reset_index()
    phenotypes = phenotypes.drop(columns=["week_num", "dose_label"])

    return X, labels_dr, labels_dr_regression, labels_w, phenotypes

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
    
    labels_dr_regression = np.array([DOSE_RATES_REGRESSION[d[1]] for d in log2fold_df['dose_rate']])
    log2fold_df['dose_rate'] = [DOSE_RATES_ACTUAL[d[1]] for d in log2fold_df['dose_rate']]
    log2fold_df['week'] = [float(w[1]) for w in log2fold_df['week']]
    if args.prune_log2fold:
        log2fold_df_na = log2fold_df.fillna(0) # if we drop na after pruning, we are left with no genes that overlap conditions
    else:
        log2fold_df_na = log2fold_df.dropna(axis=1) # identify genes that are differenitlaly expressed across dose rates while ignoring significance
    
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

    ## Build apoptosis phenotype and merge before dropping index columns
    RAD_VALUE_MAPPING = {"rad_0.001":"dF", "rad_0.01" : "dG", "rad_0.1":"dH", "rad_1":"dI", "rad_2":"dJ"}
    APOPTOSIS_CSV = "/homes/shahashka/lucid_cd/data/rpe1_experiment2/rpe1_exp2_plate8.csv"
    apop_df = pd.read_csv(APOPTOSIS_CSV, header=0)
    apop_df = apop_df.query("exp_name == 'plate8_exp_green' or exp_name == 'plate8_exp_red'").copy()
    apop_df["dose_label"] = apop_df["rad_value"].map(RAD_VALUE_MAPPING)
    apop_df["apoptosis"] = apop_df["neuclei_masked_ch3_total"] / apop_df["cell_count"]
    apop_agg = (
        apop_df.groupby(["dose_label", "week_number"])["apoptosis"]
        .mean()
        .reset_index()
        .rename(columns={"week_number": "week_num"})
    )

    phenotypes = phenotypes.merge(apop_agg, on=["dose_label", "week_num"], how="left")
    phenotypes = phenotypes.drop(columns=["week_num", "dose_label"])

    return X, labels_dr, labels_dr_regression, labels_w, phenotypes

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

def top_covariance(X, y, n_features=1000):
    """Return top genes by absolute covariance with target y."""
    X = _remove_labels(X)
    covs = np.array([np.cov(X[col], y)[0, 1] for col in X.columns])
    abs_covs = np.abs(covs)
    top_idx = np.argsort(abs_covs)[::-1][:n_features]
    return X.columns[top_idx].tolist()

def top_correlation(X, y, n_features=1000):
    """Return top genes by absolute Pearson correlation with target y."""
    X = _remove_labels(X)
    corrs = np.array([np.corrcoef(X[col], y)[0, 1] for col in X.columns])
    abs_corrs = np.abs(corrs)
    top_idx = np.argsort(abs_corrs)[::-1][:n_features]
    return X.columns[top_idx].tolist()

def top_multiple_correlation(X, y_multi, n_features=1000):
    """Return top genes by multiple correlation R with joint targets.
    R = sqrt(R²) from regressing each gene on y_multi (e.g. dose_rate + week).
    """
    from sklearn.linear_model import LinearRegression
    X = _remove_labels(X)
    r_scores = np.empty(len(X.columns))
    for i, col in enumerate(X.columns):
        reg = LinearRegression().fit(y_multi, X[col])
        r2 = reg.score(y_multi, X[col])
        r_scores[i] = np.sqrt(max(r2, 0.0))
    top_idx = np.argsort(r_scores)[::-1][:n_features]
    return X.columns[top_idx].tolist()

def variance_thresholding(X, threshold=0.1):
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
    selector = RFECV(model, step=100, cv=KFold(n_splits=4, shuffle=True, random_state=42), min_features_to_select=min_features)
    print(X.shape, y.shape)
    selector = selector.fit(X, y)
    genes = X.columns
    return genes[selector.support_]


def random_forest_features(X, y, n_features=1000):
    """Return top genes by Random Forest feature importance."""
    from sklearn.ensemble import RandomForestRegressor
    X = _remove_labels(X)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[::-1][:n_features]
    return X.columns[top_idx].tolist()

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
    model=ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=30000)
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

def sparse_features_multitask(X, y, n_features=1000):
    """
    Fit a MultiTaskElasticNet on multi-output y and return the top `n_features`
    genes by L2 norm of coefficients across tasks.
    """
    model = MultiTaskElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=30000)
    X = _remove_labels(X)
    model.fit(X, y)

    # coef_ shape: (n_tasks, n_features) → L2 norm across tasks
    coefs = np.linalg.norm(model.coef_, axis=0)
    sorted_idx = np.argsort(coefs)[::-1]

    sorted_idx = [i for i in sorted_idx if coefs[i] > 0]

    if not sorted_idx:
        return []

    k = min(n_features, len(sorted_idx))
    top_idx = sorted_idx[:k]
    return X.columns[top_idx].tolist()



def plot_scores(all_scores, norm, output_dir="./features"):
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

        # Order methods by "closeness" to Pareto front using euclidean distance from (0,0)
        if {"week", "dose_rate"}.issubset(pivot.columns):
            pivot = pivot.assign(_score=np.sqrt(pivot["week"]**2 + pivot["dose_rate"]**2))
            pivot = pivot.sort_values(by="_score", ascending=False).drop(columns="_score")
            pivot_score = pivot_score.assign(_score=np.sqrt(pivot_score["week"]**2 + pivot_score["dose_rate"]**2))
            pivot_score = pivot_score.sort_values(by="_score", ascending=False).drop(columns="_score")

        top_performers = list(pivot.head(10).iloc[0:10].index)
        print(model_name)
        print(pivot_score.head(10))

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
        # pivot = pivot.sort_index()
        # if "all_genes" in pivot.index:
        #     pivot = pivot.reindex(
        #         ["all_genes"]
        #         + [f for f in pivot.index if f != "all_genes"]
        #     )

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
        if norm:
            plt.savefig(f"{output_dir}/{model_name}_feature_selection_barplot_norm.png")
        else:
            plt.savefig(f"{output_dir}/{model_name}_feature_selection_barplot.png")
        plt.close()
        

        # --- Scatter plot: Pareto-style week vs dose_rate ---
        fig, (ax1, ax2) = plt.subplots(
            2,
            figsize=(15, 12),
            gridspec_kw={"hspace": 0.12},
        )

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
            alpha = 0.3
            if feature_name in top_performers:
                alpha=1
            ax1.scatter(
                row["week"],
                row["dose_rate"],
                s=size,
                color=color,
                marker=marker,
                label=label,
                alpha=alpha
            )
            row_holdout = pivot_score.loc[feature_name]
            ax2.scatter(
                row_holdout["week"],
                row_holdout["dose_rate"],
                s=size,
                color=color,
                marker=marker,
                label=label,
                alpha=alpha
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


        fig.tight_layout(h_pad=0.35)
        if norm:
            plt.savefig(f"{output_dir}/{model_name}_pareto_front_norm.png", bbox_inches="tight")
        else:
            plt.savefig(f"{output_dir}/{model_name}_pareto_front.png", bbox_inches="tight")
        plt.close()

        # --- Bar plot: combined R² by feature set ---
        if "combined" in set(df["label"].unique()):
            df_comb = df[df["label"] == "combined"].copy()
            df_comb = df_comb.sort_values(by="mean", ascending=False)
            baseline_comb = model_scores["combined"]["all_genes"]["mean"]

            fig, ax = plt.subplots(figsize=(15, 6))
            x = np.arange(len(df_comb))
            ax.bar(x, df_comb["mean"], yerr=df_comb["std"], capsize=3, color="tab:green", alpha=0.7, label="CV mean")
            ax.scatter(x, df_comb["score"], color="tab:red", zorder=3, s=30, label="Holdout")
            ax.axhline(baseline_comb, linestyle="--", color="gray", label="all_genes baseline")
            ax.set_xticks(x)
            ax.set_xticklabels(df_comb["feature"], rotation=45, ha="right")
            ax.set_ylabel("Combined R²")
            ax.set_title(f"{model_name} combined R² by feature set")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            suffix = "_norm" if norm else ""
            plt.savefig(f"{output_dir}/{model_name}_combined_r2_barplot{suffix}.png", bbox_inches="tight")
            plt.close()

GENE_RANGE = [10, 20, 30, 50, 100]#, 200, 400, 600, 800]
#GENE_RANGE = [100, 200, 300, 400, 500, 600, 700]
INCREASING_NUM_GENES_METHODS = [
    "random",
    "variance",
    "recursive",
    "sparse",
] + TOP_PERFORMERS


def run_increasing_num_genes(X, labels, phenotypes, holdout, normalize=False, output_dir="./features"):
    """
    For each number of genes in GENE_RANGE and each feature selection method,
    fit all models on both labels (dose_rate, week) and save accuracies.
    """
    SUPERVISED_METHODS = {"recursive", "sparse", "rf"}
    def _make_selector(m, ng):
        def selector(X, y):
            genes, _ = sample_features(ng, m, X, y, "")
            return genes
        return selector
    rows = []
    cached_data = dict()
    for num_genes in GENE_RANGE:
        for method in INCREASING_NUM_GENES_METHODS:
            for model_name, model in MODELS.items():
                for label_name, y in labels.items():
                    if method in SUPERVISED_METHODS:
                        # Defer selection to inside CV folds to avoid data leakage
                        mean_acc, std_acc, score, _, genes_used = model_fit(
                            X, y, clone(model), model_name=model_name,
                            feature_selector=_make_selector(method, num_genes),
                            holdout=(holdout["X"], holdout["y"][label_name]),
                            normalize=normalize)
                    else:
                        try:
                            genes, cached_data = sample_features(num_genes, method, X, y, label_name, cached_data)
                        except Exception as e:
                            print(f"[run_increasing_num_genes] {method} n={num_genes}: {e}")
                            continue
                        mean_acc, std_acc, score, _, genes_used = model_fit(
                            X, y, clone(model), model_name=model_name, genes=genes,
                            holdout=(holdout["X"], holdout["y"][label_name]),
                            normalize=normalize)
                    rows.append({
                        "num_genes": num_genes,
                        "feature_selection": method,
                        "model": model_name,
                        "label": label_name,
                        "accuracy_mean": float(mean_acc),
                        "accuracy_std": float(std_acc),
                        "score": float(score),
                        "n_genes_used": len(genes_used),
                    })
                    print(
                        f"num_genes={num_genes} | {method} | {model_name} | {label_name}: "
                        f"train_accuracy={mean_acc:.3f} ± {std_acc:.3f} (n_genes={len(genes_used)}) | holdout_accuracy={score:.3f}"
                    )
                        
            for p_label in list(phenotypes.columns):
                y_p = phenotypes[p_label].to_numpy().ravel()
                y_p_holdout = holdout['phenotype'][p_label].to_numpy().ravel()
                if method in SUPERVISED_METHODS:
                    # Defer selection to inside CV folds to avoid data leakage
                    mean_acc, std_acc, score, _, genes_used = model_fit(
                        X, y_p, clone(PHENOTYPE_MODEL["elastic"]), model_name="elastic",
                        feature_selector=_make_selector(method, num_genes),
                        holdout=(holdout["X"], y_p_holdout),
                        normalize=normalize)
                else:
                    try:
                        genes, cached_data = sample_features(num_genes, method, X, y, label_name, cached_data)
                    except Exception as e:
                        print(f"[run_increasing_num_genes] {method} n={num_genes}: {e}")
                        continue
                    mean_acc, std_acc, score, _, genes_used= model_fit(X, y_p, clone(PHENOTYPE_MODEL["elastic"]),model_name="elastic", genes=genes, holdout=(holdout["X"], y_p_holdout))
                rows.append({
                    "num_genes": num_genes,
                    "feature_selection": method,
                    "model": "elastic",
                    "label": p_label,
                    "accuracy_mean": float(mean_acc),
                    "accuracy_std": float(std_acc),
                    "score": float(score),
                    "n_genes_used": len(genes_used),
                })
                print(
                    f"num_genes={num_genes} | {method} | elastic | {p_label}: "
                    f"accuracy={mean_acc:.3f} ± {std_acc:.3f} (n_genes={len(genes)}) | holdout_accuracy={score:.3f}"
                )

    df = pd.DataFrame(rows)
    out_csv = f"{output_dir}/increasing_num_genes_accuracy.csv"
    df.to_csv(out_csv, index=False)
    print(f"[run_increasing_num_genes] Saved {len(df)} rows to {out_csv}")
    return df


def plot_increasing_num_genes(df, out_prefix="increasing_num_genes", output_dir="./features"):
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
            # ax.set_xscale("log")
            ax.set_xlabel("Number of genes")
            ax.set_ylabel("Accuracy")
            ax.set_title(f"Model: {model_name} — Label: {label_name}")
            ax.legend(loc="best", fontsize=8)
            ax.set_xticks(num_genes)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.05)
            plt.tight_layout()
            safe_name = f"{output_dir}/{out_prefix}_{model_name}_{label_name}_holdout.png"
            plt.savefig(safe_name, bbox_inches="tight")
            plt.close()
            print(f"[plot_increasing_num_genes] Saved {safe_name}")

def logistic_regression_elastic(X, y_dr, y_w, holdout = None):
    # Ensure y are 1D arrays
    y_dr_arr = np.ravel(y_dr)
    y_w_arr = np.ravel(y_w)
    model_dr = clone(MODELS["logistic_regression_elastic"])
    model_dr.fit(X, y_dr_arr)
    
    model_w = clone(MODELS["logistic_regression_elastic"])
    model_w.fit(X, y_w_arr)
    
    if holdout:
        print(model_dr.score(holdout["X"], holdout["y"]["dose_rate"]))
        print(model_w.score(holdout["X"], holdout["y"]["week"]))

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
        y_pred_dr = model_dr.predict(holdout["X"]).argmax(axis=1)
        y_pred_w = model_w.predict(holdout["X"]).argmax(axis=1)
        print(accuracy_score(enc_dr.transform(holdout["y"]["dose_rate"].reshape(-1, 1)).toarray().argmax(axis=1), y_pred_dr))
        print(accuracy_score(enc_w.transform(holdout["y"]["week"].reshape(-1, 1)).toarray().argmax(axis=1), y_pred_w))

        # print(model_dr.score(holdout["X"], enc_dr.transform(holdout["y"]["dose_rate"].reshape(-1, 1)).toarray()))
        # print(model_w.score(holdout["X"], enc_w.transform(holdout["y"]["week"].reshape(-1, 1)).toarray()))
        # print(model_dr.score(X,y_dr_one_hot), model_w.score(X,y_w_one_hot))

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

def split_holdout(X,y_dr,y_dr_reg, y_w, phenotypes):
    print("Splitting into a holdout set")
    holdout_inds = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    train_inds = list(set(np.arange(X.shape[0])) - set(holdout_inds))

    # Safety: splits must be disjoint (in original index space)
    assert np.intersect1d(train_inds, holdout_inds).size == 0

    # Safety: holdout must include every class for both labels
    assert set(np.unique(y_dr)).issubset(set(np.unique(y_dr[holdout_inds])))
    assert set(np.unique(y_w)).issubset(set(np.unique(y_w[holdout_inds])))

    X_holdout, y_dr_holdout, y_dr_reg_holdout, y_w_holdout, phenotypes_holdout = (
        X.loc[holdout_inds],
        y_dr[holdout_inds],
        y_dr_reg[holdout_inds],
        y_w[holdout_inds],
        phenotypes.loc[holdout_inds],
    )
    X, y_dr, y_dr_reg, y_w, phenotypes = (
        X.loc[train_inds],
        y_dr[train_inds],
        y_dr_reg[train_inds],
        y_w[train_inds],
        phenotypes.loc[train_inds],
    )
    return (
        X.reset_index(drop=True),
        y_dr,
        y_dr_reg,
        y_w,
        phenotypes.reset_index(drop=True),
        X_holdout.reset_index(drop=True),
        y_dr_holdout,
        y_dr_reg_holdout,
        y_w_holdout,
        phenotypes_holdout.reset_index(drop=True),
    )

if __name__ == "__main__":
    args = parse_arguments()
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    if args.use_tpm:
        X, y_dr, y_dr_reg, y_w, phenotypes = load_data_tpm()
    else:
        X, y_dr, y_dr_reg, y_w, phenotypes = load_data(args)
    X,y_dr,y_dr_reg, y_w, phenotypes, X_holdout, y_dr_holdout,y_dr_reg_holdout, y_w_holdout, phenotypes_holdout = split_holdout(X,y_dr,y_dr_reg, y_w, phenotypes)

    # Define labels (tasks)
    labels = {
            "dose_rate": np.ravel(y_dr),
            "week": np.ravel(y_w)    }
    labels_holdout = {
        "dose_rate": np.ravel(y_dr_holdout),
        "week": np.ravel(y_w_holdout)    }
    
    # Collect all feature sets produced by the various algorithms
    feature_sets = {}
    
    # 0) All genes
    feature_sets["all_genes"] = X.columns.tolist()
    
    # 1) Random features
    feature_sets["random_10"] = random(X, size=10)
    feature_sets["random_100"] = random(X, size=100)
    feature_sets["random"] = random(X)
    feature_sets["random_5000"] = random(X, size=5000)

    # 2) Variance thresholding
    feature_sets["variance"] = variance_thresholding(X)

    # 2b) Top covariance with dose rate and week
    feature_sets["covariance_dose_rate"] = top_covariance(X, np.ravel(y_dr_reg))
    feature_sets["covariance_week"] = top_covariance(X, np.ravel(y_w))

    # 2c) Top correlation with dose rate and week
    feature_sets["correlation_dose_rate"] = top_correlation(X, np.ravel(y_dr_reg))
    feature_sets["correlation_week"] = top_correlation(X, np.ravel(y_w))

    # 2d) Top multiple correlation with joint (dose_rate, week)
    y_joint = np.column_stack([np.ravel(y_dr_reg), np.ravel(y_w)])
    feature_sets["multiple_correlation_joint"] = top_multiple_correlation(X, y_joint, n_features=400)

    # 3) Recursive feature elimination — deferred to inside CV folds to avoid data leakage
    # (selectors are defined below in deferred_selectors)

    # 4) Causal features for different dose rates
    for dr in CAUSAL_DOSE_RATES:
        feature_sets[f"causal_{dr}"] = causal_features(dr, tf=False)
        feature_sets[f"causal_tf_{dr}"] = causal_features(dr, tf=True)
    feature_sets[f"causal_dose_rate"] = causal_features("all_doses_dose_rate", tf=False)
    feature_sets[f"causal_week"] = causal_features("all_doses_week", tf=False)


    # 5) Causal feature intersections for different dose-rate sets
    # dose_rate_sets = []
    # for r in range(1, len(CAUSAL_DOSE_RATES) + 1):
    #     dose_rate_sets.extend(combinations(CAUSAL_DOSE_RATES, r))

    # for dr_set in dose_rate_sets:
    #     key_base = "_".join(dr_set)
    #     tf_set = list(
    #         causal_features_intersections(dr_set, tf=True) 
    #     )
    #     neighborhood_set = list(
    #         causal_features_intersections(dr_set, tf=False)
    #     )
    #     if len(dr_set) > 1:
    #         if len(neighborhood_set) > 0:
    #             feature_sets[f"causal_intersections_{key_base}"] = neighborhood_set
    #         if len(tf_set) > 0:
    #             feature_sets[f"causal_intersections_tf_{key_base}"] = tf_set

    # 6) Full causal graph gene sets (all nodes per graph)
    for name, gexf_path in GRAPHS.items():
        G = nx.read_gexf(gexf_path)
        feature_sets[f"causal_full_{name}"] = [n for n in G.nodes() if n not in {"radiation", "dose_rate", "week"}]

    # 7) AI-derived features with different names
    for name in ["kosmos", "chatgpt"]:
        feature_sets[f"ai_{name}"] = ai_features(name)

    # 7) Differentially expressed genes per dose rate
    deg_by_dose = deg_features()
    for dr, genes in deg_by_dose.items():
        feature_sets[f"deg_{dr}"] = genes

    # 8) Sparse features — deferred to inside CV folds to avoid data leakage
    # 8) Top-performer recursive combos — also deferred

    # Deferred selectors: supervised methods that must run inside CV folds
    # Maps (feat_name, label_name) -> callable(X_train, y_train) -> gene_list

    if args.plot_only:
        # Optionally persist results for later analysis
        all_scores = None
        if args.normalize:
            with open(f"{out_dir}/feature_selection_scores_norm.pkl", "rb") as f:
                all_scores = pickle.load(f)
        else:
            with open(f"{out_dir}/feature_selection_scores.pkl", "rb") as f:
                all_scores = pickle.load(f)

        # Generate plots for each model
        plot_scores(all_scores, args.normalize, output_dir=out_dir)
        
    
    # ============================================================
    # Pareto: predict dose_rate, week independently
    # ============================================================
    elif args.run_pareto:
        deferred_selectors = {}
        if not args.short_version:
            deferred_selectors[("recursive_dose_rate", "dose_rate")] = (
                lambda X_t, y_t: list(recursive_features(X_t, y_t, clone(MODELS["svc_linear"])))
            )
            deferred_selectors[("recursive_week", "week")] = (
                lambda X_t, y_t: list(recursive_features(X_t, y_t, clone(MODELS["svc_linear"])))
            )
            deferred_selectors[("elastic_dose_rate", "dose_rate")] = (
                lambda X_t, y_t: sparse_features(X_t, y_t)
            )
            deferred_selectors[("elastic_week", "week")] = (
                lambda X_t, y_t: sparse_features(X_t, y_t)
            )
            deferred_selectors[("rf_dose_rate", "dose_rate")] = (
                lambda X_t, y_t: random_forest_features(X_t, y_t)
            )
            deferred_selectors[("rf_week", "week")] = (
                lambda X_t, y_t: random_forest_features(X_t, y_t)
            )

            # TOP_GENES = set.intersection(*[set(feature_sets[top]) for top in TOP_PERFORMERS])
            # feature_sets["top_genes"] = TOP_GENES
            for top in TOP_PERFORMERS:
                top_genes = list(set(feature_sets[top]).intersection(set(X.columns)))
                deferred_selectors[(f"{top}_recursive_dose_rate", "dose_rate")] = (
                    lambda X_t, y_t, tg=top_genes: list(recursive_features(
                        X_t[list(set(tg).intersection(set(X_t.columns)))], y_t,
                        clone(MODELS['svc_linear']), min_features=10))
                )
                deferred_selectors[(f"{top}_recursive_week", "week")] = (
                    lambda X_t, y_t, tg=top_genes: list(recursive_features(
                        X_t[list(set(tg).intersection(set(X_t.columns)))], y_t,
                        clone(MODELS['svc_linear']), min_features=10))
                )
        #Run each model, each label, on each feature set and store mean/std scores
        all_scores = {}
        for model_name, model in MODELS.items():
            all_scores[model_name] = {}
            for label_name, y in labels.items():
                all_scores[model_name][label_name] = {}

                def _record(feat_name, mean, std, score, fitted, genes_filtered):
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
                    if model_name == "elastic_multitask":
                        result = {
                            "features": genes_filtered,
                            f"{label_name}": {
                                "classes": fitted[1].categories_[0].tolist(),
                                "coef": fitted[0].coef_.tolist(),
                                "intercept": fitted[0].intercept_.tolist(),
                                "model": fitted[0]
                            },
                        }
                    else:
                        result = {
                            "features": genes_filtered,
                            f"{label_name}": {
                                "coef": fitted.coef_.tolist(),
                                "intercept": fitted.intercept_.tolist(),
                                "model": fitted
                            }
                        }
                    out_path = f"{out_dir}/{model_name}_{label_name}_{feat_name}.pkl"
                    with open(out_path, "wb") as f:
                        pickle.dump(result, f)

                # Unsupervised / external feature sets (no leakage)
                for feat_name, genes in feature_sets.items():
                    mean, std, score, fitted, genes_filtered = model_fit(
                        X, y, clone(model), model_name=model_name, genes=genes,
                        holdout=(X_holdout, labels_holdout[label_name]),
                        normalize=args.normalize)
                    _record(feat_name, mean, std, score, fitted, genes_filtered)

                # Supervised feature selectors (run inside CV folds)
                for (feat_name, sel_label), selector in deferred_selectors.items():
                    if sel_label != label_name:
                        continue
                    mean, std, score, fitted, genes_filtered = model_fit(
                        X, y, clone(model), model_name=model_name,
                        feature_selector=selector,
                        holdout=(X_holdout, labels_holdout[label_name]),
                        normalize=args.normalize)
                    _record(feat_name, mean, std, score, fitted, genes_filtered)
                        
            # Optionally persist results for later analysis
            if args.normalize:
                with open(f"{out_dir}/feature_selection_scores_norm.pkl", "wb") as f:
                    pickle.dump(all_scores, f)
                with open(f"{out_dir}/feature_norm.pkl", "wb") as f:
                    pickle.dump(feature_sets, f)
            else:
                with open(f"{out_dir}/feature_selection_scores.pkl", "wb") as f:
                    pickle.dump(all_scores, f)
                with open(f"{out_dir}/feature.pkl", "wb") as f:
                    pickle.dump(feature_sets, f)

            # Generate plots for each model
            plot_scores(all_scores, args.normalize, output_dir=out_dir)


    # ============================================================
    # Multitask combined: predict dose_rate + week simultaneously
    # ============================================================
    if args.run_multitask_combined:
        print("Running multitask combined (dose_rate + week)...")
        y_combined = np.column_stack([np.ravel(y_dr_reg), np.ravel(y_w)])
        y_combined_holdout = np.column_stack([np.ravel(y_dr_reg_holdout), np.ravel(y_w_holdout)])
        target_names = ["dose_rate", "week"]
        multitask_model = MultiTaskElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=30000)

        all_scores_mt = {"multitask_combined": {t: {} for t in target_names + ["combined"]}}

        def _record_mt(feat_name, cv_means, cv_stds, holdout_scores, genes_filtered):
            all_target_names = target_names + ["combined"]
            for j, target in enumerate(all_target_names):
                all_scores_mt["multitask_combined"][target][feat_name] = {
                    "mean": float(cv_means[j]),
                    "std": float(cv_stds[j]),
                    "score": float(holdout_scores[j]),
                    "n_genes": int(len(genes_filtered)),
                }
            print(
                f"multitask_combined | {len(genes_filtered)} genes | {feat_name}:  "
                + "  ".join(f"{t}: mean={cv_means[j]:.3f}±{cv_stds[j]:.3f}, holdout={holdout_scores[j]:.3f}"
                            for j, t in enumerate(all_target_names))
            )

        # Unsupervised / external feature sets
        for feat_name, genes in feature_sets.items():
            cv_means, cv_stds, holdout_scores, fitted, genes_filtered = model_fit_multitask(
                X, y_combined, clone(multitask_model),
                holdout=(X_holdout, y_combined_holdout),
                target_names=target_names, genes=genes,
                normalize=args.normalize)
            _record_mt(feat_name, cv_means, cv_stds, holdout_scores, genes_filtered)

        # Supervised selectors for combined labels
        deferred_combined = {}
        if not args.short_version:
            # deferred_combined["recursive"] = (
            #     lambda X_t, y_t: list(recursive_features(
            #         X_t, y_t, clone(multitask_model)))
            # )
            deferred_combined["sparse"] = (
                lambda X_t, y_t: sparse_features_multitask(X_t, y_t)
            )
            deferred_combined["rf"] = (
            lambda X_t, y_t: random_forest_features(X_t, y_t)
            )
            # for top in TOP_PERFORMERS:
            #     top_genes = list(set(feature_sets[top]).intersection(set(X.columns)))
            #     deferred_combined[f"{top}_recursive"] = (
            #         lambda X_t, y_t, tg=top_genes: list(recursive_features(
            #             X_t[list(set(tg).intersection(set(X_t.columns)))], y_t,
            #             clone(multitask_model), min_features=10))
            #     )

        for feat_name, selector in deferred_combined.items():
            cv_means, cv_stds, holdout_scores, fitted, genes_filtered = model_fit_multitask(
                X, y_combined, clone(multitask_model),
                holdout=(X_holdout, y_combined_holdout),
                target_names=target_names, feature_selector=selector,
                normalize=args.normalize)
            _record_mt(feat_name, cv_means, cv_stds, holdout_scores, genes_filtered)

        # Save and plot
        suffix = "_norm" if args.normalize else ""
        with open(f"{out_dir}/multitask_combined_scores{suffix}.pkl", "wb") as f:
            pickle.dump(all_scores_mt, f)
        plot_scores(all_scores_mt, args.normalize, output_dir=out_dir)

    if args.run_increasing_genes:
        print("Running increasing number of genes...")
        df_rig = run_increasing_num_genes(X,labels,phenotypes, holdout={"X":X_holdout, "y": labels_holdout, "phenotype":phenotypes_holdout}, normalize=args.normalize, output_dir=out_dir)
        if args.normalize:
            plot_increasing_num_genes(df_rig, out_prefix="increasing_num_genes_norm", output_dir=out_dir)
        else:
            plot_increasing_num_genes(df_rig, output_dir=out_dir)

    # for top in TOP_PERFORMERS:
    #     print(top)
    #     genes_filtered = list(set(feature_sets[top]).intersection(set(X.columns)))
    #     logistic_regression_elastic(X[genes_filtered],y_dr,y_w,  holdout={"X":X_holdout[genes_filtered], "y": labels_holdout})
        #multi_task_elastic_net(X[genes_filtered],y_dr,y_w, out_path=f"./features/multi_task_elastic_net_coefs_{top}.pkl", holdout={"X":X_holdout[genes_filtered], "y": labels_holdout})
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