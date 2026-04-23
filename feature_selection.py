# To quantify separability, train classifiers to predict dose rate 
# and week using different feature selections
from sklearn.model_selection import  KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.feature_selection import VarianceThreshold, RFECV, RFE
from sklearn.base import clone
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import MultiTaskElasticNet
from global_variables import DOSE_RATE_LABELS, DOSE_RATES, DOSE_RATES_REGRESSION
with open("/homes/shahashka/lucid_cd/data/gene_groups.pkl", "rb") as f:
    CAUSAL_TFS, CAUSAL_NEIGHBORHOODS, KOSMOS, CHATGPT, BNL = pickle.load(f)

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
    for dr in DOSE_RATE_LABELS:
        dose_df = deg_df.loc[deg_df["Dose"] == f"d{dr}"]
        genes_by_dose[dr] = list(set(dose_df["Gene"]))
    return genes_by_dose


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
    log2fold_df['week'] = [float(w[1]) for w in log2fold_df['week']]
    if args.prune_log2fold:
        log2fold_df_na = log2fold_df.fillna(0) # if we drop na after pruning, we are left with no genes that overlap conditions
    else:
        log2fold_df_na = log2fold_df.dropna(axis=1) # identify genes that are differenitlaly expressed across dose rates while ignoring significance
    
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

    return X, labels_dr_regression, labels_w, phenotypes

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
