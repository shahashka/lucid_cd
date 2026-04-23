import pandas as pd
import numpy as np

DOSE_RATES = {"control": 0.0, "F": 0.004, "G": 0.04, "H": 0.4, "I": 4.0, "J": 8.0}
DOSE_RATES_ACTUAL = {"F": 0.38, "G": 0.28, "H": 0.55, "I": 6.66, "J": 12.11, "shared": 0.0, "control": 0.0}
DOSE_TO_GROUP = {0.0: "ctrl", 0.004: "dA", 0.04: "dB", 0.4: "dC", 4.0: "dD", 8.0: "dE"}
LFC_DOSE_TO_GROUP = {"dF": "dA", "dG": "dB", "dH": "dC", "dI": "dD", "dJ": "dE"}

_DATA_PATH_TPM = "/homes/shahashka/lucid_cd/data/rpe1_experiment2/cd_tpm_matrix_combined_dose_rate.csv"
_DATA_PATH_LFC = "/homes/shahashka/lucid_cd/data/rpe1_experiment2/rpe1_9week_study_experiment2_diffexp_deseq_vs_control_all_dG_W2_adjust.txt"


def load_data_tpm():
    df = pd.read_csv(_DATA_PATH_TPM, header=0)

    meta = pd.DataFrame({
        "week":  df["week"].values,
        "group": df["dose_rate"].map(DOSE_TO_GROUP).values,
    })

    gene_cols = [c for c in df.columns if c not in ("dose_rate", "week")]
    X = df[gene_cols].values
    return X, meta, gene_cols


def load_data_lfc():
    df = pd.read_csv(_DATA_PATH_LFC, sep="\t")

    # Pivot to (n_samples, n_genes): one row per (Dose, Week)
    pivoted = (
        df.groupby(["Dose", "Week", "Gene"])
        .mean(numeric_only=True)["log2FoldChange"]
        .unstack("Gene")
        .reset_index()
    )

    # Drop genes with any NaN across conditions
    pivoted = pivoted.dropna(axis=1)

    gene_cols = [c for c in pivoted.columns if c not in ("Dose", "Week")]
    weeks = pivoted["Week"].str[1:].astype(int).values          # "W1" -> 1
    groups = pivoted["Dose"].map(LFC_DOSE_TO_GROUP).values      # "dF" -> "dA"

    # Append synthetic ctrl rows (LFC vs ctrl = 0 by definition)
    n_weeks = sorted(set(weeks))
    ctrl_X = np.zeros((len(n_weeks), len(gene_cols)))
    ctrl_meta = pd.DataFrame({"week": n_weeks, "group": "ctrl"})

    X = np.vstack([pivoted[gene_cols].values, ctrl_X])
    meta = pd.concat(
        [pd.DataFrame({"week": weeks, "group": groups}), ctrl_meta],
        ignore_index=True,
    )

    return X, meta, gene_cols
