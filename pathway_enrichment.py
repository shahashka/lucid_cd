import os
import argparse
import pandas as pd
import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain, combinations, product
from matplotlib_venn import venn2, venn3
from upsetplot import from_memberships, UpSet
from gprofiler import GProfiler
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import to_hex
import json
from scipy.stats import fisher_exact

from global_variables import _fs_axis, _fs_leg, _fs_leg_title, _fs_tick, _fs_title, _DEFAULT_CMAPS, GRAPHS, DOSE_RATE_LABELS, DOSE_RATES_REGRESSION, DOSE_RATES_SORTED, HK_PATH

with open("/homes/shahashka/lucid_cd/data/gene_groups.pkl", "rb") as f:
    CAUSAL_TFS, CAUSAL_NEIGHBORHOODS, KOSMOS, CHATGPT, BNL = pickle.load(f)
EXPERIMENT = "/homes/shahashka/lucid_cd/data/rpe1_experiment2"
CONTEXT_SPECIFIC_GRAPHS = f"{EXPERIMENT}/bootstrap_graphs2"
INVARIANT_GRAPHS = f"{EXPERIMENT}/bootstrap_graphs3"

GOBP_PATHWAYS = [("GO:0010212",	"response to ionizing radiation"),
                 ("GO:0006974", "DNA damage response"),
                 ("GO:0007050","cell cycle arrest"),
                 ("GO:0071479","cellular response to ionizing radiation"), 
                 ("GO:0006302"," double-strand break repair"), 
                 ("GO:0006281", "DNA repair"), 
                 ("GO:0000075", "cell cycle checkpoint signaling"),
                 ("GO:0060561","apoptotic process involved in morphogenesis"), 
                 ("GO:0006979", "response to oxidative stress")]
KEGG_PATHWAYS = [("KEGG:04115", "p53 signaling pathway"), 
                 ("KEGG:04110","Cell cycle"), 
                 ("KEGG:04210","Apoptosis"), 
                 ("KEGG:03440", "Homologous recombination"), 
                 ("KEGG:03450","Non-homologous end-joining"), 
                 ("KEGG:03460","Fanconi anemia pathway"), ]
                #  ("REAC:R-HSA-2559582","Senescence-Associated Secretory Phenotype (SASP)")]
WP_PATHWAYS = [("WP:WP45","G1 to S cell cycle control"),
               ("WP:WP254", "Apoptosis"), 
               ("WP:WP707", "DNA damage response"),
               ("WP:WP710", "DNA damage response (only ATM dependent)"), 
               ("WP:WP1530", "miRNA regulation of DNA damage response"), 
               ("WP:WP1772","Apoptosis modulation and signaling"), 
               ("WP:WP3391", "Senescence-associated secretory phenotype (SASP)"), 
               ("WP:WP4946", "Genes and complexes involved in the DNA repair pathways"),
               ("WP:WP4963", "p53 transcriptional gene network"),
               ("WP:WP5434", "Cancer pathways"), 
               ("WP:WP5475", "Hallmark of cancer: sustaining proliferative signaling")]

ALL_PATHWAY_ENTRIES = GOBP_PATHWAYS + KEGG_PATHWAYS + WP_PATHWAYS
PATHWAY_DESCRIPTIONS = {pid.strip(): desc.strip() for pid, desc in ALL_PATHWAY_ENTRIES}
ALL_RADIATION_PATHWAY_IDS = [pid.strip() for pid, _ in ALL_PATHWAY_ENTRIES]
PATHWAY_CATEGORIES = {

    "DNA_damage_sensing_response": [
        "GO:0010212",
        "GO:0006974",
        "GO:0071479", 
        "WP:WP707", 
        "WP:WP710",
        "WP:WP1530",
    ],

    "DNA_repair_mechanisms": [
        "GO:0006302", 
        "GO:0006281", 
        "KEGG:03440", 
        "KEGG:03450", 
        "KEGG:03460", 
        "WP:WP4946", 
    ],

    "cell_cycle_and_checkpoints": [
        "GO:0007050", 
        "GO:0000075",
        "KEGG:04110", 
        "WP:WP45", 
    ],

    "apoptosis_p53_cell_fate": [
        "GO:0060561",
        "KEGG:04210", 
        "KEGG:04115", 
        "WP:WP254", 
        "WP:WP1772", 
        "WP:WP4963",
    ],

    "stress_senescence_cancer_signaling": [
        "GO:0006979",
        "REAC:R-HSA-2559582",
        "WP:WP3391", 
        "WP:WP5434", 
        "WP:WP5475", 
    ],
}

# First-listed category wins if a pathway appears in more than one category.
PATHWAY_TO_CATEGORY = {}
for _cat, _pws in PATHWAY_CATEGORIES.items():
    for _pw in _pws:
        PATHWAY_TO_CATEGORY.setdefault(_pw, _cat)

CATEGORY_COLORS = {
    "DNA_damage_sensing_response":        "#4E79A7",  # steel blue
    "DNA_repair_mechanisms":              "#59A14F",  # muted green
    "cell_cycle_and_checkpoints":         "#EDC948",  # amber
    "apoptosis_p53_cell_fate":            "#E15759",  # red
    "stress_senescence_cancer_signaling": "#B07AA1",  # lavender
    "uncategorized":                      "#8C8C8C",  # neutral gray
}

_CATEGORY_DISPLAY_NAMES = {
    "DNA_damage_sensing_response":        "DNA damage sensing & response",
    "DNA_repair_mechanisms":              "DNA repair mechanisms",
    "cell_cycle_and_checkpoints":         "Cell cycle & checkpoints",
    "apoptosis_p53_cell_fate":            "Apoptosis / p53 / cell fate",
    "stress_senescence_cancer_signaling": "Stress, senescence & cancer",
    "uncategorized":                      "Uncategorized",
}

CORR_GENES_PATH = "/homes/shahashka/lucid_cd/nested_cv_results_phenotype_w_apoptosis/stable_features_multiple_correlation_joint.csv"
RF_GENES_PATH = "/homes/shahashka/lucid_cd/nested_cv_results_phenotype_w_apoptosis/stable_features_rf.csv"

def load_data():
    log2fold_df = pd.read_csv(f"{EXPERIMENT}/rpe1_9week_study_experiment2_diffexp_deseq_vs_control_all_dG_W2_adjust.txt", sep="\t")
    log2fold_df = log2fold_df.loc[log2fold_df['padj'] < 0.05] # This is new, I think I should filter by p value. However this means there are no genes that DE across all dose rates
    log2fold_df = log2fold_df.loc[abs(log2fold_df['log2FoldChange']) > 1]

    log2fold_df['Week'] = [float(w[1]) for w in log2fold_df['Week']]
    genes_by_week = {}
    genes_by_dose = {} 
    
    # LOAD DIFFERENTIAL EXPRESSION DATA (temporal)
    for i in np.arange(10):
        week_i = log2fold_df.loc[log2fold_df["Week"] == i]
        genes_week_i = set(week_i["Gene"])
        for g in genes_week_i:
            if g in genes_by_week.keys():
                genes_by_week[g].add(i)
            else:
                genes_by_week[g] = set([i])
                
    # LOAD DIFFERENTIAL EXPRESSION DATA (dose rate)       
    for d in DOSE_RATE_LABELS:
        dose_i = log2fold_df.loc[log2fold_df["Dose"] == f"d{d}"]
        genes_dose_i = dose_i["Gene"]
        genes_by_dose[d] = list(set(genes_dose_i))
        
    # LOAD TPM DATA
    tpm_df = pd.read_csv(f"{EXPERIMENT}/rpe1_9week_study_experiment2_all_tpm.tsv", header=0, sep='\t')
    
    # LOAD CAUSAL DATA
    graphs = []
    graphs_genes_by_dose = {}
    for d,g in GRAPHS.items():
        G = nx.read_gexf(g)
        graphs.append(G)
        graphs_genes_by_dose[d] = list(G.nodes())
        
    # LOAD CAUSAL NEIGHBORHOOD/TF DATA
    genes_100_tfs = {}
    genes_neighborhoods = {}
    for d in DOSE_RATE_LABELS:
        genes_100_tfs[d] = pd.read_csv(f"{CONTEXT_SPECIFIC_GRAPHS}/top_100_dag_gnn_{d}.csv", header=None).iloc[:,0].to_list()
        genes_neighborhoods[d] = pd.read_csv(f"{CONTEXT_SPECIFIC_GRAPHS}/rad_sub_dag_gnn_{d}_ranked.csv", header=0).iloc[:,0].to_list()

    genes_100_tfs['all_doses'] = pd.read_csv(f"{INVARIANT_GRAPHS}/dag_gnn_combined_top_100_tfs.csv", header=None).iloc[:,0].to_list()
    genes_neighborhoods['all_doses_dose_rate'] = pd.read_csv(f"{INVARIANT_GRAPHS}/rad_sub_dag_gnn_combined_ranked.csv", header=0).iloc[:,0].to_list()
    genes_neighborhoods['all_doses_week'] = pd.read_csv(f"{INVARIANT_GRAPHS}/week_sub_dag_gnn_combined_ranked.csv", header=0).iloc[:,0].to_list()
    
    return tpm_df, log2fold_df, graphs_genes_by_dose, genes_by_dose, genes_neighborhoods, genes_100_tfs
def profile_safe(gp, genes, background, **kwargs):
      """Recursively drop unrecognised genes until the query succeeds."""
      if not genes:
          return None
      try:
          return gp.profile(organism="hsapiens", query=genes,
                            background=background, **kwargs)
      except AssertionError:
          if len(genes) == 1:
              print(f"Dropping unrecognised gene: {genes[0]}")
              return None
          mid = len(genes) // 2
          left  = profile_safe(gp, genes[:mid], background, **kwargs)
          right = profile_safe(gp, genes[mid:], background, **kwargs)
          if left is None and right is None:
              return None
          if left is None:  return right
          if right is None: return left
          import pandas as pd
          return pd.concat([left, right]).drop_duplicates()
      
def pathway_enrichment(genes,background_genes, pathways) -> pd.DataFrame:
    """Given a list of genes, perform pathway enrichment using knowledge databases

    Args:
        genes (set(str)): Set of genes with string identifiers
        
    Returns:
        (List[Any]): Return a list of named pathways and scores for each 
    """
    print('call gprofiler')
    # This is the version of gprofiler that was used to generated plots in the paper (since archived)
    # -> base_url="https://biit.cs.ut.ee/gprofiler_archive3/e113_eg59_p19/")
    # for some reason hangs on our gene set...so just using current version 
    gp = GProfiler(return_dataframe=True)
    profile_safe(gp, list(set(genes)), background_genes)
    results = gp.profile(
        organism="hsapiens",
        query=list(set(genes)),
        sources=["GO:BP", "GO:MF", "GO:CC", "KEGG", "REAC", "WP"],
        user_threshold=0.05,
        background=background_genes, 
        significance_threshold_method="fdr"
    )
    print('done call ')

    # Sort by adjusted p-value
    results = results.sort_values("p_value")
    if pathways:
        return results.query("native in @pathways")
    else:
        return results

def random_gene_sanity_check(background_genes, graphs_genes_by_dose,
                             n_random=5, output_dir="."):
    """Sanity check: pathway enrichment on random gene sets of the same size as causal sets.

    For each dose rate, draws ``n_random`` random samples from background_genes
    matching the causal gene set size, runs gProfiler enrichment, and compares
    the -log10(p) distributions against the real causal results.  Also includes
    the top correlative genes (same size) from the joint-ranked correlation list.
    """
    import seaborn as sns
    from global_variables import _fs_axis, _fs_tick, _fs_title, _fs_leg

    print("\n=== Random Gene Sanity Check ===\n")
    np.random.seed(42)
    records = []

    # Load ranked correlative genes
    corr_df = pd.read_csv(CORR_GENES_PATH, header=0)
    all_corr_genes = corr_df["gene"].tolist()

    for d in DOSE_RATES_SORTED:
        causal_genes = graphs_genes_by_dose[d]
        n_genes = len(causal_genes)
        print(f"Dose {d}: causal set size = {n_genes}")

        # Real causal enrichment — top 10 by p-value after filtering term_size < 300
        pe_real = pathway_enrichment(causal_genes, background_genes, None)
        pe_real = pe_real.query("term_size < 300").sort_values("p_value").head(10)
        for _, row in pe_real.iterrows():
            records.append({"dose": d, "source": "Causal",
                            "neg_log10_p": -np.log10(row["p_value"])})

        # Top correlative genes (same size as causal set)
        corr_genes = all_corr_genes[:n_genes]
        pe_corr = pathway_enrichment(corr_genes, background_genes, None)
        pe_corr = pe_corr.query("term_size < 300").sort_values("p_value").head(10)
        for _, row in pe_corr.iterrows():
            records.append({"dose": d, "source": "Corr.",
                            "neg_log10_p": -np.log10(row["p_value"])})
        corr_max = -np.log10(pe_corr["p_value"].min()) if len(pe_corr) > 0 else 0
        print(f"  Correlative (top {n_genes}): {len(pe_corr)} pathways "
              f"(max -log10p = {corr_max:.1f})")

        # Random samples — top 10 (term_size < 300) from each
        for r in range(n_random):
            random_genes = list(np.random.choice(background_genes, size=n_genes,
                                                 replace=False))
            pe_rand = pathway_enrichment(random_genes, background_genes, None)
            pe_rand = pe_rand.query("term_size < 300").sort_values("p_value").head(10)
            for _, row in pe_rand.iterrows():
                records.append({"dose": d, "source": "Random",
                                "neg_log10_p": -np.log10(row["p_value"])})
            max_val = -np.log10(pe_rand["p_value"].min()) if len(pe_rand) > 0 else 0
            print(f"  Random sample {r+1}/{n_random}: {len(pe_rand)} pathways "
                  f"(max -log10p = {max_val:.1f})")

    df = pd.DataFrame(records)

    # Plot: boxplots of top-10 -log10(p) distributions per dose
    source_order = ["Causal", "Corr.", "Random"]
    palette = {"Causal": "orange", "Corr.": "steelblue", "Random": "lightgray"}
    fig, axes = plt.subplots(1, len(DOSE_RATE_LABELS), figsize=(4 * len(DOSE_RATE_LABELS), 5),
                             sharey=True)
    for ax, d in zip(axes, DOSE_RATES_SORTED):
        sub = df[df["dose"] == d]
        sns.boxplot(data=sub, x="source", y="neg_log10_p", ax=ax,
                    hue="source", hue_order=source_order, palette=palette,
                    order=source_order, legend=False)
        ax.set_title(f"{DOSE_RATES_REGRESSION[d]} mGy/hr", fontsize=_fs_axis)
        ax.set_xlabel("", fontsize=_fs_axis)
        ax.set_ylabel("$-\\log_{10}(p)$" if d == DOSE_RATE_LABELS[0] else "", fontsize=_fs_axis)
        ax.tick_params(labelsize=_fs_tick)

    fig.suptitle("Top 10 Pathways (term size < 300): Causal vs Correlative vs Random",
                 fontsize=_fs_title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = os.path.join(output_dir, "random_gene_sanity_check.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved: {fig_path}")

    # Summary table
    summary = df.groupby(["dose", "source"])["neg_log10_p"].agg(
        ["count", "median", "max"]).reset_index()
    print("\nSummary:")
    print(summary.to_string(index=False, float_format="%.1f"))

    return df


def _format_dose_label(d):
    if isinstance(d, (float, np.floating)):
        return f"{float(d):g}"
    return str(d)


def _cmap_gradient_legend_handle(cmap, n_swatches=9, lw=10):
    """Horizontal multi-segment line sampling a colormap (for legend display)."""
    t = np.linspace(0.0, 1.0, n_swatches)
    return tuple(
        Line2D([0], [0], color=cmap(ti), lw=lw, solid_capstyle="butt")
        for ti in t
    )


def generate_plots(datasets, cmap_names=None, pathway_descriptions=None,
                    agnostic_datasets=None, filename="pathway_enrichment_dose_rate",
                    output_dir="."):
    """Grouped horizontal bar plot with one subplot per radiation category.

    Parameters
    ----------
    datasets : dict[str, dict]
        Map dataset display name -> data dict with ``pathways`` (native IDs) and numeric columns (dose rates).
    cmap_names : sequence of str, optional
        Matplotlib colormap names, one per dataset in ``datasets.items()`` order.
    pathway_descriptions : dict[str, str], optional
        Map native pathway id -> full description for y-axis labels. Defaults to ``PATHWAY_DESCRIPTIONS``.
    agnostic_datasets : dict[str, dict], optional
        Map display name -> dict with ``pathways`` (list of native IDs) and ``logp_values``
        (list of floats). Each is drawn as a single bar per pathway using its own colormap
        from ``_DEFAULT_CMAPS`` (continuing after dose-rate dataset colormaps).
    """
    if not isinstance(datasets, dict) or not datasets:
        raise TypeError("datasets must be a non-empty dict mapping name -> data dict")
    if agnostic_datasets is None:
        agnostic_datasets = {}

    def _format_category_legend_name(cat_key):
        return _CATEGORY_DISPLAY_NAMES.get(cat_key, cat_key.replace("_", " "))


    def _ordered_pathways_with_category_labels(pathway_ids, descriptions=None):
        """Order pathways by PATHWAY_CATEGORIES; labels are id + description (no category suffix)."""
        if descriptions is None:
            descriptions = PATHWAY_DESCRIPTIONS
        category_order = list(PATHWAY_CATEGORIES.keys())

        def sort_key(p):
            cat = PATHWAY_TO_CATEGORY.get(p)
            if cat is None:
                return (len(category_order), 999, p)
            ci = category_order.index(cat)
            idx_in_cat = PATHWAY_CATEGORIES[cat].index(p)
            return (ci, idx_in_cat, p)

        ordered = sorted(pathway_ids, key=sort_key)
        labels = []
        categories = []
        for p in ordered:
            cat = PATHWAY_TO_CATEGORY.get(p, "uncategorized")
            categories.append(cat)
            desc = descriptions.get(p, "").strip()
            if desc:
                labels.append(f"{desc} \n {p}")
            else:
                labels.append(p)
        return ordered, labels, categories

    names = list(datasets.keys())
    n_ds = len(names)
    ag_names = list(agnostic_datasets.keys())
    n_ag = len(ag_names)

    # Assign colormaps: dose-rate datasets first, then agnostic datasets
    total_cmaps_needed = n_ds + n_ag
    if cmap_names is None:
        all_cmap_names = [_DEFAULT_CMAPS[i % len(_DEFAULT_CMAPS)] for i in range(total_cmaps_needed)]
    else:
        all_cmap_names = list(cmap_names) + [
            _DEFAULT_CMAPS[i % len(_DEFAULT_CMAPS)]
            for i in range(len(cmap_names), total_cmaps_needed)
        ]
    cmaps = [cm.get_cmap(c) for c in all_cmap_names[:n_ds]]
    ag_cmaps = [cm.get_cmap(all_cmap_names[n_ds + i]) for i in range(n_ag)]
    ag_cmap_names = all_cmap_names[n_ds:n_ds + n_ag]

    first = pd.DataFrame(next(iter(datasets.values()))).set_index("pathways").sort_index(axis=1)
    ordered, display_labels, pathway_categories = _ordered_pathways_with_category_labels(
        first.index.tolist(), descriptions=pathway_descriptions
    )
    dose_cols = list(first.columns)
    print(dose_cols)
    n_col = len(dose_cols)

    dfs = {}
    for name, d in datasets.items():
        df = pd.DataFrame(d).set_index("pathways").sort_index(axis=1).reindex(ordered)
        dfs[name] = df

    # Build agnostic dataset lookups
    ag_vals = {}
    for ag_name, ag_data in agnostic_datasets.items():
        lookup = {}
        for pw, val in zip(ag_data["pathways"], ag_data["logp_values"]):
            lookup[pw] = val
        ag_vals[ag_name] = lookup

    # Group pathways by category for separate subplots
    cat_to_indices = {}
    for idx, (pw, cat) in enumerate(zip(ordered, pathway_categories)):
        cat_to_indices.setdefault(cat, []).append(idx)

    # Preserve category ordering from PATHWAY_CATEGORIES
    cat_order = [k for k in list(PATHWAY_CATEGORIES.keys()) + ["uncategorized"]
                 if k in cat_to_indices]
    n_cats = len(cat_order)

    # Compute subplot heights proportional to number of pathways per category
    cat_n_pathways = [len(cat_to_indices[c]) for c in cat_order]
    bar_height = 1.0 / n_col
    step = bar_height * n_col
    agnostic_step = bar_height * n_ag
    pathway_gap = 0.45
    group_pitch = n_ds * step + agnostic_step + pathway_gap

    height_ratios = [max(n_pw * group_pitch, 1.5) for n_pw in cat_n_pathways]
    total_fig_height = max(10, sum(height_ratios) * 0.7 + 3)

    fig, axes = plt.subplots(
        n_cats, 1,
        figsize=(24, total_fig_height),
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_cats == 1:
        axes = [axes]

    for cat_idx, cat_key in enumerate(cat_order):
        ax = axes[cat_idx]
        indices = cat_to_indices[cat_key]
        n_pw = len(indices)

        for i_local, i_global in enumerate(indices):
            pw = ordered[i_global]
            base_y = i_local * group_pitch
            # Dose-rate dataset bars
            for k, name in enumerate(names):
                cmap = cmaps[k]
                for j, col in enumerate(dose_cols):
                    y = base_y + k * step + j * bar_height
                    w = float(dfs[name].loc[pw][col])
                    t = j / max(n_col - 1, 1) if n_col > 1 else 0.5
                    color = cmap(t)
                    ax.barh(y, w, height=bar_height, color=color, edgecolor="white", linewidth=0.4)
            # Agnostic dataset bars (one bar each)
            for a, ag_name in enumerate(ag_names):
                y = base_y + n_ds * step + a * bar_height
                w = ag_vals[ag_name].get(pw, 0)
                color = ag_cmaps[a](0.5)
                ax.barh(y, w, height=bar_height, color=color, edgecolor="white", linewidth=0.4)

        total_height = n_ds * step + agnostic_step
        y_centers = [
            i_local * group_pitch + (total_height - bar_height) / 2.0
            for i_local in range(n_pw)
        ]
        local_labels = [display_labels[i_global] for i_global in indices]
        local_cats = [pathway_categories[i_global] for i_global in indices]

        ax.set_yticks(y_centers)
        ax.set_yticklabels(local_labels)
        ax.tick_params(axis="y", labelsize=_fs_tick)
        ax.tick_params(axis="x", labelsize=_fs_tick)
        for tick, cat in zip(ax.get_yticklabels(), local_cats):
            tick.set_color(CATEGORY_COLORS.get(cat, "#333333"))
        ax.invert_yaxis()
        ax.set_title(_format_category_legend_name(cat_key), fontsize=_fs_title,
                      fontweight="bold", color=CATEGORY_COLORS.get(cat_key, "#333333"))
        if cat_idx == n_cats - 1:
            ax.set_xlabel(r"$-\log_{10}(p)$", fontsize=_fs_axis)

    # Share x-axis range across all subplots
    x_max = max(a.get_xlim()[1] for a in axes)
    for a in axes:
        a.set_xlim(0, x_max)

    # Dose-rate legend handles
    dose_handles = []
    dose_labels_list = []
    for k, name in enumerate(names):
        cmap = cmaps[k]
        for j, col in enumerate(dose_cols):
            t = j / max(n_col - 1, 1) if n_col > 1 else 0.5
            dose_handles.append(
                Line2D([0], [0], color=cmap(t), lw=8, solid_capstyle="butt")
            )
            dose_labels_list.append(f"{name}: {_format_dose_label(col)} mGy/hr")

    # Agnostic dataset legend handles
    for a_idx, ag_name in enumerate(ag_names):
        dose_handles.append(
            Line2D([0], [0], color=ag_cmaps[a_idx](0.5), lw=8, solid_capstyle="butt")
        )
        dose_labels_list.append(f"{ag_name} (dose-agnostic)")

    ncol_leg = min(n_col, 5)

    fig.subplots_adjust(left=0.28, right=0.78, bottom=0.10, top=0.95, hspace=0.4)

    # Dose rate legend at bottom of last subplot
    # axes[-1].legend(
    #     dose_handles,
    #     dose_labels_list,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, -0.12),
    #     ncol=ncol_leg,
    #     frameon=True,
    #     fontsize=_fs_leg,
    #     title="Dose rate",
    #     title_fontsize=_fs_leg_title,
    # )

    # Colormap legend on right side of first subplot
    cmap_legend_handles = [
        _cmap_gradient_legend_handle(cmaps[k]) for k in range(n_ds)
    ]
    cmap_legend_labels = [
        f"{names[k]} ({all_cmap_names[k]})" for k in range(n_ds)
    ]
    for a_idx, ag_name in enumerate(ag_names):
        cmap_legend_handles.append(
            Line2D([0], [0], color=ag_cmaps[a_idx](0.5), lw=10, solid_capstyle="butt")
        )
        cmap_legend_labels.append(f"{ag_name} ({ag_cmap_names[a_idx]})")
    axes[0].legend(
        cmap_legend_handles,
        cmap_legend_labels,
        handler_map={tuple: HandlerTuple(pad=0)},
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=_fs_leg,
        title="Dataset colormap (light \u2192 dark = low \u2192 high dose)",
        title_fontsize=_fs_leg_title,
    )

    _save_kw = dict(bbox_inches="tight", pad_inches=0.5)
    plt.savefig(f"{output_dir}/{filename}.png", format="png", dpi=300, **_save_kw)
    plt.savefig(f"{output_dir}/{filename}.svg", format="svg", dpi=300, **_save_kw)


def generate_top_plots(datasets, cmap_names=None, filename="top10_pathway_enrichment",
                       agnostic_datasets=None, output_dir="."):
    """Stacked horizontal bar plots of top-10 enriched pathways per dose rate.

    Creates one subplot per entry in *datasets*. Each bar represents a
    pathway; stacked segments show the contribution from each dose rate
    where that pathway appeared in the top 10.

    Parameters
    ----------
    datasets : dict[str, dict]
        Map display name -> dict keyed by dose-rate letter. Each value has
        keys ``pathways``, ``names``, ``sources``, ``neg_log10_p``.
    cmap_names : sequence of str, optional
        Colormap names, one per dataset. Defaults cycle through _DEFAULT_CMAPS.
    filename : str, optional
        Base filename (no extension) for saved figures.
    agnostic_datasets : dict[str, dict], optional
        Map display name -> dict with ``pathways``, ``names``, ``sources``,
        ``neg_log10_p`` (not keyed by dose rate). Each gets its own subplot
        with a single bar per pathway using a solid color from _DEFAULT_CMAPS.
    """
    if agnostic_datasets is None:
        agnostic_datasets = {}
    names = list(datasets.keys())
    n_panels = len(names) + len(agnostic_datasets)
    if cmap_names is None:
        cmap_names = [_DEFAULT_CMAPS[i % len(_DEFAULT_CMAPS)] for i in range(n_panels)]
    elif len(cmap_names) < n_panels:
        cmap_names = list(cmap_names) + [
            _DEFAULT_CMAPS[i % len(_DEFAULT_CMAPS)]
            for i in range(len(cmap_names), n_panels)
        ]
    cmaps = [cm.get_cmap(c) for c in cmap_names[:n_panels]]

    n_ds = len(names)
    ag_names = list(agnostic_datasets.keys())
    panels = [(name, datasets[name], cmaps[k]) for k, name in enumerate(names)]

    first_data = next(iter(datasets.values()))
    dose_keys = sorted(first_data.keys(), key=lambda d: DOSE_RATES_REGRESSION.get(d, 0))
    print(dose_keys)
    n_dose = len(dose_keys)

    # --- build per-panel DataFrames (union of pathways x dose rates) ---
    panel_dfs = []
    panel_labels = []
    panel_is_agnostic = []
    for title, top_data, _ in panels:
        # Collect union of pathways; map pathway id -> display label
        pw_to_label = {}
        for d in dose_keys:
            for pw, nm, src in zip(
                top_data[d]["pathways"],
                top_data[d]["names"],
                top_data[d]["sources"],
            ):
                if pw not in pw_to_label:
                    pw_to_label[pw] = f"{nm} \n {pw}"

        all_pws = list(pw_to_label.keys())
        # Build a DataFrame: rows = pathways, cols = dose rates
        rows = {pw: {d: 0.0 for d in dose_keys} for pw in all_pws}
        for d in dose_keys:
            for pw, val in zip(top_data[d]["pathways"], top_data[d]["neg_log10_p"]):
                rows[pw][d] = val

        df = pd.DataFrame(rows).T  # index = pathway id, columns = dose letters
        # Sort by total enrichment (most significant at top)
        df["_total"] = df.sum(axis=1)
        df = df.sort_values("_total", ascending=False).drop(columns="_total").iloc[0:5] # switch to 5
        labels = [pw_to_label[pw] for pw in df.index]
        panel_dfs.append(df)
        panel_labels.append(labels)
        panel_is_agnostic.append(False)

    # --- build agnostic panel DataFrames ---
    ag_panels = []
    for a_idx, ag_name in enumerate(ag_names):
        ag_data = agnostic_datasets[ag_name]
        ag_cmap = cmaps[n_ds + a_idx]
        ag_panels.append((ag_name, ag_data, ag_cmap))
        pw_to_label = {}
        for pw, nm, src in zip(ag_data["pathways"], ag_data["names"], ag_data["sources"]):
            if pw not in pw_to_label:
                pw_to_label[pw] = f"{nm} \n {pw}"
        vals = {pw: v for pw, v in zip(ag_data["pathways"], ag_data["neg_log10_p"])}
        df = pd.DataFrame({"value": vals}).sort_values("value", ascending=True)
        labels = [pw_to_label[pw] for pw in df.index]
        panel_dfs.append(df)
        panel_labels.append(labels)
        panel_is_agnostic.append(True)

    max_pathways = max(len(df) for df in panel_dfs)
    fig_height = max(20, 0.4 * max_pathways)
    fig_width = 20 * n_panels
    print(fig_height, fig_width)
    fig, axes = plt.subplots(n_panels, 1, figsize=(fig_height, fig_width), sharex=True)
    if n_panels == 1:
        axes = [axes]

    all_panels = panels + ag_panels
    for ax, (title, _, cmap_obj), df, labels, is_ag in zip(
        axes, all_panels, panel_dfs, panel_labels, panel_is_agnostic
    ):
        n_pw = len(df)
        y_pos = np.arange(n_pw)

        if is_ag:
            # Single bar per pathway using mid-colormap color
            color = cmap_obj(0.5)
            ax.barh(
                y_pos, df["value"].values, height=0.7,
                color=color, edgecolor="white", linewidth=0.4,
            )
        else:
            bar_height = 0.7 / n_dose
            for j, d in enumerate(dose_keys):
                t = j / max(n_dose - 1, 1) if n_dose > 1 else 0.5
                color = cmap_obj(t)
                widths = df[d].values
                offset = (j - (n_dose - 1) / 2) * bar_height
                ax.barh(
                    y_pos + offset, widths, height=bar_height,
                    color=color, edgecolor="white", linewidth=0.4,
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=_fs_axis+5)
        ax.invert_yaxis()
        ax.set_xlabel(r"$-\log_{10}(p)$", fontsize=_fs_axis+2)
        ax.set_title(title, fontsize=_fs_title+6, fontweight="bold")
        ax.tick_params(axis="x", labelsize=_fs_axis+2)

    # Shared x-axis range so magnitudes are directly comparable
    x_max = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(0, x_max)

    # Dose-rate legend matching generate_plots style (Line2D handles, per-panel colors)
    dose_handles = []
    dose_labels = []
    panel_names = [title for title, _, _ in panels]
    for k, (_, _, cmap_obj) in enumerate(panels):
        for j, d in enumerate(dose_keys):
            t = j / max(n_dose - 1, 1)
            dose_handles.append(
                Line2D([0], [0], color=cmap_obj(t), lw=8, solid_capstyle="butt")
            )
            dose_labels.append(f"{panel_names[k]}: {_format_dose_label(DOSE_RATES_REGRESSION[d])} mGy/hr")

    ncol_leg = min(n_dose, 5)
    # fig.legend(
    #     dose_handles, dose_labels,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, -0.01),
    #     ncol=ncol_leg,
    #     frameon=True,
    #     fontsize=_fs_leg,
    #     title="Dose rate",
    #     title_fontsize=_fs_leg_title,
    # )
    # Agnostic dataset legend handles
    for a_idx, ag_name in enumerate(ag_names):
        dose_handles.append(
            Line2D([0], [0], color=cmaps[n_ds + a_idx](0.5), lw=8, solid_capstyle="butt")
        )
        dose_labels.append(f"{ag_name} (dose-agnostic)")

    # Colormap legend — right side, below category legend
    cmap_legend_handles = [
        _cmap_gradient_legend_handle(cmaps[k]) for k in range(n_ds)
    ]
    cmap_legend_labels = [
        f"{names[k]} ({cmap_names[k]})" for k in range(n_ds)
    ]
    for a_idx, ag_name in enumerate(ag_names):
        cmap_legend_handles.append(
            Line2D([0], [0], color=cmaps[n_ds + a_idx](0.5), lw=10, solid_capstyle="butt")
        )
        cmap_legend_labels.append(f"{ag_name} ({cmap_names[n_ds + a_idx]})")
    # leg_cmap = ax.legend(
    #     cmap_legend_handles,
    #     cmap_legend_labels,
    #     handler_map={tuple: HandlerTuple(pad=0)},
    #     loc="upper left",
    #     bbox_to_anchor=(1.02, 0.55),
    #     borderaxespad=0.0,
    #     frameon=True,
    #     fontsize=_fs_leg,
    #     title="Dataset colormap (light \u2192 dark = low \u2192 high dose)",
    #     title_fontsize=_fs_leg_title,
    # )

    fig.subplots_adjust(wspace=0.55, bottom=0.12, top=0.93)
    _save_kw = dict(bbox_inches="tight", pad_inches=0.2)
    # plt.savefig(f"{output_dir}/{filename}.pdf", format="pdf", **_save_kw)
    plt.savefig(f"{output_dir}/{filename}.png", format="png", dpi=300, **_save_kw)
    plt.savefig(f"{output_dir}/{filename}.svg", format="svg", dpi=300, **_save_kw)


def venn_diagrams(graphs_genes_by_dose, genes_by_dose, rf_genes=None, corr_genes=None, output_dir="."):
    """Venn diagrams comparing gene sets across dose rates and methods.

    Produces three figures:
    1. Pairwise Venn diagrams across dose rates for Causal Graph gene sets
    2. Pairwise Venn diagrams across dose rates for DE gene sets
    3. Causal Graph vs DE at each dose rate

    Parameters
    ----------
    graphs_genes_by_dose : dict[str, list]
        Causal graph gene sets keyed by dose rate letter.
    genes_by_dose : dict[str, list]
        DE gene sets keyed by dose rate letter.
    """
    _save_kw = dict(bbox_inches="tight", pad_inches=0.2)
    
    pairs = list(combinations(DOSE_RATES_SORTED, 2))  # 10 pairs

    def _dose_label(d):
        return f"{DOSE_RATES_REGRESSION[d]} mGy/hr"

    # --- Figure 1 & 2: pairwise across dose rates for each method ---
    _venn_color_causal = to_hex(cm.get_cmap("OrangesDark")(0.5))
    _venn_color_de = to_hex(cm.get_cmap("Blues")(0.5))
    _venn_method_colors = {
        "Causal Graph": _venn_color_causal,
        "Differential Expression": _venn_color_de,
    }
    method_data = [
        ("Causal Graph", graphs_genes_by_dose, "venn_causal_pairwise"),
        ("Differential Expression", genes_by_dose, "venn_de_pairwise"),
    ]
    for method_name, gene_dict, fname in method_data:
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        axes_flat = axes.flatten()
        for idx, (d1, d2) in enumerate(pairs):
            ax = axes_flat[idx]
            set1 = set(gene_dict[d1])
            set2 = set(gene_dict[d2])
            _mc = _venn_method_colors[method_name]
            v = venn2(
                [set1, set2],
                set_labels=(_dose_label(d1), _dose_label(d2)),
                set_colors=(_mc, _mc),
                ax=ax,
            )
            # for text in v.set_labels:
            #     if text:
            #         text.set_fontsize(14)
            # for text in v.subset_labels:
            #     if text:
                    # text.set_fontsize(18)
            ax.set_title(f"{_dose_label(d1)} vs {_dose_label(d2)}", fontsize=12, fontweight="bold")
        fig.suptitle(f"{method_name} — Pairwise Gene Set Overlap", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        # plt.savefig(f"{output_dir}/{fname}.pdf", format="pdf", **_save_kw)
        plt.savefig(f"{output_dir}/{fname}.png", format="png", dpi=300, **_save_kw)
        plt.savefig(f"{output_dir}/{fname}.svg", format="svg", dpi=300, **_save_kw)

    # --- Figure 3: Causal vs DE at each dose rate ---
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for idx, d in enumerate(DOSE_RATES_SORTED):
        ax = axes[idx]
        set_causal = set(graphs_genes_by_dose[d])
        set_de = set(genes_by_dose[d])
        v = venn2(
            [set_causal, set_de],
            set_labels=("Causal Graph", "Diff. Expression"),
            set_colors=(_venn_color_causal, _venn_color_de),
            ax=ax,
        )
        for text in v.set_labels:
            if text:
                text.set_fontsize(18)
        for text in v.subset_labels:
            if text:
                text.set_fontsize(18)
        ax.set_title(_dose_label(d), fontsize=_fs_leg_title, fontweight="bold")
    fig.suptitle("Causal Graph vs Differential Expression", fontsize=_fs_title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    # plt.savefig(f"{output_dir}/venn_causal_vs_de.pdf", format="pdf", **_save_kw)
    plt.savefig(f"{output_dir}/venn_causal_vs_de.png", format="png", dpi=300, **_save_kw)
    plt.savefig(f"{output_dir}/venn_causal_vs_de.svg", format="svg", dpi=300, **_save_kw)

    # --- Figure 4: Causal vs DE vs RandomForest at each dose rate ---
    if rf_genes is not None:
        _venn_color_rf = to_hex(cm.get_cmap("Greens")(0.5))
        set_rf = set(rf_genes)
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for idx, d in enumerate(DOSE_RATES_SORTED):
            ax = axes[idx]
            set_causal = set(graphs_genes_by_dose[d])
            set_de = set(genes_by_dose[d])
            v = venn3(
                [set_causal, set_de, set_rf],
                set_labels=("Causal Graph", "Diff. Expression", "RandomForest"),
                set_colors=(_venn_color_causal, _venn_color_de, _venn_color_rf),
                ax=ax,
            )
            for text in v.set_labels:
                if text:
                    text.set_fontsize(14)
            for text in v.subset_labels:
                if text:
                    text.set_fontsize(14)
            ax.set_title(_dose_label(d), fontsize=_fs_leg_title, fontweight="bold")
        fig.suptitle("Causal Graph vs Differential Expression vs RandomForest", fontsize=_fs_title, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(f"{output_dir}/venn_causal_vs_de_vs_rf.png", format="png", dpi=300, **_save_kw)
        plt.savefig(f"{output_dir}/venn_causal_vs_de_vs_rf.svg", format="svg", dpi=300, **_save_kw)

    # --- Figure 5: Causal vs DE vs Correlation at each dose rate ---
    if corr_genes is not None:
        _venn_color_corr = to_hex(cm.get_cmap("Purples")(0.5))
        set_corr = set(corr_genes)
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        for idx, d in enumerate(DOSE_RATES_SORTED):
            ax = axes[idx]
            set_causal = set(graphs_genes_by_dose[d])
            set_de = set(genes_by_dose[d])
            v = venn3(
                [set_causal, set_de, set_corr],
                set_labels=("Causal Graph", "Diff. Expression", "Correlation"),
                set_colors=(_venn_color_causal, _venn_color_de, _venn_color_corr),
                ax=ax,
            )
            for text in v.set_labels:
                if text:
                    text.set_fontsize(14)
            for text in v.subset_labels:
                if text:
                    text.set_fontsize(14)
            ax.set_title(_dose_label(d), fontsize=_fs_leg_title, fontweight="bold")
        fig.suptitle("Causal Graph vs Differential Expression vs Correlation", fontsize=_fs_title, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        plt.savefig(f"{output_dir}/venn_causal_vs_de_vs_corr.png", format="png", dpi=300, **_save_kw)
        plt.savefig(f"{output_dir}/venn_causal_vs_de_vs_corr.svg", format="svg", dpi=300, **_save_kw)

    # --- UpSet plots: all dose rates at once ---
    _label_to_dose_val = {_dose_label(d): DOSE_RATES_REGRESSION[d] for d in DOSE_RATES_SORTED}

    def _build_memberships(gene_dict):
        """Return a list of (membership_tuple, gene) pairs for upsetplot."""
        gene_to_sets = {}
        for d in DOSE_RATES_SORTED:
            for g in gene_dict[d]:
                gene_to_sets.setdefault(g, set()).add(_dose_label(d))
        memberships = []
        for g, sets in gene_to_sets.items():
            memberships.append(tuple(sorted(sets, key=lambda s: _label_to_dose_val[s])))
        return memberships

    for method_name, gene_dict, fname in method_data:
        memberships = _build_memberships(gene_dict)
        # Count unique genes per membership group to avoid non-unique index
        from collections import Counter
        counts = Counter(memberships)
        upset_data = from_memberships(
            list(counts.keys()),
            data=list(counts.values()),
        )
        upset = UpSet(upset_data, show_counts=True, sort_by="cardinality",
                      facecolor=_venn_method_colors[method_name],
                      element_size=None)
        fig = plt.figure(figsize=(14, 8))
        upset.plot(fig=fig)
        for ax in fig.axes:
            ax.tick_params(labelsize=10)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontsize(10)
            if ax.get_ylabel():
                ax.yaxis.label.set_fontsize(10)
            if ax.get_xlabel():
                ax.xaxis.label.set_fontsize(10)
            for txt in ax.texts:
                txt.set_fontsize(10)
        fig.suptitle(f"{method_name} — Gene Set Intersections Across Dose Rates",
                     fontsize=16, fontweight="bold", y=1.02)
        # plt.savefig(f"{output_dir}/{fname}_upset.pdf", format="pdf", **_save_kw)
        plt.savefig(f"{output_dir}/{fname}_upset.png", format="png", dpi=300, **_save_kw)
        plt.savefig(f"{output_dir}/{fname}_upset.svg", format="svg", dpi=300, **_save_kw)

# RESHAPE DATA FOR PLOTTING (native IDs only; descriptions come from PATHWAY_DESCRIPTIONS)
def _build_radiation_logp(pe_results):
    logp_values = []
    for p in ALL_RADIATION_PATHWAY_IDS:
        match = pe_results.loc[pe_results['native'] == p]
        if len(match) > 0:
            logp_values.append(-np.log10(match['p_value'].values[0]))
        else:
            logp_values.append(0)
    return logp_values

def create_latex_table(pe, outdir, fname, name):
    vals = -np.log10(pe['p_value'].values)
    # --- Save top-10 to CSV and LaTeX ---
    top10_df = pe[['name', 'native', 'source', 'p_value', 'term_size',
                    'intersection_size']].copy()
    top10_df['-log10(p)'] = vals
    top10_df.to_csv(f"{outdir}/{fname}.csv", index=False)

    latex_df = top10_df[['-log10(p)', 'term_size',
                            'intersection_size']].copy()
    latex_df.insert(0, 'Pathway',
                    top10_df['name'] + r' \newline {\small ('
                    + top10_df['native'] + ')}')
    latex_df.columns = ['Pathway', '$-\\log_{10}(p)$',
                        'Term Size', 'Intersection']
    latex_df['$-\\log_{10}(p)$'] = latex_df['$-\\log_{10}(p)$'].apply(
        lambda x: f"{x:.2f}")
    latex_str = latex_df.to_latex(
        index=False, escape=False,
        caption=f"Top 10 Enriched Pathways — {name} Gene Set",
        label=f"tab:{fname}",
        column_format="p{6cm} r r r",
    )
    with open(f"{outdir}/{fname}.tex", "w") as f:
        f.write(latex_str)
            
def invariant_plots(outdir, causal_invariant_genes, de_invariant_genes,
                             background_genes):
    """Venn diagrams and pathway enrichment for invariant gene sets.

    Produces:
    1. Venn: causal invariant vs DE invariant genes
    2. Radiation pathway bar plot for each invariant set
    3. Top-10 enriched pathway bar plot for each invariant set
    """
    _save_kw = dict(bbox_inches="tight", pad_inches=0.2)
    _venn_color_causal = to_hex(cm.get_cmap("OrangesDark")(0.5))
    _venn_color_de = to_hex(cm.get_cmap("Blues")(0.5))

    causal_set = set(causal_invariant_genes)
    de_set = set(de_invariant_genes)

    # --- Venn: Causal invariant vs DE invariant ---
    fig, ax = plt.subplots(figsize=(6, 6))
    v = venn2(
        [causal_set, de_set],
        set_labels=("Causal Invariant", "Diff. Exp. Invariant"),
        set_colors=(_venn_color_causal, _venn_color_de),
        ax=ax,
    )
    for text in v.set_labels:
        if text:
            text.set_fontsize(18)
    for text in v.subset_labels:
        if text:
            text.set_fontsize(18)
    ax.set_title("Causal Invariant vs DE Invariant Genes",
                 fontsize=_fs_title, fontweight="bold")
    fig.tight_layout()
    plt.savefig(f"{outdir}/venn_causal_invariant_vs_de_invariant.png", format="png", dpi=300, **_save_kw)
    plt.savefig(f"{outdir}/venn_causal_invariant_vs_de_invariant.svg", format="svg", dpi=300, **_save_kw)
    plt.close()

    # --- Pathway enrichment ---
    pe_causal_inv = pathway_enrichment(list(causal_set), background_genes, None)
    pe_de_inv = pathway_enrichment(list(de_set), background_genes, None)
    print(f"Causal invariant enrichment: {len(pe_causal_inv)} pathways")
    print(f"DE invariant enrichment: {len(pe_de_inv)} pathways")


    pw_labels = [PATHWAY_DESCRIPTIONS.get(p, p) for p in ALL_RADIATION_PATHWAY_IDS]
    causal_logp = _build_radiation_logp(pe_causal_inv)
    de_logp = _build_radiation_logp(pe_de_inv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    y_pos = np.arange(len(pw_labels))
    for ax, vals, name, color in [
        (axes[0], causal_logp, "Causal Invariant", _venn_color_causal),
        (axes[1], de_logp, "DE Invariant", _venn_color_de),
    ]:
        ax.barh(y_pos, vals, color=color, edgecolor="white", linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(pw_labels, fontsize=_fs_tick)
        ax.invert_yaxis()
        ax.set_xlabel(r"$-\log_{10}(p)$", fontsize=_fs_axis)
        ax.set_title(name, fontsize=_fs_leg_title, fontweight="bold")
    fig.suptitle("Radiation Pathways — Invariant Gene Sets",
                 fontsize=_fs_title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{outdir}/invariant_radiation_pathways.png", format="png", dpi=300, **_save_kw)
    plt.savefig(f"{outdir}/invariant_radiation_pathways.svg", format="svg", dpi=300, **_save_kw)
    plt.close()

    # --- Top-10 enriched pathways for each invariant set ---
    for name, pe_results, color in [
        ("Causal Invariant", pe_causal_inv, _venn_color_causal),
        ("DE Invariant", pe_de_inv, _venn_color_de),
    ]:
        pe = pe_results.sort_values(by='p_value', ascending=True).query('term_size < 300')
        pe = pe.iloc[0:10]
        top_labels = [f"{nm} \n {pw}" for pw, nm in
                      zip(pe['native'].values, pe['name'].values)]
        vals = -np.log10(pe['p_value'].values)

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(top_labels))
        ax.barh(y_pos, vals, color=color, edgecolor="white", linewidth=0.4)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_labels, fontsize=_fs_tick)
        ax.invert_yaxis()
        ax.set_xlabel(r"$-\log_{10}(p)$", fontsize=_fs_axis)
        ax.set_title(f"Top 10 Enriched Pathways — {name} Gene Set",
                     fontsize=_fs_title, fontweight="bold")
        fig.tight_layout()
        fname = f"invariant_top10_{name.lower().replace(' ', '_')}"
        plt.savefig(f"{outdir}/{fname}.png", format="png", dpi=300, **_save_kw)
        plt.savefig(f"{outdir}/{fname}.svg", format="svg", dpi=300, **_save_kw)
        plt.close()

        create_latex_table(pe, outdir, fname, name)

    # --- Top correlation genes (matched size to causal invariant) ---
    n_causal_inv = len(causal_set)
    corr_genes = pd.read_csv(CORR_GENES_PATH, header=0)['gene'].values[:n_causal_inv]

    pe_corr = pathway_enrichment(corr_genes, background_genes, None)
    print(f"Correlation genes enrichment: {len(pe_corr)} pathways")

    _venn_color_corr = to_hex(cm.get_cmap("Purples")(0.5))


    # Top-10 pathways
    pe_corr_top = pe_corr.sort_values(by='p_value', ascending=True).query('term_size < 300')
    pe_corr_top = pe_corr_top.iloc[0:10]
    fname = "invariant_top10_corr"
    # create_latex_table(pe_corr_top, outdir, fname, name='Corr.')

    top_labels = [f"{nm} \n {pw}" for pw, nm in
                  zip(pe_corr_top['native'].values, pe_corr_top['name'].values)]
    vals = -np.log10(pe_corr_top['p_value'].values)
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(top_labels))
    ax.barh(y_pos, vals, color=_venn_color_corr,
            edgecolor="white", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels, fontsize=_fs_tick)
    ax.invert_yaxis()
    ax.set_xlabel(r"$-\log_{10}(p)$", fontsize=_fs_axis)
    ax.set_title(f"Top 10 Enriched Pathways — Top {n_causal_inv} Correlation Genes",
                 fontsize=_fs_title, fontweight="bold")
    fig.tight_layout()
    plt.savefig(f"{outdir}/correlation_top10_pathways.png", format="png", dpi=300, **_save_kw)
    plt.savefig(f"{outdir}/correlation_top10_pathways.svg", format="svg", dpi=300, **_save_kw)
    plt.close()

    # --- Housekeeping gene overlap ---
    with open(HK_PATH, "r") as f:
        hk_data = json.load(f)
    hk_genes = set(hk_data["HSIAO_HOUSEKEEPING_GENES"]["geneSymbols"])

    # Load perfect genes per dose rate
    perfect_genes_by_dose = {}
    for d in DOSE_RATE_LABELS:
        path = os.path.join("structure_analysis", f"perfect_genes_{d}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as fh:
                perfect_genes_by_dose[d] = set(pickle.load(fh))

    # Compute overlaps
    causal_hk = causal_set & hk_genes
    de_hk = de_set & hk_genes
    print(f"\n--- Housekeeping Gene Overlap ---")
    print(f"Housekeeping genes: {len(hk_genes)}")
    print(f"Causal invariant ∩ housekeeping: {len(causal_hk)}/{len(causal_set)} "
          f"({100*len(causal_hk)/len(causal_set):.1f}%)")
    print(f"DE invariant ∩ housekeeping: {len(de_hk)}/{len(de_set)} "
          f"({100*len(de_hk)/len(de_set):.1f}%)")

    for d in DOSE_RATE_LABELS:
        if d in perfect_genes_by_dose:
            perf = perfect_genes_by_dose[d]
            perf_hk = perf & hk_genes
            print(f"Perfect {d} ({DOSE_RATES_REGRESSION[d]} mGy/hr) ∩ housekeeping: "
                  f"{len(perf_hk)}/{len(perf)} ({100*len(perf_hk)/len(perf):.1f}%)")

    # Fisher's exact test: enrichment/depletion of housekeeping genes
    from scipy.stats import fisher_exact
    bg_set = set(background_genes)
    n_bg = len(bg_set)
    n_hk_in_bg = len(hk_genes & bg_set)

    rows = []
    for name, gene_set in [("Causal invariant", causal_set), ("Diff. Exp. invariant", de_set)]:
        overlap = len(gene_set & hk_genes)
        n_set = len(gene_set)
        # contingency table: [[overlap, set_only], [hk_only, neither]]
        table = [[overlap, n_set - overlap],
                 [n_hk_in_bg - overlap, n_bg - n_set - n_hk_in_bg + overlap]]
        odds, p_enrich = fisher_exact(table, alternative="greater")
        _, p_deplete = fisher_exact(table, alternative="less")
        rows.append({"Gene Set": name, "Size": n_set,
                      "HK Overlap": overlap,
                      "HK Fraction": f"{100*overlap/n_set:.1f}%",
                      "Odds Ratio": f"{odds:.2f}",
                      "p (enriched)": f"{p_enrich:.2e}",
                      "p (depleted)": f"{p_deplete:.2e}"})

    hk_df = pd.DataFrame(rows)
    print("\n" + hk_df.to_string(index=False))
    hk_df.to_csv(f"{outdir}/housekeeping_overlap.csv", index=False)

    # Bar chart: housekeeping fraction per gene set
    bar_names = [r["Gene Set"] for r in rows]
    bar_fracs = [float(r["HK Fraction"].strip("%")) for r in rows]
    bar_colors = ([_venn_color_causal, _venn_color_de])
                #   + [to_hex(cm.get_cmap("Reds")(0.4 + 0.1 * i))
                #      for i in range(len(rows) - 2)])

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(bar_names))
    ax.barh(y_pos, bar_fracs, color=bar_colors, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_names, fontsize=_fs_tick)
    ax.invert_yaxis()
    ax.set_xlabel("% Housekeeping Genes", fontsize=_fs_axis)
    ax.set_title("Housekeeping Gene Fraction",
                 fontsize=_fs_title, fontweight="bold")
    # Reference line: background rate
    # bg_rate = 100 * n_hk_in_bg / n_bg
    # ax.axvline(bg_rate, color="grey", linestyle="--", linewidth=1.2)
    # ax.text(bg_rate + 0.3, len(bar_names) - 0.5,
    #         f"Background: {bg_rate:.1f}%", fontsize=_fs_tick, color="grey")
    fig.tight_layout()
    plt.savefig(f"{outdir}/housekeeping_fraction.png", format="png", dpi=300, **_save_kw)
    plt.savefig(f"{outdir}/housekeeping_fraction.svg", format="svg", dpi=300, **_save_kw)
    plt.close()


def housekeeping_parent_enrichment(output_dir=".", graph_pattern="invariant_subgraph_{d}_annotated.gexf",
                                 filename="top10_hk_parents_enrichment"):
    """Pathway enrichment of HK-specific ancestors vs non-ancestors in causal graphs.

    HK-specific ancestors are nodes whose descendants are significantly
    enriched for housekeeping genes (Fisher's exact test, p < 0.05).
    Differential enrichment then identifies pathways over-represented
    in HK-specific ancestors relative to non-ancestors.
    """

    subgraph_dir = os.path.join(output_dir, "..", "structure_analysis")

    # Load housekeeping genes
    with open(HK_PATH, "r") as f:
        hk_data = json.load(f)
    hk_genes = set(hk_data["HSIAO_HOUSEKEEPING_GENES"]["geneSymbols"])

    # Background: all genes in the TPM expression matrix
    tpm_df = pd.read_csv(
        f"{EXPERIMENT}/cd_tpm_matrix_combined_dose_rate.csv",
        header=0, nrows=0,
    )
    background_genes = [c for c in tpm_df.columns if c not in ("dose_rate", "week")]

    top_data_specific = {}
    top_data_non = {}
    top_data_diff = {}
    _save_kw = dict(bbox_inches="tight", pad_inches=0.2)

    for d in DOSE_RATE_LABELS:
        sub_path = os.path.join(subgraph_dir, graph_pattern.format(d=d))
        if not os.path.exists(sub_path):
            print(f"  Dose {d}: annotated subgraph not found, skipping")
            continue
        G = nx.read_gexf(sub_path)

        hk_nodes = set(n for n in G.nodes() if n in hk_genes)
        gene_nodes = set(n for n in G.nodes() if n not in {"radiation", "dose_rate", "week"})
        n_total = len(gene_nodes)
        n_hk_total = len(hk_nodes)

        # --- Step 1: Identify HK-specific ancestors ---
        # For each ancestor, test if its descendants are enriched for HK genes
        all_ancestors = set()
        for node in hk_nodes:
            all_ancestors.update(nx.ancestors(G, node))
        all_ancestors -= hk_nodes

        hk_specific_ancestors = set()
        for anc in all_ancestors:
            descendants = nx.descendants(G, anc) & gene_nodes
            n_desc = len(descendants)
            if n_desc == 0:
                continue
            n_desc_hk = len(descendants & hk_nodes)
            n_non_desc = n_total - n_desc
            n_non_desc_hk = n_hk_total - n_desc_hk
            _, p = fisher_exact([
                [n_desc_hk, n_desc - n_desc_hk],
                [n_non_desc_hk, n_non_desc - n_non_desc_hk],
            ], alternative="greater")
            if p < 0.05:
                hk_specific_ancestors.add(anc)

        non_ancestors = gene_nodes - all_ancestors - hk_nodes
        print(f"Dose {d}: {len(hk_nodes)} HK nodes, {len(all_ancestors)} total ancestors, "
              f"{len(hk_specific_ancestors)} HK-specific ancestors (p<0.05), "
              f"{len(non_ancestors)} non-ancestors")

        # Save gene roles to CSV
        all_nodes = sorted(G.nodes())
        gene_df = pd.DataFrame({
            "gene": all_nodes,
            "housekeeping": [g in hk_nodes for g in all_nodes],
            "ancestor_of_hk": [g in all_ancestors for g in all_nodes],
            "hk_specific_ancestor": [g in hk_specific_ancestors for g in all_nodes],
        })
        gene_df.to_csv(os.path.join(output_dir, f"{filename}_gene_roles_{d}.csv"), index=False)

        # --- Step 2: Pathway enrichment for each group ---
        pe_specific = None
        pe_non = None
        if len(hk_specific_ancestors) >= 2:
            pe_specific = pathway_enrichment(list(hk_specific_ancestors), background_genes, None)
            pe_specific = pe_specific.sort_values(by="p_value").query("term_size < 300")
            top = pe_specific.iloc[0:10]
            if len(top) > 0:
                top_data_specific[d] = {
                    "pathways": top["native"].values,
                    "names": top["name"].values,
                    "sources": top["source"].values,
                    "neg_log10_p": -np.log10(top["p_value"].values),
                }
        if len(non_ancestors) >= 2:
            pe_non = pathway_enrichment(list(non_ancestors), background_genes, None)
            pe_non = pe_non.sort_values(by="p_value").query("term_size < 300")
            top = pe_non.iloc[0:10]
            if len(top) > 0:
                top_data_non[d] = {
                    "pathways": top["native"].values,
                    "names": top["name"].values,
                    "sources": top["source"].values,
                    "neg_log10_p": -np.log10(top["p_value"].values),
                }

        # --- Step 3: Differential enrichment ---
        # For each pathway found in either group, test if it's differentially
        # represented in HK-specific ancestors vs non-ancestors
        if pe_specific is not None and pe_non is not None:
            # Union of all pathways enriched in either group (p < 0.05)
            all_pathways = set(pe_specific["native"].values) | set(pe_non["native"].values)

            # Get intersection sizes from gProfiler results
            spec_intersections = dict(zip(pe_specific["native"], pe_specific["intersection_size"]))
            non_intersections = dict(zip(pe_non["native"], pe_non["intersection_size"]))
            pathway_terms = dict(zip(
                pd.concat([pe_specific, pe_non])["native"],
                pd.concat([pe_specific, pe_non])["term_size"],
            ))
            pathway_names = dict(zip(
                pd.concat([pe_specific, pe_non])["native"],
                pd.concat([pe_specific, pe_non])["name"],
            ))
            pathway_sources = dict(zip(
                pd.concat([pe_specific, pe_non])["native"],
                pd.concat([pe_specific, pe_non])["source"],
            ))

            n_spec = len(hk_specific_ancestors)
            n_non = len(non_ancestors)

            diff_rows = []
            for pw in all_pathways:
                # Genes in pathway within each group
                a = spec_intersections.get(pw, 0)   # HK-specific ancestors in pathway
                b = n_spec - a                       # HK-specific ancestors not in pathway
                c = non_intersections.get(pw, 0)     # non-ancestors in pathway
                dd = n_non - c                       # non-ancestors not in pathway

                _, p_diff = fisher_exact(
                    [[a, b], [c, dd]], alternative="greater"
                )
                # Log odds ratio (enrichment in ancestors relative to non-ancestors)
                frac_spec = a / max(n_spec, 1)
                frac_non = c / max(n_non, 1)
                log_or = np.log2((frac_spec + 1e-6) / (frac_non + 1e-6))

                diff_rows.append({
                    "pathway": pw,
                    "name": pathway_names.get(pw, pw),
                    "source": pathway_sources.get(pw, ""),
                    "term_size": pathway_terms.get(pw, 0),
                    "n_in_ancestors": a,
                    "n_in_non_ancestors": c,
                    "frac_ancestors": frac_spec,
                    "frac_non_ancestors": frac_non,
                    "log2_odds_ratio": log_or,
                    "p_value_diff": p_diff,
                })

            diff_df = pd.DataFrame(diff_rows)
            # FDR correction
            from statsmodels.stats.multitest import multipletests
            if len(diff_df) > 0:
                _, diff_df["p_adj"], _, _ = multipletests(diff_df["p_value_diff"], method="fdr_bh")
                diff_df = diff_df.sort_values("p_value_diff")
                diff_df.to_csv(os.path.join(output_dir, f"{filename}_differential_{d}.csv"), index=False)

                sig = diff_df.query("p_adj < 0.05 and log2_odds_ratio > 0").head(10)
                print(f"  Dose {d}: {len(sig)} differentially enriched pathways (FDR<0.05, log2OR>0)")
                if len(sig) > 0:
                    top_data_diff[d] = {
                        "pathways": sig["pathway"].values,
                        "names": sig["name"].values,
                        "sources": sig["source"].values,
                        "neg_log10_p": -np.log10(sig["p_value_diff"].values),
                    }

    # --- Plots ---
    if top_data_specific:
        generate_top_plots(
            {"HK-Specific Ancestors": top_data_specific},
            filename=filename + "_hk_specific",
            output_dir=output_dir, cmap_names=["OrangesDark"]
        )
    if top_data_non:
        generate_top_plots(
            {"Non-Ancestors": top_data_non},
            filename=filename + "_non_ancestors",
            output_dir=output_dir, cmap_names=["OrangesDark"]
        )
    if top_data_diff:
        generate_top_plots(
            {"Differential: HK-Specific Ancestors vs Non-Ancestors": top_data_diff},
            filename=filename + "_differential",
            output_dir=output_dir, cmap_names=["OrangesDark"]
        )

def main():
    parser = argparse.ArgumentParser(description="Pathway enrichment analysis")
    parser.add_argument("--output_dir", "-o", default="./pathway_enrichment",
                        help="Output directory for figures (default: ./pathway_enrichment)")
    args = parser.parse_args()
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    tpm_df, _, graphs_genes_by_dose, genes_by_dose, genes_neighborhoods, genes_100_tfs = load_data()
    background_genes = list(set(tpm_df["Gene"]))
    
    
    # random_gene_sanity_check(background_genes, graphs_genes_by_dose,
    #                          n_random=5, output_dir=out_dir)
    de_gene_intersection = set.intersection(*[set(genes) for genes in genes_by_dose.values()])
    causal_gene_intersection = set.intersection(*[set(genes) for genes in graphs_genes_by_dose.values()])
    # causal_gene_intersection = set.intersection(*[set(graphs_genes_by_dose[d]) for d in DOSE_RATE_LABELS])

    print(len(set(causal_gene_intersection)))
    invariant_plots(out_dir, causal_gene_intersection, de_gene_intersection,
                             background_genes)
    housekeeping_parent_enrichment(output_dir=out_dir)

    # CORRELATIVE FEATURES
    threshold=0.5
    corr_features = pd.read_csv(CORR_GENES_PATH, header=0)
    corr_genes = corr_features.loc[corr_features["fold_fraction"]>=threshold]["gene"].values
    corr_check = pathway_enrichment(corr_genes, background_genes, None)
    radiation_corr = {"pathways": ALL_RADIATION_PATHWAY_IDS}
    radiation_corr['logp_values'] = _build_radiation_logp(corr_check)
    pe_corr = corr_check.sort_values(by='p_value', ascending=True).query('term_size < 300')

    top_data_corr = {
        "pathways": pe_corr.iloc[0:10]['native'].values,
        "names": pe_corr.iloc[0:10]['name'].values,
        "sources": pe_corr.iloc[0:10]['source'].values,
        "neg_log10_p": -np.log10(pe_corr.iloc[0:10]['p_value'].values),
    }
    create_latex_table(pe_corr.iloc[0:10], args.output_dir, fname="top10_corr.csv", name='Corr.')

    # RANDOM FOREST FEATURES 
    threshold=0.5
    rf_features = pd.read_csv(RF_GENES_PATH, header=0)
    rf_genes = rf_features.loc[rf_features["fold_fraction"]>=threshold]["gene"].values
    rf_check = pathway_enrichment(rf_genes, background_genes, None)
    radiation_rf = {"pathways": ALL_RADIATION_PATHWAY_IDS}
    radiation_rf['logp_values'] = _build_radiation_logp(rf_check)
    pe = rf_check.sort_values(by='p_value', ascending=True).query('term_size < 300')

    top_data_rf = {
        "pathways": pe.iloc[0:10]['native'].values,
        "names": pe.iloc[0:10]['name'].values,
        "sources": pe.iloc[0:10]['source'].values,
        "neg_log10_p": -np.log10(pe.iloc[0:10]['p_value'].values),
    }
    create_latex_table(pe.iloc[0:10], "./pathway_enrichment", fname="top10_rf.csv", name='Corr.')

    # RUN PATHWAY ENRICHMENT FOR EACH DOSE RATE
    all_causal = [pathway_enrichment(graphs_genes_by_dose[d], background_genes, None) for d in DOSE_RATE_LABELS]
    all_de = [pathway_enrichment(genes_by_dose[d], background_genes, None) for d in DOSE_RATE_LABELS]

    # RADIATION SPECIFIC BAR PLOTS 
    radiation_data_causal = {"pathways": ALL_RADIATION_PATHWAY_IDS}
    radiation_data_de = {"pathways": ALL_RADIATION_PATHWAY_IDS}

    for i, d in enumerate(DOSE_RATE_LABELS):
        radiation_data_causal[DOSE_RATES_REGRESSION[d]] = _build_radiation_logp(all_causal[i])
        radiation_data_de[DOSE_RATES_REGRESSION[d]] = _build_radiation_logp(all_de[i])
    generate_plots(
        {
            "Differential Expression": radiation_data_de,
            "Causal Graph": radiation_data_causal,
        },
        agnostic_datasets={"Supervised ML (Random Forest)": radiation_rf, "Corr. (LinearRegression)": radiation_corr},
        output_dir=out_dir,
    )

    # TOP ENRICHED PATHWAYS FOR EACH METHOD
    top_data_causal = {}
    top_data_de = {}
    for i, d in enumerate(DOSE_RATE_LABELS):
        pe_causal = all_causal[i].sort_values(by='p_value', ascending=True).query('term_size < 300')
        pe_de = all_de[i].sort_values(by='p_value', ascending=True).query('term_size < 300')

        top_data_causal[d] = {
            "pathways": pe_causal.iloc[0:10]['native'].values,
            "names": pe_causal.iloc[0:10]['name'].values,
            "sources": pe_causal.iloc[0:10]['source'].values,
            "neg_log10_p": -np.log10(pe_causal.iloc[0:10]['p_value'].values),
        }

        top_data_de[d] = {
            "pathways": pe_de.iloc[0:10]['native'].values,
            "names": pe_de.iloc[0:10]['name'].values,
            "sources": pe_de.iloc[0:10]['source'].values,
            "neg_log10_p": -np.log10(pe_de.iloc[0:10]['p_value'].values),
        }

    generate_top_plots({
            "Differential Expression": top_data_de,
            "Causal Graph": top_data_causal,
        },
        agnostic_datasets={"Supervised ML (Random Forest)": top_data_rf, "Correlation (Linear Regression)":top_data_corr},
        output_dir=out_dir)
    venn_diagrams(graphs_genes_by_dose, genes_by_dose, rf_genes=rf_genes, corr_genes=corr_genes, output_dir=out_dir)


if __name__ == "__main__":
    main()