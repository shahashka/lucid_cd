import pandas as pd
import numpy as np
import pickle
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import chain, combinations, product
from Bio import Entrez
from matplotlib_venn import venn2
from upsetplot import from_memberships, UpSet
from gprofiler import GProfiler
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple


GRAPHS = {"invariant": "/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs3/dag_gnn_combined.gexf",
          "F":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseF.gexf",
          "G":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseG.gexf",
          "H":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseH.gexf",
          "I":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseI.gexf",
          "J":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseJ.gexf" 
          }
with open("/homes/shahashka/lucid_cd/data/gene_groups.pkl", "rb") as f:
    CAUSAL_TFS, CAUSAL_NEIGHBORHOODS, KOSMOS, CHATGPT, BNL = pickle.load(f)
EXPERIMENT = "./data/rpe1_experiment2"
CONTEXT_SPECIFIC_GRAPHS = f"{EXPERIMENT}/bootstrap_graphs2"
INVARIANT_GRAPHS = f"{EXPERIMENT}/bootstrap_graphs3"
CAUSAL_DOSE_RATES = ["dF", "dG", "dH", "dI", "dJ"]
DOSE_RATES = ["F", "G", "H", "I", "J"]
DOSE_RATES_ACTUAL = {"F": 0.38, "G": 0.28, "H": 0.55, "I": 6.66, "J": 12.11, "shared":0}
WEEKS = np.arange(10)
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
            labels.append(f"{p} — {desc}")
        else:
            labels.append(p)
    return ordered, labels, categories
def load_data():
    log2fold_df = pd.read_csv(f"/homes/shahashka/lucid_cd/data/rpe1_experiment2/rpe1_9week_study_experiment2_diffexp_deseq_vs_control_all_dG_W2_adjust.txt", sep="\t")
    log2fold_df = log2fold_df.loc[log2fold_df['padj'] < 0.05] # This is new, I think I should filter by p value. However this means there are no genes that DE across all dose rates
    log2fold_df = log2fold_df.loc[abs(log2fold_df['log2FoldChange']) > 1]
    log2fold_df['Week'] = [float(w[1]) for w in log2fold_df['Week']]
    genes_by_week = {}
    genes_by_dose = {} 
    
    # LOAD DIFFERENTIAL EXPRESSION DATA (temporal)
    for i in WEEKS:
        week_i = log2fold_df.loc[log2fold_df["Week"] == i]
        genes_week_i = set(week_i["Gene"])
        for g in genes_week_i:
            if g in genes_by_week.keys():
                genes_by_week[g].add(i)
            else:
                genes_by_week[g] = set([i])
                
    # LOAD DIFFERENTIAL EXPRESSION DATA (dose rate)       
    for d in CAUSAL_DOSE_RATES:
        dose_i = log2fold_df.loc[log2fold_df["Dose"] == d]
        genes_dose_i = dose_i["Gene"]
        genes_by_dose[d[1]] = list(set(genes_dose_i))
        
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
    for d in DOSE_RATES:
        genes_100_tfs[d] = pd.read_csv(f"{CONTEXT_SPECIFIC_GRAPHS}/top_100_dag_gnn_{d}.csv", header=None).iloc[:,0].to_list()
        genes_neighborhoods[d] = pd.read_csv(f"{CONTEXT_SPECIFIC_GRAPHS}/rad_sub_dag_gnn_{d}_ranked.csv", header=0).iloc[:,0].to_list()

    genes_100_tfs['all_doses'] = pd.read_csv(f"{INVARIANT_GRAPHS}/dag_gnn_combined_top_100_tfs.csv", header=None).iloc[:,0].to_list()
    genes_neighborhoods['all_doses_dose_rate'] = pd.read_csv(f"{INVARIANT_GRAPHS}/rad_sub_dag_gnn_combined_ranked.csv", header=0).iloc[:,0].to_list()
    genes_neighborhoods['all_doses_week'] = pd.read_csv(f"{INVARIANT_GRAPHS}/week_sub_dag_gnn_combined_ranked.csv", header=0).iloc[:,0].to_list()
    
    return tpm_df, log2fold_df, graphs_genes_by_dose, genes_by_dose, genes_neighborhoods, genes_100_tfs


def bootstrap_enrichment_CI(genes, background_genes, n_boot, pathways) ->  pd.DataFrame:
    stats = {}
    for p in pathways:
        stats[p] = []
    
    for _ in range(n_boot):
        boot_genes = np.random.choice(genes, size=len(genes), replace=True)
        df = pathway_enrichment(boot_genes, background_genes, pathways)
        for _, row in df.iterrows():
                stats[row["native"]].append(-np.log10(row["p_value"]))

    ci = {}
    for pathway, values in stats.items():
        if len(values) > 10:
            ci[pathway] = {
                "mean": np.mean(values),
                "ci_low": np.percentile(values, 2.5),
                "ci_high": np.percentile(values, 97.5)
            }
        

    return ci

def pathway_enrichment(genes,background_genes, pathways) -> pd.DataFrame:
    """Given a list of genes, perform pathway enrichment using knowledge databases

    Args:
        genes (set(str)): Set of genes with string identifiers
        
    Returns:
        (List[Any]): Return a list of named pathways and scores for each 
    """
    gp = GProfiler(return_dataframe=True)

    results = gp.profile(
        organism="hsapiens",
        query=list(set(genes)),
        sources=["GO:BP", "GO:MF", "GO_:CC", "KEGG", "REAC", "WP"],
        user_threshold=0.05,
        background=background_genes, 
        significance_threshold_method="fdr"
    )

    # Sort by adjusted p-value
    results = results.sort_values("p_value")
    if pathways:
        return results.query("native in @pathways")
    else:
        return results

# Perceptually separated hue families (avoid viridis/plasma/inferno cluster).
_DEFAULT_CMAPS = (
    "Blues",
    "Oranges",
    "Greens",
    "Purples",
    "Reds",
    "YlOrBr",
    "BuPu",
)


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


def generate_plots(datasets, cmap_names=None, pathway_descriptions=None):
    """Single stacked horizontal bar plot: one colormap per dataset (name).

    Parameters
    ----------
    datasets : dict[str, dict]
        Map dataset display name -> data dict with ``pathways`` (native IDs) and numeric columns (dose rates).
    cmap_names : sequence of str, optional
        Matplotlib colormap names, one per dataset in ``datasets.items()`` order.
    pathway_descriptions : dict[str, str], optional
        Map native pathway id -> full description for y-axis labels. Defaults to ``PATHWAY_DESCRIPTIONS``.
    """
    if not isinstance(datasets, dict) or not datasets:
        raise TypeError("datasets must be a non-empty dict mapping name -> data dict")

    names = list(datasets.keys())
    n_ds = len(names)
    if cmap_names is None:
        cmap_names = [_DEFAULT_CMAPS[i % len(_DEFAULT_CMAPS)] for i in range(n_ds)]
    elif len(cmap_names) < n_ds:
        cmap_names = list(cmap_names) + [
            _DEFAULT_CMAPS[i % len(_DEFAULT_CMAPS)]
            for i in range(len(cmap_names), n_ds)
        ]
    cmaps = [cm.get_cmap(c) for c in cmap_names[:n_ds]]

    first = pd.DataFrame(next(iter(datasets.values()))).set_index("pathways").sort_index(axis=1)
    ordered, display_labels, pathway_categories = _ordered_pathways_with_category_labels(
        first.index.tolist(), descriptions=pathway_descriptions
    )
    dose_cols = list(first.columns)
    n_col = len(dose_cols)
    n_pathways = len(ordered)

    dfs = {}
    for name, d in datasets.items():
        df = pd.DataFrame(d).set_index("pathways").sort_index(axis=1).reindex(ordered)
        dfs[name] = df

    # Touching bars within each pathway group: bar height equals vertical step.
    bar_height = 1.0
    step = bar_height
    pathway_gap = 0.45
    group_pitch = n_ds * step + pathway_gap

    _fs_axis = 23
    _fs_tick = 21
    _fs_leg = 20
    _fs_leg_title = 21

    fig_height = max(8, 0.55 * n_pathways * max(n_ds, 1))
    fig, ax = plt.subplots(figsize=(24, fig_height))

    for i, pw in enumerate(ordered):
        base_y = i * group_pitch
        for k, name in enumerate(names):
            y = base_y + k * step
            row = dfs[name].loc[pw]
            left = 0.0
            cmap = cmaps[k]
            for j, col in enumerate(dose_cols):
                w = float(row[col])
                t = j / max(n_col - 1, 1) if n_col > 1 else 0.5
                color = cmap(t)
                ax.barh(y, w, left=left, height=bar_height, color=color, edgecolor="white", linewidth=0.4)
                left += w

    y_centers = [
        i * group_pitch + (n_ds - 1) * step / 2.0 for i in range(n_pathways)
    ]
    ax.set_yticks(y_centers)
    ax.set_yticklabels(display_labels)
    ax.tick_params(axis="y", labelsize=_fs_tick)
    ax.tick_params(axis="x", labelsize=_fs_tick)
    for tick, cat in zip(ax.get_yticklabels(), pathway_categories):
        tick.set_color(CATEGORY_COLORS.get(cat, "#333333"))
    ax.invert_yaxis()
    ax.set_xlabel(r"$-\log_{10}(p)$", fontsize=_fs_axis)

    dose_handles = []
    dose_labels = []
    for k, name in enumerate(names):
        cmap = cmaps[k]
        for j, col in enumerate(dose_cols):
            t = j / max(n_col - 1, 1) if n_col > 1 else 0.5
            dose_handles.append(
                Line2D([0], [0], color=cmap(t), lw=8, solid_capstyle="butt")
            )
            dose_labels.append(f"{name}: {_format_dose_label(col)} mGy/min")

    ncol_leg = min(n_col, 5)

    # Leave room: left for y-labels, bottom for dose legend, right for category legend
    fig.subplots_adjust(left=0.28, right=0.78, bottom=0.14, top=0.95)

    # Dose rate legend — bottom centre, below the axes
    leg_dose = ax.legend(
        dose_handles,
        dose_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=ncol_leg,
        frameon=True,
        fontsize=_fs_leg,
        title="Dose rate",
        title_fontsize=_fs_leg_title,
    )
    ax.add_artist(leg_dose)

    # Category legend — right side, anchored to axes
    seen_cats = set(pathway_categories)
    cat_order = [k for k in list(PATHWAY_CATEGORIES.keys()) + ["uncategorized"] if k in seen_cats]
    cat_handles = [
        Patch(facecolor=CATEGORY_COLORS[c], edgecolor="none", label=_format_category_legend_name(c))
        for c in cat_order
    ]
    leg_cat = ax.legend(
        handles=cat_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
        fontsize=_fs_leg,
        title="Pathway category",
        title_fontsize=_fs_leg_title,
    )
    ax.add_artist(leg_cat)

    # Colormap legend — right side, below category legend
    cmap_legend_handles = [
        _cmap_gradient_legend_handle(cmaps[k]) for k in range(n_ds)
    ]
    cmap_legend_labels = [
        f"{names[k]} ({cmap_names[k]})" for k in range(n_ds)
    ]
    leg_cmap = ax.legend(
        cmap_legend_handles,
        cmap_legend_labels,
        handler_map={tuple: HandlerTuple(pad=0)},
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        borderaxespad=0.0,
        frameon=True,
        fontsize=_fs_leg,
        title="Dataset colormap (light \u2192 dark = low \u2192 high dose)",
        title_fontsize=_fs_leg_title,
    )
    _save_kw = dict(bbox_inches="tight", pad_inches=0.2)
    plt.savefig("./pathway_enrichment_dose_rate.pdf", format="pdf", **_save_kw)
    plt.savefig("./pathway_enrichment_dose_rate.png", format="png", dpi=300, **_save_kw)


def generate_top_plots(datasets, cmap_names=None, filename="top10_pathway_enrichment"):
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
    """
    names = list(datasets.keys())
    n_panels = len(names)
    if cmap_names is None:
        cmap_names = [_DEFAULT_CMAPS[i % len(_DEFAULT_CMAPS)] for i in range(n_panels)]
    elif len(cmap_names) < n_panels:
        cmap_names = list(cmap_names) + [
            _DEFAULT_CMAPS[i % len(_DEFAULT_CMAPS)]
            for i in range(len(cmap_names), n_panels)
        ]
    cmaps = [cm.get_cmap(c) for c in cmap_names[:n_panels]]

    panels = [(name, datasets[name], cmaps[k]) for k, name in enumerate(names)]

    first_data = next(iter(datasets.values()))
    dose_keys = list(first_data.keys())
    n_dose = len(dose_keys)

    # --- build per-panel DataFrames (union of pathways x dose rates) ---
    panel_dfs = []
    panel_labels = []
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
                    pw_to_label[pw] = f"{nm} ({src})"

        all_pws = list(pw_to_label.keys())
        # Build a DataFrame: rows = pathways, cols = dose rates
        rows = {pw: {d: 0.0 for d in dose_keys} for pw in all_pws}
        for d in dose_keys:
            for pw, val in zip(top_data[d]["pathways"], top_data[d]["neg_log10_p"]):
                rows[pw][d] = val

        df = pd.DataFrame(rows).T  # index = pathway id, columns = dose letters
        # Sort by total enrichment (most significant at top)
        df["_total"] = df.sum(axis=1)
        df = df.sort_values("_total", ascending=True).drop(columns="_total")
        labels = [pw_to_label[pw] for pw in df.index]
        panel_dfs.append(df)
        panel_labels.append(labels)

    # --- plot ---
    _fs_axis = 13
    _fs_tick = 10
    _fs_leg = 10
    _fs_leg_title = 11
    _fs_title = 14

    max_pathways = max(len(df) for df in panel_dfs)
    fig_height = max(8, 0.4 * max_pathways)
    fig_width = 14 * n_panels
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_width, fig_height), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, (title, _, cmap_obj), df, labels in zip(
        axes, panels, panel_dfs, panel_labels
    ):
        n_pw = len(df)
        y_pos = np.arange(n_pw)
        left = np.zeros(n_pw)

        for j, d in enumerate(dose_keys):
            t = 0.5 if n_dose == 1 else j / (n_dose - 1)
            color = cmap_obj(t)
            widths = df[d].values
            ax.barh(
                y_pos, widths, left=left, height=0.7,
                color=color, edgecolor="white", linewidth=0.4,
            )
            left += widths

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=_fs_tick)
        ax.set_xlabel(r"$-\log_{10}(p)$", fontsize=_fs_axis)
        ax.set_title(title, fontsize=_fs_title, fontweight="bold")
        ax.tick_params(axis="x", labelsize=_fs_tick)

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
            dose_labels.append(f"{panel_names[k]}: {_format_dose_label(DOSE_RATES_ACTUAL[d])} mGy/min")

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
    # Colormap legend — right side, below category legend
    names = list(datasets.keys())
    cmap_legend_handles = [
        _cmap_gradient_legend_handle(cmaps[k]) for k in range(len(names))
    ]
    cmap_legend_labels = [
        f"{names[k]} ({cmap_names[k]})" for k in range(len(names))
    ]
    leg_cmap = ax.legend(
        cmap_legend_handles,
        cmap_legend_labels,
        handler_map={tuple: HandlerTuple(pad=0)},
        loc="upper left",
        bbox_to_anchor=(1.02, 0.55),
        borderaxespad=0.0,
        frameon=True,
        fontsize=_fs_leg,
        title="Dataset colormap (light \u2192 dark = low \u2192 high dose)",
        title_fontsize=_fs_leg_title,
    )

    fig.subplots_adjust(wspace=0.55, bottom=0.12, top=0.93)
    _save_kw = dict(bbox_inches="tight", pad_inches=0.2)
    plt.savefig(f"./{filename}.pdf", format="pdf", **_save_kw)
    plt.savefig(f"./{filename}.png", format="png", dpi=300, **_save_kw)


def venn_diagrams(graphs_genes_by_dose, genes_by_dose):
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
    pairs = list(combinations(DOSE_RATES, 2))  # 10 pairs

    def _dose_label(d):
        return f"{d} ({DOSE_RATES_ACTUAL[d]} mGy/min)"

    # --- Figure 1 & 2: pairwise across dose rates for each method ---
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
            venn2(
                [set1, set2],
                set_labels=(_dose_label(d1), _dose_label(d2)),
                ax=ax,
            )
            ax.set_title(f"{d1} vs {d2}", fontsize=12, fontweight="bold")
        fig.suptitle(f"{method_name} — Pairwise Gene Set Overlap", fontsize=16, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"./{fname}.pdf", format="pdf", **_save_kw)
        plt.savefig(f"./{fname}.png", format="png", dpi=300, **_save_kw)

    # --- Figure 3: Causal vs DE at each dose rate ---
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for idx, d in enumerate(DOSE_RATES):
        ax = axes[idx]
        set_causal = set(graphs_genes_by_dose[d])
        set_de = set(genes_by_dose[d])
        venn2(
            [set_causal, set_de],
            set_labels=("Causal Graph", "Diff. Expression"),
            ax=ax,
        )
        ax.set_title(_dose_label(d), fontsize=12, fontweight="bold")
    fig.suptitle("Causal Graph vs Differential Expression", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig("./venn_causal_vs_de.pdf", format="pdf", **_save_kw)
    plt.savefig("./venn_causal_vs_de.png", format="png", dpi=300, **_save_kw)

    # --- UpSet plots: all dose rates at once ---
    def _build_memberships(gene_dict):
        """Return a list of (membership_tuple, gene) pairs for upsetplot."""
        gene_to_sets = {}
        for d in DOSE_RATES:
            for g in gene_dict[d]:
                gene_to_sets.setdefault(g, set()).add(_dose_label(d))
        memberships = []
        for g, sets in gene_to_sets.items():
            memberships.append(tuple(sorted(sets, key=lambda s: DOSE_RATES.index(s[0]))))
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
        upset = UpSet(upset_data, show_counts=True, sort_by="cardinality")
        fig = plt.figure(figsize=(14, 8))
        upset.plot(fig=fig)
        fig.suptitle(f"{method_name} — Gene Set Intersections Across Dose Rates",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.savefig(f"./{fname}_upset.pdf", format="pdf", **_save_kw)
        plt.savefig(f"./{fname}_upset.png", format="png", dpi=300, **_save_kw)


def main():
    tpm_df, log2fold_df, graphs_genes_by_dose, genes_by_dose, genes_neighborhoods, genes_100_tfs = load_data()
    background_genes = list(set(tpm_df["Gene"]))
    # RUN PATHWAY ENRICHMENT FOR EACH DOSE RATE
    all_causal = [pathway_enrichment(graphs_genes_by_dose[d], background_genes, None) for d in DOSE_RATES]
    all_de = [pathway_enrichment(genes_by_dose[d], background_genes, None) for d in DOSE_RATES]
    all_causal_neighborhoods = [pathway_enrichment(genes_neighborhoods[d], background_genes, None) for d in DOSE_RATES]
    print(all_causal[0]['source'])
    
    # RESHAPE DATA TO FOR PLOTTING (native IDs only; descriptions come from PATHWAY_DESCRIPTIONS)
    radiation_data_causal = {"pathways": ALL_RADIATION_PATHWAY_IDS}
    radiation_data_de = {"pathways": ALL_RADIATION_PATHWAY_IDS}
    radiation_data_neighborhoods = {"pathways": ALL_RADIATION_PATHWAY_IDS}

    for i,d in enumerate(DOSE_RATES):
        causal_list = []
        de_list = []
        neighborhood_list = []
        for p in ALL_RADIATION_PATHWAY_IDS:
            pe_causal = all_causal[i]
            pe_de = all_de[i]
            pe_neighborhood = all_causal_neighborhoods[i]
            if p in list(pe_causal['native']):
                causal_list.append(-np.log10(pe_causal.loc[pe_causal['native']==p]['p_value'].values[0]))
            else:
                causal_list.append(0)
            
            if p in list(pe_de['native']):
                de_list.append(-np.log10(pe_de.loc[pe_de['native']==p]['p_value'].values[0]))
            else:
                de_list.append(0)
                
            if p in list(pe_neighborhood['native']):
                neighborhood_list.append(-np.log10(pe_neighborhood.loc[pe_neighborhood['native']==p]['p_value'].values[0]))
            else:
                neighborhood_list.append(0)
        radiation_data_causal[DOSE_RATES_ACTUAL[d]] = causal_list
        radiation_data_de[DOSE_RATES_ACTUAL[d]] = de_list
        radiation_data_neighborhoods[DOSE_RATES_ACTUAL[d]] = neighborhood_list
        
    # BAR PLOTS
    generate_plots(
        {
            "Differential Expression": radiation_data_de,
            "Causal Graph": radiation_data_causal,
            # "Causal Neighborhood": all_data_neighborhoods,
        }
    )

    top_data_causal = {}
    top_data_de = {}
    for i,d in enumerate(DOSE_RATES):
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
        })

    # PATHWAY ENRICHMENT FOR INTERSECTION OF DE AND CAUSAL GENES
    intersection_genes_by_dose = {}
    for d in DOSE_RATES:
        intersection_genes_by_dose[d] = list(
            set(graphs_genes_by_dose[d]) & set(genes_by_dose[d])
        )
    all_intersection = [
        pathway_enrichment(intersection_genes_by_dose[d], background_genes, None)
        for d in DOSE_RATES
    ]
    top_data_intersection = {}
    for i, d in enumerate(DOSE_RATES):
        pe = all_intersection[i].sort_values(by='p_value', ascending=True).query('term_size < 300')
        top_data_intersection[d] = {
            "pathways": pe.iloc[0:10]['native'].values,
            "names": pe.iloc[0:10]['name'].values,
            "sources": pe.iloc[0:10]['source'].values,
            "neg_log10_p": -np.log10(pe.iloc[0:10]['p_value'].values),
        }

    generate_top_plots(
        {
            "Differential Expression": top_data_de,
            "Causal Graph": top_data_causal,
            "Intersection (DE ∩ Causal)": top_data_intersection,
        },
        filename="top10_pathway_enrichment_with_intersection",
    )

    # PATHWAY ENRICHMENT FOR GENES SHARED ACROSS ALL DOSE RATES (per method)
    causal_intersection = set(graphs_genes_by_dose[DOSE_RATES[0]])
    de_intersection = set(genes_by_dose[DOSE_RATES[0]])
    for d in DOSE_RATES[1:]:
        causal_intersection &= set(graphs_genes_by_dose[d])
        de_intersection &= set(genes_by_dose[d])

    pe_causal_shared = pathway_enrichment(
        list(causal_intersection), background_genes, None
    ).sort_values(by='p_value', ascending=True).query('term_size < 300')
    pe_de_shared = pathway_enrichment(
        list(de_intersection), background_genes, None
    ).sort_values(by='p_value', ascending=True).query('term_size < 300')

    shared_key = "shared"
    top_shared_causal = {shared_key: {
        "pathways": pe_causal_shared.iloc[0:10]['native'].values,
        "names": pe_causal_shared.iloc[0:10]['name'].values,
        "sources": pe_causal_shared.iloc[0:10]['source'].values,
        "neg_log10_p": -np.log10(pe_causal_shared.iloc[0:10]['p_value'].values),
    }}
    top_shared_de = {shared_key: {
        "pathways": pe_de_shared.iloc[0:10]['native'].values,
        "names": pe_de_shared.iloc[0:10]['name'].values,
        "sources": pe_de_shared.iloc[0:10]['source'].values,
        "neg_log10_p": -np.log10(pe_de_shared.iloc[0:10]['p_value'].values),
    }}

    generate_top_plots(
        {
            "Causal Graph": top_shared_causal,
            "Differential Expression": top_shared_de,
        },
        filename="top10_shared_across_doses",
    )

    venn_diagrams(graphs_genes_by_dose, genes_by_dose)


if __name__ == "__main__":
    main()