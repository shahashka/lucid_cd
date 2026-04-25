import json
import os
import pickle
from collections import Counter
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.preprocessing import StandardScaler
from feature_selection import load_data, load_data_tpm
from pathway_enrichment import pathway_enrichment as run_pathway_enrichment, DOSE_RATES_REGRESSION
import matplotlib.cm as cm
from global_variables import _fs_axis, _fs_leg, _fs_leg_title, _fs_tick, _fs_title, _DEFAULT_CMAPS, GRAPHS, DOSE_RATE_LABELS, DOSE_RATES_SORTED, HK_PATH
import seaborn as sns
import argparse

PARTITION_PICKLE = {
                    "F":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/cd_partition_dF_new.pickle",
                    "G":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/cd_partition_dG_new.pickle",
                    "H":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/cd_partition_dH_new.pickle",
                    "I":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/cd_partition_dI_new.pickle",
                    "J": "/homes/shahashka/lucid_cd/data/rpe1_experiment2/cd_partition_dJ_new.pickle"
}
TRRUST_PATH = "/homes/shahashka/lucid_cd/data/prior_knowledge/trrust.tsv"
STRING_PATH = "/homes/shahashka/lucid_cd/data/prior_knowledge/string_ppi_gene_names.csv"
CORUM_PATH =  "/homes/shahashka/lucid_cd/data/prior_knowledge/corum_complexes.txt"
CHIPSEQ_PATH = "/homes/shahashka/lucid_cd/data/prior_knowledge/Hep_G2_ChipSeq.csv"


EXCLUDE_NODES = {"radiation", "dose_rate", "week"}
TPM_PATH = "/homes/shahashka/lucid_cd/data/rpe1_experiment2/rpe1_9week_study_experiment2_all_tpm.tsv"

N_GRAPHS = len(GRAPHS.keys())
_CMAP = cm.get_cmap("OrangesDark")
COLORS_BY_DOSE = {k: _CMAP((i + 1) / (N_GRAPHS + 1)) for i, k in enumerate(DOSE_RATES_SORTED)}

def load_tf_set():
    """Load known TF names from TRRUST database."""
    df = pd.read_csv(TRRUST_PATH, sep="\t", header=None)
    print(df.columns)
    return set(df[0])

def load_corum_gene_clusters():
    df = pd.read_csv(
        CORUM_PATH,
        sep="\t",
    )
    # Create a dictionary mapping gene clusters (as frozensets) to GO names
    mapping = {}
    for _, row in df.iterrows():
        genes = row['subunits_gene_name']
        go = row['functions_go_name']
        if pd.isna(genes) or pd.isna(go):
            continue
        complex_members = frozenset(genes.split(";"))
        go_names = go.split(";")
        mapping[complex_members] = go_names
    return mapping

def corum_enrichment(gene_cluster, mapping):
    """Find the CORUM complex with the most overlap with gene_cluster.
    Returns (best_complex, overlap_genes, go_terms) or (None, set(), [])
    if no overlap is found.
    """
    gene_set = set(gene_cluster)
    best_complex = None
    best_overlap = set()
    best_go = []
    for complex_members, go_names in mapping.items():
        overlap = gene_set & complex_members
        if len(overlap) > len(best_overlap):
            best_overlap = overlap
            best_complex = complex_members
            best_go = go_names
    return best_complex, best_overlap, best_go

def load_corum():
    df = pd.read_csv(
        CORUM_PATH,
        #index_col="subunits(Gene name)",
        index_col = "subunits_gene_name",
        sep="\t",
    ) 
    row_names = df.index.values.tolist()
    gene_from, gene_to = [], []
    for row_name in row_names:
        complex_members = row_name.split(";")
        for i in range(len(complex_members)):
            for other in complex_members[:i] + complex_members[i + 1 :]:
                gene_to.append(other)
                gene_from.append(complex_members[i])
    dataset_corum = set(zip(gene_from, gene_to))
    return dataset_corum

def load_chipseq():
    df = pd.read_csv(CHIPSEQ_PATH)
    dataset_chip_seq = set()
    for row in df[["source", "target"]].values:
        if not pd.isna(row[0]) and not pd.isna(row[1]):
            gene1, gene2 = (
                    row[0],
                    row[1],
                )
            dataset_chip_seq.add((gene1, gene2))
    return dataset_chip_seq


def load_string_hubs(top_n=500, min_score=400):
    """Load top STRING PPI hub genes by degree (filtered by combined_score)."""
    df = pd.read_csv(STRING_PATH)
    df = df[df.combined_score >= min_score]
    degree = Counter()
    degree.update(df.protein1)
    degree.update(df.protein2)
    top_genes = [g for g, _ in degree.most_common(top_n)]
    return set(top_genes), degree


def load_chipseq_tf_set():
    """Load TF names from ChIP-seq data (source column)."""
    df = pd.read_csv(CHIPSEQ_PATH)
    return set(df["source"].dropna().unique())


def load_bootstrap_graphs(d, bootstrap_dir, max_density=0.1):
    """Load bootstrap .npy matrices and merge by summing weights.

    Edges get a 'weight' (sum of weights across bootstraps) and 'count'
    (number of bootstraps the edge appeared in), matching the GEXF convention.

    Returns (combined_graph, per_partition_graphs, bootstrap_data, gene_map).
    """
    with open(PARTITION_PICKLE[d], "rb") as f:
        gene_map = pickle.load(f)

    n_parts = len(gene_map)
    n_boots = 10
    per_partition = {}
    bootstrap_data = {}

    for part in range(n_parts):
        genes = gene_map[part] + ['radiation']
        n_genes = len(genes)
        matrices = []

        for boot in range(n_boots):
            path = os.path.join(
                bootstrap_dir,
                # f"dag_gnn_dose_combined_part_{part}_boot_{boot}.npy",
                f"dag_gnn_dose_{d}_part_{part}_boot_{boot}.npy"
            )
            m = np.load(path)
            m_genes = m[:n_genes, :n_genes]

            # Skip anomalous dense bootstraps
            n_possible = n_genes * (n_genes - 1)
            density = np.count_nonzero(m) / max(n_possible, 1)
            if density > max_density:
                continue
            matrices.append(m)

        bootstrap_data[part] = matrices
        if not matrices:
            continue

        # Sum weights and count occurrences across bootstraps
        stack = np.array(matrices)
        weight_sum = stack.sum(axis=0)
        count = (stack != 0).sum(axis=0)
        np.fill_diagonal(weight_sum, 0)
        np.fill_diagonal(count, 0)

        # Build graph — keep all edges that appear in at least 1 bootstrap
        G = nx.DiGraph()
        G.add_nodes_from(genes)
        for i in range(n_genes):
            for j in range(n_genes):
                if count[i, j] > 0:
                    G.add_edge(genes[i], genes[j],
                            weight=float(weight_sum[i, j]),
                            count=int(count[i, j]))

        per_partition[part] = G
        print(f" Dose {d} Part {part}: {len(genes)} genes, {len(matrices)} valid boots, "
            f"{G.number_of_edges()} edges")

    # Combine across partitions — sum weights and counts for overlapping edges
    combined = nx.DiGraph()
    for part, G in per_partition.items():
        combined.add_nodes_from(G.nodes)
        for u, v, data in G.edges(data=True):
            if combined.has_edge(u, v):
                combined[u][v]["weight"] += data["weight"]
                combined[u][v]["count"] += data["count"]
            else:
                combined.add_edge(u, v, weight=data["weight"], count=data["count"])

    print(f"\nCombined: {combined.number_of_nodes()} nodes, "
        f"{combined.number_of_edges()} edges")

    return combined, per_partition, bootstrap_data, gene_map


def build_bootstrap_graph(boot, bootstrap_data, gene_map):
    """Build a combined directed graph for a single bootstrap replicate."""
    G_boot = nx.DiGraph()
    for part in sorted(bootstrap_data.keys()):
        matrices = bootstrap_data[part]
        if boot >= len(matrices):
            continue
        genes = gene_map[part]
        n_genes = len(genes)
        m = matrices[boot][:n_genes, :n_genes]
        G_boot.add_nodes_from(genes)
        for i in range(n_genes):
            for j in range(n_genes):
                if m[i, j] != 0:
                    if G_boot.has_edge(genes[i], genes[j]):
                        G_boot[genes[i]][genes[j]]["weight"] += abs(m[i, j])
                    else:
                        G_boot.add_edge(genes[i], genes[j], weight=abs(m[i, j]))
    return G_boot

def hub_tf_analysis(output_dir="."):
    """Check whether high out-degree hub nodes correspond to known TFs (TRRUST + ChIP-seq)."""
    trrust_tf_set = load_tf_set()
    chipseq_tf_set = load_chipseq_tf_set()
    combined_tf_set = trrust_tf_set | chipseq_tf_set
    top_ks = [10, 25, 50, 100]

    rows = []
    for graph_name, path in GRAPHS.items():
        G = nx.read_gexf(path)
        gene_nodes = [n for n in G.nodes if n not in EXCLUDE_NODES]
        n_total = len(gene_nodes)
        n_trrust_total = sum(1 for n in gene_nodes if n in trrust_tf_set)
        n_chipseq_total = sum(1 for n in gene_nodes if n in chipseq_tf_set)
        n_combined_total = sum(1 for n in gene_nodes if n in combined_tf_set)

        out_deg = {n: G.out_degree(n) for n in gene_nodes}
        ranked = sorted(out_deg.items(), key=lambda x: x[1], reverse=True)

        for k in top_ks:
            if k > len(ranked):
                continue
            top_k_genes = [g for g, _ in ranked[:k]]

            # TRRUST TF enrichment
            n_trrust_in = sum(1 for g in top_k_genes if g in trrust_tf_set)
            n_trrust_rest = n_trrust_total - n_trrust_in
            _, pval_trrust = fisher_exact([
                [n_trrust_in, k - n_trrust_in],
                [n_trrust_rest, (n_total - k) - n_trrust_rest],
            ], alternative="greater")

            # ChIP-seq TF enrichment
            n_chipseq_in = sum(1 for g in top_k_genes if g in chipseq_tf_set)
            n_chipseq_rest = n_chipseq_total - n_chipseq_in
            _, pval_chipseq = fisher_exact([
                [n_chipseq_in, k - n_chipseq_in],
                [n_chipseq_rest, (n_total - k) - n_chipseq_rest],
            ], alternative="greater")
            rows.append({
                "graph": graph_name,
                "k": k,
                "n_trrust_tf": n_trrust_in,
                "trrust_tf_precision": n_trrust_in / k,
                "trrust_tf_p_value": pval_trrust,
                "trrust_tf_baseline": n_trrust_total / n_total,
                "n_chipseq_tf": n_chipseq_in,
                "chipseq_tf_precision": n_chipseq_in / k,
                "chipseq_tf_p_value": pval_chipseq,
                "chipseq_tf_baseline": n_chipseq_total / n_total,
                "n_nodes": n_total,
            })

    df = pd.DataFrame(rows)

    # Print summary
    print("\n=== Hub TF Enrichment (TRRUST + ChIP-seq) ===\n")
    for graph_name in GRAPHS:
        sub = df[df.graph == graph_name]
        if sub.empty:
            continue
        print(f"{graph_name} ({sub.iloc[0]['n_nodes']} nodes, "
              f"TRRUST baseline: {sub.iloc[0]['trrust_tf_baseline']:.3f}, "
              f"ChIP-seq baseline: {sub.iloc[0]['chipseq_tf_baseline']:.3f}, ")
        for _, r in sub.iterrows():
            print(f"  top-{r.k:>3d}: "
                  f"TRRUST={r.n_trrust_tf:>3d} (prec={r.trrust_tf_precision:.2f}, p={r.trrust_tf_p_value:.4f}){'*' if r.trrust_tf_p_value < 0.05 else ''}  "
                  f"ChIP-seq={r.n_chipseq_tf:>3d} (prec={r.chipseq_tf_precision:.2f}, p={r.chipseq_tf_p_value:.4f}){'*' if r.chipseq_tf_p_value < 0.05 else ''}  ")
        print()

    # Plot: TF enrichment — line plot per dose rate (Oranges palette)
    graph_names = sorted(DOSE_RATE_LABELS, key=lambda d: DOSE_RATES_REGRESSION.get(d, 0))
    oranges_cmap = cm.get_cmap("OrangesDark")
    dose_colors = {gn: oranges_cmap(0.3 + 0.7 * i / (len(graph_names) - 1))
                   for i, gn in enumerate(graph_names)}

    fig, axes = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    for i,(ax, prec_col, bl_col, title) in enumerate([
        (axes[0], "trrust_tf_precision", "trrust_tf_baseline", "TRRUST TF Enrichment"),
        (axes[1], "chipseq_tf_precision", "chipseq_tf_baseline", "ChIP-seq TF Enrichment"),
    ]):
        for gn in graph_names:
            sub = df[df.graph == gn].sort_values("k")
            ax.plot(sub["k"], sub[prec_col], marker="o", color=dose_colors[gn],
                    label=f"{DOSE_RATES_REGRESSION[gn]} mGy/hr", linewidth=5, markersize=10)
            bl = sub.iloc[0][bl_col]
            ax.axhline(bl, color=dose_colors[gn], linestyle="--", linewidth=3, alpha=0.5)

        ax.set_xlabel("Top-k hub cutoff", fontsize=_fs_axis)
        ax.set_xticks(top_ks)
        ax.set_title(title, fontsize=_fs_title+3, fontweight="bold")
        if i==0:
            ax.set_ylabel("Precision (fraction of hubs that are TFs)", fontsize=_fs_axis)
        else:
            ax.legend( fontsize=_fs_leg)
            # ax.text(0.02, -0.15, "Dashed lines: |known TFs ∩ graph genes| / |graph genes|",
            #     transform=ax.transAxes, fontsize=(_fs_leg -5), fontstyle="italic", color="gray")
        ax.tick_params(axis='both', labelsize=_fs_tick)


    plt.tight_layout()
    plt.savefig(f"{output_dir}/hub_tf_enrichment.png", dpi=300)
    plt.close()
    print(f"Plot saved to {output_dir}/hub_tf_enrichment.png")

    return df

def load_housekeeping_set():
    """Load housekeeping gene symbols from HSIAO gene set."""
    with open(HK_PATH, "r") as f:
        data = json.load(f)
    return set(data["HSIAO_HOUSEKEEPING_GENES"]["geneSymbols"])


def sink_housekeeping_analysis(output_dir="."):
    """Check whether high in-degree sink nodes are enriched for housekeeping genes."""
    hk_set = load_housekeeping_set()
    top_ks = [10, 25, 50, 100]

    rows = []
    for graph_name, path in GRAPHS.items():
        G = nx.read_gexf(path)
        gene_nodes = [n for n in G.nodes if n not in EXCLUDE_NODES]
        n_total = len(gene_nodes)
        n_hk_total = sum(1 for n in gene_nodes if n in hk_set)

        in_deg = {n: G.in_degree(n) for n in gene_nodes}
        ranked = sorted(in_deg.items(), key=lambda x: x[1], reverse=True)

        for k in top_ks:
            if k > len(ranked):
                continue
            top_k_genes = [g for g, _ in ranked[:k]]

            n_hk_in = sum(1 for g in top_k_genes if g in hk_set)
            n_hk_rest = n_hk_total - n_hk_in
            _, pval = fisher_exact([
                [n_hk_in, k - n_hk_in],
                [n_hk_rest, (n_total - k) - n_hk_rest],
            ], alternative="greater")

            rows.append({
                "graph": graph_name,
                "k": k,
                "n_hk": n_hk_in,
                "hk_precision": n_hk_in / k,
                "hk_p_value": pval,
                "hk_baseline": n_hk_total / n_total,
                "n_nodes": n_total,
            })

    df = pd.DataFrame(rows)

    # Print summary
    print("\n=== Sink Housekeeping Enrichment ===\n")
    for graph_name in GRAPHS:
        sub = df[df.graph == graph_name]
        if sub.empty:
            continue
        print(f"{graph_name} ({sub.iloc[0]['n_nodes']} nodes, "
              f"HK baseline: {sub.iloc[0]['hk_baseline']:.3f})")
        for _, r in sub.iterrows():
            print(f"  top-{r.k:>3d}: "
                  f"HK={r.n_hk:>3d} (prec={r.hk_precision:.2f}, "
                  f"p={r.hk_p_value:.4f}){'*' if r.hk_p_value < 0.05 else ''}")
        print()

    # Plot: line plot per dose rate
    graph_names = sorted(DOSE_RATE_LABELS, key=lambda d: DOSE_RATES_REGRESSION.get(d, 0))
    oranges_cmap = cm.get_cmap("OrangesDark")
    dose_colors = {gn: oranges_cmap(0.3 + 0.7 * i / (len(graph_names) - 1))
                   for i, gn in enumerate(graph_names)}

    fig, ax = plt.subplots(figsize=(10, 10))
    for gn in graph_names:
        sub = df[df.graph == gn].sort_values("k")
        ax.plot(sub["k"], sub["hk_precision"], marker="o", color=dose_colors[gn],
                label=f"{DOSE_RATES_REGRESSION[gn]} mGy/hr", linewidth=5, markersize=10)
        bl = sub.iloc[0]["hk_baseline"]
        ax.axhline(bl, color=dose_colors[gn], linestyle="--", linewidth=3, alpha=0.5)

    ax.set_xlabel("Top-k sink cutoff", fontsize=_fs_axis)
    ax.set_xticks(top_ks)
    ax.set_ylabel("Precision (fraction of sinks that are housekeeping)", fontsize=_fs_axis)
    ax.set_title("Housekeeping Gene Enrichment", fontsize=_fs_title+3,fontweight="bold")
    ax.legend(fontsize=_fs_leg)
    # ax.text(0.02, -0.12, "Dashed lines: |housekeeping ∩ graph genes| / |graph genes|",
    #         transform=ax.transAxes, fontsize=(_fs_leg - 5), fontstyle="italic", color="gray")
    ax.tick_params(axis='both', labelsize=_fs_tick)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sink_housekeeping_enrichment.png", dpi=300)
    plt.savefig(f"{output_dir}/sink_housekeeping_enrichment.svg", dpi=300)
    plt.close()
    print(f"Plot saved to {output_dir}/sink_housekeeping_enrichment.png")

    return df


def bootstrap_tf_analysis(bootstrap_dir, output_dir="."):
    """Compute TF enrichment (TRRUST + ChIP-seq) across bootstrap replicates per dose."""
    trrust_tf_set = load_tf_set()
    chipseq_tf_set = load_chipseq_tf_set()
    combined_tf_set = trrust_tf_set | chipseq_tf_set
    top_ks = [10, 25, 50, 100]

    rows = []

    for d in DOSE_RATE_LABELS:
        _, _, bootstrap_data, gene_map = load_bootstrap_graphs(d, bootstrap_dir)

        n_boots = min(len(mats) for mats in bootstrap_data.values() if mats)

        for boot in range(n_boots):
            G_boot = build_bootstrap_graph(boot, bootstrap_data, gene_map)

            gene_nodes = [n for n in G_boot.nodes if n not in EXCLUDE_NODES]
            n_total = len(gene_nodes)
            if n_total == 0:
                continue

            n_trrust_total = sum(1 for n in gene_nodes if n in trrust_tf_set)
            n_chipseq_total = sum(1 for n in gene_nodes if n in chipseq_tf_set)
            n_combined_total = sum(1 for n in gene_nodes if n in combined_tf_set)

            out_deg = {n: G_boot.out_degree(n) for n in gene_nodes}
            ranked = sorted(out_deg.items(), key=lambda x: x[1], reverse=True)

            for k in top_ks:
                if k > len(ranked):
                    continue
                top_k_genes = [g for g, _ in ranked[:k]]

                n_trrust_in = sum(1 for g in top_k_genes if g in trrust_tf_set)
                n_trrust_rest = n_trrust_total - n_trrust_in
                _, pval_trrust = fisher_exact([
                    [n_trrust_in, k - n_trrust_in],
                    [n_trrust_rest, (n_total - k) - n_trrust_rest],
                ], alternative="greater")

                n_chipseq_in = sum(1 for g in top_k_genes if g in chipseq_tf_set)
                n_chipseq_rest = n_chipseq_total - n_chipseq_in
                _, pval_chipseq = fisher_exact([
                    [n_chipseq_in, k - n_chipseq_in],
                    [n_chipseq_rest, (n_total - k) - n_chipseq_rest],
                ], alternative="greater")

               
                rows.append({
                    "dose": d, "boot": boot, "k": k,
                    "trrust_precision": n_trrust_in / k,
                    "trrust_p_value": pval_trrust,
                    "trrust_baseline": n_trrust_total / n_total,
                    "chipseq_precision": n_chipseq_in / k,
                    "chipseq_p_value": pval_chipseq,
                    "chipseq_baseline": n_chipseq_total / n_total,
                })

        print(f"Dose {d}: processed {n_boots} bootstrap replicates")

    df = pd.DataFrame(rows)

    # Plot: 3 rows (TRRUST, ChIP-seq, Combined) x len(top_ks) columns
    tf_sources = [
        ("trrust_precision", "trrust_baseline", "TRRUST"),
        ("chipseq_precision", "chipseq_baseline", "ChIP-seq"),
    ]
    fig, axes = plt.subplots(len(tf_sources), len(top_ks),
                             figsize=(4 * len(top_ks), 4 * len(tf_sources)),
                             sharey="row")

    for row_idx, (prec_col, bl_col, source_name) in enumerate(tf_sources):
        for col_idx, k in enumerate(top_ks):
            ax = axes[row_idx][col_idx]
            data_by_dose = []
            colors = []
            for d in DOSE_RATES_SORTED:
                sub = df[(df.dose == d) & (df.k == k)]
                data_by_dose.append(sub[prec_col].values)
                colors.append(COLORS_BY_DOSE[d])

            bp = ax.boxplot(data_by_dose, positions=range(len(DOSE_RATES_SORTED)),
                            patch_artist=True, widths=0.6)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            for j, d in enumerate(DOSE_RATES_SORTED):
                sub = df[df.dose == d]
                if not sub.empty:
                    bl = sub[bl_col].iloc[0]
                    ax.plot([j - 0.35, j + 0.35], [bl, bl], "k--", linewidth=0.8)

            ax.set_xticks(range(len(DOSE_RATES_SORTED)))
            ax.set_xticklabels([f"{DOSE_RATES_REGRESSION[d]}" for d in DOSE_RATES_SORTED])
            if row_idx == 0:
                ax.set_title(f"top-{k}")
            if row_idx == len(tf_sources) - 1:
                ax.set_xlabel("Dose rate (mGy/hr)")
            if col_idx == 0:
                ax.set_ylabel(f"{source_name}\nTF precision")

    fig.suptitle("TF Enrichment Across Bootstrap Replicates", fontsize=_fs_title)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bootstrap_tf_enrichment.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_dir}/bootstrap_tf_enrichment.png")

    return df


def degree_distribution(output_dir="."):
    """Plot in-degree and out-degree distributions for each causal graph."""
    graph_names = list(GRAPHS.keys())
    n_graphs = len(graph_names)

    fig, axes = plt.subplots(n_graphs, figsize=(10, 3 * n_graphs))

    for i, (name, path) in enumerate(GRAPHS.items()):
        G = nx.read_gexf(path)
        gene_nodes = [n for n in G.nodes if n not in EXCLUDE_NODES]

        # in_deg = [G.in_degree(n) for n in gene_nodes]
        out_deg = [G.out_degree(n) for n in gene_nodes]
        ax = axes[i]
        max_deg = max(out_deg) if out_deg else 1
        n_bins = min(50, max_deg + 1)
        bins = np.linspace(-0.5, max_deg + 0.5, n_bins + 1)
        color='gray' if (name == 'invariant') else COLORS_BY_DOSE[name]
        ax.hist(out_deg, bins=bins, color=color, edgecolor="white", linewidth=0.5)
        ax.set_yscale("log")
        ax.set_ylabel(name, fontsize=_fs_axis, fontweight="bold")
        ax.set_xlabel("Out-degree" if i == n_graphs - 1 else "")
        if i == 0:
            ax.set_title("Out-degree distribution")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/degree_distributions.png", dpi=300)
    plt.close()
    print(f"\nDegree distribution plot saved to {output_dir}/degree_distributions.png")


def edge_overlap_analysis(output_dir="."):
    """Check overlap of causal graph edges with TRRUST, STRING, CORUM, and ChIP-seq."""
    # Load reference datasets
    trrust_df = pd.read_csv(TRRUST_PATH, sep="\t", header=None)
    trrust_directed = set(zip(trrust_df[0], trrust_df[1]))

    string_df = pd.read_csv(STRING_PATH)
    string_df = string_df[string_df.combined_score >= 400]
    string_edges = set(frozenset({r.protein1, r.protein2}) for _, r in string_df.iterrows())

    corum_edges = load_corum()
    corum_undirected = set(frozenset(e) for e in corum_edges)

    chipseq_directed = load_chipseq()

    print(f"\n=== Edge Overlap Analysis ===\n")
    print(f"Reference: TRRUST {len(trrust_directed)} directed, "
          f"STRING {len(string_edges)} undirected (score>=400), "
          f"CORUM {len(corum_undirected)} undirected, "
          f"ChIP-seq {len(chipseq_directed)} directed\n")

    rows = []
    weight_data = {}
    for graph_name, path in GRAPHS.items():
        G = nx.read_gexf(path)
        gene_nodes = set(n for n in G.nodes if n not in EXCLUDE_NODES)
        edges = [(u, v) for u, v in G.edges()
                 if u not in EXCLUDE_NODES and v not in EXCLUDE_NODES]
        predicted_edges = set(edges)
        n_edges = len(edges)

        # Reference edges scoped to genes in this graph (for recall)
        trrust_in_graph = {(u, v) for u, v in trrust_directed
                           if u in gene_nodes and v in gene_nodes}
        string_in_graph = {e for e in string_edges
                           if e.issubset(gene_nodes)}
        corum_in_graph = {e for e in corum_undirected
                          if e.issubset(gene_nodes)}
        chipseq_in_graph = {(u, v) for u, v in chipseq_directed
                            if u in gene_nodes and v in gene_nodes}

        w_trrust, w_string, w_corum, w_chipseq, w_novel = [], [], [], [], []
        n_trrust_dir, n_trrust_rev, n_string, n_corum, n_chipseq_dir, n_chipseq_rev = 0, 0, 0, 0, 0, 0
        for u, v in edges:
            w = abs(float(G[u][v].get("weight", 0)))
            in_trrust_dir = (u, v) in trrust_directed
            in_trrust_rev = (v, u) in trrust_directed and not in_trrust_dir
            in_string = frozenset({u, v}) in string_edges
            in_corum = frozenset({u, v}) in corum_undirected
            in_chipseq_dir = (u, v) in chipseq_directed
            in_chipseq_rev = (v, u) in chipseq_directed and not in_chipseq_dir

            if in_trrust_dir:
                n_trrust_dir += 1
                w_trrust.append(w)
            elif in_trrust_rev:
                n_trrust_rev += 1
                w_trrust.append(w)
            if in_string:
                n_string += 1
                w_string.append(w)
            if in_corum:
                n_corum += 1
                w_corum.append(w)
            if in_chipseq_dir:
                n_chipseq_dir += 1
                w_chipseq.append(w)
            elif in_chipseq_rev:
                n_chipseq_rev += 1
                w_chipseq.append(w)
            if not any([in_trrust_dir, in_trrust_rev, in_string, in_corum,
                        in_chipseq_dir, in_chipseq_rev]):
                w_novel.append(w)

        n_trrust_either = n_trrust_dir + n_trrust_rev
        n_chipseq_either = n_chipseq_dir + n_chipseq_rev
        weight_data[graph_name] = {
            "TRRUST": w_trrust, "STRING": w_string, "CORUM": w_corum,
            "ChIP-seq": w_chipseq, "Novel": w_novel
        }

        # F1 scores: precision = TP / predicted, recall = TP / reference_in_graph
        # For directed databases, TP = edges matching direction
        # For undirected databases, compare as frozensets
        def _f1(tp, n_predicted, n_ref):
            prec = tp / max(n_predicted, 1)
            rec = tp / max(n_ref, 1)
            return 2 * prec * rec / max(prec + rec, 1e-12), prec, rec

        f1_trrust, prec_trrust, rec_trrust = _f1(
            n_trrust_dir, n_edges, len(trrust_in_graph))
        f1_string, prec_string, rec_string = _f1(
            n_string, n_edges, len(string_in_graph))
        f1_corum, prec_corum, rec_corum = _f1(
            n_corum, n_edges, len(corum_in_graph))
        f1_chipseq, prec_chipseq, rec_chipseq = _f1(
            n_chipseq_dir, n_edges, len(chipseq_in_graph))

        rows.append({
            "graph": graph_name,
            "n_edges": n_edges,
            "trrust_directed": n_trrust_dir,
            "trrust_reversed": n_trrust_rev,
            "trrust_either": n_trrust_either,
            "trrust_dir_frac": n_trrust_dir / max(n_edges, 1),
            "trrust_rev_frac": n_trrust_rev / max(n_edges, 1),
            "trrust_either_frac": n_trrust_either / max(n_edges, 1),
            "trrust_recall": rec_trrust,
            "trrust_f1": f1_trrust,
            "string": n_string,
            "string_frac": n_string / max(n_edges, 1),
            "string_recall": rec_string,
            "string_f1": f1_string,
            "corum": n_corum,
            "corum_frac": n_corum / max(n_edges, 1),
            "corum_recall": rec_corum,
            "corum_f1": f1_corum,
            "chipseq_directed": n_chipseq_dir,
            "chipseq_reversed": n_chipseq_rev,
            "chipseq_either": n_chipseq_either,
            "chipseq_dir_frac": n_chipseq_dir / max(n_edges, 1),
            "chipseq_rev_frac": n_chipseq_rev / max(n_edges, 1),
            "chipseq_either_frac": n_chipseq_either / max(n_edges, 1),
            "chipseq_recall": rec_chipseq,
            "chipseq_f1": f1_chipseq,
        })

        print(f"{graph_name} ({n_edges} edges):")
        print(f"  TRRUST directed:  {n_trrust_dir:>4d} (prec={prec_trrust:.3f}, rec={rec_trrust:.3f}, F1={f1_trrust:.3f})")
        print(f"  TRRUST reversed:  {n_trrust_rev:>4d} ({n_trrust_rev/max(n_edges,1):.3f})")
        print(f"  STRING:           {n_string:>4d} (prec={prec_string:.3f}, rec={rec_string:.3f}, F1={f1_string:.3f})")
        print(f"  CORUM:            {n_corum:>4d} (prec={prec_corum:.3f}, rec={rec_corum:.3f}, F1={f1_corum:.3f})")
        print(f"  ChIP-seq directed:{n_chipseq_dir:>4d} (prec={prec_chipseq:.3f}, rec={rec_chipseq:.3f}, F1={f1_chipseq:.3f})")
        print(f"  ChIP-seq reversed:{n_chipseq_rev:>4d} ({n_chipseq_rev/max(n_edges,1):.3f})")

        # Weight comparison
        for src, w_src in [("TRRUST", w_trrust), ("STRING", w_string),
                           ("CORUM", w_corum), ("ChIP-seq", w_chipseq)]:
            if w_src and w_novel:
                _, pval = mannwhitneyu(w_src, w_novel, alternative="greater")
                print(f"  |weight| {src} vs novel: "
                      f"median {np.median(w_src):.3f} vs {np.median(w_novel):.3f} "
                      f"(Mann-Whitney p={pval:.4f}{'*' if pval < 0.05 else ''})")
        print()

    df = pd.DataFrame(rows)
    df.to_csv(f"{output_dir}/overlap.csv")
    # Sort by actual dose rate for plotting
    graph_names = sorted(DOSE_RATE_LABELS, key=lambda d: DOSE_RATES_REGRESSION.get(d, 0))
    df = df.set_index("graph").loc[graph_names].reset_index()
    # Plot: Edge overlap fractions
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(graph_names))
    n_bars = 6
    width = 0.8 / n_bars

    _eo_colors = [_CMAP(v) for v in np.linspace(0.3, 1.0, n_bars)]
    ax.bar(x - 2.5*width, df["trrust_dir_frac"], width, label="TRRUST directed", color=_eo_colors[0])
    ax.bar(x - 1.5*width, df["trrust_rev_frac"], width, label="TRRUST reversed", color=_eo_colors[1])
    ax.bar(x - 0.5*width, df["string_frac"], width, label="STRING", color=_eo_colors[2])
    ax.bar(x + 0.5*width, df["corum_frac"], width, label="CORUM", color=_eo_colors[3])
    ax.bar(x + 1.5*width, df["chipseq_dir_frac"], width, label="ChIP-seq directed", color=_eo_colors[4])
    ax.bar(x + 2.5*width, df["chipseq_rev_frac"], width, label="ChIP-seq reversed", color=_eo_colors[5])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{DOSE_RATES_REGRESSION[g]} mGy/hr" for g in graph_names])
    ax.set_ylabel("Fraction of causal edges in reference")
    ax.set_title("Edge Overlap with Known Interaction Databases")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/edge_overlap.png", dpi=300)
    plt.close()
    print(f"Plot saved to {output_dir}/edge_overlap.png")

    # Plot 2: F1 scores per database
    fig, ax = plt.subplots(figsize=(12, 5))
    n_bars = 4
    width = 0.8 / n_bars

    _f1_colors = [_CMAP(v) for v in np.linspace(0.3, 1.0, n_bars)]
    ax.bar(x - 1.5*width, df["trrust_f1"], width, label="TRRUST", color=_f1_colors[0])
    ax.bar(x - 0.5*width, df["string_f1"], width, label="STRING", color=_f1_colors[1])
    ax.bar(x + 0.5*width, df["corum_f1"], width, label="CORUM", color=_f1_colors[2])
    ax.bar(x + 1.5*width, df["chipseq_f1"], width, label="ChIP-seq", color=_f1_colors[3])

    ax.set_xticks(x)
    ax.set_xticklabels([f"{DOSE_RATES_REGRESSION[g]} mGy/hr" for g in graph_names])
    ax.set_ylabel("F1 Score")
    ax.set_title("Edge Overlap F1 Scores (precision vs recall against reference)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/edge_overlap_f1.png", dpi=300)
    plt.close()
    print(f"Plot saved to {output_dir}/edge_overlap_f1.png")

    # Plot 3: Edge weight distributions by source
    n_graphs = len(graph_names)
    fig, axes = plt.subplots(1, n_graphs, figsize=(3.5 * n_graphs, 4), sharey=True)
    for i, gn in enumerate(graph_names):
        ax = axes[i]
        wd = weight_data[gn]
        all_sources = ["TRRUST", "STRING", "CORUM", "ChIP-seq", "Novel"]
        data = [v for v in [wd[s] for s in all_sources] if v]
        labels = [l for l, v in zip(all_sources, [wd[s] for s in all_sources]) if v]
        ax.boxplot(data, tick_labels=labels, showfliers=False)
        ax.set_title(f"{DOSE_RATES_REGRESSION[gn]} mGy/hr")
        if i == 0:
            ax.set_ylabel("|Edge weight|")
    fig.suptitle("Edge Weight by Reference Match", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/edge_weight_by_source.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_dir}/edge_weight_by_source.png")

    return df


def bootstrap_edge_overlap(bootstrap_dir, output_dir="."):
    """Compute edge overlap precision and F1 across bootstrap replicates."""
    # Load reference datasets
    trrust_df = pd.read_csv(TRRUST_PATH, sep="\t", header=None)
    trrust_directed = set(zip(trrust_df[0], trrust_df[1]))

    string_df = pd.read_csv(STRING_PATH)
    string_df = string_df[string_df.combined_score >= 400]
    string_edges = set(frozenset({r.protein1, r.protein2}) for _, r in string_df.iterrows())

    corum_edges = load_corum()
    corum_undirected = set(frozenset(e) for e in corum_edges)

    chipseq_directed = load_chipseq()

    ref_sources = ["TRRUST", "STRING", "CORUM", "ChIP-seq"]
    rows = []

    for d in DOSE_RATE_LABELS:
        _, _, bootstrap_data, gene_map = load_bootstrap_graphs(d, bootstrap_dir)
        n_boots = min(len(mats) for mats in bootstrap_data.values() if mats)

        for boot in range(n_boots):
            G_boot = build_bootstrap_graph(boot, bootstrap_data, gene_map)

            gene_nodes = set(n for n in G_boot.nodes if n not in EXCLUDE_NODES)
            edges = [(u, v) for u, v in G_boot.edges()
                     if u not in EXCLUDE_NODES and v not in EXCLUDE_NODES]
            n_edges = len(edges)
            if n_edges == 0:
                continue

            # Scope reference to genes in this graph
            trrust_in_graph = {(u, v) for u, v in trrust_directed
                               if u in gene_nodes and v in gene_nodes}
            string_in_graph = {e for e in string_edges if e.issubset(gene_nodes)}
            corum_in_graph = {e for e in corum_undirected if e.issubset(gene_nodes)}
            chipseq_in_graph = {(u, v) for u, v in chipseq_directed
                                if u in gene_nodes and v in gene_nodes}

            # Count TPs
            n_trrust = sum(1 for u, v in edges if (u, v) in trrust_directed)
            n_string = sum(1 for u, v in edges if frozenset({u, v}) in string_edges)
            n_corum = sum(1 for u, v in edges if frozenset({u, v}) in corum_undirected)
            n_chipseq = sum(1 for u, v in edges if (u, v) in chipseq_directed)

            for src, tp, n_ref in [
                ("TRRUST", n_trrust, len(trrust_in_graph)),
                ("STRING", n_string, len(string_in_graph)),
                ("CORUM", n_corum, len(corum_in_graph)),
                ("ChIP-seq", n_chipseq, len(chipseq_in_graph)),
            ]:
                prec = tp / n_edges
                rec = tp / max(n_ref, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-12)
                rows.append({
                    "dose": d, "boot": boot, "source": src,
                    "precision": prec, "recall": rec, "f1": f1,
                })

        print(f"Dose {d}: processed {n_boots} bootstrap replicates for edge overlap")

    df = pd.DataFrame(rows)

    # Plot: 2 rows (precision, F1) x 4 columns (one per reference)
    fig, axes = plt.subplots(2, len(ref_sources),
                             figsize=(4.5 * len(ref_sources), 8),
                             sharey="row")

    for col, src in enumerate(ref_sources):
        for row, (metric, ylabel) in enumerate([
            ("precision", "Precision"),
            ("f1", "F1 Score"),
        ]):
            ax = axes[row][col]
            data_by_dose = []
            colors = []
            for dose in DOSE_RATES_SORTED:
                sub = df[(df.dose == dose) & (df.source == src)]
                data_by_dose.append(sub[metric].values)
                colors.append(COLORS_BY_DOSE[dose])

            bp = ax.boxplot(data_by_dose, positions=range(len(DOSE_RATES_SORTED)),
                            patch_artist=True, widths=0.6)
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xticks(range(len(DOSE_RATES_SORTED)))
            ax.set_xticklabels([f"{DOSE_RATES_REGRESSION[d]}" for d in DOSE_RATES_SORTED])
            if row == 0:
                ax.set_title(src)
            if row == 1:
                ax.set_xlabel("Dose rate (mGy/hr)")
            if col == 0:
                ax.set_ylabel(ylabel)

    fig.suptitle("Edge Overlap Across Bootstrap Replicates", fontsize=_fs_title)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/bootstrap_edge_overlap.png", dpi=300,
                bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_dir}/bootstrap_edge_overlap.png")

    return df

def bootstrap_stability_analysis(d, bootstrap_dir, output_dir="."):
    """Analyze gene-level stability across bootstrap replicates."""
    print(f"\n=== Bootstrap Stability Analysis Does Rate {d} ===\n")

    combined, per_partition, bootstrap_data, gene_map = load_bootstrap_graphs(
        d, bootstrap_dir
    )

    all_stability = []  # (gene, part, mean_degree, degree_cv, stable_edge_frac)
    all_edge_freqs = []  # per-edge bootstrap frequencies across all partitions
    all_edge_records = []  # (source, target, frequency) for every edge

    for part in sorted(bootstrap_data.keys()):
        matrices = bootstrap_data[part]
        if len(matrices) < 2:
            continue
        genes = gene_map[part]
        n_genes = len(genes)
        n_boots = len(matrices)

        # Degree per bootstrap (out + in)
        degrees = np.zeros((n_boots, n_genes))
        for b, m in enumerate(matrices):
            m_gene = m[:n_genes, :n_genes]
            degrees[b] = (m_gene != 0).sum(axis=1) + (m_gene != 0).sum(axis=0)

        mean_deg = degrees.mean(axis=0)
        std_deg = degrees.std(axis=0)
        cv_deg = np.where(mean_deg > 0, std_deg / mean_deg, 0)

        # Edge stability per gene: mean bootstrap frequency of each gene's edges
        stack = np.array([m[:n_genes, :n_genes] for m in matrices])
        edge_freq = (stack != 0).mean(axis=0)  # fraction of boots each edge appears in

        # Collect per-edge frequencies (off-diagonal nonzero entries)
        mask = edge_freq > 0
        np.fill_diagonal(mask, False)
        all_edge_freqs.extend(edge_freq[mask].tolist())
        for i, j in zip(*np.where(mask)):
            all_edge_records.append((genes[i], genes[j], edge_freq[i, j]))

        for i in range(n_genes):
            row_freq = edge_freq[i, :]
            col_freq = edge_freq[:, i]
            all_freq = np.concatenate([row_freq, col_freq])
            total_edges = (all_freq > 0).sum()
            mean_edge_freq = all_freq[all_freq > 0].mean() if total_edges > 0 else 0

            all_stability.append({
                "gene": genes[i],
                "part": part,
                "mean_degree": mean_deg[i],
                "degree_cv": cv_deg[i],
                "mean_edge_freq": mean_edge_freq,
                "n_total_edges": total_edges,
            })

    df = pd.DataFrame(all_stability)

    # Print top stable genes (by mean edge frequency, filtered to genes with edges)
    has_edges = df[df.n_total_edges > 0].copy()
    if not has_edges.empty:
        gene_agg = has_edges.groupby("gene").agg(
            mean_degree=("mean_degree", "mean"),
            degree_cv=("degree_cv", "mean"),
            mean_edge_freq=("mean_edge_freq", "mean"),
            n_total_edges=("n_total_edges", "sum"),
            n_partitions=("part", "count"),
        ).sort_values("mean_edge_freq", ascending=False)

        print("Top 20 most stable genes (highest mean edge bootstrap frequency):")
        print(gene_agg.head(20).to_string())
        print()

        print("Top 20 least stable genes (highest degree CV, min 2 mean degree):")
        unstable = gene_agg[gene_agg.mean_degree >= 2].sort_values(
            "degree_cv", ascending=False
        )
        print(unstable.head(20).to_string())
        print()

    # Report edges present in 100% of bootstraps with TRRUST/STRING annotation
    trrust_df = pd.read_csv(TRRUST_PATH, sep="\t", header=None)
    trrust_directed = set(zip(trrust_df[0], trrust_df[1]))
    string_df = pd.read_csv(STRING_PATH)
    string_df = string_df[string_df.combined_score >= 400]
    string_edges = set(
        frozenset({r.protein1, r.protein2}) for _, r in string_df.iterrows()
    )

    perfect_edges = [(s, t) for s, t, f in all_edge_records if f >= 1.0]
    n_trrust = sum(1 for s, t in perfect_edges if (s, t) in trrust_directed)
    n_trrust_rev = sum(1 for s, t in perfect_edges
                       if (t, s) in trrust_directed and (s, t) not in trrust_directed)
    n_string = sum(1 for s, t in perfect_edges if frozenset({s, t}) in string_edges)

    print(f"Edges in 100% of bootstraps (dose {d}): {len(perfect_edges)}")
    print(f"  TRRUST directed: {n_trrust}  |  TRRUST reversed: {n_trrust_rev}  |  STRING: {n_string}")
    for src, tgt in sorted(perfect_edges):
        flags = []
        if (src, tgt) in trrust_directed:
            flags.append("TRRUST")
        elif (tgt, src) in trrust_directed:
            flags.append("TRRUST(rev)")
        if frozenset({src, tgt}) in string_edges:
            flags.append("STRING")
        annotation = f"  [{', '.join(flags)}]" if flags else ""
        print(f"  {src} -> {tgt}{annotation}")
    print()

    # Save per-edge bootstrap frequencies to CSV
    edge_freq_df = pd.DataFrame(all_edge_records, columns=["source", "target", "frequency"])
    edge_freq_path = f"{output_dir}/bootstrap_edge_freq_{d}.csv"
    edge_freq_df.to_csv(edge_freq_path, index=False)
    print(f"Edge frequencies saved: {edge_freq_path} ({len(edge_freq_df)} edges)")

    # Load predefined clusters from cluster_enrichment_analysis
    mapping_path = os.path.join(output_dir, "cluster_gene_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            cluster_gene_mapping = json.load(f)
    else:
        cluster_gene_mapping = {}

    # Load all validation sources
    corum_edges = load_corum()
    corum_undirected = set(frozenset(e) for e in corum_edges)
    chipseq_directed = load_chipseq()

    # Plot 1: Per-edge bootstrap frequency distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    if all_edge_freqs:
        ax.hist(all_edge_freqs, bins=20, color=_CMAP(0.6), edgecolor="white")
    ax.set_xlabel("Bootstrap frequency (fraction of replicates)")
    ax.set_ylabel("Number of edges")
    ax.set_title("Edge Bootstrap Frequency Distribution")

    # Plot 2: Degree CV distribution
    ax = axes[1]
    vals = has_edges["degree_cv"]
    ax.hist(vals[vals > 0], bins=30, color=_CMAP(0.8), edgecolor="white")
    ax.set_xlabel("Degree coefficient of variation")
    ax.set_ylabel("Number of genes")
    ax.set_title("Degree Stability Distribution")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/bootstrap_stability_{d}.png", dpi=300)
    plt.close()
    print(f"Plot saved to {output_dir}/bootstrap_stability_{d}.png")

    return combined, df


def run_bootstrap_structural_analysis(bootstrap_dir, output_dir="."):
    """Run full structural analysis on bootstrap consensus graph."""
    print("\n=== Bootstrap Consensus Graph Structural Analysis ===\n")

    for d in DOSE_RATE_LABELS:
        combined, stability_df = bootstrap_stability_analysis(d, bootstrap_dir, output_dir)

        # Run hub-TF and edge overlap on the combined graph directly
        tf_set = load_tf_set()
        string_hub_set, _ = load_string_hubs()
        top_ks = [10, 25, 50, 100]

        gene_nodes = [n for n in combined.nodes if n not in EXCLUDE_NODES]
        n_total = len(gene_nodes)
        n_tf_total = sum(1 for n in gene_nodes if n in tf_set)

        out_deg = {n: combined.out_degree(n) for n in gene_nodes}
        ranked = sorted(out_deg.items(), key=lambda x: x[1], reverse=True)

        print(f"\nBootstrap consensus Dose {d}: {n_total} genes, {combined.number_of_edges()} edges")
        print(f"TF baseline Dose {d} : {n_tf_total/n_total:.3f}\n")

        print(f"Hub-TF enrichment (bootstrap consensus) Dose {d}:")
        for k in top_ks:
            if k > len(ranked):
                continue
            top_k = [g for g, _ in ranked[:k]]
            n_tf = sum(1 for g in top_k if g in tf_set)
            n_tf_rest = n_tf_total - n_tf
            _, pval = fisher_exact([
                [n_tf, k - n_tf],
                [n_tf_rest, (n_total - k) - n_tf_rest],
            ], alternative="greater")
            sig = "*" if pval < 0.05 else ""
            print(f"  top-{k:>3d}: {n_tf:>3d} TFs "
                f"(precision={n_tf/k:.2f}, p={pval:.4f}){sig}")
        print()

        # Top hubs
        print("Top 20 hubs by out-degree:")
        for gene, deg in ranked[:20]:
            is_tf = "TF" if gene in tf_set else ""
            print(f"  {gene:>15s}: out-degree={deg:>4d} {is_tf}")
        print()

        # Edge overlap
        trrust_df = pd.read_csv(TRRUST_PATH, sep="\t", header=None)
        trrust_directed = set(zip(trrust_df[0], trrust_df[1]))
        string_df = pd.read_csv(STRING_PATH)
        string_df = string_df[string_df.combined_score >= 400]
        string_edges = set(
            frozenset({r.protein1, r.protein2}) for _, r in string_df.iterrows()
        )

        edges = [(u, v) for u, v in combined.edges()
                if u not in EXCLUDE_NODES and v not in EXCLUDE_NODES]
        n_edges = len(edges)
        n_trrust_dir = sum(1 for u, v in edges if (u, v) in trrust_directed)
        n_trrust_rev = sum(1 for u, v in edges
                        if (v, u) in trrust_directed and (u, v) not in trrust_directed)
        n_string = sum(1 for u, v in edges if frozenset({u, v}) in string_edges)

        print(f"Edge overlap (bootstrap consensus, {n_edges} edges) Dose Rate {d}:")
        print(f"  TRRUST directed:  {n_trrust_dir:>4d} ({n_trrust_dir/max(n_edges,1):.3f})")
        print(f"  TRRUST reversed:  {n_trrust_rev:>4d} ({n_trrust_rev/max(n_edges,1):.3f})")
        print(f"  STRING:           {n_string:>4d} ({n_string/max(n_edges,1):.3f})")


def cross_dose_invariant_analysis(output_dir="."):
    """Analyze the structure of gene sets and edges that are shared across all dose rates.
    """
    print("\n=== Cross-Dose Invariant Subgraph Analysis ===\n")

    # Load all dose graphs and extract gene nodes
    dose_graphs = {}
    dose_gene_sets = {}
    for d in DOSE_RATE_LABELS:
        G = nx.read_gexf(GRAPHS[d])
        gene_nodes = set(n for n in G.nodes() if n not in EXCLUDE_NODES)
        dose_graphs[d] = G
        dose_gene_sets[d] = gene_nodes

    # Invariant gene set = intersection across all doses
    invariant_genes = set.intersection(*dose_gene_sets.values())
    print(f"Genes per dose: {', '.join(f'{d}={len(g)}' for d, g in dose_gene_sets.items())}")
    print(f"Invariant gene set (intersection): {len(invariant_genes)} genes\n")

    # Extract subgraphs induced by invariant genes
    subgraphs = {}
    edge_sets_directed = {}    # {dose: set of (u,v) tuples}
    edge_sets_undirected = {}  # {dose: set of frozenset({u,v})}
    for d in DOSE_RATE_LABELS:
        sub = dose_graphs[d].subgraph(invariant_genes).copy()
        subgraphs[d] = sub
        edges_d = set((u, v) for u, v in sub.edges())
        edge_sets_directed[d] = edges_d
        edge_sets_undirected[d] = set(frozenset({u, v}) for u, v in edges_d)
        print(f"  Dose {d} subgraph: {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges")

    # ---  Edge conservation spectrum ---
    all_directed_edges = set()
    for edges in edge_sets_directed.values():
        all_directed_edges |= edges
    all_undirected_edges = set()
    for edges in edge_sets_undirected.values():
        all_undirected_edges |= edges

    # Count how many doses each undirected edge appears in
    edge_dose_count = {}
    for e in all_undirected_edges:
        count = sum(1 for d in DOSE_RATE_LABELS if e in edge_sets_undirected[d])
        edge_dose_count[e] = count

    spectrum = Counter(edge_dose_count.values())
    print(f"\nEdge conservation spectrum (undirected):")
    for k in sorted(spectrum.keys()):
        print(f"  Present in {k}/{len(DOSE_RATE_LABELS)} doses: {spectrum[k]} edges")

    n_conserved = sum(1 for c in edge_dose_count.values() if c == len(DOSE_RATE_LABELS))
    n_total = len(all_undirected_edges)
    print(f"\n  Fully conserved (all doses): {n_conserved}/{n_total} "
          f"({100*n_conserved/max(n_total,1):.1f}%)")
    print(f"  Dose-specific (1 dose only): {spectrum.get(1, 0)}/{n_total} "
          f"({100*spectrum.get(1,0)/max(n_total,1):.1f}%)")

    # ---  Direction consistency ---
    # For edges present in >=2 doses, check if direction is always the same
    multi_dose_edges = {e for e, c in edge_dose_count.items() if c >= 2}
    n_consistent = 0
    n_inconsistent = 0
    for e in multi_dose_edges:
        u, v = tuple(e)
        doses_with_edge = [d for d in DOSE_RATE_LABELS if e in edge_sets_undirected[d]]
        directions = set()
        for d in doses_with_edge:
            if (u, v) in edge_sets_directed[d]:
                directions.add((u, v))
            if (v, u) in edge_sets_directed[d]:
                directions.add((v, u))
        if len(directions) == 1:
            n_consistent += 1
        else:
            n_inconsistent += 1

    print(f"\nDirection consistency (edges in >=2 doses):")
    print(f"  Consistent direction: {n_consistent}/{n_consistent + n_inconsistent}")
    print(f"  Inconsistent (flipped): {n_inconsistent}/{n_consistent + n_inconsistent}")

    # ---  Validation enrichment by conservation level ---
    trrust_df = pd.read_csv(TRRUST_PATH, sep="\t", header=None)
    trrust_directed = set(zip(trrust_df[0], trrust_df[1]))
    trrust_undirected = set(frozenset({a, b}) for a, b in trrust_directed)
    string_df = pd.read_csv(STRING_PATH)
    string_df = string_df[string_df.combined_score >= 400]
    string_undirected = set(frozenset({r.protein1, r.protein2})
                            for _, r in string_df.iterrows())
    corum_edges = load_corum()
    corum_undirected = set(frozenset(e) for e in corum_edges)
    chipseq_directed = load_chipseq()
    chipseq_undirected = set(frozenset({a, b}) for a, b in chipseq_directed)

    print(f"\nValidation enrichment by conservation level:")
    print(f"  {'Doses':>5s}  {'N edges':>7s}  {'TRRUST':>7s}  {'STRING':>7s}  "
          f"{'CORUM':>7s}  {'ChIP':>7s}  {'Any':>7s}")
    for k in sorted(spectrum.keys()):
        edges_at_k = [e for e, c in edge_dose_count.items() if c == k]
        n = len(edges_at_k)
        n_tr = sum(1 for e in edges_at_k if e in trrust_undirected)
        n_st = sum(1 for e in edges_at_k if e in string_undirected)
        n_co = sum(1 for e in edges_at_k if e in corum_undirected)
        n_ch = sum(1 for e in edges_at_k if e in chipseq_undirected)
        n_any = sum(1 for e in edges_at_k
                    if e in trrust_undirected or e in string_undirected
                    or e in corum_undirected or e in chipseq_undirected)
        print(f"  {k:>5d}  {n:>7d}  {n_tr/max(n,1):>7.1%}  {n_st/max(n,1):>7.1%}  "
              f"{n_co/max(n,1):>7.1%}  {n_ch/max(n,1):>7.1%}  {n_any/max(n,1):>7.1%}")

    # --- Save invariant-gene subgraphs as GEXF per dose ---
    mapping_path = os.path.join(output_dir, "cluster_gene_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            cluster_gene_mapping = json.load(f)
    else:
        cluster_gene_mapping = {}

    for d in DOSE_RATE_LABELS:
        G_sub = subgraphs[d]

        # Build gene -> cluster_id mapping
        gene_to_cluster = {}
        if d in cluster_gene_mapping:
            for cluster_name, genes_in_cluster in cluster_gene_mapping[d].items():
                for gene in genes_in_cluster:
                    gene_to_cluster[gene] = cluster_name

        # Node attributes: cluster_id
        for node in G_sub.nodes():
            G_sub.nodes[node]["cluster_id"] = gene_to_cluster.get(node, "unassigned")

        # Edge attributes: validation sources
        for u, v in G_sub.edges():
            G_sub.edges[u, v]["trrust"] = (u, v) in trrust_directed
            G_sub.edges[u, v]["trrust_rev"] = (v, u) in trrust_directed and (u, v) not in trrust_directed
            G_sub.edges[u, v]["string"] = frozenset({u, v}) in string_undirected
            G_sub.edges[u, v]["corum"] = frozenset({u, v}) in corum_undirected
            G_sub.edges[u, v]["chipseq"] = (u, v) in chipseq_directed
            G_sub.edges[u, v]["chipseq_rev"] = (v, u) in chipseq_directed and (u, v) not in chipseq_directed

        gexf_path = f"{output_dir}/invariant_subgraph_{d}.gexf"
        nx.write_gexf(G_sub, gexf_path)
        print(f"Invariant subgraph saved: {gexf_path} "
              f"({G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges)")

    # Compare invariant subgraph edges with perfect (100% bootstrap) edges
    print(f"\n--- Invariant subgraph vs perfect bootstrap edges ---")
    for d in DOSE_RATE_LABELS:
        perfect_path = f"{output_dir}/perfect_edges_{d}.gexf"
        if not os.path.exists(perfect_path):
            print(f"  Dose {d}: no perfect edges file found, skipping")
            continue
        G_perf = nx.read_gexf(perfect_path)
        perfect_set = set(G_perf.edges())
        invariant_set = set(subgraphs[d].edges())
        overlap = perfect_set & invariant_set
        pct_of_invariant = 100 * len(overlap) / max(len(invariant_set), 1)
        pct_of_perfect = 100 * len(overlap) / max(len(perfect_set), 1)
        print(f"  Dose {d}: invariant={len(invariant_set)} edges, "
              f"perfect={len(perfect_set)} edges, overlap={len(overlap)} "
              f"({pct_of_invariant:.1f}% of invariant, {pct_of_perfect:.1f}% of perfect)")

    # Plot bootstrap frequency distributions: invariant vs non-invariant edges
    # Two rows per dose: top = other, bottom = invariant
    n_doses = len(DOSE_RATES_SORTED)
    fig, axes = plt.subplots(2, n_doses, figsize=(4 * n_doses, 7), sharey="row")
    if n_doses == 1:
        axes = axes.reshape(2, 1)
    bins = np.linspace(0, 1, 21)

    for ci, d in enumerate(DOSE_RATES_SORTED):
        freq_path = f"{output_dir}/bootstrap_edge_freq_{d}.csv"
        dose_label = f"{DOSE_RATES_REGRESSION[d]} mGy/hr"
        if not os.path.exists(freq_path):
            axes[0, ci].set_title(f"{dose_label}\n(no data)")
            continue
        edge_freq_df = pd.read_csv(freq_path)
        invariant_edges_d = set((u, v) for u, v in subgraphs[d].edges())

        freq_invariant = []
        freq_other = []
        for _, row in edge_freq_df.iterrows():
            if (row["source"], row["target"]) in invariant_edges_d:
                freq_invariant.append(row["frequency"])
            else:
                freq_other.append(row["frequency"])

        axes[0, ci].hist(freq_other, bins=bins, color=COLORS_BY_DOSE[d], alpha=0.5,edgecolor="white")
        med_oth = np.median(freq_other) if freq_other else 0
        axes[0, ci].set_title(f"{dose_label} — Other (n={len(freq_other)})\nmedian={med_oth:.2f}",
                              fontsize=_fs_axis-7)
        axes[0, ci].set_xlabel("Bootstrap freq")

        axes[1, ci].hist(freq_invariant, bins=bins, color=COLORS_BY_DOSE[d], edgecolor="white")
        med_inv = np.median(freq_invariant) if freq_invariant else 0
        axes[1, ci].set_title(f"{dose_label} — Invariant (n={len(freq_invariant)})\nmedian={med_inv:.2f}",
                              fontsize=_fs_axis-7)
        axes[1, ci].set_xlabel("Bootstrap freq")

        print(f"  Dose {d}: invariant median={med_inv:.2f} (n={len(freq_invariant)}), "
              f"other median={med_oth:.2f} (n={len(freq_other)})")

    axes[0, 0].set_ylabel("Number of edges")
    axes[1, 0].set_ylabel("Number of edges")
    plt.suptitle("Bootstrap Edge Frequency: Other vs Invariant Edges", fontsize=_fs_title)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/invariant_bootstrap_freq.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_dir}/invariant_bootstrap_freq.png")

    # Save invariant gene set
    invariant_path = f"{output_dir}/invariant_genes.pkl"
    with open(invariant_path, "wb") as f:
        pickle.dump(sorted(invariant_genes), f)
    print(f"Invariant gene set saved: {invariant_path} ({len(invariant_genes)} genes)")

    return invariant_genes, edge_dose_count


def _cluster_graph(G, exclude_nodes):
    """Cluster an undirected view of G after removing exclude_nodes."""
    G_clean = G.copy()
    G_clean.remove_nodes_from([n for n in exclude_nodes if n in G_clean])
    G_und = G_clean.to_undirected()
    communities = list(nx.community.greedy_modularity_communities(G_und))
    communities = sorted(communities, key=len, reverse=True)
    return communities, G_und


def cluster_enrichment_analysis(output_dir="."):
    """Cluster per-dose Variant B graphs, run pathway enrichment, and produce
    a single combined plot with dose-specific color schemes per cluster."""
    tpm_df = pd.read_csv(TPM_PATH, header=0, sep="\t")
    background_genes = list(set(tpm_df["Gene"]))

    print("\n=== Graph Clustering & Pathway Enrichment ===\n")

    cluster_gene_mapping = {}  # {dose: {cluster_name: [genes]}}
    exclude_b = EXCLUDE_NODES - {"radiation"}
    mapping_path = os.path.join(output_dir, "cluster_gene_mapping.json")
    if os.path.exists(mapping_path):
         with open(mapping_path, "r") as f:
            cluster_gene_mapping = json.load(f)

    if not os.path.exists(mapping_path):
        for dose in DOSE_RATE_LABELS:
            G = nx.read_gexf(GRAPHS[dose])
            communities, G_und = _cluster_graph(G, exclude_b)

            # Note which cluster has radiation
            for ci, comm in enumerate(communities):
                if "radiation" in comm:
                    print(f"  'radiation' is in cluster {ci} (size {len(comm)})")
                    break

            cluster_gene_mapping[dose] = {}

            for ci, comm in enumerate(communities):
                genes = sorted(g for g in comm if g not in EXCLUDE_NODES)
                cluster_name = f"{dose}_C{ci}"
                cluster_gene_mapping[dose][cluster_name] = genes
                print(f"  {cluster_name}: {len(genes)} genes")
        # Save cluster-to-genes mapping
        with open(mapping_path, "w") as f:
            json.dump(cluster_gene_mapping, f, indent=2)
        print(f"\nCluster-gene mapping saved to {mapping_path}")


    # Save GEXF files with cluster_id node attribute
    for dose in DOSE_RATE_LABELS:
        G = nx.read_gexf(GRAPHS[dose])
        gene_to_cluster = {}
        for cluster_name, genes in cluster_gene_mapping[dose].items():
            for gene in genes:
                gene_to_cluster[gene] = cluster_name
        for node in G.nodes():
            G.nodes[node]["cluster_id"] = gene_to_cluster.get(node, "excluded")
        out_path = os.path.join(output_dir, f"graph_clustered_dose{dose}.gexf")
        nx.write_gexf(G, out_path)
        print(f"GEXF saved: {out_path}")

    # Collect results across all doses: list of (dose, cluster_name, n_genes, top10_df)
    all_results_pe = []
    all_results_corum = []
    corum_mapping = load_corum_gene_clusters()
    
    for d in DOSE_RATE_LABELS:
        clusters = cluster_gene_mapping[d]
        for i, genes in clusters.items():
            if len(genes) < 10:
                continue

            try:
                pe = run_pathway_enrichment(genes, background_genes, None)
                pe = pe.sort_values("p_value").query("term_size < 300")
                top5 = pe.head(5)
            except Exception as e:
                print(f"    Enrichment failed: {e}")
                continue

            if top5.empty:
                print("    No significant pathways")
                continue

            all_results_pe.append((d, i, len(genes), top5))
            print(f"    Top pathways:")
            for _, row in top5.iterrows():
                print(f"      {row['name']} ({row['source']}) p={row['p_value']:.2e}")

            best_complex, overlap, go_terms = corum_enrichment(genes, mapping=corum_mapping)

            # Fisher's exact test for CORUM overlap significance
            if best_complex is not None and len(overlap) > 0:
                n_background = len(background_genes)
                cluster_size = len(genes)
                complex_size = len(best_complex)
                overlap_size = len(overlap)
                table = [
                    [overlap_size, cluster_size - overlap_size],
                    [complex_size - overlap_size,
                     n_background - cluster_size - complex_size + overlap_size],
                ]
                _, corum_p = fisher_exact(table, alternative="greater")
            else:
                corum_p = 1.0

            all_results_corum.append((d, i, len(genes), best_complex, overlap, go_terms, corum_p))
            print(f"    CORUM best match: {len(overlap)}/{len(best_complex) if best_complex else 0} "
                  f"overlap (Fisher p={corum_p:.2e})")

    # Combined plot: 5 columns (one per dose), clusters stacked as rows
    if not all_results_pe:
        return

    # Build CORUM lookup: (dose, cluster_name) -> (overlap_ratio_str, go_terms_str, p_str)
    corum_info = {}
    for dose, cluster_name, n_genes, best_complex, overlap, go_terms, corum_p in all_results_corum:
        if best_complex is not None:
            ratio = f"{len(overlap)}/{len(best_complex)}"
            go_str = "; ".join(go_terms[:3])
            p_str = f"p={corum_p:.2e}"
        else:
            ratio = "0/0"
            go_str = "none"
            p_str = ""
        corum_info[(dose, cluster_name)] = (ratio, go_str, p_str)

    # Group results by dose
    results_by_dose = {d: [] for d in DOSE_RATE_LABELS}
    for dose, cluster_name, n_genes, top5 in all_results_pe:
        results_by_dose[dose].append((cluster_name, n_genes, top5))

    # Save per-dose CSVs
    for dose in DOSE_RATE_LABELS:
        csv_rows = []
        for cluster_name, n_genes, top5 in results_by_dose[dose]:
            ratio, go_str, p_str = corum_info.get((dose, cluster_name), ("0/0", "none", ""))
            corum_p_val = ""
            for d, cn, ng, bc, ov, gt, cp in all_results_corum:
                if d == dose and cn == cluster_name:
                    corum_p_val = cp
                    break
            top_row = top5.iloc[0] if not top5.empty else None
            csv_rows.append({
                "cluster_id": cluster_name,
                "corum_complex_name": go_str,
                "corum_overlap_ratio": ratio,
                "corum_p_value": corum_p_val,
                "top_pathway_name": top_row["name"] if top_row is not None else "",
                "top_pathway_id": top_row["native"] if top_row is not None else "",
                "top_pathway_p_value": top_row["p_value"] if top_row is not None else "",
            })
        pd.DataFrame(csv_rows).to_csv(
            f"{output_dir}/cluster_enrichment_dose{dose}.csv", index=False
        )
        print(f"CSV saved: cluster_enrichment_dose{dose}.csv")

    max_rows = max(len(v) for v in results_by_dose.values())
    n_cols = len(DOSE_RATE_LABELS)
    fig, axes = plt.subplots(max_rows, n_cols,
                             figsize=(6 * n_cols, 5 * max_rows),
                             squeeze=False)

    for col, dose in enumerate(DOSE_RATE_LABELS):
        dose_results = results_by_dose[dose]
        n_clusters = len(cluster_gene_mapping[dose])

        for row, (cluster_name, n_genes, top5) in enumerate(dose_results):
            ax = axes[row][col]
            labels = [f"{r['name']} ({r['source']})" for _, r in top5.iterrows()]
            vals = [-np.log10(r["p_value"]) for _, r in top5.iterrows()]
            y = np.arange(len(labels))

            base_rgba = COLORS_BY_DOSE[dose]
            ci = int(cluster_name.split("_C")[1])
            alpha = 1.0 - 0.5 * (ci / max(n_clusters - 1, 1))
            color = (*base_rgba[:3], alpha)

            ax.barh(y, vals, color=color, edgecolor="white")
            ax.set_xlim(0, 30)
            ax.set_yticks(y)
            ax.set_yticklabels(labels, fontsize=_fs_tick)
            ratio, go_str, p_str = corum_info.get((dose, cluster_name), ("0/0", "none", ""))
            ax.set_title(f"{cluster_name} ({n_genes} genes)\nCORUM overlap: {ratio} {p_str} — {go_str}",
                         fontsize=_fs_title, fontweight="bold")
            ax.invert_yaxis()
            if row == max_rows - 1:
                ax.set_xlabel(r"$-\log_{10}(p)$")

        # Hide unused axes in this column
        for row in range(len(dose_results), max_rows):
            axes[row][col].axis("off")

    fig.suptitle("Cluster Pathway Enrichment", fontsize=_fs_title)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_enrichment_all_doses.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: cluster_enrichment_all_doses.png")


def perfect_edge_enrichment(output_dir="."):
    """Test whether perfect bootstrap edges (freq=1.0) are enriched in true
    positives compared to non-perfect edges, using Fisher's exact test.

    For each dose rate, loads bootstrap edge frequencies from CSV, splits edges
    into perfect vs non-perfect, checks each against TRRUST, STRING, CORUM, and
    ChIP-seq, then runs a one-sided Fisher's exact test (alternative='greater')
    to test if perfect edges have a higher TP rate.

    Outputs:
      - Console summary table
      - perfect_edge_enrichment.csv  (per-dose, per-source results)
      - perfect_edge_enrichment.png  (grouped bar plot of TP rates)
    """
    # Load prior knowledge
    trrust_df = pd.read_csv(TRRUST_PATH, sep="\t", header=None)
    trrust_directed = set(zip(trrust_df[0], trrust_df[1]))

    string_df = pd.read_csv(STRING_PATH)
    string_df = string_df[string_df.combined_score >= 400]
    string_edges = set(frozenset({r.protein1, r.protein2}) for _, r in string_df.iterrows())

    corum_edges = load_corum()
    corum_undirected = set(frozenset(e) for e in corum_edges)

    chipseq_directed = load_chipseq()

    print(f"\n=== Perfect Edge Enrichment in True Positives (Invariant Subgraph) ===\n")

    # Compute invariant gene set (genes present in all dose-rate graphs)
    dose_gene_sets = {}
    for d in DOSE_RATE_LABELS:
        G = nx.read_gexf(GRAPHS[d])
        dose_gene_sets[d] = set(n for n in G.nodes() if n not in EXCLUDE_NODES)
    invariant_genes = set.intersection(*dose_gene_sets.values())
    print(f"Invariant gene set: {len(invariant_genes)} genes\n")

    dose_keys = sorted(DOSE_RATE_LABELS, key=lambda d: DOSE_RATES_REGRESSION.get(d, 0))
    rows = []

    for d in dose_keys:
        freq_path = os.path.join(output_dir, f"bootstrap_edge_freq_{d}.csv")
        if not os.path.exists(freq_path):
            print(f"  Skipping dose {d}: {freq_path} not found")
            continue

        edge_df = pd.read_csv(freq_path)
        # Filter to edges within the invariant subgraph
        edge_df = edge_df[
            edge_df["source"].isin(invariant_genes) &
            edge_df["target"].isin(invariant_genes)
        ]
        perfect = edge_df[edge_df["frequency"] >= 1.0]
        non_perfect = edge_df[edge_df["frequency"] < 1.0]

        def _count_tp(edges_df, source_name):
            tp = 0
            for _, row in edges_df.iterrows():
                s, t = row["source"], row["target"]
                if source_name == "TRRUST":
                    if (s, t) in trrust_directed or (t, s) in trrust_directed:
                        tp += 1
                elif source_name == "STRING":
                    if frozenset({s, t}) in string_edges:
                        tp += 1
                elif source_name == "CORUM":
                    if frozenset({s, t}) in corum_undirected:
                        tp += 1
                elif source_name == "ChIP-seq":
                    if (s, t) in chipseq_directed or (t, s) in chipseq_directed:
                        tp += 1
            return tp

        n_perf = len(perfect)
        n_non = len(non_perfect)

        for source in ["TRRUST", "STRING", "CORUM", "ChIP-seq"]:
            tp_perf = _count_tp(perfect, source)
            tp_non = _count_tp(non_perfect, source)
            fp_perf = n_perf - tp_perf
            fp_non = n_non - tp_non

            # Fisher's exact test: are perfect edges enriched in TPs?
            # Contingency table:
            #              TP    FP
            # Perfect    [ a ,  b ]
            # Non-perf   [ c ,  d ]
            table = [[tp_perf, fp_perf], [tp_non, fp_non]]
            odds_ratio, p_value = fisher_exact(table, alternative="greater")

            rate_perf = tp_perf / n_perf if n_perf > 0 else 0
            rate_non = tp_non / n_non if n_non > 0 else 0

            rows.append({
                "dose": d,
                "dose_rate": DOSE_RATES_REGRESSION.get(d, d),
                "source": source,
                "n_perfect": n_perf,
                "n_non_perfect": n_non,
                "tp_perfect": tp_perf,
                "tp_non_perfect": tp_non,
                "rate_perfect": rate_perf,
                "rate_non_perfect": rate_non,
                "odds_ratio": odds_ratio,
                "p_value": p_value,
            })

        print(f"Dose {d} ({DOSE_RATES_REGRESSION.get(d, d)} mGy/hr): "
              f"{n_perf} perfect, {n_non} non-perfect edges "
              f"({n_perf + n_non} invariant subgraph edges)")

    results = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "perfect_edge_enrichment.csv")
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved: {csv_path}")

    # Print summary table
    print(f"\n{'Dose':>6}  {'Source':>8}  {'Perfect TP rate':>16}  "
          f"{'Non-perf TP rate':>17}  {'OR':>8}  {'p-value':>10}  Sig")
    print("-" * 80)
    for _, r in results.iterrows():
        sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else \
              "*" if r.p_value < 0.05 else ""
        print(f"{r.dose_rate:>6.2f}  {r.source:>8}  "
              f"{r.tp_perfect:>4d}/{r.n_perfect:<5d} ({r.rate_perfect:.3f})  "
              f"{r.tp_non_perfect:>4d}/{r.n_non_perfect:<5d} ({r.rate_non_perfect:.3f})  "
              f"{r.odds_ratio:>8.2f}  {r.p_value:>10.2e}  {sig}")

    # --- Grouped bar plot: TP rates for perfect vs non-perfect ---
    sources = ["TRRUST", "STRING", "CORUM", "ChIP-seq"]
    n_sources = len(sources)
    n_doses = len(dose_keys)

    fig, axes = plt.subplots(1, n_sources, figsize=(4 * n_sources, 5), sharey=True)
    if n_sources == 1:
        axes = [axes]

    x = np.arange(n_doses)
    width = 0.35

    for ax, source in zip(axes, sources):
        sub = results[results["source"] == source].sort_values("dose_rate")
        if sub.empty:
            continue
        ax.bar(x - width / 2, sub["rate_perfect"].values, width,
               label="Perfect (freq=1.0)", color="tab:green", alpha=0.8)
        ax.bar(x + width / 2, sub["rate_non_perfect"].values, width,
               label="Non-perfect (freq<1.0)", color="tab:gray", alpha=0.8)

        # Add significance stars
        for i, (_, r) in enumerate(sub.iterrows()):
            if r.p_value < 0.05:
                star = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*"
                max_val = max(r.rate_perfect, r.rate_non_perfect)
                ax.text(i, max_val + 0.005, star, ha="center", fontsize=10,
                        fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:.2f}" for r in sub["dose_rate"].values],
                           fontsize=_fs_tick)
        ax.set_xlabel("Dose rate (mGy/hr)", fontsize=_fs_axis)
        ax.set_title(source, fontsize=_fs_leg_title, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("TP rate (fraction of edges in KB)", fontsize=_fs_axis)
            ax.legend(fontsize=_fs_leg)

    fig.suptitle("Perfect vs Non-Perfect Bootstrap Edges: True Positive Rates",
                 fontsize=_fs_title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(output_dir, "perfect_edge_enrichment.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.savefig(fig_path.replace(".png", ".svg"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {fig_path}")


def invariant_gene_variance(output_dir="."):
    """Compute per-gene variance for invariant genes in TPM and log2-fold-change data.

    Saves ranked CSV files (descending variance) to output_dir.
    """
    invariant_path = os.path.join(output_dir, "invariant_genes.pkl")
    with open(invariant_path, "rb") as f:
        invariant_genes = pickle.load(f)
    invariant_genes = set(invariant_genes)

    # --- TPM ---
    X_tpm, _, _, _ = load_data_tpm()
    tpm_shared = sorted(invariant_genes & set(X_tpm.columns))
    tpm_var = X_tpm[tpm_shared].var().rename("variance").to_frame()
    tpm_var.index.name = "gene"
    tpm_var = tpm_var.sort_values("variance", ascending=False)
    tpm_var['variance'] = tpm_var['variance']/np.max(tpm_var['variance'].values)
    # --- Log2 fold-change ---
    X_l2fc, _, _, _ = load_data(None)
    l2fc_shared = sorted(invariant_genes & set(X_l2fc.columns))
    l2fc_var = X_l2fc[l2fc_shared].var().rename("variance").to_frame()
    l2fc_var.index.name = "gene"
    l2fc_var = l2fc_var.sort_values("variance", ascending=False)

    # --- Add degree columns from each invariant subgraph ---
    for d in DOSE_RATE_LABELS:
        gexf_path = os.path.join(output_dir, f"invariant_subgraph_{d}.gexf")
        if not os.path.exists(gexf_path):
            continue
        G = nx.read_gexf(gexf_path)
        degree_dict = dict(G.degree())
        tpm_var[f"degree_{d}"] = [degree_dict.get(g, 0) for g in tpm_var.index]
        l2fc_var[f"degree_{d}"] = [degree_dict.get(g, 0) for g in l2fc_var.index]

    tpm_out = os.path.join(output_dir, "invariant_gene_variance_tpm.csv")
    tpm_var.to_csv(tpm_out)
    print(f"TPM variance saved: {tpm_out}  ({len(tpm_var)} genes)")
    print(tpm_var.head(10).to_string())

    l2fc_out = os.path.join(output_dir, "invariant_gene_variance_log2fc.csv")
    l2fc_var.to_csv(l2fc_out)
    print(f"\nLog2FC variance saved: {l2fc_out}  ({len(l2fc_var)} genes)")
    print(l2fc_var.head(10).to_string())

    # --- Correlation between variance and degree in invariant subgraphs ---
    print("\n--- Variance vs Degree correlation (Spearman) ---")
    degree_cols = [c for c in tpm_var.columns if c.startswith("degree_")]
    for col in degree_cols:
        d = col.split("_", 1)[1]
        rho_tpm, p_tpm = spearmanr(tpm_var["variance"], tpm_var[col])
        rho_l2fc, p_l2fc = spearmanr(l2fc_var["variance"], l2fc_var[col])
        print(f"  Dose {d}: TPM rho={rho_tpm:.3f} (p={p_tpm:.2e}, n={len(tpm_var)}), "
              f"Log2FC rho={rho_l2fc:.3f} (p={p_l2fc:.2e}, n={len(l2fc_var)})")

    # --- Load housekeeping genes ---
    hk_genes = load_housekeeping_set()

    # --- Annotate perfect_edges graphs with TPM variance + housekeeping ---
    tpm_var_all = X_tpm.var()
    for d in DOSE_RATE_LABELS:
        perf_path = os.path.join(output_dir, f"perfect_edges_{d}.gexf")
        if not os.path.exists(perf_path):
            print(f"  Dose {d}: perfect_edges not found, skipping")
            continue
        G = nx.read_gexf(perf_path)
        for node in G.nodes():
            G.nodes[node]["tpm_variance"] = float(tpm_var_all[node]) if node in tpm_var_all.index else 0.0
            G.nodes[node]["housekeeping"] = node in hk_genes
        out_path = os.path.join(output_dir, f"perfect_edges_{d}_annotated.gexf")
        nx.write_gexf(G, out_path)
        print(f"Annotated graph saved: {out_path}")

    # --- Annotate invariant_subgraph graphs with TPM variance + housekeeping ---
    for d in DOSE_RATE_LABELS:
        sub_path = os.path.join(output_dir, f"invariant_subgraph_{d}.gexf")
        if not os.path.exists(sub_path):
            print(f"  Dose {d}: invariant_subgraph not found, skipping")
            continue
        G = nx.read_gexf(sub_path)
        for node in G.nodes():
            G.nodes[node]["tpm_variance"] = float(tpm_var_all[node]) if node in tpm_var_all.index else 0.0
            G.nodes[node]["housekeeping"] = node in hk_genes
        out_path = os.path.join(output_dir, f"invariant_subgraph_{d}_annotated.gexf")
        nx.write_gexf(G, out_path)
        print(f"Annotated graph saved: {out_path}")

    return tpm_var, l2fc_var


def invariant_eigenvector_centrality(output_dir=".", top_k=20):
    """Compute eigenvector centrality per dose rate on the invariant subgraphs.

    Loads the pre-computed annotated invariant subgraphs, computes eigenvector
    centrality on the undirected projection, and reports top-k genes per dose
    plus a combined ranking table with Spearman rank correlations across doses.
    """

    print("\n=== Eigenvector Centrality on Invariant Subgraphs ===\n")

    ec_per_dose = {}
    for d in DOSE_RATE_LABELS:
        gexf_path = os.path.join(output_dir, f"invariant_subgraph_{d}_annotated.gexf")
        sub = nx.read_gexf(gexf_path)
        gene_nodes = [n for n in sub.nodes() if n not in EXCLUDE_NODES]
        sub = sub.subgraph(gene_nodes).copy()

        U = sub.to_undirected()
        try:
            ec = nx.eigenvector_centrality(U, max_iter=1000, weight="weight")
        except nx.PowerIterationFailedConvergence:
            ec = nx.eigenvector_centrality_numpy(U, weight="weight")

        ec_per_dose[d] = ec
        ranked = sorted(ec.items(), key=lambda x: x[1], reverse=True)
        dose_label = f"{DOSE_RATES_REGRESSION[d]} mGy/hr"
        print(f"Dose {d} ({dose_label}) — top {top_k}:")
        for rank, (gene, val) in enumerate(ranked[:top_k], 1):
            print(f"  {rank:3d}. {gene:<15s} {val:.4f}")
        print()

    # Combined DataFrame: genes x doses
    all_genes = sorted(set().union(*[set(ec.keys()) for ec in ec_per_dose.values()]))
    ec_df = pd.DataFrame(
        {d: [ec_per_dose[d].get(g, 0.0) for g in all_genes] for d in DOSE_RATE_LABELS},
        index=all_genes,
    )
    ec_df.index.name = "gene"
    ec_df.columns = [f"ec_{d}" for d in DOSE_RATE_LABELS]
    ec_df["ec_mean"] = ec_df.mean(axis=1)
    ec_df = ec_df.sort_values("ec_mean", ascending=False)

    out_path = os.path.join(output_dir, "invariant_eigenvector_centrality.csv")
    ec_df.to_csv(out_path)
    print(f"Full table saved: {out_path}  ({len(ec_df)} genes)")
    print(f"\nTop {top_k} by mean eigenvector centrality across doses:")
    print(ec_df.head(top_k).to_string(float_format="%.4f"))

    # Rank correlation across dose pairs
    print(f"\nEigenvector-centrality rank correlation (Spearman):")
    rho_matrix = np.zeros((len(DOSE_RATE_LABELS), len(DOSE_RATE_LABELS)))
    for i, d1 in enumerate(DOSE_RATE_LABELS):
        for j, d2 in enumerate(DOSE_RATE_LABELS):
            if i == j:
                rho_matrix[i][j] = 1.0
            else:
                rho, _ = spearmanr(ec_df[f"ec_{d1}"], ec_df[f"ec_{d2}"])
                rho_matrix[i][j] = rho
    rho_df = pd.DataFrame(rho_matrix, index=DOSE_RATE_LABELS, columns=DOSE_RATE_LABELS)
    print(rho_df.to_string(float_format="%.3f"))

    # --- Bar plot: top 5 eigencentral genes per dose ---
    dose_labels = {d: f"{DOSE_RATES_REGRESSION[d]} mGy/hr" for d in DOSE_RATE_LABELS}
    top_n = 5
    fig, axes = plt.subplots(1, len(DOSE_RATE_LABELS), figsize=(4 * len(DOSE_RATE_LABELS), 5),
                             sharey=False)
    # sort by dose rate value
    
    for ax, d in zip(axes, DOSE_RATES_SORTED):
        col = f"ec_{d}"
        top_genes = ec_df[col].nlargest(top_n)
        ax.barh(range(top_n), top_genes.values, color=COLORS_BY_DOSE[d])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top_genes.index, fontsize=_fs_tick)
        ax.invert_yaxis()
        if d=="H":
            ax.set_xlabel("Eigenvector Centrality", fontsize=_fs_axis)
        ax.set_title(f"{dose_labels[d]}", fontsize=_fs_leg_title-2)
        ax.tick_params(axis="x", labelsize=_fs_tick)
    fig.suptitle(f"Top {top_n} Eigencentral Genes by Dose Rate (Invariant Gene Set)",
                 fontsize=_fs_title, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    bar_path = os.path.join(output_dir, "invariant_eigenvector_centrality_top5.png")
    plt.savefig(bar_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\nBar plot saved: {bar_path}")

    # --- Heatmap: Spearman rank correlation of eigenvector centrality ---
    fig, ax = plt.subplots(figsize=(6, 5))
    dose_tick_labels = [f"{dose_labels[d]}" for d in DOSE_RATE_LABELS]
    sns.heatmap(rho_df.values, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True,
                xticklabels=dose_tick_labels, yticklabels=dose_tick_labels,
                ax=ax, annot_kws={"fontsize": _fs_tick-3},
                cbar_kws={"label": "Spearman ρ"})
    ax.set_title("Eigenvector Centrality Rank Correlation\n(Invariant Gene Set)",
                 fontsize=_fs_title-7, fontweight="bold")
    ax.tick_params(axis="both", labelsize=_fs_tick-7)
    fig.tight_layout()
    heatmap_path = os.path.join(output_dir, "invariant_eigenvector_centrality_corr.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Heatmap saved: {heatmap_path}")

    return ec_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", "-o", default="./structure_analysis")
    parser.add_argument("--bootstrap_dir", "-b",
                        default="./data/rpe1_experiment2/bootstrap_graphs2/")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # DEGREE EVALS
    degree_distribution(output_dir=args.output_dir)
    hub_tf_analysis(output_dir=args.output_dir)
    sink_housekeeping_analysis(output_dir=args.output_dir)
    
    # # EDGE WISE EVALS
    edge_overlap_analysis(output_dir=args.output_dir)

    # EVALS OVER BOOTSTRAP DIST
    bootstrap_tf_analysis(args.bootstrap_dir, output_dir=args.output_dir)
    bootstrap_edge_overlap(args.bootstrap_dir, output_dir=args.output_dir)
    run_bootstrap_structural_analysis(args.bootstrap_dir, output_dir=args.output_dir)
    perfect_edge_enrichment(output_dir=args.output_dir)

    # CLUSTERING FUNCTIONAL EVALS
    cluster_enrichment_analysis(output_dir=args.output_dir)
    
    # INVARIANT GENE SET EVALS
    cross_dose_invariant_analysis(output_dir=args.output_dir)
    invariant_gene_variance(output_dir=args.output_dir)
    invariant_eigenvector_centrality(output_dir=args.output_dir)