
## Project Overview

**lucid_cd** is a computational biology research project for causal discovery and feature selection in radiation biology. It infers and analyzes gene regulatory networks in RPE1 cells exposed to ionizing radiation across 5 dose rates over 9 weeks.

## Data 
- Gene expression data is in TPM (transcripts per million) stored in `data/rpe_experiment2/rpe1_9week_study_experiment2_all_tpm.tsv`.
- Differential Expression analysis with DESeq2 is stored in `data/rpe1_experiment2/rpe1_9week_study_experiment2_diffexp_deseq_vs_control_all_dG_W2_adjust.txt`
- To generate datasets for causal discovery see `data/generate_data_matrices.ipynb`. The expected data format for causal discovery is an n x p matrix with columns corresponding to genes expression (in TPM) and radiation levels for each sample.

## Causal Discovery

`run_cd.py` orchestrates causal discovery on gene expression data with bootstrap resampling and partitioning for large gene sets. Methods include: 
- **DAG-GNN** (VAE/GNN via `gCastle`, `https://github.com/huawei-noah/trustworthyAI/blob/master/gcastle/`)
- **GENIE3** (Tree-based ensemble models `https://github.com/vahuynh/GENIE3/blob/master/GENIE3_python/GENIE3.py`) 
- **GENELink** (Graph Attention model`https://github.com/vahuynh/GENIE3/blob/master/GENIE3_python/GENIE3.py` ) 
- **Tetrad algorithms** (GES with BIC score, DirectLinGAM, PC with Fisher z-test for conditional independence `https://github.com/cmu-phil/tetrad`)

## Three Analysis Pipelines

1. **Bioinformatics Pipeline** (`pathway_enrichment.py`) — Compares genes from Differential Expression vs Causal Graph analysis for pathway enrichment using `gProfiler`. Generates stacked bar plots and Venn diagrams per dose rate.

2. **Supervised ML Pipeline** (`feature_selection.py`, `nested_cv_feature_selection.py`) — Trains ElasticNet linear models to predict phenotypes from gene expression with downselected genes. Uses nested repeated cross-validation. Compares feature selection methods: causal graph nodes, differentially expressed genes, AI-curated lists, variance thresholding. Also includes feature selection via supervised stability analysis of top feature importance genes with linear models and random forest models.

3. **Causal Structure Analysis** (`structure_analysis.py`) — Validates biological plausibility of causal DAGs via hub/TF enrichment (Fisher's exact test against TRRUST/ChIP-seq), edge overlap with prior knowledge graphs stored in `data/prior_knowledge/` (STRING PPI, CORUM, ChIP-seq), and bootstrap stability analysis of edge weights.
