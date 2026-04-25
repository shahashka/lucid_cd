import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Register a truncated Oranges colormap that starts at a darker shade
_orig_oranges = plt.get_cmap("Oranges")
_oranges_dark = mcolors.LinearSegmentedColormap.from_list(
    "OrangesDark", _orig_oranges(np.linspace(0.25, 1.0, 256))
)
plt.colormaps.register(_oranges_dark)

# Perceptually separated hue families (avoid viridis/plasma/inferno cluster).
_DEFAULT_CMAPS = (
    "Blues",
    "OrangesDark",
    "Greens",
    "Purples",
    "Reds",
    "YlOrBr",
    "BuPu",
)
_fs_axis = 21
_fs_tick = 18
_fs_leg = 18
_fs_leg_title = 28
_fs_title = 25

METHOD_TO_CMAP = {"Causal Graph": "OrangesDark", "Differential Expression": "Blues", "Supervised ML": "Greens"}

DOSE_RATE_LABELS = ["F", "G", "H", "I", "J"]
DOSE_RATES_SORTED = ["G", "F", "H", "I", "J"]

DOSE_RATES = {"control": 0.0, "F": 0.004, "G": 0.04, "H": 0.4, "I": 4.0, "J": 8.0}
DOSE_RATES_REGRESSION = {"control":0.0, "F": 0.38, "G": 0.28, "H": 0.55, "I": 6.66, "J": 12.11, "shared":0}

GRAPHS = {"invariant": "/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs3/dag_gnn_combined.gexf",
          "F":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseF.gexf",
          "G":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseG.gexf",
          "H":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseH.gexf",
          "I":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseI.gexf",
          "J":"/homes/shahashka/lucid_cd/data/rpe1_experiment2/bootstrap_graphs2/dag_gnn_full_doseJ.gexf" 
          }
HK_PATH = "/homes/shahashka/lucid_cd/data/prior_knowledge/HSIAO_HOUSEKEEPING_GENES.v2026.1.Hs.json"
