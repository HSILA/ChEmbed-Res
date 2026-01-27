
import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from adjustText import adjust_text
import numpy as np

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "chemrxiv", "results")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "figure2.pdf")

# Constants
EXAMPLES = 74457

# Models to skip (from notebook)
UNNECESSARY_MODELS = [
    "allenai__scibert_scivocab_uncased",
    "answerdotai__ModernBERT-base",
    "answerdotai__ModernBERT-large",
    "google-bert__bert-base-uncased",
    "m3rg-iitd__matscibert",
    "nomic-ai__nomic-bert-2048",
    "recobo__chemical-bert-uncased",
    "sentence-transformers__all-MiniLM-L12-v2", # Fixed name prefix
    "sentence-transformers__all-mpnet-base-v2", # Fixed name prefix
    "BAAI__bge-small-en",
    "BAAI__bge-base-en",
    "BAAI__bge-large-en",
    "intfloat__e5-small",
    "intfloat__e5-base",
    "intfloat__e5-large",
    "BASF-AI__ChEmbed-plug",
    "BASF-AI__ChEmbed-full"
]

def load_data():
    data = []
    
    # Load pre-processed metadata
    METADATA_FILE = os.path.join(PROJECT_ROOT, "artifacts", "models_metadata.json")
    with open(METADATA_FILE, 'r') as f:
        models_meta = json.load(f)

    # Iterate over model directories
    for model_dir in os.listdir(RESULTS_DIR):
        model_path = os.path.join(RESULTS_DIR, model_dir)
        if not os.path.isdir(model_path):
            continue
            
        # Check blacklist
        skip = False
        for bad_name in UNNECESSARY_MODELS:
            if model_dir == bad_name or model_dir.endswith(bad_name):
                skip = True
                break
        if skip:
            continue

        # Find results
        hash_dirs = glob.glob(os.path.join(model_path, "*"))
        for hash_dir in hash_dirs:
            if not os.path.isdir(hash_dir):
                continue
                
            res_file = os.path.join(hash_dir, "ChemRxivRetrieval.json")
            
            if os.path.exists(res_file):
                try:
                    with open(res_file, 'r') as f:
                        res = json.load(f)
                        
                    ndcg = res["scores"]["test"][0]["ndcg_at_10"]
                    time = res["evaluation_time"]
                    
                    # Get metadata from JSON
                    model_meta = models_meta.get(model_dir, {})
                    params = model_meta.get("n_parameters")
                    embed_dim = model_meta.get("embed_dim")
                    
                    data.append({
                        "exp": model_dir,
                        "ndcgat10": ndcg,
                        "evaluation_time": time,
                        "n_parameters": params,
                        "embedding_size": embed_dim
                    })
                    break # Take first hash found
                except Exception as e:
                    print(f"Error reading {model_dir}: {e}")
                    
    return pd.DataFrame(data)

def plot_model_scatter(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        name_col: str = "exp",
        embed_col: str = "embedding_size",
        param_col: str = "n_parameters",
        figsize: tuple = (13, 9),
        min_dot: int = 150,
        max_dot: int = 800,
        y_cutoff: float | None = 0.2,
    ):

    best_models = {"BASF-AI__ChEmbed-prog"} # Updated name
    
    data = df.copy()
    
    # Calculate bubble area based on parameters
    pvals = data[param_col].dropna()
    p_min, p_max = (1, 1) if pvals.empty else (pvals.min(), pvals.max())
    sqrt_min, sqrt_max = np.sqrt(p_min), np.sqrt(p_max)

    def area(p):
        return np.nan if np.isnan(p) else np.interp(np.sqrt(p),
                                                    (sqrt_min, sqrt_max),
                                                    (min_dot, max_dot))

    data["dot_area"] = data[param_col].apply(area)
    
    # Setup Colors
    bins = [384, 768, 1024, 1536, 3072]
    colors = ["#6baed6", "#74c476", "#9e9ac8", "#fdae6b", "#ffd92f"]
    cmap = mcolors.ListedColormap(colors)
    upper_bound = bins[-1] + sum(np.diff(bins)) 
    norm = mcolors.BoundaryNorm(bins + [upper_bound], cmap.N)

    def colour(dim):
        return cmap(norm(dim))

    fig, ax = plt.subplots(figsize=figsize)
    texts = []
    
    # Font settings
    plt.rcParams.update({'font.size': 12})
    plt.rc('axes', labelsize=14)
    plt.rc('axes', titlesize=16)
    plt.rc('legend', fontsize=12)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)

    # Legend entries
    ax.scatter([], [], marker="o", s=200, color="white", edgecolor="black", linewidths=0.7, alpha=0.90, label="Circle size reflects model size")
    ax.scatter([], [], marker="*", s=200, color="white", edgecolor="black", linewidths=0.7, alpha=0.90, label="Model size unavailable")
    ax.scatter([], [], color="gray", s=200, hatch='...', edgecolor="black", label="ChEmbed models")

    for _, row in data.iterrows():
        x, y = row[x_col], row[y_col]
        if (y_cutoff is not None) and (y < y_cutoff):
            continue

        label_full = row[name_col]
        is_chembed = "ChEmbed" in label_full
        
        # Simplify label for display: get model name after provider prefix
        label_show = label_full.split("__", 1)[-1]
        
        dim_val = row[embed_col]
        # Handle cases where dim might be missing or list
        if isinstance(dim_val, list): dim_val = dim_val[0]
        if pd.isna(dim_val): dim_val = 768 # Default fallback
        
        dim_color = colour(dim_val)
        is_closed = pd.isna(row[param_col])

        if is_closed:
            marker, size = "*", 150
        else:
            marker, size = "o", row["dot_area"]

        ax.scatter(
            x, y,
            marker=marker,
            s=size,
            color=dim_color,
            edgecolor="black",
            linewidths=0.7,
            alpha=0.90,
            zorder=3,
            hatch='...' if is_chembed else None,
        )

        texts.append(ax.text(x, y, label_show, ha="center", va="center", fontsize=9))

    # Adjust text
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", lw=0.25))

    # Axes labels
    ax.set_xlabel("Speed (examples per sec)")
    ax.set_ylabel("nDCG@10")
    ax.grid(True, ls="--", alpha=0.3)

    xmin, xmax = ax.get_xlim()
    if xmin < 0: ax.set_xlim(left=0)

    # Colorbar
    bounds = bins + [upper_bound]
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05, boundaries=bounds, spacing="proportional")
    
    tick_locs = [(lo + hi) / 2 for lo, hi in zip(bounds[:-1], bounds[1:])]
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([str(b) for b in bins])
    cbar.minorticks_off()
    cbar.ax.tick_params(axis="y", which="both", length=3)
    cbar.set_label("Embedding Dimension")

    ax.legend(loc="upper right", frameon=True, borderpad=0.6)

    # A gap before x=0
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(left=xmin - 0.02 * (xmax - xmin))

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, format="pdf", bbox_inches='tight')
    print(f"Generated {OUTPUT_FILE}")

def main():
    df = load_data()
    # Calculate speed
    df["examples_per_sec"] = EXAMPLES / df["evaluation_time"]
    
    plot_model_scatter(
        df, 
        x_col="examples_per_sec", 
        y_col="ndcgat10"
    )

if __name__ == "__main__":
    main()
