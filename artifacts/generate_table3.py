
import os
import glob
import sys
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "chemrxiv", "results")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "table3.tex")

# Model directory name -> Display name
MODELS = [
    ("nomic-ai__nomic-embed-text-v1", r"\texttt{nomic-embed-text-v1} (baseline)"),
    ("BASF-AI__ChEmbed-vanilla", r"\texttt{ChEmbed\textsubscript{vanilla}}"),
    ("BASF-AI__ChEmbed-full", r"\texttt{ChEmbed\textsubscript{full}}"),
    ("BASF-AI__ChEmbed-plug", r"\texttt{ChEmbed\textsubscript{plug}}"),
    ("BASF-AI__ChEmbed-prog", r"\textbf{\texttt{ChEmbed\textsubscript{progressive}}}")
]

def get_ndcg_at_10(model_dir_name):
    model_path = os.path.join(RESULTS_DIR, model_dir_name)
    # Find the hash directory
    hash_dirs = glob.glob(os.path.join(model_path, "*"))
    for hash_dir in hash_dirs:
        if os.path.isdir(hash_dir):
            json_file = os.path.join(hash_dir, "ChemRxivRetrieval.json")
            if os.path.exists(json_file):
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    try:
                        return data["scores"]["test"][0]["ndcg_at_10"]
                    except (KeyError, IndexError):
                        continue
    return None

def generate_table():
    results = {}
    baseline_score = None
    
    # First pass to get scores
    for model_dir, display_name in MODELS:
        score = get_ndcg_at_10(model_dir)
        if score is None:
            print(f"Warning: Could not find results for {model_dir}")
            continue
        results[model_dir] = score
        if model_dir == "nomic-ai__nomic-embed-text-v1":
            baseline_score = score

    if baseline_score is None:
        raise ValueError("Baseline score (nomic-ai__nomic-embed-text-v1) not found!")

    # Generate LaTeX
    latex_lines = [
        r"\subsection{Domain-Specific Retrieval Performance}",
        r"\begin{table}[!b]",
        r"    \centering",
        r"    \caption{Impact of different tokenizer-adaptation variants on ChemRxiv retrieval performance}",
        r"    \label{tab:tok-ablation}",
        r"    \begin{tabular}{lcc}",
        r"    \toprule",
        r"    \textbf{Variant} & $nDCG@10$ & $\Delta$ vs.\ baseline \\",
        r"    \midrule"
    ]

    for model_dir, display_name in MODELS:
        if model_dir not in results:
            continue
        
        score = results[model_dir]
        
        # Calculate Delta
        if model_dir == "nomic-ai__nomic-embed-text-v1":
            delta_str = "--"
        else:
            diff = score - baseline_score
            sign = "+" if diff > 0 else ""
            delta_str = f"{sign}{diff*100:.1f} \%"

        # Format Score
        score_str = f"{score:.3f}"
        if model_dir == "BASF-AI__ChEmbed-prog":
            score_str = f"\\textbf{{{score_str}}}"
        
        line = f"    {display_name:<45} & {score_str} & {delta_str:<7} \\\\"
        latex_lines.append(line)

    latex_lines.extend([
        r"    \bottomrule",
        r"    \end{tabular}",
        r"\end{table}"
    ])

    with open(OUTPUT_FILE, 'w') as f:
        f.write("\n".join(latex_lines))
    
    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_table()
