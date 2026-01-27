
import os
import json
import glob
import numpy as np

# Configuration
RESULTS_DIR = "results/ChEmbed"
CHEMTEB_DIR = os.path.join(RESULTS_DIR, "chemteb", "results")
MTEB_DIR = os.path.join(RESULTS_DIR, "mteb", "results")
BENCHMARK_MAP_FILE = "artifacts/benchmark_tasks_map.json"

# Models to include in table
MODELS = [
    ("nomic-embed-text-v1", "nomic-ai__nomic-embed-text-v1"),
    (r"\texttt{ChEmbed\textsubscript{vanilla}}", "BASF-AI__ChEmbed-vanilla"),
    (r"\texttt{ChEmbed\textsubscript{progressive}}", "BASF-AI__ChEmbed-prog"),
]

def load_benchmark_map():
    with open(BENCHMARK_MAP_FILE, 'r') as f:
        return json.load(f)

def get_retrieval_tasks(benchmark_name, benchmark_data):
    """Identify Retrieval tasks from the benchmark map."""
    retrieval_tasks = []
    
    if benchmark_name not in benchmark_data:
        return retrieval_tasks

    for task_info in benchmark_data[benchmark_name]:
        if task_info['type'] in ["Retrieval", "Reranking"]: # Should strict Retrieval be used? Table says Retrieval.
             # Only Type "Retrieval"
             if task_info['type'] == "Retrieval":
                 retrieval_tasks.append(task_info['name'])
    
    return retrieval_tasks

def get_ndcg_at_10(base_dir, model_name, task_name):
    """Retrieve ndcg_at_10 for a task."""
    model_path = os.path.join(base_dir, model_name)
    hash_dirs = glob.glob(os.path.join(model_path, "*"))
    
    json_path = None
    for hd in hash_dirs:
        cand = os.path.join(hd, f"{task_name}.json")
        if os.path.exists(cand):
            json_path = cand
            break
        if not json_path:
             candidates = glob.glob(os.path.join(hd, f"{task_name}*.json"))
             if candidates:
                 json_path = candidates[0]
                 break
    
    if not json_path or not os.path.exists(json_path): 
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # MTEB structure
        if "scores" in data and "test" in data["scores"]:
            test_res = data["scores"]["test"][0]
            if "ndcg_at_10" in test_res:
                return test_res["ndcg_at_10"]
            
    except Exception as e:
        pass
    return None

def compute_mean_ndcg(tasks, base_dir, model_name):
    scores = []
    for task in tasks:
        s = get_ndcg_at_10(base_dir, model_name, task)
        if s is not None:
            scores.append(s)
    
    if not scores:
        return 0.0
    return np.mean(scores)

def format_score(val, is_best=False):
    if val <= 0: return "-"
    s = f"{val:.3f}"
    if is_best:
        return f"\\textbf{{{s}}}"
    return s

def main():
    benchmark_data = load_benchmark_map()
    
    # 1. ChemTEB Retrieval
    chem_tasks = get_retrieval_tasks("ChemTEB", benchmark_data)
    # Manually add ChemRxivRetrieval if not present
    if "ChemRxivRetrieval" not in chem_tasks:
        chem_tasks.append("ChemRxivRetrieval")
    
    # 2. MTEB Retrieval
    mteb_tasks = get_retrieval_tasks("MTEB(eng, v2)", benchmark_data)
    
    print(f"ChemTEB Retrieval Tasks: {len(chem_tasks)} ({chem_tasks})")
    print(f"MTEB Retrieval Tasks: {len(mteb_tasks)} ({mteb_tasks})")

    # Compute Scores
    # Row 1: ChemTEB
    row1_scores = []
    for _, dir_name in MODELS:
        row1_scores.append(compute_mean_ndcg(chem_tasks, CHEMTEB_DIR, dir_name))

    # Row 2: MTEB Retrieval
    row2_scores = []
    for _, dir_name in MODELS:
        row2_scores.append(compute_mean_ndcg(mteb_tasks, MTEB_DIR, dir_name))

    # Determine Bests
    best1 = max(row1_scores)
    best2 = max(row2_scores)

    # Generate LaTeX
    latex_lines = []
    latex_lines.append(r"\begin{table*}[!t]")
    latex_lines.append(r"\centering")
    latex_lines.append(r"\caption{Retrieval performance comparison across chemistry-specific and open‑domain tasks. Dataset descriptions and performance metrics for each model–dataset pair are provided.}")
    latex_lines.append(r"\label{tab:retrieval-wide}")
    latex_lines.append(r"\setlength{\tabcolsep}{5pt}")
    latex_lines.append(r"\resizebox{\textwidth}{!}{")
    # Columns: Task | Dataset | Domain-Specific | Model1 | Model2 | Model3
    
    latex_lines.append(r"\begin{tabular}{ll c ccc}")
    latex_lines.append(r"\toprule")
    # Header row
    # Task | Dataset | Domain-Specific | nDCG@10 (header spanning 3 cols)
    latex_lines.append(r" & & & \multicolumn{3}{c}{\textbf{nDCG@10 $\uparrow$}} \\ ")
    latex_lines.append(r"\cmidrule(lr){4-6}")
    
    # Model Names Header
    model_headers = [m[0] for m in MODELS]
    latex_lines.append(r"Task & Dataset & Domain-Specific & " + " & ".join(model_headers) + r" \\ ")
    latex_lines.append(r"\midrule")
    
    # Row 1
    # Check best
    cells1 = []
    for s in row1_scores:
        is_b = (s >= best1 - 1e-9) and (s > 0)
        cells1.append(format_score(s, is_b))
        
    latex_lines.append(r"Chemistry Retrieval & ChemTEB(latest)   & \cmark  & " + " & ".join(cells1) + r" \\")

    # Row 2
    cells2 = []
    for s in row2_scores:
        is_b = (s >= best2 - 1e-9) and (s > 0)
        cells2.append(format_score(s, is_b))
        
    latex_lines.append(r"Open-domain Retrieval & MTEB Retrieval       & \xmark & " + " & ".join(cells2) + r" \\")

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table*}")
    
    with open("artifacts/table6.tex", "w") as f:
        f.write("\n".join(latex_lines))
    print("Generated artifacts/table6.tex")

if __name__ == "__main__":
    main()
