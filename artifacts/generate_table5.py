
import os
import json
import glob
import numpy as np

# Configuration
RESULTS_DIR = "results/ChEmbed"
CHEMTEB_DIR = os.path.join(RESULTS_DIR, "chemteb", "results")
MTEB_DIR = os.path.join(RESULTS_DIR, "mteb", "results")
BENCHMARK_MAP_FILE = "artifacts/benchmark_tasks_map.json"

# Model Mapping: (Display Name, Directory Name)
MODELS = [
    ("nomic-embed-text-v1-unsupervised", "nomic-ai__nomic-embed-text-v1-unsupervised"),
    ("nomic-embed-text-v1", "nomic-ai__nomic-embed-text-v1"),
    (r"\texttt{ChEmbed\textsubscript{vanilla}}", "BASF-AI__ChEmbed-vanilla"),
    (r"\texttt{ChEmbed\textsubscript{full}}", "BASF-AI__ChEmbed-full"),
    (r"\texttt{ChEmbed\textsubscript{plug}}", "BASF-AI__ChEmbed-plug"),
    (r"\texttt{ChEmbed\textsubscript{progressive}}", "BASF-AI__ChEmbed-prog"),
]

def load_benchmark_map():
    with open(BENCHMARK_MAP_FILE, 'r') as f:
        return json.load(f)

def get_tasks_from_map(benchmark_name, benchmark_data):
    """Identify tasks and categories from the benchmark map."""
    tasks = {'Cls': [], 'Clust': [], 'Pair': []}
    
    if benchmark_name not in benchmark_data:
        print(f"Warning: Benchmark {benchmark_name} not found in map.")
        return tasks

    for task_info in benchmark_data[benchmark_name]:
        task_name = task_info['name']
        task_type = task_info['type']
        
        if task_type == "Classification":
            tasks['Cls'].append(task_name)
        elif task_type == "Clustering":
            tasks['Clust'].append(task_name)
        elif task_type == "PairClassification":
            tasks['Pair'].append(task_name)
            
    return tasks

def get_score(base_dir, model_name, task_name):
    """Retrieve main_score or relevant metric for a task."""
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
        
        if "test" in data.get("scores", {}):
            test_res = data["scores"]["test"][0]
            if "main_score" in test_res:
                return test_res["main_score"]
            
    except Exception as e:
        pass
    return None

def compute_metrics(category_tasks, base_dir, model_name):
    """Compute per-category means and overall means."""
    scores = {'Cls': [], 'Clust': [], 'Pair': []}
    all_scores = []
    
    for cat in ['Cls', 'Clust', 'Pair']:
        for task in category_tasks[cat]:
            s = get_score(base_dir, model_name, task)
            if s is not None:
                scores[cat].append(s)
                all_scores.append(s)
    
    # Means
    mean_cls = np.mean(scores['Cls']) if scores['Cls'] else 0.0
    mean_clust = np.mean(scores['Clust']) if scores['Clust'] else 0.0
    mean_pair = np.mean(scores['Pair']) if scores['Pair'] else 0.0
    
    mean_task = np.mean(all_scores) if all_scores else 0.0
    
    valid_cat_means = []
    if scores['Cls']: valid_cat_means.append(mean_cls)
    if scores['Clust']: valid_cat_means.append(mean_clust)
    if scores['Pair']: valid_cat_means.append(mean_pair)
    
    mean_type = np.mean(valid_cat_means) if valid_cat_means else 0.0
    
    return mean_cls, mean_clust, mean_pair, mean_task, mean_type

def format_score(val, is_best=False):
    s = f"{val:.3f}"
    if is_best:
        return f"\\textbf{{{s}}}"
    return s

def main():
    benchmark_data = load_benchmark_map()
    
    chem_tasks = get_tasks_from_map("ChemTEB", benchmark_data)
    mteb_tasks = get_tasks_from_map("MTEB(eng, v2)", benchmark_data) 
    
    print(f"ChemTEB Tasks Configured: Cls={len(chem_tasks['Cls'])}, Clust={len(chem_tasks['Clust'])}, Pair={len(chem_tasks['Pair'])}")
    print(f"MTEB Tasks Configured: Cls={len(mteb_tasks['Cls'])}, Clust={len(mteb_tasks['Clust'])}, Pair={len(mteb_tasks['Pair'])}")

    # Collect data
    results = [] 
    
    for display_name, dir_name in MODELS:
        c_metrics = compute_metrics(chem_tasks, CHEMTEB_DIR, dir_name)
        m_metrics = compute_metrics(mteb_tasks, MTEB_DIR, dir_name)
        results.append((display_name, c_metrics, m_metrics))

    # Determine bests
    best_chem = [-1.0] * 5
    best_mteb = [-1.0] * 5
    
    for _, c, m in results:
        for i in range(5):
            if c[i] > best_chem[i]: best_chem[i] = c[i]
            if m[i] > best_mteb[i]: best_mteb[i] = m[i]

    # Generate LaTeX
    latex_lines = []
    latex_lines.append(r"\begin{table*}[!b]")
    latex_lines.append(r"\centering \small")
    latex_lines.append(r"\caption{Performance comparison across shared non‑retrieval task categories in MTEB and ChemTEB, covering open‑domain and chemistry‑specific data, respectively. Evaluation metrics are: accuracy for Classification, V‑measure for Clustering, and average precision (AP) for Pair Classification. Mean (Task) reports the average score across all individual tasks, while Mean (Task Type) first averages results within each task category and then computes the mean across the category‑level averages.}")
    latex_lines.append(r"\label{tab:nonretrieval-wide}")
    latex_lines.append(r"\setlength{\tabcolsep}{5pt}")
    latex_lines.append(r"\resizebox{\textwidth}{!}{")
    latex_lines.append(r"\begin{tabular}{l")
    latex_lines.append(r"              ccc cc")
    latex_lines.append(r"              ccc cc}")
    latex_lines.append(r"\toprule")
    latex_lines.append(r"& \multicolumn{5}{c}{\textbf{ChemTEB}} & \multicolumn{5}{c}{\textbf{MTEB}}\\")
    latex_lines.append(r"\cmidrule(lr){2-6}\cmidrule(lr){7-11}")
    latex_lines.append(r"Model")
    latex_lines.append(r"& Cls & Clust & Pair & Mean (T) & Mean (T-type)")
    latex_lines.append(r"& Cls & Clust & Pair & Mean (T) & Mean (T-type)\\")
    latex_lines.append(r"\midrule")
    
    for display_name, c, m in results:
        cells_chem = []
        cells_mteb = []
        
        # Chem
        for i in range(5):
            if c[i] <= 0:
                cells_chem.append("-")
            else:
                is_b = (c[i] >= best_chem[i] - 1e-9)
                cells_chem.append(format_score(c[i], is_b))
                
        # MTEB
        for i in range(5):
            if m[i] <= 0:
                cells_mteb.append("-")
            else:
                is_b = (m[i] >= best_mteb[i] - 1e-9)
                cells_mteb.append(format_score(m[i], is_b))
        
        # Split row formatting
        row_chem = f"{display_name} & " + " & ".join(cells_chem)
        # Align second row with spaces
        row_mteb = f"               & " + " & ".join(cells_mteb) + r" \\"
        
        latex_lines.append(row_chem)
        latex_lines.append(row_mteb)
        
    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")
    latex_lines.append(r"}")
    latex_lines.append(r"\end{table*}")
    
    # Save to artifacts/table5.tex
    with open("artifacts/table5.tex", "w") as f:
        f.write("\n".join(latex_lines))
    print("Generated artifacts/table5.tex")

if __name__ == "__main__":
    main()
