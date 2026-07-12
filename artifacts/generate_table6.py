
import os
import json
import glob
import numpy as np

# Configuration
RESULTS_DIR = "results/ChEmbed"
CHEMTEB_DIR = os.path.join(RESULTS_DIR, "chemteb", "results")
MTEB_DIR = os.path.join(RESULTS_DIR, "mteb", "results")
BENCHMARK_MAP_FILE = "artifacts/benchmark_tasks_map.json"
MODELS_FILE = "models/ChEmbed.json"

# Models to include in table
MODELS = [
    ("nomic-embed-text-v1", "nomic-ai__nomic-embed-text-v1"),
    (r"\texttt{ChEmbed\textsubscript{vanilla}}", "BASF-AI__ChEmbed-vanilla"),
    (r"\texttt{ChEmbed\textsubscript{progressive}}", "BASF-AI__ChEmbed-prog"),
]

# Dataset statistics (Task, Dataset name, Domain-Specific, #Tasks, Avg Queries, Avg Corpus).
# These describe the benchmark composition itself, not model scores, so they are not
# derivable from the result JSONs -- fixed here rather than hand-merged into paper.tex.
DATASET_STATS = {
    "Chemistry Retrieval": {
        "dataset": "ChemTEB(latest)",
        "domain_specific": r"\cmark",
        "n_tasks": 3,
        "avg_queries": 5116,
        "avg_corpus": r"$85,958$",
    },
    "Open-domain Retrieval": {
        "dataset": "MTEB Retrieval",
        "domain_specific": r"\xmark",
        "n_tasks": 10,
        "avg_queries": 1482,
        "avg_corpus": r"$109,645$",
    },
}

def load_benchmark_map():
    with open(BENCHMARK_MAP_FILE, 'r') as f:
        return json.load(f)

def load_model_revisions():
    """Map dir-style model names (org__model) to the exact pinned revision."""
    with open(MODELS_FILE, 'r') as f:
        raw = json.load(f)
    return {k.replace("/", "__"): v for k, v in raw.items()}

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

def get_ndcg_at_10(base_dir, model_name, task_name, revision=None):
    """Retrieve ndcg_at_10 for a task, pinned to an exact revision."""
    model_path = os.path.join(base_dir, model_name)

    if revision:
        hash_dirs = [os.path.join(model_path, revision)]
    else:
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

def compute_mean_ndcg(tasks, base_dir, model_name, revision=None):
    scores = []
    for task in tasks:
        s = get_ndcg_at_10(base_dir, model_name, task, revision)
        if s is not None:
            scores.append(s)

    if not scores:
        return 0.0
    return np.mean(scores)

def get_mrr_at_10(base_dir, model_name, task_name, revision=None):
    """Retrieve mrr_at_10 for a task, pinned to an exact revision."""
    model_path = os.path.join(base_dir, model_name)

    if revision:
        hash_dirs = [os.path.join(model_path, revision)]
    else:
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

        if "scores" in data and "test" in data["scores"]:
            test_res = data["scores"]["test"][0]
            if "mrr_at_10" in test_res:
                return test_res["mrr_at_10"]

    except Exception as e:
        pass
    return None

def compute_mean_mrr(tasks, base_dir, model_name, revision=None):
    scores = []
    for task in tasks:
        s = get_mrr_at_10(base_dir, model_name, task, revision)
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
    model_revisions = load_model_revisions()

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
    # Row 1: ChemTEB (ChemNQRetrieval, ChemHotpotQARetrieval, ChemRxivRetrieval) --
    # dominated by single-relevant-document-per-query tasks, so we report MRR@10
    # here rather than NDCG@10 (see generate_table3.py / limitations-notes.md).
    row1_scores = []
    for _, dir_name in MODELS:
        revision = model_revisions.get(dir_name)
        if revision is None:
            print(f"Warning: no pinned revision found for {dir_name} in {MODELS_FILE}; falling back to scanning all revision folders.")
        row1_scores.append(compute_mean_mrr(chem_tasks, CHEMTEB_DIR, dir_name, revision))

    # Row 2: MTEB Retrieval -- genuinely multi-relevant-document retrieval tasks
    # (ArguAna, FEVER, TRECCOVID, etc.), so NDCG@10 remains the appropriate metric.
    row2_scores = []
    for _, dir_name in MODELS:
        revision = model_revisions.get(dir_name)
        row2_scores.append(compute_mean_ndcg(mteb_tasks, MTEB_DIR, dir_name, revision))

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
    
    latex_lines.append(r"\begin{tabular}{ll c ccc | ccc}")
    latex_lines.append(r"\toprule")
    # Header row
    # Task | Dataset | Domain-Specific | Dataset statistics (3 cols) | Retrieval Score (3 cols)
    # Metric differs by row: MRR@10 for Chemistry Retrieval (single-positive),
    # NDCG@10 for Open-domain Retrieval (multi-relevant) -- see footnote below.
    latex_lines.append(r" & & & \multicolumn{3}{c|}{\textbf{Dataset statistics}} & \multicolumn{3}{c}{\textbf{Retrieval Score$^{\ddagger}$ $\uparrow$}} \\ ")
    latex_lines.append(r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}")

    # Model Names Header
    model_headers = [m[0] for m in MODELS]
    latex_lines.append(r"Task & Dataset & Domain-Specific & \#Tasks & Avg Queries & Avg Corpus & " + " & ".join(model_headers) + r" \\ ")
    latex_lines.append(r"\midrule")

    # Row 1
    # Check best
    cells1 = []
    for s in row1_scores:
        is_b = (s >= best1 - 1e-9) and (s > 0)
        cells1.append(format_score(s, is_b))

    stats1 = DATASET_STATS["Chemistry Retrieval"]
    latex_lines.append(
        f"Chemistry Retrieval & {stats1['dataset']} & {stats1['domain_specific']} & "
        f"{stats1['n_tasks']} & {stats1['avg_queries']} & {stats1['avg_corpus']} & "
        + " & ".join(cells1) + r" \\"
    )

    # Row 2
    cells2 = []
    for s in row2_scores:
        is_b = (s >= best2 - 1e-9) and (s > 0)
        cells2.append(format_score(s, is_b))

    stats2 = DATASET_STATS["Open-domain Retrieval"]
    latex_lines.append(
        f"Open-domain Retrieval & {stats2['dataset']} & {stats2['domain_specific']} & "
        f"{stats2['n_tasks']} & {stats2['avg_queries']} & {stats2['avg_corpus']} & "
        + " & ".join(cells2) + r" \\"
    )

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\multicolumn{9}{l}{\footnotesize $^{\ddagger}$\,MRR@10 for Chemistry Retrieval (single relevant document per query); NDCG@10 for Open-domain Retrieval (multiple relevant documents per query).}\\")
    latex_lines.append(r"\end{tabular}}")
    latex_lines.append(r"\end{table*}")
    
    with open("artifacts/table6.tex", "w") as f:
        f.write("\n".join(latex_lines))
    print("Generated artifacts/table6.tex")

if __name__ == "__main__":
    main()
