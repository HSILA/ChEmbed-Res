
import json
import os
import glob
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "chemrxiv", "results")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "table4.tex")

# Defines the table structure: (directory__name, latex_display_name)
# We preserve the exact order from the original table4-old.tex
OPEN_SOURCE_MODELS = [
    ("recobo__chemical-bert-uncased", r"\texttt{chemical-bert-uncased}"),
    ("m3rg-iitd__matscibert", r"\texttt{matscibert}"),
    ("nomic-ai__nomic-bert-2048", r"\texttt{nomic-bert-2048}"),
    ("answerdotai__ModernBERT-base", r"\texttt{ModernBERT-base}"),
    ("answerdotai__ModernBERT-large", r"\texttt{ModernBERT-large}"),
    ("allenai__scibert_scivocab_uncased", r"\texttt{scibert\_scivocab\_uncased}"),
    ("google-bert__bert-base-uncased", r"\texttt{bert-base-uncased}"),
    ("sentence-transformers__all-MiniLM-L12-v2", r"\texttt{all-MiniLM-L12-v2}"),
    ("sentence-transformers__all-MiniLM-L6-v2", r"\texttt{all-MiniLM-L6-v2}"),
    ("sentence-transformers__all-mpnet-base-v2", r"\texttt{all-mpnet-base-v2}"),
    ("sentence-transformers__multi-qa-mpnet-base-dot-v1", r"\texttt{multi-qa-mpnet-base-dot-v1}"),
    ("intfloat__e5-small", r"\texttt{e5-small}"),
    ("intfloat__e5-base", r"\texttt{e5-base}"),
    ("intfloat__e5-large", r"\texttt{e5-large}"),
    ("intfloat__e5-small-v2", r"\texttt{e5-small-v2}"),
    ("intfloat__e5-base-v2", r"\texttt{e5-base-v2}"),
    ("intfloat__e5-large-v2", r"\texttt{e5-large-v2}"),
    ("intfloat__multilingual-e5-small", r"\texttt{multilingual-e5-small}"),
    ("intfloat__multilingual-e5-base", r"\texttt{multilingual-e5-base}"),
    ("intfloat__multilingual-e5-large", r"\texttt{multilingual-e5-large}"),
    ("thenlper__gte-small", r"\texttt{gte-small}"),
    ("thenlper__gte-base", r"\texttt{gte-base}"),
    ("thenlper__gte-large", r"\texttt{gte-large}"),
    ("Alibaba-NLP__gte-multilingual-base", r"\texttt{gte-multilingual-base}"),
    ("BAAI__bge-small-en", r"\texttt{bge-small-en}"),
    ("BAAI__bge-base-en", r"\texttt{bge-base-en}"),
    ("BAAI__bge-large-en", r"\texttt{bge-large-en}"),
    ("BAAI__bge-small-en-v1.5", r"\texttt{bge-small-en-v1.5}"),
    ("BAAI__bge-base-en-v1.5", r"\texttt{bge-base-en-v1.5}"),
    ("BAAI__bge-large-en-v1.5", r"\texttt{bge-large-en-v1.5}"),
    ("BAAI__bge-m3", r"\texttt{bge-m3}"),
    ("nomic-ai__nomic-embed-text-v1-unsupervised", r"\texttt{nomic-embed-text-v1-unsupervised}"),
    ("nomic-ai__nomic-embed-text-v1", r"\texttt{nomic-embed-text-v1}"),
    ("nomic-ai__nomic-embed-text-v1.5", r"\texttt{nomic-embed-text-v1.5}"),
    ("nomic-ai__nomic-embed-text-v2-moe", r"\texttt{nomic-embed-text-v2-moe}"),
    ("nomic-ai__modernbert-embed-base", r"\texttt{modernbert-embed-base}"),
    ("NovaSearch__stella_en_1.5B_v5", r"\texttt{stella\_en\_1.5B\_v5}"),
    ("jinaai__jina-embeddings-v3", r"\texttt{jina-embeddings-v3}"),
    ("Qwen__Qwen3-Embedding-0.6B", r"\texttt{Qwen3-Embedding-0.6B}$^{\dagger}$"),
    ("Qwen__Qwen3-Embedding-4B", r"\texttt{Qwen3-Embedding-4B}$^{\dagger}$"),
    ("Qwen__Qwen3-Embedding-8B", r"\texttt{Qwen3-Embedding-8B}$^{\dagger}$"),
    ("BASF-AI__ChEmbed-vanilla", r"\texttt{ChEmbed\textsubscript{vanilla}}"),
    ("BASF-AI__ChEmbed-prog", r"\texttt{ChEmbed\textsubscript{progressive}}"),
]

PROPRIETARY_MODELS = [
    ("openai__text-embedding-ada-002", r"\texttt{text-embedding-ada-002}"),
    ("openai__text-embedding-3-small", r"\texttt{text-embedding-3-small}"),
    ("openai__text-embedding-3-large", r"\texttt{text-embedding-3-large}"),
    ("bedrock__amazon-titan-embed-text-v1", r"\texttt{amazon-titan-embed-text-v1}"),
    ("bedrock__amazon-titan-embed-text-v2", r"\texttt{amazon-titan-embed-text-v2}"),
    ("bedrock__cohere-embed-english-v3", r"\texttt{cohere-embed-english-v3}"),
    ("bedrock__cohere-embed-multilingual-v3", r"\texttt{cohere-embed-multilingual-v3}"),
]

def get_model_data(model_dir_name):
    model_path = os.path.join(RESULTS_DIR, model_dir_name)
    data = {
        "emb_size": "N/A",
        "params": "N/A",
        "map_at_10": 0.0,
        "mrr_at_10": 0.0,
        "ndcg_at_10": 0.0
    }
    
    # Check if directory exists
    if not os.path.exists(model_path):
        # try simple name match if mapped incorrectly
        possible_dirs = glob.glob(os.path.join(RESULTS_DIR, f"*{model_dir_name.split('__')[-1]}*"))
        if possible_dirs:
            model_path = possible_dirs[0]
        else:
            return None

    # Find the hash directory (usually just one)
    hash_dirs = glob.glob(os.path.join(model_path, "*"))
    
    found_data = False
    
    for hash_dir in hash_dirs:
        if not os.path.isdir(hash_dir):
            continue
            
        # 1. Get Metrics from ChemRxivRetrieval.json
        res_file = os.path.join(hash_dir, "ChemRxivRetrieval.json")
        if os.path.exists(res_file):
            with open(res_file, 'r') as f:
                res = json.load(f)
                try:
                    scores = res["scores"]["test"][0]
                    data["map_at_10"] = scores.get("map_at_10", 0.0)
                    data["mrr_at_10"] = scores.get("mrr_at_10", 0.0)
                    data["ndcg_at_10"] = scores.get("ndcg_at_10", 0.0)
                    found_data = True
                except (KeyError, IndexError):
                    continue
        
        # 2. Get Metadata from model_meta.json
        meta_file = os.path.join(hash_dir, "model_meta.json")
        if os.path.exists(meta_file):
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                data["emb_size"] = meta.get("embed_dim", "N/A")
                n_params = meta.get("n_parameters", None)
                if n_params:
                    # Convert to Millions (M)
                    data["params"] = f"{n_params / 1e6:.1f}"
                else:
                    data["params"] = "N/A"
    
    return data if found_data else None

def generate_table():
    latex_lines = [
        r"\begin{table*}[!t]",
        r"\centering \small",
        r"\caption{Performance of embedding models on the \textbf{ChemRxiv Retrieval} task. “N/A” means the provider has not released parameter counts. Best scores per metric are shown in bold.}",
        r"\label{tab:chemrxiv-results}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Model Name & Emb. size & \#Params (M) & MAP@10 & MRR@10 & NDCG@10 \\",
        r"\midrule",
        r"    \multicolumn{6}{l}{\textbf{Open-Source Models}}\\"
    ]

    all_scores = {"map": [], "mrr": [], "ndcg": []}
    
    # Store processed rows to format bolding later
    rows = []

    # Process all models first
    for model_list in [OPEN_SOURCE_MODELS, PROPRIETARY_MODELS]:
        group_rows = []
        for model_id, model_name in model_list:
            data = get_model_data(model_id)
            if data:
                all_scores["map"].append(data["map_at_10"])
                all_scores["mrr"].append(data["mrr_at_10"])
                all_scores["ndcg"].append(data["ndcg_at_10"])
                group_rows.append((model_name, data))
            else:
                # Keep placeholder if missing? Or skip? The old table had them.
                # If missing, maybe fill with existing table values or skip. 
                # For this task, we assume we regenerate based on results. 
                # If result is missing, we print warning but maybe skip row to avoid errors.
                print(f"Warning: No data for {model_id}")
                group_rows.append((model_name, None))
        rows.append(group_rows)

    # Find max scores
    max_map = max(all_scores["map"]) if all_scores["map"] else 0
    max_mrr = max(all_scores["mrr"]) if all_scores["mrr"] else 0
    max_ndcg = max(all_scores["ndcg"]) if all_scores["ndcg"] else 0

    # Format Rows
    def format_row(name, d):
        if d is None:
            return f"    {name} & N/A & N/A & - & - & - \\\\"
        
        # Emb Size
        emb = str(d['emb_size'])
        
        # Params
        params = str(d['params'])
        
        # Metrics
        map_val = d['map_at_10']
        mrr_val = d['mrr_at_10']
        ndcg_val = d['ndcg_at_10']
        
        s_map = f"{map_val:.3f}"
        s_mrr = f"{mrr_val:.3f}"
        s_ndcg = f"{ndcg_val:.3f}"
        
        if map_val >= max_map: s_map = f"\\textbf{{{s_map}}}"
        if mrr_val >= max_mrr: s_mrr = f"\\textbf{{{s_mrr}}}"
        if ndcg_val >= max_ndcg: s_ndcg = f"\\textbf{{{s_ndcg}}}"
        
        return f"    {name} & {emb} & {params} & {s_map} & {s_mrr} & {s_ndcg} \\\\"

    # Add Open Source Rows
    for r in rows[0]:
        latex_lines.append(format_row(r[0], r[1]))
    
    latex_lines.append(r"    \midrule")
    latex_lines.append(r"    \multicolumn{6}{l}{\textbf{Proprietary Models}}\\")
    
    # Add Proprietary Rows
    for r in rows[1]:
        latex_lines.append(format_row(r[0], r[1]))

    latex_lines.extend([
        r"    \bottomrule",
        r"\multicolumn{6}{l}{\footnotesize $^{\dagger}$\,Loaded the model with bfloat16 to fit into GPU VRAM.}\\"
        r"\end{tabular}",
        r"\end{table*}"
    ])

    with open(OUTPUT_FILE, 'w') as f:
        f.write("\n".join(latex_lines))
    
    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_table()
