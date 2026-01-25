#!/usr/bin/env python3
"""
Quick status check for benchmark runs.
"""
import json
from pathlib import Path
from collections import defaultdict


def load_models(json_file: str) -> dict:
    """Load model definitions from JSON file."""
    with open(json_file, 'r') as f:
        return json.load(f)


def check_chemrxiv_status():
    """Check ChemRxiv benchmark completion status."""
    print("=" * 60)
    print("ChemRxiv Benchmark Status")
    print("=" * 60)
    
    results_dir = Path("results/chemrxiv/results")
    models_dict = load_models("models/models.json")
    
    completed = []
    incomplete = []
    
    for model_name, revision in models_dict.items():
        model_folder = model_name.replace("/", "__")
        result_file = results_dir / model_folder / revision / "ChemRxivRetrieval.json"
        
        if result_file.exists():
            completed.append(model_name)
        else:
            incomplete.append(model_name)
    
    print(f"Completed: {len(completed)}/{len(models_dict)}")
    print(f"Remaining: {len(incomplete)}")
    
    if incomplete and len(incomplete) <= 20:
        print(f"\nIncomplete models:")
        for model in incomplete:
            print(f"  - {model}")
    print()


def check_chembed_status():
    """Check ChEmbed (Nomic) benchmark completion status."""
    print("=" * 60)
    print("ChEmbed Benchmark Status")
    print("=" * 60)
    
    models_dict = load_models("models/ChEmbed.json")
    benchmarks = {
        "mteb": Path("results/ChEmbed/mteb"),
        "chemteb": Path("results/ChEmbed/chemteb")
    }
    
    # Track status per model per benchmark
    model_status = defaultdict(dict)
    
    for bench_name, bench_dir in benchmarks.items():
        for model_name, revision in models_dict.items():
            model_folder = model_name.replace("/", "__")
            # Standardize on the MTEB ResultCache output path:
            # results/ChEmbed/{benchmark}/results/{model_name}/{revision}/
            
            task_files = []
            
            # Target Path (Standard MTEB nested results)
            path_standard = bench_dir / "results" / model_folder / revision
            
            if path_standard.exists():
                 task_files.extend([f for f in path_standard.glob("*.json") if f.name != "model_meta.json"])
            
            unique_tasks = set(f.stem for f in task_files)
            
            model_status[model_name][bench_name] = len(unique_tasks)
    
    # Print summary per benchmark
    for bench_name in benchmarks.keys():
        completed = sum(1 for m in model_status.values() if m.get(bench_name, 0) > 0)
        print(f"{bench_name.upper()}: {completed}/{len(models_dict)} models")
    
    print()
    print("Per-Model Task Completion:")
    print("-" * 60)
    
    for model_name in sorted(model_status.keys()):
        model_short = model_name.split("/")[-1]
        mteb_tasks = model_status[model_name].get("mteb", 0)
        chemteb_tasks = model_status[model_name].get("chemteb", 0)
        
        print(f"{model_short:40} MTEB: {mteb_tasks:3d}  ChemTEB: {chemteb_tasks:3d}")
    print()


def main():
    check_chemrxiv_status()
    check_chembed_status()


if __name__ == "__main__":
    main()
