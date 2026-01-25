import sys
import json
import gc
import torch
from pathlib import Path
from tqdm import tqdm
from loguru import logger
import mteb
from mteb.cache import ResultCache
from ChEmbedWrapper import ChEmbedWrapper
import argparse

# Logging setup
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(
    LOG_DIR / "nomic_{time:YYMMDD_HHmmss}.log",
    level="DEBUG",
    format="{time:HH:mm:ss} | {level} | {message}",
)

RESULTS_DIR = Path("results/ChEmbed")

BENCHMARKS_MAP = {"chemteb": "ChemTEB(latest)", "mteb": "MTEB(eng, v2)"}


def load_models(json_file: str) -> dict:
    with open(json_file, "r") as f:
        return json.load(f)


def get_missing_tasks(model_name, revision, bench_key):
    benchmark_name = BENCHMARKS_MAP[bench_key]

    # Map model name to folder name (slashes to underscores)
    model_folder = model_name.replace("/", "__")

    # MTEB ResultCache explicitly appends 'results' to the cache_path.
    # We must check this specific subdirectory to match the library's behavior.
    results_path = RESULTS_DIR / bench_key / "results" / model_folder / revision

    # Get all tasks
    try:
        benchmark = mteb.get_benchmark(benchmark_name)
    except Exception as e:
        logger.error(f"Failed to load benchmark {benchmark_name}: {e}")
        return []

    all_tasks = set(t.metadata.name for t in benchmark.tasks)

    if not results_path.exists():
        # Nothing completed in the standard path
        return list(all_tasks)

    completed_files = set(
        f.stem for f in results_path.glob("*.json") if f.name != "model_meta.json"
    )

    missing = list(all_tasks - completed_files)
    return missing


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on MTEB/ChemTEB benchmarks robustly"
    )
    parser.add_argument(
        "-b",
        "--benchmark",
        choices=["mteb", "chemteb"],
        help="Specific benchmark to run (default: runs both)",
    )
    parser.add_argument(
        "--num_proc", type=int, default=64, help="Number of processes for processing"
    )
    args = parser.parse_args()

    benchmarks_to_run = list(BENCHMARKS_MAP.keys())
    if args.benchmark:
        benchmarks_to_run = [args.benchmark]

    models_dict = load_models("models/ChEmbed.json")

    for model_name, revision in tqdm(models_dict.items(), desc="Checking Models"):

        # Determine all missing tasks for this model across benchmarks
        model_tasks_queue = []

        for bench_key in benchmarks_to_run:
            missing = get_missing_tasks(model_name, revision, bench_key)
            if missing:
                logger.info(
                    f"Model {model_name} missing {len(missing)} tasks in {bench_key}"
                )
                model_tasks_queue.append((bench_key, missing))

        if not model_tasks_queue:
            logger.info(f"Model {model_name} has completed selected benchmarks.")
            continue

        # Load model ONCE
        try:
            logger.info(f"Loading {model_name}...")
            if "ChEmbed" in model_name:
                model = ChEmbedWrapper(
                    model_name, revision=revision, trust_remote_code=True
                )
            else:
                model = mteb.get_model(
                    model_name, revision=revision, trust_remote_code=True
                )

            # Access underlying model for attributes
            inner_model = model.model if hasattr(model, "model") else model
            inner_model.max_seq_length = 2048
            logger.info(f"Context length confirmed: {inner_model.max_seq_length}")

            # Run tasks
            for bench_key, missing_tasks in model_tasks_queue:
                benchmark_name = BENCHMARKS_MAP[bench_key]
                benchmark = mteb.get_benchmark(benchmark_name)

                # Filter task objects
                tasks_to_run = [
                    t for t in benchmark.tasks if t.metadata.name in missing_tasks
                ]

                # Sort to ensure consistent order (optional, but good)
                tasks_to_run.sort(key=lambda t: t.metadata.name)

                bench_res_dir = RESULTS_DIR / bench_key
                bench_res_dir.mkdir(exist_ok=True, parents=True)
                cache = ResultCache(cache_path=bench_res_dir)

                logger.info(
                    f"Running {len(tasks_to_run)} missing tasks for {benchmark_name}"
                )

                for task_instance in tasks_to_run:
                    logger.info(f"Running Task: {task_instance.metadata.name}")
                    try:
                        # The MTEB library might cache task instances.
                        # If a previous run called unload_data(), self.dataset might be None.
                        # We force a FRESH instance of the task to ensure clean state (self.dataset = {}).
                        TaskClass = type(task_instance)
                        task = TaskClass()

                        # Run single task
                        mteb.evaluate(model, tasks=[task], cache=cache, num_proc=4)
                        logger.info(f"Completed Task: {task.metadata.name}")

                        # Aggressive cleanup
                        del task
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e:
                        import traceback

                        logger.error(
                            f"Failed Task {task_instance.metadata.name}: {e}\n{traceback.format_exc()}"
                        )

            # Cleanup model
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback

            logger.error(
                f"Failed to load/run {model_name}: {e}\n{traceback.format_exc()}"
            )
            continue


if __name__ == "__main__":
    main()
