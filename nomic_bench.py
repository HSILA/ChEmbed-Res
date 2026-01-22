from dotenv import load_dotenv

load_dotenv()

import sys
import json
from pathlib import Path
from tqdm import tqdm

from loguru import logger
import mteb
from mteb.cache import ResultCache
from wrapper import ChEmbedWrapper

# Logging setup
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(
    sys.stderr, level="ERROR", format="<red>{time:HH:mm:ss} | {level} | {message}</red>"
)
logger.add(
    LOG_DIR / "nomic_{time:YY-MM-DD_HH-mm-ss}.log",
    level="DEBUG",
    format="{time:HH:mm:ss} | {level} | {message}",
)

RESULTS_DIR = Path("results/ChEmbed")
BENCHMARKS = ["MTEB(eng, v2)", "ChemTEB(latest)"]


def load_models(json_file: str) -> dict:
    with open(json_file, "r") as f:
        return json.load(f)


def main():
    models_dict = load_models("models/ChEmbed.json")
    cache = ResultCache(cache_path=RESULTS_DIR)

    logger.info(f"Running {len(BENCHMARKS)} benchmarks on {len(models_dict)} models")
    logger.info(f"Results: {RESULTS_DIR}")

    for model_name, revision in tqdm(models_dict.items(), desc="Evaluating Models"):
        logger.info(f"Processing: {model_name}")

        try:
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

            for benchmark_name in BENCHMARKS:
                logger.info(f"Running: {benchmark_name}")

                try:
                    benchmark = mteb.get_benchmark(benchmark_name)
                    mteb.evaluate(model, tasks=benchmark.tasks, cache=cache)
                    logger.info(f"Completed: {benchmark_name}")

                except Exception as e:
                    logger.error(f"Failed {benchmark_name}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed {model_name}: {e}")
            continue

    logger.info(f"All evaluations completed. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
