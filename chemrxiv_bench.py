from dotenv import load_dotenv

load_dotenv()

from tqdm import tqdm
import sys
import json
from pathlib import Path

from loguru import logger
import mteb
from mteb.cache import ResultCache
from ChEmbedWrapper import ChEmbedWrapper
import torch
import gc
from transformers.cache_utils import DynamicCache

# Monkey-patch DynamicCache for compatibility with cached Qwen code
if not hasattr(DynamicCache, "get_usable_length"):

    def get_usable_length(self, input_seq_len: int, layer_idx: int = 0) -> int:
        return self.get_seq_length(layer_idx)

    DynamicCache.get_usable_length = get_usable_length

# Monkey-patch from_legacy_cache to silence warning when passed None
if hasattr(DynamicCache, "from_legacy_cache"):
    original_from_legacy_cache = DynamicCache.from_legacy_cache

    @classmethod
    def from_legacy_cache_silenced(cls, past_key_values):
        if past_key_values is None:
            return cls()
        return original_from_legacy_cache(past_key_values)

    DynamicCache.from_legacy_cache = from_legacy_cache_silenced


# Logging setup
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(
    sys.stderr, level="ERROR", format="<red>{time:HH:mm:ss} | {level} | {message}</red>"
)
logger.add(
    LOG_DIR / "chemrxiv_{time:YYMMDD_HHMMSS}.log",
    level="DEBUG",
    format="{time:HH:mm:ss} | {level} | {message}",
)

RESULTS_DIR = Path("results/chemrxiv")


def load_models(json_file: str) -> dict:
    with open(json_file, "r") as f:
        return json.load(f)


def main():
    models_dict = load_models("models/models.json")

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    cache = ResultCache(cache_path=RESULTS_DIR)

    logger.info(f"Running ChemRxivRetrieval on {len(models_dict)} models")
    logger.info(f"Results: {RESULTS_DIR}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    for model_name, revision in tqdm(models_dict.items()):
        logger.info(f"Evaluating: {model_name}")

        try:
            tasks_meta = mteb.get_tasks(tasks=["ChemRxivRetrieval"])
            if not tasks_meta:
                logger.error("ChemRxivRetrieval task not found!")
                continue
            task_instance_ref = tasks_meta[0]

            # Prepare kwargs for model loading
            kwargs = {"revision": revision}

            # Default trust_remote_code to True, but exclude for Stella to avoid collision
            if "stella_en_1.5B_v5" not in model_name:
                kwargs["trust_remote_code"] = True
                if "model_kwargs" not in kwargs:
                    kwargs["model_kwargs"] = {}
                kwargs["model_kwargs"]["trust_remote_code"] = True

            encode_kwargs = {}

            # Specific handling for Qwen to use bfloat16 and limited context
            if "Qwen" in model_name:
                if "model_kwargs" not in kwargs:
                    kwargs["model_kwargs"] = {}
                kwargs["model_kwargs"]["dtype"] = torch.bfloat16
                kwargs["max_seq_length"] = 2048
                encode_kwargs["batch_size"] = 8

            if "nomic" in model_name:
                if "model_kwargs" not in kwargs:
                    kwargs["model_kwargs"] = {}
                kwargs["model_kwargs"]["dtype"] = torch.bfloat16
                kwargs["model_kwargs"]["trust_remote_code"] = True
                encode_kwargs["batch_size"] = 16

            if "ChEmbed" in model_name:
                model = ChEmbedWrapper(model_name, device=device, **kwargs)
            else:
                model = mteb.get_model(model_name, device=device, **kwargs)

            TaskClass = type(task_instance_ref)
            task = TaskClass()

            # We rely on MTEB's internal cache checking to skip if done,
            mteb.evaluate(
                model,
                tasks=[task],
                cache=cache,
                encode_kwargs=encode_kwargs,
                num_proc=4,
            )
            logger.info(f"Completed: {model_name}")

            del task

        except Exception as e:
            import traceback

            logger.error(f"Failed {model_name}: {e}\n{traceback.format_exc()}")
            continue

        finally:
            # Extensive Garbage Collection
            if "model" in locals():
                del model
            gc.collect()
            torch.cuda.empty_cache()

    logger.info(f"All evaluations completed. Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
