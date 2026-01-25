import torch
from transformers import AutoTokenizer
import torch.nn.functional as F
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import PromptType
from loguru import logger

_NOMIC_PROMPTS = {
    "Classification": "classification: ",
    "MultilabelClassification": "classification: ",
    "Clustering": "clustering: ",
    "PairClassification": "classification: ",
    "Reranking": "classification: ",
    "STS": "classification: ",
    "Summarization": "classification: ",
    PromptType.query.value: "search_query: ",
    PromptType.document.value: "search_document: ",
}


class ChEmbedWrapper(SentenceTransformerEncoderWrapper):
    def __init__(self, model_name: str, **kwargs):
        # Automatically inject prompts if they aren't provided
        if "model_prompts" not in kwargs:
            kwargs["model_prompts"] = _NOMIC_PROMPTS

        super().__init__(model_name, **kwargs)
        self.model_name = model_name

        if "BASF-AI/ChEmbed" in model_name and "vanilla" not in model_name:
            logger.info(f"Replacing tokenizer for {model_name} with BASF-AI/ChemVocab")
            new_tokenizer = AutoTokenizer.from_pretrained(
                "BASF-AI/ChemVocab", trust_remote_code=True
            )
            # Replace the tokenizer in the first module (Transformer)
            transformer_module = self.model._first_module()
            transformer_module.tokenizer = new_tokenizer
            # Also update the auto_model config if possible, although not strictly required for inference if input_ids are correct
            if hasattr(transformer_module, "auto_model"):
                # Optional: print vocab size match check
                logger.info(f"New tokenizer vocab size: {len(new_tokenizer)}")

    def to(self, device: torch.device) -> None:
        self.model.to(device)

    def encode(
        self,
        inputs,
        *,
        task_metadata,
        hf_split,
        hf_subset,
        prompt_type=None,
        batch_size=8,
        **kwargs,
    ):
        prompt_name = (
            self.get_prompt_name(task_metadata, prompt_type)
            or PromptType.document.value
        )
        sentences = [text for batch in inputs for text in batch["text"]]

        normalize = task_metadata not in (
            "Classification",
            "MultilabelClassification",
            "PairClassification",
            "Reranking",
            "STS",
            "Summarization",
        )

        emb = self.model.encode(
            sentences,
            prompt_name=prompt_name,
            batch_size=batch_size,
            **kwargs,
        )

        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().detach().float().numpy()
        return emb
