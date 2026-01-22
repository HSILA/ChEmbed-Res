import torch
import torch.nn.functional as F
from mteb.models.sentence_transformer_wrapper import SentenceTransformerEncoderWrapper
from mteb.types import PromptType

# Protected internal prompts, baked into the wrapper
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

        # Ensure trust_remote_code is preserved
        if "trust_remote_code" in kwargs:
            print(
                f"DEBUG: ChEmbedWrapper received and passing trust_remote_code={kwargs['trust_remote_code']}"
            )
        else:
            print("DEBUG: ChEmbedWrapper did NOT receive trust_remote_code!")

        super().__init__(model_name, **kwargs)
        self.model_name = model_name

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

        if self.model_name == "nomic-ai/nomic-embed-text-v1.5":
            if not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)
            emb = F.layer_norm(emb, normalized_shape=(emb.shape[1],))
            if normalize:
                emb = F.normalize(emb, p=2, dim=1)

        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().detach().float().numpy()
        return emb
