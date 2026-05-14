from __future__ import annotations

import threading

import numpy as np
from sentence_transformers import CrossEncoder


class NLIScorer:
    """CrossEncoder-based NLI scorer for textual entailment.

    Uses a fine-tuned NLI model to judge whether a context sentence entails
    a claim, rather than relying on cosine similarity between their embeddings.
    Cosine similarity measures topic proximity; NLI measures logical support —
    "Jupiter is the largest planet" and "Jupiter is the smallest planet" have
    near-identical embeddings but opposite entailment relationships.

    The default model (nli-deberta-v3-small) is ~85 MB, runs on CPU, and needs
    no API key. Larger variants trade speed for accuracy:
      cross-encoder/nli-deberta-v3-base   (~180 MB, higher accuracy)
      cross-encoder/nli-deberta-v3-large  (~380 MB, best accuracy)
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        self._lock = threading.Lock()
        self._entailment_idx: int | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = CrossEncoder(self.model_name)
                    self._entailment_idx = self._resolve_entailment_idx(self._model)
        return self._model

    @property
    def entailment_idx(self) -> int:
        _ = self.model  # ensure model (and index) is initialised
        if self._entailment_idx is None:
            raise RuntimeError(
                f"Entailment index was not resolved after loading {self.model_name!r}. "
                "The model config may be missing an 'entailment' label."
            )
        return self._entailment_idx

    @staticmethod
    def _resolve_entailment_idx(model: CrossEncoder) -> int:
        """Read the entailment label index from the model's own config.

        Hardcoding a fixed index is fragile — different NLI checkpoints use
        different orderings (e.g. cross-encoder/nli-deberta-v3-small has
        entailment at index 1, not 2). Reading from id2label makes this
        work correctly for any compliant NLI model.
        """
        id2label: dict = getattr(getattr(model.model, "config", None), "id2label", {}) or {}
        for idx, label in id2label.items():
            if label.lower() == "entailment":
                return int(idx)
        # Fallback: scan the model's labels attribute if present
        labels = getattr(model, "labels", None) or []
        for idx, label in enumerate(labels):
            if str(label).lower() == "entailment":
                return idx
        raise ValueError(
            f"Cannot determine entailment label index for '{model}'. "
            f"id2label={id2label}"
        )

    def entailment_scores(self, premises: list[str], hypothesis: str) -> np.ndarray:
        """Return entailment probability for each (premise, hypothesis) pair."""
        pairs = [(p, hypothesis) for p in premises]
        logits = self.model.predict(pairs)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs[:, self.entailment_idx]

    def is_entailed(self, premises: list[str], hypothesis: str, threshold: float = 0.5) -> bool:
        """True if any premise entails the hypothesis above threshold."""
        if not premises:
            return False
        return bool(self.entailment_scores(premises, hypothesis).max() >= threshold)
