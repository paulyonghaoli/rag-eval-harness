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

    # DeBERTa NLI label order: [contradiction, neutral, entailment]
    _ENTAILMENT_IDX = 2

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-small") -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        self._lock = threading.Lock()

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = CrossEncoder(self.model_name)
        return self._model

    def entailment_scores(self, premises: list[str], hypothesis: str) -> np.ndarray:
        """Return entailment probability for each (premise, hypothesis) pair."""
        pairs = [(p, hypothesis) for p in premises]
        logits = self.model.predict(pairs)
        # softmax over [contradiction, neutral, entailment]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs[:, self._ENTAILMENT_IDX]

    def is_entailed(self, premises: list[str], hypothesis: str, threshold: float = 0.5) -> bool:
        """True if any premise entails the hypothesis above threshold."""
        if not premises:
            return False
        return bool(self.entailment_scores(premises, hypothesis).max() >= threshold)
