from __future__ import annotations

import threading
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Lazy-loading sentence-transformer wrapper with a per-text embedding cache.

    The cache avoids re-encoding identical strings when multiple scorers process
    the same record (faithfulness and recall both embed the same context sentences).
    A threading lock makes encode() safe to call from concurrent threads.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)
        with self._lock:
            missing = [t for t in texts_list if t not in self._cache]
            if missing:
                vecs = self.model.encode(
                    missing, convert_to_numpy=True, normalize_embeddings=True
                )
                for text, vec in zip(missing, vecs):
                    self._cache[text] = vec
            return np.stack([self._cache[t] for t in texts_list])


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return np.matmul(a, b.T)
