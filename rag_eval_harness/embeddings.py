from __future__ import annotations

import threading
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Lazy-loading sentence-transformer wrapper with a per-text embedding cache.

    The cache avoids re-encoding identical strings when multiple scorers process
    the same record (faithfulness and recall both embed the same context sentences).

    Threading design: the lock guards only cache reads and writes, not the model
    encoding call itself.  Two threads may redundantly encode the same text if
    both see it as missing before either updates the cache — that is safe because
    SentenceTransformer.encode() is deterministic (last write wins, values are
    identical).  Keeping encoding outside the lock lets threads that need
    *different* texts encode concurrently instead of serialising behind the lock.

    Note: ThreadPoolExecutor in the pipeline mainly speeds up LLM-judge mode,
    where each record's async API calls are I/O-bound.  For offline cosine-only
    scoring, CPython's GIL still serialises CPU-bound PyTorch work, so threading
    reduces scheduling overhead but does not achieve true parallelism on encode().
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def model(self) -> SentenceTransformer:
        # Double-checked locking: avoids re-acquiring the lock on every encode()
        # call once the model is loaded, while preventing concurrent threads from
        # each calling SentenceTransformer() during the first load.
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        texts_list = list(texts)

        with self._lock:
            missing = [t for t in texts_list if t not in self._cache]

        if missing:
            vecs = self.model.encode(missing, convert_to_numpy=True, normalize_embeddings=True)
            with self._lock:
                for text, vec in zip(missing, vecs):
                    self._cache[text] = vec

        with self._lock:
            return np.stack([self._cache[t] for t in texts_list])


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    return np.matmul(a, b.T)
