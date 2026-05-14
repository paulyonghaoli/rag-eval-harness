from __future__ import annotations

import pytest

from evaluator.embeddings import Embedder


@pytest.fixture(scope="session")
def embedder() -> Embedder:
    """Single Embedder instance shared across the whole test session.

    Loading SentenceTransformer is slow; sharing the instance across all test
    modules avoids reloading the model for every test file.
    """
    return Embedder()
