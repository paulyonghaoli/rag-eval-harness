from __future__ import annotations

from typing import List

import numpy as np

from rag_eval_harness.embeddings import Embedder, cosine_similarity_matrix
from rag_eval_harness.text_utils import split_into_claims


def score_context_precision(
    answer: str,
    contexts: List[str],
    embedder: Embedder,
    threshold: float = 0.5,
) -> float:
    if not contexts:
        return 0.0

    answer_sentences = split_into_claims(answer)
    if not answer_sentences:
        return 0.0

    answer_embeddings = embedder.encode(answer_sentences)
    chunk_embeddings = embedder.encode(contexts)
    similarity = cosine_similarity_matrix(chunk_embeddings, answer_embeddings)

    return float((similarity.max(axis=1) >= threshold).sum()) / len(contexts)
