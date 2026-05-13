from __future__ import annotations

from typing import List

import numpy as np

from evaluator.embeddings import Embedder, cosine_similarity_matrix
from evaluator.text_utils import split_into_claims


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

    supported = 0
    for chunk_idx in range(similarity.shape[0]):
        if float(similarity[chunk_idx].max()) >= threshold:
            supported += 1
    return float(supported) / len(contexts)
