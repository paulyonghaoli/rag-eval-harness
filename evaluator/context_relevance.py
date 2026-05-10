from __future__ import annotations

from typing import List

import numpy as np

from evaluator.embeddings import Embedder, cosine_similarity_matrix


def score_context_relevance(
    question: str,
    contexts: List[str],
    embedder: Embedder,
) -> float:
    """Mean cosine similarity between the question and each retrieved chunk.

    Measures whether the retriever pulled on-topic chunks, independent of
    whether those chunks ended up in the answer (precision) or cover the
    ground truth (recall). Low scores indicate the retriever is returning
    semantically distant documents regardless of their answer utility.
    """
    if not contexts:
        return 0.0
    q_emb = embedder.encode([question])
    ctx_embs = embedder.encode(contexts)
    sims = cosine_similarity_matrix(q_emb, ctx_embs)[0]
    return float(np.mean(sims))
