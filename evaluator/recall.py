from __future__ import annotations

from typing import List

from evaluator.embeddings import Embedder, cosine_similarity_matrix
from evaluator.text_utils import split_into_claims


def score_context_recall(
    ground_truth: str,
    contexts: List[str],
    embedder: Embedder,
    threshold: float = 0.7,
) -> float:
    claims = split_into_claims(ground_truth)
    if not claims:
        return 0.0

    context_sentences = []
    for context in contexts:
        context_sentences.extend(split_into_claims(context))

    if not context_sentences:
        return 0.0

    claim_embeddings = embedder.encode(claims)
    context_embeddings = embedder.encode(context_sentences)
    similarity = cosine_similarity_matrix(claim_embeddings, context_embeddings)

    covered = 0
    for claim_idx in range(similarity.shape[0]):
        if float(similarity[claim_idx].max()) >= threshold:
            covered += 1
    return float(covered) / len(claims)
