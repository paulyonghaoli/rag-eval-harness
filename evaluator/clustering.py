from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from evaluator.embeddings import Embedder
from evaluator.types import ScoredRecord, Scores


@dataclass
class ClusterSummary:
    label: str
    size: int
    mean_scores: Dict[str, float]
    centroid_example: str
    worst_examples: List[str]


def _failure_label(mean_scores: Dict[str, float]) -> str:
    lowest = min(mean_scores, key=mean_scores.__getitem__)
    return {
        "faithfulness": "hallucination-dominant",
        "recall": "retrieval-dominant",
        "relevance": "relevance-dominant",
    }.get(lowest, "mixed")


def _severity(scores: Scores) -> float:
    return sum(
        1.0 - v
        for v in (
            scores.faithfulness, scores.relevance, scores.precision,
            scores.context_relevance, scores.recall,
        )
        if v is not None
    )


def _best_k(embeddings: np.ndarray, min_k: int, max_k: int) -> int:
    best_k, best_score = min_k, float("-inf")
    for k in range(min_k, max_k + 1):
        if k >= len(embeddings):
            break
        labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score, best_k = score, k
    return best_k


def cluster_failures(
    records: List[ScoredRecord],
    embedder: Embedder,
    min_k: int = 3,
    max_k: int = 5,
) -> List[ClusterSummary]:
    if len(records) < min_k:
        return []

    examples = [f"{r.question} {r.answer}" for r in records]
    embeddings = embedder.encode(examples)
    chosen_k = _best_k(embeddings, min_k, max_k)
    kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(embeddings)

    summaries: List[ClusterSummary] = []
    for cluster_id in sorted(set(labels)):
        indices = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
        mean_scores = {
            dim: float(np.mean(
                [v for i in indices if (v := getattr(records[i].scores, dim)) is not None] or [0.0]
            ))
            for dim in ("faithfulness", "relevance", "precision", "context_relevance", "recall")
        }

        centroid = kmeans.cluster_centers_[cluster_id]
        distances = np.linalg.norm(embeddings[indices] - centroid, axis=1)
        closest = indices[int(np.argmin(distances))]
        centroid_example = f"Q: {records[closest].question} | A: {records[closest].answer}"

        worst_indices = sorted(indices, key=lambda i: _severity(records[i].scores), reverse=True)[:5]
        worst_examples = [
            f"Q: {records[i].question} | A: {records[i].answer}" for i in worst_indices
        ]

        summaries.append(ClusterSummary(
            label=_failure_label(mean_scores),
            size=len(indices),
            mean_scores=mean_scores,
            centroid_example=centroid_example,
            worst_examples=worst_examples,
        ))

    return summaries
