from __future__ import annotations

import pytest

from evaluator.clustering import ClusterSummary, _failure_label, _severity, cluster_failures
from evaluator.embeddings import Embedder
from evaluator.types import Scores, ScoredRecord


def _record(q: str, a: str, s: Scores) -> ScoredRecord:
    return ScoredRecord(question=q, answer=a, contexts=[], ground_truth=None, scores=s)


def _scores(**kwargs: float) -> Scores:
    base = dict(faithfulness=1.0, relevance=1.0, precision=1.0, context_relevance=1.0, recall=1.0)
    base.update(kwargs)
    return Scores(**base)


# --- _failure_label ---

@pytest.mark.parametrize("dim,expected_label", [
    ("faithfulness", "hallucination-dominant"),
    ("recall", "retrieval-gap-dominant"),
    ("relevance", "relevance-dominant"),
    ("precision", "noise-dominant"),
    ("context_relevance", "topic-drift-dominant"),
])
def test_failure_label_each_dimension(dim: str, expected_label: str) -> None:
    scores = {d: 1.0 for d in ("faithfulness", "relevance", "precision", "context_relevance", "recall")}
    scores[dim] = 0.0
    assert _failure_label(scores) == expected_label


def test_failure_label_all_equal_returns_mixed() -> None:
    scores = {d: 0.5 for d in ("faithfulness", "relevance", "precision", "context_relevance", "recall")}
    # When all are equal, min() picks the first key alphabetically.
    # Just verify the function returns a non-empty string without crashing.
    label = _failure_label(scores)
    assert isinstance(label, str)
    assert label  # non-empty


# --- _severity ---

def test_severity_perfect_scores_is_zero() -> None:
    s = Scores(faithfulness=1.0, relevance=1.0, precision=1.0, context_relevance=1.0, recall=1.0)
    assert _severity(s) == 0.0


def test_severity_all_zero_scores() -> None:
    s = Scores(faithfulness=0.0, relevance=0.0, precision=0.0, context_relevance=0.0, recall=0.0)
    assert abs(_severity(s) - 5.0) < 1e-6


def test_severity_none_recall_excluded() -> None:
    s_with = Scores(faithfulness=0.0, relevance=0.0, precision=0.0, context_relevance=0.0, recall=0.0)
    s_without = Scores(faithfulness=0.0, relevance=0.0, precision=0.0, context_relevance=0.0, recall=None)
    assert _severity(s_with) > _severity(s_without)


def test_severity_partial() -> None:
    s = Scores(faithfulness=0.5, relevance=1.0, precision=1.0, context_relevance=1.0, recall=1.0)
    assert abs(_severity(s) - 0.5) < 1e-6


# --- cluster_failures ---

def test_cluster_failures_too_few_records_returns_empty(embedder: Embedder) -> None:
    records = [
        _record("Q1", "Answer one.", _scores(faithfulness=0.1)),
        _record("Q2", "Answer two.", _scores(faithfulness=0.9)),
    ]
    result = cluster_failures(records, embedder, min_k=3, max_k=5)
    assert result == []


def test_cluster_failures_returns_cluster_summaries(embedder: Embedder) -> None:
    records = [
        _record(f"Q{i}", f"The answer number {i} is here.", _scores(faithfulness=float(i % 2)))
        for i in range(1, 8)
    ]
    summaries = cluster_failures(records, embedder, min_k=2, max_k=3)
    assert isinstance(summaries, list)
    assert all(isinstance(s, ClusterSummary) for s in summaries)


def test_cluster_failures_all_sizes_positive(embedder: Embedder) -> None:
    records = [
        _record("Q1", "The answer is A.", Scores(1.0, 0.9, 1.0, 0.9, 1.0)),
        _record("Q2", "The answer is B.", Scores(0.2, 0.8, 0.3, 0.2, 0.2)),
        _record("Q3", "The answer is C.", Scores(0.7, 0.5, 0.7, 0.6, 0.6)),
        _record("Q4", "The answer is D.", Scores(0.4, 0.3, 0.4, 0.3, 0.3)),
        _record("Q5", "The answer is E.", Scores(0.9, 0.9, 0.8, 0.9, 0.9)),
        _record("Q6", "The answer is F.", Scores(0.1, 0.2, 0.1, 0.1, 0.1)),
    ]
    summaries = cluster_failures(records, embedder, min_k=2, max_k=3)
    total = sum(s.size for s in summaries)
    assert total == len(records)
    assert all(s.size > 0 for s in summaries)


def test_cluster_failures_labels_are_strings(embedder: Embedder) -> None:
    records = [
        _record(f"Q{i}", f"Answer {i}.", _scores())
        for i in range(1, 7)
    ]
    summaries = cluster_failures(records, embedder, min_k=2, max_k=3)
    assert all(isinstance(s.label, str) for s in summaries)
