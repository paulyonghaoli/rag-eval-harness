from evaluator.clustering import cluster_failures
from evaluator.embeddings import Embedder
from evaluator.types import Scores, ScoredRecord


def _record(q: str, a: str, s: Scores) -> ScoredRecord:
    return ScoredRecord(question=q, answer=a, contexts=[], ground_truth=None, scores=s)


def test_clustering_selects_a_valid_cluster_count() -> None:
    records = [
        _record("Q1", "The answer is A.", Scores(1.0, 0.9, 1.0, 1.0)),
        _record("Q2", "The answer is B.", Scores(0.2, 0.8, 0.3, 0.2)),
        _record("Q3", "The answer is C.", Scores(0.7, 0.5, 0.7, 0.6)),
        _record("Q4", "The answer is D.", Scores(0.4, 0.3, 0.4, 0.3)),
        _record("Q5", "The answer is E.", Scores(0.9, 0.9, 0.8, 0.9)),
        _record("Q6", "The answer is F.", Scores(0.1, 0.2, 0.1, 0.1)),
    ]
    embedder = Embedder()
    summaries = cluster_failures(records, embedder, min_k=3, max_k=5)

    assert summaries
    assert all(s.size > 0 for s in summaries)
    assert any("dominant" in s.label for s in summaries)
