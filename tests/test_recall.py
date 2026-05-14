from __future__ import annotations

from evaluator.embeddings import Embedder
from evaluator.recall import score_context_recall


def test_recall_empty_contexts(embedder: Embedder) -> None:
    assert score_context_recall("Some ground truth.", [], embedder) == 0.0


def test_recall_empty_ground_truth(embedder: Embedder) -> None:
    assert score_context_recall("", ["Some context."], embedder) == 0.0


def test_recall_whitespace_ground_truth(embedder: Embedder) -> None:
    assert score_context_recall("   ", ["Some context."], embedder) == 0.0


def test_recall_full_coverage(embedder: Embedder) -> None:
    ground_truth = "The Eiffel Tower is located in Paris, France."
    contexts = ["The Eiffel Tower is a famous landmark in Paris, France."]
    score = score_context_recall(ground_truth, contexts, embedder, threshold=0.6)
    assert score == 1.0


def test_recall_no_coverage(embedder: Embedder) -> None:
    ground_truth = "Quantum entanglement links particles regardless of distance."
    contexts = ["The French baguette is a type of long thin bread."]
    score = score_context_recall(ground_truth, contexts, embedder, threshold=0.9)
    assert score == 0.0


def test_recall_partial_coverage(embedder: Embedder) -> None:
    ground_truth = "Water is H2O. It boils at 100 degrees."
    contexts = ["Water has the molecular formula H2O."]
    score = score_context_recall(ground_truth, contexts, embedder, threshold=0.6)
    assert 0.0 < score < 1.0


def test_recall_returns_float_in_range(embedder: Embedder) -> None:
    score = score_context_recall("Some fact.", ["Some context."], embedder)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_recall_multiple_contexts_better_than_one(embedder: Embedder) -> None:
    ground_truth = "Dogs are mammals. Cats are also mammals."
    single = ["Dogs are mammals."]
    both = ["Dogs are mammals.", "Cats belong to the mammal family."]
    score_single = score_context_recall(ground_truth, single, embedder, threshold=0.6)
    score_both = score_context_recall(ground_truth, both, embedder, threshold=0.6)
    assert score_both >= score_single
