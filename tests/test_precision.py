from __future__ import annotations

from evaluator.embeddings import Embedder
from evaluator.precision import score_context_precision


def test_precision_empty_contexts(embedder: Embedder) -> None:
    assert score_context_precision("Any answer.", [], embedder) == 0.0


def test_precision_empty_answer(embedder: Embedder) -> None:
    assert score_context_precision("", ["Some context."], embedder) == 0.0


def test_precision_whitespace_answer(embedder: Embedder) -> None:
    assert score_context_precision("   ", ["Some context."], embedder) == 0.0


def test_precision_all_contexts_relevant(embedder: Embedder) -> None:
    answer = "The dog chased the cat across the yard."
    contexts = [
        "A dog was running after a cat in a yard.",
        "The dog pursued the cat outside.",
    ]
    score = score_context_precision(answer, contexts, embedder, threshold=0.4)
    assert score == 1.0


def test_precision_no_contexts_relevant(embedder: Embedder) -> None:
    answer = "The quantum field collapses into a superposition state."
    contexts = [
        "I enjoy eating pasta on rainy days.",
        "The French Revolution started in 1789.",
    ]
    score = score_context_precision(answer, contexts, embedder, threshold=0.9)
    assert score == 0.0


def test_precision_partial(embedder: Embedder) -> None:
    answer = "Water boils at 100 degrees Celsius."
    contexts = [
        "Water boils at 100 degrees Celsius at sea level.",  # relevant
        "The Eiffel Tower is 330 metres tall.",               # irrelevant
    ]
    score = score_context_precision(answer, contexts, embedder, threshold=0.6)
    assert score == 0.5


def test_precision_returns_float_in_range(embedder: Embedder) -> None:
    score = score_context_precision("Some answer.", ["Some context."], embedder)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_precision_single_context_match(embedder: Embedder) -> None:
    answer = "Paris is the capital of France."
    contexts = ["Paris is the capital and most populous city of France."]
    score = score_context_precision(answer, contexts, embedder, threshold=0.5)
    assert score == 1.0
