from __future__ import annotations

from evaluator.context_relevance import score_context_relevance
from evaluator.embeddings import Embedder


def test_context_relevance_empty_contexts(embedder: Embedder) -> None:
    assert score_context_relevance("What is AI?", [], embedder) == 0.0


def test_context_relevance_returns_float_in_range(embedder: Embedder) -> None:
    score = score_context_relevance("What is Python?", ["Python is a programming language."], embedder)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_context_relevance_on_topic_higher_than_off_topic(embedder: Embedder) -> None:
    question = "What is machine learning?"
    relevant_ctx = ["Machine learning is a branch of artificial intelligence."]
    irrelevant_ctx = ["The Eiffel Tower is located in Paris, France."]
    relevant_score = score_context_relevance(question, relevant_ctx, embedder)
    irrelevant_score = score_context_relevance(question, irrelevant_ctx, embedder)
    assert relevant_score > irrelevant_score


def test_context_relevance_single_context(embedder: Embedder) -> None:
    score = score_context_relevance(
        "Who invented the telephone?",
        ["Alexander Graham Bell invented the telephone."],
        embedder,
    )
    assert score > 0.5


def test_context_relevance_multiple_contexts_averaged(embedder: Embedder) -> None:
    question = "What is the speed of light?"
    contexts = [
        "The speed of light in vacuum is approximately 299,792 km/s.",
        "The French Alps are a popular skiing destination.",
    ]
    score = score_context_relevance(question, contexts, embedder)
    score_relevant_only = score_context_relevance(question, contexts[:1], embedder)
    score_irrelevant_only = score_context_relevance(question, contexts[1:], embedder)
    # Mean of relevant + irrelevant should lie between them
    assert score_irrelevant_only < score < score_relevant_only
