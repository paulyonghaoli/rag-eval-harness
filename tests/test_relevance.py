from __future__ import annotations

from evaluator.embeddings import Embedder
from evaluator.relevance import score_answer_relevance


def test_relevance_on_topic_answer_scores_above_half(embedder: Embedder) -> None:
    question = "What is the capital of France?"
    answer = "The capital of France is Paris, a major European city."
    score = score_answer_relevance(question, answer, embedder)
    assert score > 0.5


def test_relevance_off_topic_scores_lower_than_on_topic(embedder: Embedder) -> None:
    question = "What is the capital of France?"
    on_topic = "Paris is the capital of France."
    off_topic = "Quantum entanglement is a fascinating physics phenomenon."
    on_score = score_answer_relevance(question, on_topic, embedder)
    off_score = score_answer_relevance(question, off_topic, embedder)
    assert on_score > off_score


def test_relevance_returns_float_in_range(embedder: Embedder) -> None:
    score = score_answer_relevance("Some question?", "Some answer.", embedder)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_relevance_empty_answer_does_not_crash(embedder: Embedder) -> None:
    # split_into_claims on empty string returns [] — score_answer_relevance
    # should handle gracefully and return a valid float.
    score = score_answer_relevance("What is X?", "", embedder)
    assert isinstance(score, float)


def test_relevance_llm_flag_false_uses_offline_path(embedder: Embedder) -> None:
    # use_llm=False must never raise even without an API key present.
    score = score_answer_relevance(
        "What causes tides?",
        "Tides are caused by the gravity of the Moon.",
        embedder,
        use_llm=False,
    )
    assert 0.0 <= score <= 1.0
