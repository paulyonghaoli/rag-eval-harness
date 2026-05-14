from __future__ import annotations

import pytest

from evaluator.nli import NLIScorer


@pytest.fixture(scope="module")
def scorer() -> NLIScorer:
    return NLIScorer()


def test_entailment_idx_is_valid(scorer: NLIScorer) -> None:
    assert scorer.entailment_idx in (0, 1, 2)


def test_clear_paraphrase_is_supported(scorer: NLIScorer) -> None:
    # This exact pair was returning confidence=0.003 due to a wrong _ENTAILMENT_IDX.
    premise = "Japan's capital city is Tokyo."
    hypothesis = "Tokyo is the capital of Japan"
    scores = scorer.entailment_scores([premise], hypothesis)
    assert scores[0] > 0.5, (
        f"Entailment confidence {scores[0]:.4f} is too low for a clear paraphrase — "
        f"check NLIScorer._resolve_entailment_idx (wrong label index?)"
    )


def test_contradiction_is_not_supported(scorer: NLIScorer) -> None:
    premise = "Jupiter is the largest planet in the solar system."
    hypothesis = "Jupiter is the smallest planet in the solar system."
    assert not scorer.is_entailed([premise], hypothesis)


def test_empty_premises_returns_false(scorer: NLIScorer) -> None:
    assert not scorer.is_entailed([], "Any claim.")


def test_entailment_scores_length_matches_premises(scorer: NLIScorer) -> None:
    premises = [
        "Water is a liquid at room temperature.",
        "The sky appears blue during the day.",
        "Dogs are domesticated mammals.",
    ]
    scores = scorer.entailment_scores(premises, "Water is H2O.")
    assert len(scores) == len(premises)


def test_entailment_scores_are_probabilities(scorer: NLIScorer) -> None:
    premises = ["The cat sat on the mat.", "Cats like to sit on mats."]
    scores = scorer.entailment_scores(premises, "A cat is resting on a mat.")
    assert all(0.0 <= float(s) <= 1.0 for s in scores)


def test_is_entailed_zero_threshold_always_true(scorer: NLIScorer) -> None:
    # Any probability >= 0.0 is True — even for an unrelated pair.
    assert scorer.is_entailed(
        ["Bananas are yellow."], "The moon is made of cheese.", threshold=0.0
    )


def test_is_entailed_unit_threshold_always_false(scorer: NLIScorer) -> None:
    # Softmax outputs never reach exactly 1.0; threshold=1.0 must always fail.
    assert not scorer.is_entailed(
        ["Tokyo is the capital of Japan."],
        "Tokyo is the capital of Japan.",
        threshold=1.0,
    )


def test_entailment_scores_single_premise_shape(scorer: NLIScorer) -> None:
    scores = scorer.entailment_scores(["One premise."], "One hypothesis.")
    assert scores.ndim == 1
    assert len(scores) == 1
