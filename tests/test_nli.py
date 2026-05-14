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
