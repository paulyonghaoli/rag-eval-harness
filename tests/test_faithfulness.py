from evaluator.embeddings import Embedder
from evaluator.faithfulness import _parse_judge_verdict, score_faithfulness, split_into_claims


def test_split_into_claims_separates_sentences() -> None:
    claims = split_into_claims("The sky is blue. Water is wet.")
    assert len(claims) == 2


def test_split_into_claims_splits_compound_independent_clauses() -> None:
    # ", and " joining two independent clauses must become two separate claims.
    text = "Photosynthesis converts light to energy, and it occurs in chloroplasts."
    claims = split_into_claims(text)
    assert len(claims) == 2
    assert any("Photosynthesis" in c for c in claims)
    assert any("chloroplasts" in c for c in claims)


def test_split_into_claims_preserves_noun_phrase_and() -> None:
    # "and" inside a noun phrase (no leading comma) must NOT split.
    claims = split_into_claims("Cats and dogs are common household pets.")
    assert len(claims) == 1


def test_split_into_claims_preserves_oxford_comma_enumeration() -> None:
    # Oxford-comma list must stay as one claim — "and reflected" is not an independent clause.
    claims = split_into_claims(
        "Rainbows form when sunlight is refracted, dispersed, and reflected inside water droplets."
    )
    assert len(claims) == 1


def test_split_into_claims_splits_pronoun_led_clause() -> None:
    # ", and it/they/this" is the independent-clause signal that warrants splitting.
    claims = split_into_claims(
        "The water cycle moves water through evaporation, and it returns via precipitation."
    )
    assert len(claims) == 2


def test_faithfulness_detects_unsupported_claims() -> None:
    # Two-sentence answer: first claim is supported, second is hallucinated.
    answer = "The Eiffel Tower is in Paris. It is painted bright green."
    contexts = ["The Eiffel Tower is in Paris."]
    embedder = Embedder()

    score = score_faithfulness(answer, contexts, embedder, threshold=0.7)
    assert 0.3 <= score <= 0.7


def test_faithfulness_fully_supported() -> None:
    answer = "Water boils at 100 degrees Celsius."
    contexts = ["Water boils at 100 degrees Celsius at sea level."]
    embedder = Embedder()

    score = score_faithfulness(answer, contexts, embedder, threshold=0.7)
    assert score == 1.0


def test_faithfulness_empty_contexts() -> None:
    answer = "The sky is blue."
    embedder = Embedder()

    score = score_faithfulness(answer, [], embedder)
    assert score == 0.0


# --- LLM verdict parser ---

def test_parse_verdict_supported() -> None:
    assert _parse_judge_verdict('{"verdict": "SUPPORTED"}') is True


def test_parse_verdict_not_supported() -> None:
    assert _parse_judge_verdict('{"verdict": "NOT_SUPPORTED"}') is False


def test_parse_verdict_rejects_unsupported_substring() -> None:
    # Old bug: "unsupported" contains "supported" — strict matching must reject it.
    assert _parse_judge_verdict('"unsupported"') is None
    assert _parse_judge_verdict('{"verdict": "unsupported"}') is None


def test_parse_verdict_malformed_returns_none() -> None:
    assert _parse_judge_verdict("not json at all") is None
    assert _parse_judge_verdict("{}") is None
    assert _parse_judge_verdict('{"verdict": "MAYBE"}') is None
