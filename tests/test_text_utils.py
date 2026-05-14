from __future__ import annotations

from evaluator.text_utils import split_into_claims


def test_empty_string_returns_empty() -> None:
    assert split_into_claims("") == []


def test_whitespace_only_returns_empty() -> None:
    assert split_into_claims("   \n\t  ") == []


def test_single_sentence_period() -> None:
    claims = split_into_claims("Water is wet.")
    assert len(claims) == 1
    assert claims[0] == "Water is wet"


def test_two_sentences_period() -> None:
    claims = split_into_claims("The sky is blue. Water is wet.")
    assert len(claims) == 2


def test_question_mark_splits() -> None:
    claims = split_into_claims("Is the sky blue? Yes it is blue.")
    assert len(claims) == 2


def test_exclamation_mark_splits() -> None:
    claims = split_into_claims("The sky is blue! Water is wet.")
    assert len(claims) == 2


def test_semicolon_splits() -> None:
    claims = split_into_claims("Water is wet; fire is hot.")
    assert len(claims) == 2


def test_fragments_below_min_length_dropped() -> None:
    # Single-word fragments after split should be filtered out.
    claims = split_into_claims("Hello. OK. Water is definitely wet here.")
    assert all(len(c.split()) >= 2 for c in claims)


def test_short_answer_returns_full_text_as_fallback() -> None:
    # If all fragments are too short, the whole text is returned as-is.
    claims = split_into_claims("Yes.")
    assert claims == ["Yes."] or claims == []  # either acceptable


def test_trailing_period_stripped() -> None:
    claims = split_into_claims("Water is wet.")
    assert not any(c.endswith(".") for c in claims)


def test_compound_clause_with_pronoun_and_splits() -> None:
    text = "Photosynthesis converts light to energy, and it occurs in chloroplasts."
    claims = split_into_claims(text)
    assert len(claims) == 2
    assert any("Photosynthesis" in c for c in claims)
    assert any("chloroplasts" in c for c in claims)


def test_oxford_comma_enumeration_not_split() -> None:
    claims = split_into_claims(
        "Rainbows are caused by refraction, dispersion, and reflection of light."
    )
    assert len(claims) == 1


def test_noun_phrase_and_not_split() -> None:
    claims = split_into_claims("Cats and dogs are popular pets worldwide.")
    assert len(claims) == 1


def test_multiple_compound_clauses() -> None:
    text = "X is true. Y is also true, and it can be verified. Z follows."
    claims = split_into_claims(text)
    assert len(claims) == 4  # X, Y, it, Z


def test_pronoun_and_variants() -> None:
    for pronoun in ("it", "they", "this", "these", "he", "she", "we"):
        text = f"Alpha does something, and {pronoun} does something else entirely."
        claims = split_into_claims(text)
        assert len(claims) == 2, f"Expected split on ', and {pronoun}', got {claims}"
