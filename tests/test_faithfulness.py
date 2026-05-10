from evaluator.embeddings import Embedder
from evaluator.faithfulness import score_faithfulness, split_into_claims


def test_split_into_claims_separates_sentences() -> None:
    claims = split_into_claims("The sky is blue. Water is wet.")
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
