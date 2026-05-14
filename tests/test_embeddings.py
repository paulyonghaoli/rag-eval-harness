from __future__ import annotations

import numpy as np
import pytest

from evaluator.embeddings import Embedder, cosine_similarity_matrix


# --- cosine_similarity_matrix ---

def test_cosine_similarity_identical_vectors() -> None:
    v = np.array([[1.0, 0.0, 0.0]])
    result = cosine_similarity_matrix(v, v)
    assert result.shape == (1, 1)
    assert abs(result[0, 0] - 1.0) < 1e-6


def test_cosine_similarity_orthogonal_vectors() -> None:
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    result = cosine_similarity_matrix(a, b)
    assert result.shape == (1, 1)
    assert abs(result[0, 0]) < 1e-6


def test_cosine_similarity_1d_inputs_reshaped() -> None:
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 0.0])
    result = cosine_similarity_matrix(a, b)
    assert result.shape == (1, 1)
    assert abs(result[0, 0] - 1.0) < 1e-6


def test_cosine_similarity_matrix_shape() -> None:
    rng = np.random.default_rng(42)
    a = rng.standard_normal((3, 8))
    b = rng.standard_normal((4, 8))
    result = cosine_similarity_matrix(a, b)
    assert result.shape == (3, 4)


def test_cosine_similarity_known_values() -> None:
    # 45° angle → cos = sqrt(2)/2 ≈ 0.7071
    a = np.array([[1.0, 0.0]])
    b = np.array([[1.0, 1.0]]) / np.sqrt(2)
    result = cosine_similarity_matrix(a, b)
    assert abs(result[0, 0] - np.sqrt(2) / 2) < 1e-5


def test_cosine_similarity_opposite_vectors() -> None:
    a = np.array([[1.0, 0.0]])
    b = np.array([[-1.0, 0.0]])
    result = cosine_similarity_matrix(a, b)
    assert abs(result[0, 0] - (-1.0)) < 1e-6


# --- Embedder ---

def test_embedder_returns_2d_ndarray(embedder: Embedder) -> None:
    vecs = embedder.encode(["hello world"])
    assert isinstance(vecs, np.ndarray)
    assert vecs.ndim == 2
    assert vecs.shape[0] == 1


def test_embedder_vectors_are_unit_normalized(embedder: Embedder) -> None:
    vecs = embedder.encode(["test sentence for normalization check"])
    norm = float(np.linalg.norm(vecs[0]))
    assert abs(norm - 1.0) < 1e-5


def test_embedder_multiple_texts_correct_shape(embedder: Embedder) -> None:
    texts = ["first", "second sentence here", "third one also"]
    vecs = embedder.encode(texts)
    assert vecs.shape == (3, vecs.shape[1])


def test_embedder_same_text_same_vector(embedder: Embedder) -> None:
    text = "deterministic encoding test"
    v1 = embedder.encode([text])
    v2 = embedder.encode([text])
    np.testing.assert_array_almost_equal(v1, v2)


def test_embedder_cache_populated_after_encode(embedder: Embedder) -> None:
    text = "unique cache population string xyz987"
    embedder.encode([text])
    assert text in embedder._cache


def test_embedder_similar_sentences_higher_similarity_than_dissimilar(embedder: Embedder) -> None:
    q = embedder.encode(["What is the capital of France?"])
    on_topic = embedder.encode(["Paris is the capital city of France."])
    off_topic = embedder.encode(["Quantum physics describes subatomic behaviour."])
    sim_on = float(cosine_similarity_matrix(q, on_topic)[0, 0])
    sim_off = float(cosine_similarity_matrix(q, off_topic)[0, 0])
    assert sim_on > sim_off


def test_embedder_different_texts_different_vectors(embedder: Embedder) -> None:
    v1 = embedder.encode(["the sky is blue"])
    v2 = embedder.encode(["quantum entanglement is weird"])
    sim = float(cosine_similarity_matrix(v1, v2)[0, 0])
    assert sim < 0.99  # must not be identical
