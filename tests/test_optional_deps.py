from __future__ import annotations

"""Tests that LLM judge paths degrade gracefully when openai is not installed or misconfigured."""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


def test_faithfulness_llm_falls_back_without_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate openai not installed by poisoning sys.modules.
    monkeypatch.setitem(sys.modules, "openai", None)  # type: ignore[arg-type]

    from evaluator.faithfulness import _batch_llm_judge

    similarity = np.array([[0.9, 0.1]])  # max = 0.9, above threshold 0.7 → True
    with pytest.warns(UserWarning, match="openai"):
        result = asyncio.run(
            _batch_llm_judge(["some claim"], ["ctx"], similarity, 0.7, openai_api_key="fake")
        )
    assert result == [True]  # cosine fallback applied correctly


def test_faithfulness_llm_raises_on_auth_error() -> None:
    """AuthenticationError must propagate as RuntimeError, not be silently swallowed."""
    import openai

    similarity = np.array([[0.9, 0.1]])

    mock_response = MagicMock()
    mock_response.status_code = 401
    auth_error = openai.AuthenticationError("bad key", response=mock_response, body={})

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=auth_error)
    mock_client.close = AsyncMock()

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        from evaluator.faithfulness import _batch_llm_judge
        with pytest.raises(RuntimeError, match="authentication failed"):
            asyncio.run(
                _batch_llm_judge(["some claim"], ["ctx"], similarity, 0.7, openai_api_key="bad-key")
            )


def test_relevance_llm_falls_back_without_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "openai", None)  # type: ignore[arg-type]

    from evaluator.relevance import _generate_questions_llm

    with pytest.warns(UserWarning, match="openai"):
        result = _generate_questions_llm("Some answer text.", count=3, openai_api_key="fake")
    assert result == []
