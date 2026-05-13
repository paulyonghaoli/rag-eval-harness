from __future__ import annotations

import asyncio
import json
import warnings
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from evaluator.embeddings import Embedder, cosine_similarity_matrix
from evaluator.nli import NLIScorer
from evaluator.text_utils import split_into_claims
from evaluator.types import ClaimVerdict

if TYPE_CHECKING:
    from openai import AsyncOpenAI as _AsyncOpenAIClient


def _make_judge_prompt(claim: str, contexts: List[str]) -> str:
    return (
        "Given the contexts and claim below, respond with exactly one JSON object.\n"
        'If the claim is supported by the contexts: {"verdict": "SUPPORTED"}\n'
        'Otherwise: {"verdict": "NOT_SUPPORTED"}\n\n'
        f"Contexts:\n{chr(10).join(contexts)}\n\nClaim:\n{claim}\n\nJSON:"
    )


def _parse_judge_verdict(raw: str) -> Optional[bool]:
    """Parse a JSON verdict string from the LLM judge.

    Returns True for SUPPORTED, False for NOT_SUPPORTED, None if the response
    is malformed or contains an unexpected value (caller falls back to cosine).
    Strict enum matching prevents 'unsupported' or other substrings being
    misclassified as SUPPORTED.
    """
    try:
        verdict = json.loads(raw).get("verdict", "").upper()
    except (json.JSONDecodeError, AttributeError):
        return None
    if verdict == "SUPPORTED":
        return True
    if verdict == "NOT_SUPPORTED":
        return False
    return None


async def _async_judge_claim(
    claim: str,
    contexts: List[str],
    client: _AsyncOpenAIClient,
) -> Optional[bool]:
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _make_judge_prompt(claim, contexts)}],
            max_tokens=32,
            temperature=0.0,
        )
        return _parse_judge_verdict(response.choices[0].message.content.strip())
    except Exception:
        return None


async def _batch_llm_judge(
    claims: List[str],
    contexts: List[str],
    similarity: np.ndarray,
    threshold: float,
    openai_api_key: Optional[str] = None,
) -> List[bool]:
    """Fire all claim judgments concurrently; fall back to cosine on failure."""
    cosine_fallback = [float(similarity[i].max()) >= threshold for i in range(len(claims))]
    if not openai_api_key:
        return cosine_fallback
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_api_key)
        try:
            raw = await asyncio.gather(
                *[_async_judge_claim(c, contexts, client) for c in claims]
            )
        finally:
            await client.close()
    except ImportError:
        warnings.warn(
            "The 'openai' package is not installed. "
            "Install it with: pip install 'rag-eval-harness[llm]'. "
            "Falling back to cosine similarity for faithfulness.",
            UserWarning,
            stacklevel=2,
        )
        return cosine_fallback
    return [
        v if v is not None else float(similarity[i].max()) >= threshold
        for i, v in enumerate(raw)
    ]


def score_faithfulness_detailed(
    answer: str,
    contexts: List[str],
    embedder: Embedder,
    threshold: float = 0.7,
    use_llm_judge: bool = False,
    nli_scorer: Optional[NLIScorer] = None,
    openai_api_key: Optional[str] = None,
) -> tuple[float, List[ClaimVerdict]]:
    """Return (score, per-claim verdicts) so callers can trace which claims failed.

    Scoring priority per claim:
      1. NLI CrossEncoder (if nli_scorer provided) — detects contradictions, not
         just topic drift; most accurate offline method.
      2. LLM-as-judge (if use_llm_judge and API key set) — all claims judged
         concurrently via AsyncOpenAI to minimise wall-clock latency.
      3. Cosine similarity fallback — fast, offline, topic-level only.
    """
    claims = split_into_claims(answer)
    if not claims:
        return 0.0, []

    context_sentences: List[str] = []
    for ctx in contexts:
        context_sentences.extend(split_into_claims(ctx))
    if not context_sentences:
        return 0.0, [ClaimVerdict(claim=c, supported=False) for c in claims]

    verdicts: List[ClaimVerdict] = []

    if nli_scorer is not None:
        # NLI path: no cosine fallback needed, skip embedding computation entirely.
        for claim in claims:
            scores = nli_scorer.entailment_scores(context_sentences, claim)
            confidence = float(scores.max())
            verdicts.append(ClaimVerdict(
                claim=claim, supported=confidence >= 0.5, confidence=confidence,
            ))
    else:
        # Cosine / LLM paths both need the similarity matrix (LLM uses it as a fallback).
        claim_embeddings = embedder.encode(claims)
        context_embeddings = embedder.encode(context_sentences)
        similarity = cosine_similarity_matrix(claim_embeddings, context_embeddings)

        if use_llm_judge:
            # asyncio.run works from a ThreadPoolExecutor worker because each worker
            # thread has no running event loop of its own.
            try:
                supported_list = asyncio.run(
                    _batch_llm_judge(claims, contexts, similarity, threshold, openai_api_key)
                )
            except RuntimeError:
                supported_list = [float(similarity[i].max()) >= threshold for i in range(len(claims))]
            verdicts = [ClaimVerdict(claim=c, supported=s) for c, s in zip(claims, supported_list)]
        else:
            for claim_idx, claim in enumerate(claims):
                confidence = float(similarity[claim_idx].max())
                verdicts.append(ClaimVerdict(
                    claim=claim, supported=confidence >= threshold, confidence=confidence,
                ))

    score = sum(v.supported for v in verdicts) / len(verdicts)
    return score, verdicts


def score_faithfulness(
    answer: str,
    contexts: List[str],
    embedder: Embedder,
    threshold: float = 0.7,
    use_llm_judge: bool = False,
    nli_scorer: Optional[NLIScorer] = None,
    openai_api_key: Optional[str] = None,
) -> float:
    score, _ = score_faithfulness_detailed(
        answer, contexts, embedder, threshold, use_llm_judge, nli_scorer, openai_api_key
    )
    return score
