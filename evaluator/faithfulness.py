from __future__ import annotations

import asyncio
import os
import re
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from evaluator.embeddings import Embedder, cosine_similarity_matrix
from evaluator.nli import NLIScorer
from evaluator.types import ClaimVerdict

if TYPE_CHECKING:
    from openai import AsyncOpenAI as _AsyncOpenAIClient


def split_into_claims(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    fragments = re.split(r"[.?!;]\s*", text)
    claims = []
    for fragment in fragments:
        fragment = fragment.strip().rstrip(".")
        if len(fragment.split()) < 3:
            continue
        claims.append(fragment)
    return claims or [text]


def _make_judge_prompt(claim: str, contexts: List[str]) -> str:
    return (
        "Given the following context chunks and a claim, answer only 'SUPPORTED' if the claim is "
        "clearly supported by the contexts, otherwise answer 'NOT SUPPORTED'.\n\n"
        f"Contexts:\n{chr(10).join(contexts)}\n\nClaim:\n{claim}\n\nAnswer:"
    )


async def _async_judge_claim(
    claim: str,
    contexts: List[str],
    client: _AsyncOpenAIClient,
) -> Optional[bool]:
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": _make_judge_prompt(claim, contexts)}],
            max_tokens=16,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip().lower()
        return "supported" in text and "not supported" not in text
    except Exception:
        return None


async def _batch_llm_judge(
    claims: List[str],
    contexts: List[str],
    similarity: np.ndarray,
    threshold: float,
) -> List[bool]:
    """Fire all claim judgments concurrently; fall back to cosine on failure."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return [float(similarity[i].max()) >= threshold for i in range(len(claims))]
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key)
        try:
            raw = await asyncio.gather(
                *[_async_judge_claim(c, contexts, client) for c in claims]
            )
        finally:
            await client.close()
    except ImportError:
        return [float(similarity[i].max()) >= threshold for i in range(len(claims))]
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

    # Pre-compute cosine similarity matrix once — used as fallback in all paths.
    claim_embeddings = embedder.encode(claims)
    context_embeddings = embedder.encode(context_sentences)
    similarity = cosine_similarity_matrix(claim_embeddings, context_embeddings)

    verdicts: List[ClaimVerdict] = []

    if nli_scorer is not None:
        for claim in claims:
            is_supported = nli_scorer.is_entailed(context_sentences, claim, threshold=0.5)
            verdicts.append(ClaimVerdict(claim=claim, supported=is_supported))
    elif use_llm_judge:
        # asyncio.run works from a ThreadPoolExecutor worker because each worker
        # thread has no running event loop of its own.
        try:
            supported_list = asyncio.run(
                _batch_llm_judge(claims, contexts, similarity, threshold)
            )
        except RuntimeError:
            supported_list = [float(similarity[i].max()) >= threshold for i in range(len(claims))]
        verdicts = [ClaimVerdict(claim=c, supported=s) for c, s in zip(claims, supported_list)]
    else:
        for claim_idx, claim in enumerate(claims):
            is_supported = float(similarity[claim_idx].max()) >= threshold
            verdicts.append(ClaimVerdict(claim=claim, supported=is_supported))

    score = sum(v.supported for v in verdicts) / len(verdicts)
    return score, verdicts


def score_faithfulness(
    answer: str,
    contexts: List[str],
    embedder: Embedder,
    threshold: float = 0.7,
    use_llm_judge: bool = False,
    nli_scorer: Optional[NLIScorer] = None,
) -> float:
    score, _ = score_faithfulness_detailed(
        answer, contexts, embedder, threshold, use_llm_judge, nli_scorer
    )
    return score
