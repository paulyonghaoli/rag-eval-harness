from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from evaluator.embeddings import Embedder, cosine_similarity_matrix
from evaluator.types import ClaimVerdict

if TYPE_CHECKING:
    from openai import OpenAI as _OpenAIClient

# Instantiated once on first use; avoids per-claim client construction overhead.
_openai_client: Optional[_OpenAIClient] = None


def _get_openai_client() -> Optional[_OpenAIClient]:
    global _openai_client
    if _openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                _openai_client = OpenAI(api_key=api_key)
            except ImportError:
                pass
    return _openai_client


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


def _llm_support_judge(claim: str, contexts: List[str]) -> Optional[bool]:
    client = _get_openai_client()
    if client is None:
        return None
    context_text = "\n".join(contexts)
    prompt = (
        "Given the following context chunks and a claim, answer only 'SUPPORTED' if the claim is "
        "clearly supported by the contexts, otherwise answer 'NOT SUPPORTED'.\n\n"
        f"Contexts:\n{context_text}\n\nClaim:\n{claim}\n\nAnswer:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip().lower()
        return "supported" in text and "not supported" not in text
    except Exception:
        return None


def score_faithfulness_detailed(
    answer: str,
    contexts: List[str],
    embedder: Embedder,
    threshold: float = 0.7,
    use_llm_judge: bool = False,
) -> tuple[float, List[ClaimVerdict]]:
    """Return (score, per-claim verdicts) so callers can trace which claims failed."""
    claims = split_into_claims(answer)
    if not claims:
        return 0.0, []

    context_sentences: List[str] = []
    for ctx in contexts:
        context_sentences.extend(split_into_claims(ctx))
    if not context_sentences:
        return 0.0, [ClaimVerdict(claim=c, supported=False) for c in claims]

    claim_embeddings = embedder.encode(claims)
    context_embeddings = embedder.encode(context_sentences)
    similarity = cosine_similarity_matrix(claim_embeddings, context_embeddings)

    verdicts: List[ClaimVerdict] = []
    for claim_idx, claim in enumerate(claims):
        is_supported = False
        if use_llm_judge:
            verdict = _llm_support_judge(claim, contexts)
            if verdict is not None:
                is_supported = verdict
            else:
                is_supported = float(similarity[claim_idx].max()) >= threshold
        else:
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
) -> float:
    score, _ = score_faithfulness_detailed(answer, contexts, embedder, threshold, use_llm_judge)
    return score
