from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional

import numpy as np

from evaluator.embeddings import Embedder, cosine_similarity_matrix
from evaluator.faithfulness import split_into_claims

if TYPE_CHECKING:
    from openai import OpenAI as _OpenAIClient

# Instantiated once on first use; avoids per-call client construction overhead.
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


def _generate_questions_llm(answer: str, count: int = 3) -> List[str]:
    client = _get_openai_client()
    if client is None:
        return []
    prompt = (
        "Generate three concise, distinct questions that would be answered by the following text. "
        "Output one question per line, with no numbering or bullet points.\n\n"
        f"Text:\n{answer}\n"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
        questions = [ln.strip(" -0123456789.)") for ln in text.splitlines() if ln.strip()]
        return questions[:count] if len(questions) >= count else []
    except Exception:
        return []


def score_answer_relevance(
    question: str,
    answer: str,
    embedder: Embedder,
    use_llm: bool = False,
) -> float:
    """Score how well the answer addresses the question.

    LLM path (preferred): generates 3 synthetic questions from the answer and
    measures their cosine similarity to the original question. A relevant answer
    produces questions that closely resemble the real question.

    Offline fallback: measures sentence-level semantic alignment between the
    answer and the question. Template-string heuristics produce near-zero
    variance and are deliberately avoided here.
    """
    if use_llm:
        synthetic_qs = _generate_questions_llm(answer, count=3)
        if synthetic_qs:
            q_emb = embedder.encode([question])
            sq_embs = embedder.encode(synthetic_qs)
            return float(np.mean(cosine_similarity_matrix(q_emb, sq_embs)[0]))

    sentences = split_into_claims(answer)
    if not sentences:
        return 0.0
    q_emb = embedder.encode([question])
    ans_embs = embedder.encode(sentences)
    return float(np.mean(cosine_similarity_matrix(q_emb, ans_embs)[0]))
