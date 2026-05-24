"""rag-eval-harness: production-quality RAG evaluation in Python."""

from rag_eval_harness.pipeline import (
    build_report,
    compute_summary,
    load_jsonl,
    run_evaluation,
    write_jsonl,
)
from rag_eval_harness.faithfulness import score_faithfulness, score_faithfulness_detailed
from rag_eval_harness.relevance import score_answer_relevance
from rag_eval_harness.precision import score_context_precision
from rag_eval_harness.recall import score_context_recall
from rag_eval_harness.context_relevance import score_context_relevance
from rag_eval_harness.embeddings import Embedder
from rag_eval_harness.nli import NLIScorer
from rag_eval_harness.types import ClaimVerdict, EvalRecord, Scores, ScoredRecord

try:
    from importlib.metadata import version as _version
    __version__ = _version("rag-eval-harness")
except Exception:
    __version__ = "0.0.0.dev0"

__all__ = [
    # Pipeline
    "run_evaluation",
    "compute_summary",
    "build_report",
    "load_jsonl",
    "write_jsonl",
    # Individual scorers
    "score_faithfulness_detailed",
    "score_faithfulness",
    "score_answer_relevance",
    "score_context_precision",
    "score_context_recall",
    "score_context_relevance",
    # Infrastructure
    "Embedder",
    "NLIScorer",
    # Types
    "EvalRecord",
    "ScoredRecord",
    "Scores",
    "ClaimVerdict",
    "__version__",
]
