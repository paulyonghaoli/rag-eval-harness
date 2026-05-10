from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalRecord:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


@dataclass
class ClaimVerdict:
    claim: str
    supported: bool


@dataclass
class Scores:
    faithfulness: float
    relevance: float
    precision: float
    recall: Optional[float]
    faithfulness_claims: List[ClaimVerdict] = field(default_factory=list)


@dataclass
class ScoredRecord:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]
    scores: Scores
