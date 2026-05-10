from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EvalRecord:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None


@dataclass
class Scores:
    faithfulness: float
    relevance: float
    precision: float
    recall: Optional[float]


@dataclass
class ScoredRecord:
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str]
    scores: Scores
