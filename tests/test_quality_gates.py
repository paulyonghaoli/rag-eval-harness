from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator.pipeline import compute_summary, run_evaluation
from evaluator.types import Scores, ScoredRecord


def _record(q: str, faith: float, rel: float) -> ScoredRecord:
    return ScoredRecord(
        question=q, answer="A.", contexts=[], ground_truth=None,
        scores=Scores(faithfulness=faith, relevance=rel, precision=1.0,
                      context_relevance=1.0, recall=None),
    )


def test_summary_keys_present() -> None:
    records = [_record("Q1", 0.9, 0.8), _record("Q2", 0.7, 0.6)]
    summary = compute_summary(records, {})
    assert summary["records_evaluated"] == 2
    assert "faithfulness" in summary["metrics"]
    assert summary["quality_gates"] is None
    assert summary["passed"] is None


def test_quality_gate_passes() -> None:
    records = [_record("Q1", 0.9, 0.8), _record("Q2", 0.9, 0.8)]
    summary = compute_summary(records, {"faithfulness": 0.85})
    assert summary["passed"] is True
    assert summary["quality_gates"]["faithfulness"]["passed"] is True


def test_quality_gate_fails() -> None:
    records = [_record("Q1", 0.5, 0.4), _record("Q2", 0.5, 0.4)]
    summary = compute_summary(records, {"faithfulness": 0.85})
    assert summary["passed"] is False
    assert summary["quality_gates"]["faithfulness"]["passed"] is False


def test_multiple_gates_all_must_pass() -> None:
    records = [_record("Q1", 0.9, 0.3)]
    summary = compute_summary(records, {"faithfulness": 0.85, "relevance": 0.70})
    assert summary["passed"] is False  # relevance 0.3 < 0.70


def test_summary_json_written_to_output_dir(tmp_path: Path) -> None:
    config = {
        "embedder_model": "all-MiniLM-L6-v2",
        "similarity_thresholds": {"faithfulness": 0.7, "precision": 0.5},
        "clustering": {"min_k": 3, "max_k": 5},
        "llm_as_judge": {"enabled": False},
    }
    summary = run_evaluation(Path("data/sample_eval_set.jsonl"), tmp_path, config)
    summary_file = tmp_path / "summary.json"
    assert summary_file.exists()
    data = json.loads(summary_file.read_text())
    assert data["records_evaluated"] >= 20
    assert "faithfulness" in data["metrics"]
    assert data["quality_gates"] is None   # no gates configured
    assert data["passed"] is None
    assert summary == data
