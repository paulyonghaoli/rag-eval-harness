from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator.pipeline import build_report, load_jsonl, run_evaluation, write_jsonl
from evaluator.types import Scores, ScoredRecord


def _scored(q: str = "Q?", faith: float = 0.9) -> ScoredRecord:
    return ScoredRecord(
        question=q, answer="A.", contexts=[], ground_truth=None,
        scores=Scores(faithfulness=faith, relevance=0.8, precision=0.7,
                      context_relevance=0.6, recall=None),
    )


def test_pipeline_generates_scores_and_report(tmp_path: Path) -> None:
    input_path = Path("data/sample_eval_set.jsonl")
    output_dir = tmp_path / "results"
    config = {
        "embedder_model": "all-MiniLM-L6-v2",
        "similarity_thresholds": {"faithfulness": 0.7, "precision": 0.5},
        "clustering": {"min_k": 3, "max_k": 5},
        "llm_as_judge": {"enabled": False},
    }

    run_evaluation(input_path, output_dir, config)

    scores_file = output_dir / "scores.jsonl"
    report_file = output_dir / "report.md"
    assert scores_file.exists()
    assert report_file.exists()

    lines = scores_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 20

    for line in lines:
        record = json.loads(line)
        assert "scores" in record
        assert all(d in record["scores"] for d in ["faithfulness", "relevance", "precision", "context_relevance", "recall"])


def test_write_jsonl_roundtrip(tmp_path: Path) -> None:
    records = [
        ScoredRecord(
            question="What is X?", answer="X is Y.",
            contexts=["X is a type of Y."], ground_truth=None,
            scores=Scores(faithfulness=0.9, relevance=0.8, precision=0.7,
                          context_relevance=0.6, recall=None),
        ),
        ScoredRecord(
            question="Define Z.", answer="Z is W.",
            contexts=[], ground_truth="Z equals W.",
            scores=Scores(faithfulness=0.5, relevance=0.5, precision=0.5,
                          context_relevance=0.5, recall=1.0),
        ),
    ]
    out = tmp_path / "out.jsonl"
    write_jsonl(out, records)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["question"] == "What is X?"
    assert first["scores"]["faithfulness"] == 0.9
    assert first["scores"]["recall"] is None
    second = json.loads(lines[1])
    assert second["scores"]["recall"] == 1.0
    assert second["ground_truth"] == "Z equals W."


def test_build_report_contains_all_dimension_sections(tmp_path: Path) -> None:
    records = [_scored("What is A?", 0.9), _scored("What is B?", 0.4)]
    report_path = tmp_path / "report.md"
    build_report(report_path, records, [])

    content = report_path.read_text(encoding="utf-8")
    for heading in ("Faithfulness", "Relevance", "Precision", "Context Relevance", "Recall"):
        assert heading in content, f"Missing section heading: {heading!r}"
    assert "Failure mode clusters" in content
    assert "No clusters generated" in content


def test_build_report_with_clusters_includes_cluster_info(tmp_path: Path) -> None:
    from evaluator.clustering import ClusterSummary

    cluster = ClusterSummary(
        label="hallucination-dominant",
        size=3,
        mean_scores={"faithfulness": 0.2, "relevance": 0.8, "precision": 0.9,
                     "context_relevance": 0.7, "recall": 0.9},
        centroid_example="The capital of France is Berlin.",
        worst_examples=["Paris is in Germany.", "Berlin is the capital of France."],
    )
    records = [_scored() for _ in range(3)]
    report_path = tmp_path / "report.md"
    build_report(report_path, records, [cluster])

    content = report_path.read_text(encoding="utf-8")
    assert "hallucination-dominant" in content
    assert "Size: 3" in content
    assert "Berlin" in content


def test_load_jsonl_invalid_json_raises_with_line_number(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text(
        json.dumps({"question": "Q?", "answer": "A.", "contexts": []}) + "\n"
        + "{ not valid json }\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Line 2"):
        load_jsonl(p)


def test_load_jsonl_invalid_json_on_first_line(tmp_path: Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text("{ broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Line 1"):
        load_jsonl(p)
