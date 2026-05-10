import json
from pathlib import Path

from evaluator.pipeline import run_evaluation


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
