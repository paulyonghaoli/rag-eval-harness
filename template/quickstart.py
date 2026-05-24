"""
Minimal Python API example for rag-eval-harness.
Run: python quickstart.py
"""
from pathlib import Path
from rag_eval_harness import run_evaluation
from rag_eval_harness.data import sample_eval_set

config = {
    "embedder_model": "all-MiniLM-L6-v2",
    "faithfulness_method": "cosine",        # "nli" for better accuracy (offline)
    "similarity_thresholds": {"faithfulness": 0.7, "precision": 0.5},
    "clustering": {"min_k": 3, "max_k": 5},
    "llm_as_judge": {"enabled": False},
}

summary = run_evaluation(
    input_path=sample_eval_set(),           # swap with Path("your_eval_set.jsonl")
    output_dir=Path("results/"),
    config=config,
)

print(f"Faithfulness : {summary['metrics']['faithfulness']['mean']:.3f}")
print(f"Relevance    : {summary['metrics']['relevance']['mean']:.3f}")
print(f"Precision    : {summary['metrics']['precision']['mean']:.3f}")
