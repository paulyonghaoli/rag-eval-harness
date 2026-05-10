from __future__ import annotations

import dataclasses
import functools
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List

from tqdm import tqdm

from evaluator.clustering import cluster_failures, ClusterSummary
from evaluator.embeddings import Embedder
from evaluator.faithfulness import score_faithfulness_detailed
from evaluator.precision import score_context_precision
from evaluator.recall import score_context_recall
from evaluator.relevance import score_answer_relevance
from evaluator.types import EvalRecord, Scores, ScoredRecord


def load_jsonl(path: Path) -> List[EvalRecord]:
    records: List[EvalRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            records.append(EvalRecord(
                question=str(raw.get("question", "")),
                answer=str(raw.get("answer", "")),
                contexts=list(raw.get("contexts", [])),
                ground_truth=raw.get("ground_truth"),
            ))
    return records


def write_jsonl(path: Path, records: Iterable[ScoredRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(dataclasses.asdict(record), ensure_ascii=False) + "\n")


def _mean_std(values: List[float]) -> str:
    if not values:
        return "n/a"
    mean = statistics.mean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.3f} ± {std:.3f}"


def _score_record(
    record: EvalRecord,
    *,
    embedder: Embedder,
    faith_threshold: float,
    precision_threshold: float,
    use_llm: bool,
) -> ScoredRecord:
    faithfulness, claim_verdicts = score_faithfulness_detailed(
        record.answer, record.contexts, embedder, faith_threshold, use_llm
    )
    relevance = score_answer_relevance(record.question, record.answer, embedder, use_llm=use_llm)
    precision = score_context_precision(
        record.answer, record.contexts, embedder, precision_threshold
    )
    recall: float | None = None
    if record.ground_truth is not None:
        recall = score_context_recall(
            record.ground_truth, record.contexts, embedder, faith_threshold
        )
    return ScoredRecord(
        question=record.question,
        answer=record.answer,
        contexts=record.contexts,
        ground_truth=record.ground_truth,
        scores=Scores(
            faithfulness=round(faithfulness, 4),
            relevance=round(relevance, 4),
            precision=round(precision, 4),
            recall=None if recall is None else round(recall, 4),
            faithfulness_claims=claim_verdicts,
        ),
    )


def build_report(
    output_path: Path,
    records: List[ScoredRecord],
    clusters: List[ClusterSummary],
) -> None:
    dims = ("faithfulness", "relevance", "precision", "recall")
    lines: List[str] = ["# RAG Evaluation Report", ""]
    for dim in dims:
        values = [v for r in records if (v := getattr(r.scores, dim)) is not None]
        lines += [f"## {dim.title()}", f"- Mean ± std: {_mean_std(values)}", ""]

    lines.append("## Failure mode clusters")
    if not clusters:
        lines.append("No clusters generated (evaluation set too small).")
    else:
        for cluster in clusters:
            lines += [
                f"### {cluster.label}",
                f"- Size: {cluster.size}",
                "- Mean scores:",
                *[f"  - {d}: {v:.3f}" for d, v in cluster.mean_scores.items()],
                f"- Centroid example: {cluster.centroid_example}",
                "- Worst examples:",
                *[f"  - {ex}" for ex in cluster.worst_examples],
                "",
            ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_evaluation(input_path: Path, output_dir: Path, config: Dict[str, Any]) -> None:
    records = load_jsonl(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(model_name=config.get("embedder_model", "all-MiniLM-L6-v2"))
    thresholds = config.get("similarity_thresholds", {})
    faith_threshold = float(thresholds.get("faithfulness", 0.7))
    precision_threshold = float(thresholds.get("precision", 0.5))
    min_k = int(config.get("clustering", {}).get("min_k", 3))
    max_k = int(config.get("clustering", {}).get("max_k", 5))
    use_llm = bool(config.get("llm_as_judge", {}).get("enabled", False))

    score_fn = functools.partial(
        _score_record,
        embedder=embedder,
        faith_threshold=faith_threshold,
        precision_threshold=precision_threshold,
        use_llm=use_llm,
    )

    # ThreadPoolExecutor parallelises the per-record numpy scoring work.
    # The Embedder cache + lock ensures each unique text is encoded exactly once,
    # with subsequent calls served from memory across all threads.
    with ThreadPoolExecutor() as executor:
        scored_records: List[ScoredRecord] = list(
            tqdm(executor.map(score_fn, records), total=len(records), desc="Scoring", unit="rec")
        )

    scores_path = output_dir / "scores.jsonl"
    write_jsonl(scores_path, scored_records)

    clusters = cluster_failures(scored_records, embedder, min_k, max_k)
    report_path = output_dir / "report.md"
    build_report(report_path, scored_records, clusters)
    _print_summary(scored_records, scores_path, report_path)


def _print_summary(
    records: List[ScoredRecord],
    scores_path: Path,
    report_path: Path,
) -> None:
    dims = ("faithfulness", "relevance", "precision", "recall")
    print("\n" + "=" * 50)
    print("RAG Evaluation Summary")
    print("=" * 50)
    print(f"Records evaluated : {len(records)}")
    print()
    for dim in dims:
        values = [v for r in records if (v := getattr(r.scores, dim)) is not None]
        label = _mean_std(values) if values else "n/a"
        print(f"  {dim:<18} {label}")
    print()
    print(f"Scores  -> {scores_path}")
    print(f"Report  -> {report_path}")
    print("=" * 50 + "\n")
