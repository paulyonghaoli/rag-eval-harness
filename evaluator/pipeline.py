from __future__ import annotations

import dataclasses
import functools
import json
import os
import statistics
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from evaluator.clustering import cluster_failures, ClusterSummary
from evaluator.embeddings import Embedder
from evaluator.context_relevance import score_context_relevance
from evaluator.faithfulness import score_faithfulness_detailed
from evaluator.nli import NLIScorer
from evaluator.precision import score_context_precision
from evaluator.recall import score_context_recall
from evaluator.relevance import score_answer_relevance
from evaluator.types import EvalRecord, Scores, ScoredRecord


def _validate_record(raw: dict, line_num: int) -> EvalRecord:
    for field in ("question", "answer"):
        val = raw.get(field)
        if not isinstance(val, str):
            raise ValueError(
                f"Line {line_num}: '{field}' must be a non-null string, "
                f"got {type(val).__name__!r}"
            )
    contexts = raw.get("contexts")
    if not isinstance(contexts, list):
        raise ValueError(
            f"Line {line_num}: 'contexts' must be a list of strings, "
            f"got {type(contexts).__name__!r}"
        )
    for i, ctx in enumerate(contexts):
        if not isinstance(ctx, str):
            raise ValueError(
                f"Line {line_num}: 'contexts[{i}]' must be a string, "
                f"got {type(ctx).__name__!r}"
            )
    ground_truth = raw.get("ground_truth")
    if ground_truth is not None and not isinstance(ground_truth, str):
        raise ValueError(
            f"Line {line_num}: 'ground_truth' must be a string or null, "
            f"got {type(ground_truth).__name__!r}"
        )
    return EvalRecord(
        question=raw["question"],
        answer=raw["answer"],
        contexts=contexts,
        ground_truth=ground_truth,
    )


def load_jsonl(path: Path) -> List[EvalRecord]:
    records: List[EvalRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Line {line_num}: invalid JSON — {exc}") from exc
            records.append(_validate_record(raw, line_num))
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
    nli_scorer: NLIScorer | None,
    openai_api_key: Optional[str],
) -> ScoredRecord:
    faithfulness, claim_verdicts = score_faithfulness_detailed(
        record.answer, record.contexts, embedder, faith_threshold, use_llm, nli_scorer,
        openai_api_key=openai_api_key,
    )
    relevance = score_answer_relevance(
        record.question, record.answer, embedder,
        use_llm=use_llm, openai_api_key=openai_api_key,
    )
    precision = score_context_precision(
        record.answer, record.contexts, embedder, precision_threshold
    )
    ctx_relevance = score_context_relevance(record.question, record.contexts, embedder)
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
            context_relevance=round(ctx_relevance, 4),
            recall=None if recall is None else round(recall, 4),
            faithfulness_claims=claim_verdicts,
        ),
    )


_DIMS: Tuple[str, ...] = (
    "faithfulness", "relevance", "precision", "context_relevance", "recall"
)


def compute_summary(
    records: List[ScoredRecord],
    quality_gates: Dict[str, float],
) -> Dict[str, Any]:
    """Build a machine-readable summary dict suitable for writing to summary.json."""
    metrics: Dict[str, Any] = {}
    for dim in _DIMS:
        values = [v for r in records if (v := getattr(r.scores, dim)) is not None]
        if values:
            metrics[dim] = {
                "mean": round(statistics.mean(values), 4),
                "std": round(statistics.pstdev(values) if len(values) > 1 else 0.0, 4),
                "n": len(values),
            }
        else:
            metrics[dim] = None

    gate_results: Dict[str, Any] = {}
    passed: Optional[bool] = None
    if quality_gates:
        passed = True
        for metric, threshold in quality_gates.items():
            if metric not in _DIMS:
                warnings.warn(
                    f"Unknown quality gate metric {metric!r}. Valid metrics: {list(_DIMS)}",
                    UserWarning,
                    stacklevel=2,
                )
            m = metrics.get(metric)
            mean = m["mean"] if m is not None else 0.0
            gate_ok = mean >= threshold
            gate_results[metric] = {"threshold": threshold, "mean": mean, "passed": gate_ok}
            if not gate_ok:
                passed = False

    return {
        "records_evaluated": len(records),
        "metrics": metrics,
        "quality_gates": gate_results if quality_gates else None,
        "passed": passed,
    }


def build_report(
    output_path: Path,
    records: List[ScoredRecord],
    clusters: List[ClusterSummary],
) -> None:
    dims = _DIMS
    lines: List[str] = ["# RAG Evaluation Report", ""]
    for dim in dims:
        values = [v for r in records if (v := getattr(r.scores, dim)) is not None]
        lines += [f"## {dim.replace('_', ' ').title()}", f"- Mean ± std: {_mean_std(values)}", ""]

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


def load_baseline_means(path: Path) -> Dict[str, float]:
    """Read per-metric means from a previous scores.jsonl (candidate vs baseline comparison)."""
    values: Dict[str, List[float]] = {dim: [] for dim in _DIMS}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scores = json.loads(line).get("scores", {})
            for dim in _DIMS:
                v = scores.get(dim)
                if v is not None:
                    values[dim].append(float(v))
    return {dim: statistics.mean(vals) if vals else 0.0 for dim, vals in values.items()}


def run_evaluation(input_path: Path, output_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
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
    llm_cfg = config.get("llm_as_judge", {})
    use_llm = bool(llm_cfg.get("enabled", False))
    api_key_env = llm_cfg.get("openai_api_key_env", "OPENAI_API_KEY")
    openai_api_key: Optional[str] = os.environ.get(api_key_env) if use_llm else None
    faith_method = config.get("faithfulness_method", "cosine")
    nli_scorer: NLIScorer | None = None
    if faith_method == "nli":
        nli_scorer = NLIScorer(model_name=config.get("nli_model", "cross-encoder/nli-deberta-v3-small"))

    score_fn = functools.partial(
        _score_record,
        embedder=embedder,
        faith_threshold=faith_threshold,
        precision_threshold=precision_threshold,
        use_llm=use_llm,
        nli_scorer=nli_scorer,
        openai_api_key=openai_api_key,
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

    quality_gates: Dict[str, float] = {
        k: float(v) for k, v in config.get("quality_gates", {}).items()
    }
    summary = compute_summary(scored_records, quality_gates)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    _print_summary(scores_path, report_path, summary_path, summary)
    return summary


def _print_summary(
    scores_path: Path,
    report_path: Path,
    summary_path: Path,
    summary: Dict[str, Any],
) -> None:
    print("\n" + "=" * 50)
    print("RAG Evaluation Summary")
    print("=" * 50)
    print(f"Records evaluated : {summary['records_evaluated']}")
    print()
    for dim in _DIMS:
        m = summary["metrics"].get(dim)
        label = f"{m['mean']:.3f} ± {m['std']:.3f}" if m else "n/a"
        print(f"  {dim:<18} {label}")
    print()
    if summary.get("quality_gates"):
        print("Quality gates:")
        for metric, result in summary["quality_gates"].items():
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  {metric:<18} {result['mean']:.3f} >= {result['threshold']} [{status}]")
        print()
    print(f"Scores   -> {scores_path}")
    print(f"Report   -> {report_path}")
    print(f"Summary  -> {summary_path}")
    print("=" * 50 + "\n")
