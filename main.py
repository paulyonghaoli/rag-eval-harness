import argparse
import sys
from pathlib import Path

import yaml

from evaluator.pipeline import load_baseline_means, run_evaluation


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the RAG evaluation harness on a JSONL evaluation set."
    )
    parser.add_argument("--input", required=True, help="Path to input JSONL evaluation file.")
    parser.add_argument("--output", required=True, help="Directory for scores and report output.")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        default=None,
        help="Enable LLM-as-judge for answer relevance and faithfulness (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--faithfulness-method",
        choices=["cosine", "nli"],
        default=None,
        help="Faithfulness scoring method: 'cosine' (default) or 'nli' (CrossEncoder).",
    )
    parser.add_argument(
        "--baseline-scores",
        default=None,
        metavar="PATH",
        help="Path to a previous scores.jsonl for regression comparison.",
    )
    parser.add_argument(
        "--min-delta-faithfulness",
        type=float,
        default=None,
        metavar="DELTA",
        help="Fail if faithfulness mean drops below baseline by more than DELTA (e.g. -0.02).",
    )
    return parser.parse_args()


def main() -> None:
    args = build_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    config_path = Path(args.config)

    config = load_config(config_path)

    # CLI flags override config file when explicitly provided.
    if args.llm_judge is not None:
        config.setdefault("llm_as_judge", {})["enabled"] = args.llm_judge
    if args.faithfulness_method is not None:
        config["faithfulness_method"] = args.faithfulness_method

    summary = run_evaluation(input_path, output_dir, config)

    if summary.get("passed") is False:
        sys.exit(1)

    if args.baseline_scores and args.min_delta_faithfulness is None:
        print(
            "Warning: --baseline-scores has no effect without --min-delta-faithfulness",
            file=sys.stderr,
        )

    if args.baseline_scores and args.min_delta_faithfulness is not None:
        baseline = load_baseline_means(Path(args.baseline_scores))
        current = (summary["metrics"].get("faithfulness") or {}).get("mean", 0.0)
        base = baseline.get("faithfulness", 0.0)
        delta = current - base
        if delta < args.min_delta_faithfulness:
            print(
                f"REGRESSION: faithfulness {base:.4f} -> {current:.4f} "
                f"(delta {delta:+.4f}, allowed >= {args.min_delta_faithfulness:+.4f})"
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
