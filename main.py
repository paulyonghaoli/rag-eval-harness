import argparse
from pathlib import Path

import yaml

from evaluator.pipeline import run_evaluation


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

    run_evaluation(input_path, output_dir, config)


if __name__ == "__main__":
    main()
