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
    return parser.parse_args()


def main() -> None:
    args = build_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    config_path = Path(args.config)

    config = load_config(config_path)
    run_evaluation(input_path, output_dir, config)


if __name__ == "__main__":
    main()
