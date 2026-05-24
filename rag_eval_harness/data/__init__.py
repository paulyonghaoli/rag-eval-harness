from importlib.resources import files
from pathlib import Path


def sample_eval_set() -> Path:
    """Return the path to the bundled sample evaluation set (20 records)."""
    return Path(str(files("rag_eval_harness.data").joinpath("sample_eval_set.jsonl")))


def adversarial_eval_set() -> Path:
    """Return the path to the bundled adversarial evaluation set (17 records, 9 failure modes)."""
    return Path(str(files("rag_eval_harness.data").joinpath("adversarial_eval_set.jsonl")))
