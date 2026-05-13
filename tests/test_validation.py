from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluator.pipeline import load_jsonl


def _write_jsonl(tmp_path: Path, records: list) -> Path:
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    return p


def test_valid_record_loads(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [
        {"question": "What is ML?", "answer": "Machine learning.", "contexts": ["ML is..."], "ground_truth": "ML."}
    ])
    records = load_jsonl(p)
    assert len(records) == 1
    assert records[0].question == "What is ML?"
    assert records[0].contexts == ["ML is..."]


def test_missing_question_raises(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"answer": "A.", "contexts": ["ctx"]}])
    with pytest.raises(ValueError, match="question"):
        load_jsonl(p)


def test_missing_answer_raises(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"question": "Q?", "contexts": ["ctx"]}])
    with pytest.raises(ValueError, match="answer"):
        load_jsonl(p)


def test_contexts_as_plain_string_raises(tmp_path: Path) -> None:
    # A bare string would silently iterate to individual characters — must be rejected.
    p = _write_jsonl(tmp_path, [{"question": "Q?", "answer": "A.", "contexts": "some string"}])
    with pytest.raises(ValueError, match="contexts"):
        load_jsonl(p)


def test_contexts_with_non_string_element_raises(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"question": "Q?", "answer": "A.", "contexts": [1, 2, 3]}])
    with pytest.raises(ValueError, match=r"contexts\[0\]"):
        load_jsonl(p)


def test_ground_truth_null_accepted(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"question": "Q?", "answer": "A.", "contexts": [], "ground_truth": None}])
    records = load_jsonl(p)
    assert records[0].ground_truth is None


def test_ground_truth_absent_accepted(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"question": "Q?", "answer": "A.", "contexts": []}])
    records = load_jsonl(p)
    assert records[0].ground_truth is None


def test_ground_truth_non_string_raises(tmp_path: Path) -> None:
    p = _write_jsonl(tmp_path, [{"question": "Q?", "answer": "A.", "contexts": [], "ground_truth": 42}])
    with pytest.raises(ValueError, match="ground_truth"):
        load_jsonl(p)


def test_error_includes_line_number(tmp_path: Path) -> None:
    lines = [
        json.dumps({"question": "Q?", "answer": "A.", "contexts": []}),
        json.dumps({"question": 42, "answer": "A.", "contexts": []}),
    ]
    p = tmp_path / "test.jsonl"
    p.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="Line 2"):
        load_jsonl(p)


def test_empty_lines_skipped(tmp_path: Path) -> None:
    p = tmp_path / "test.jsonl"
    p.write_text(
        json.dumps({"question": "Q?", "answer": "A.", "contexts": []}) + "\n\n",
        encoding="utf-8",
    )
    records = load_jsonl(p)
    assert len(records) == 1
