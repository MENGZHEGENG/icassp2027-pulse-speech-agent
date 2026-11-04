import json
import os
import subprocess
from pathlib import Path

from pulse.merge_jsonl import expand_inputs, merge_jsonl, sort_key


def write(path, records) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_expand_inputs_globs_and_sorts(tmp_path) -> None:
    second = tmp_path / "b.jsonl"
    first = tmp_path / "a.jsonl"
    write(second, [])
    write(first, [])
    assert expand_inputs([str(tmp_path / "*.jsonl")]) == [first, second]


def test_merge_jsonl_sorts_by_numeric_sample_index(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write(first, [{"sample_index": 10}, {"sample_index": 2}])
    write(second, [{"sample_index": 1}])
    assert [record["sample_index"] for record in merge_jsonl([str(first), str(second)])] == [1, 2, 10]


def test_merge_jsonl_cli_requires_inputs(tmp_path) -> None:
    output = tmp_path / "out.jsonl"
    result = subprocess.run(
        [
            "python3",
            "-m",
            "pulse.merge_jsonl",
            str(tmp_path / "missing-*.jsonl"),
            "--output",
            str(output),
            "--require-inputs",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": "src"},
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "no input files matched" in result.stderr


def test_sort_key_handles_missing_and_text() -> None:
    assert sort_key({"id": "2"}, "id") < sort_key({"id": "10"}, "id")
    assert sort_key({}, "id") > sort_key({"id": "x"}, "id")
