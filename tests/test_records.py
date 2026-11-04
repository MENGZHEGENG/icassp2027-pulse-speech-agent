from pathlib import Path

from pulse.records import read_jsonl, write_jsonl


def test_write_jsonl_round_trips_and_cleans_unique_temp(tmp_path: Path) -> None:
    output = tmp_path / "records.jsonl"
    rows = [{"id": "a", "text": "hello"}, {"id": "b", "text": "世界"}]

    write_jsonl(output, rows)

    assert read_jsonl(output) == rows
    assert list(tmp_path.glob(".records.jsonl.*.tmp")) == []


def test_write_jsonl_replaces_existing_file_atomically(tmp_path: Path) -> None:
    output = tmp_path / "records.jsonl"
    write_jsonl(output, [{"id": "old"}])

    write_jsonl(output, [{"id": "new"}])

    assert read_jsonl(output) == [{"id": "new"}]
