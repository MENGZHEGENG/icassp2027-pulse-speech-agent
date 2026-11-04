import json

from pulse.collect_asr_run import audit_shards, collect_run, expand_patterns, parse_shard_name, select_shard_set


def write_record(path, record_id: str, sample_index: int = 0) -> None:
    record = {
        "id": record_id,
        "sample_index": sample_index,
        "reference": "hello world",
        "hypotheses": [{"text": "hello world", "score": -0.1}],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_parse_shard_name_extracts_index_and_total(tmp_path) -> None:
    path = tmp_path / "run.shard-3-of-12.jsonl"
    assert parse_shard_name(path) == (3, 12)


def test_collect_run_marks_complete_shards(tmp_path) -> None:
    first = tmp_path / "run.shard-0-of-2.jsonl"
    second = tmp_path / "run.shard-1-of-2.jsonl"
    write_record(first, "a", 0)
    write_record(second, "b", 1)

    payload = collect_run([first, second], expected_count=2)

    assert payload["audit"]["complete"] is True
    assert payload["audit"]["missing_shards"] == []
    assert payload["summary"]["count"] == 2
    assert payload["summary"]["mean_wer"] == 0.0


def test_audit_shards_reports_missing_and_duplicate_records(tmp_path) -> None:
    first = tmp_path / "run.shard-0-of-3.jsonl"
    second = tmp_path / "run.shard-2-of-3.jsonl"
    write_record(first, "dup", 0)
    write_record(second, "dup", 2)

    records = []
    for path in [first, second]:
        records.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    audit = audit_shards([first, second], records)

    assert audit["complete"] is False
    assert audit["missing_shards"] == [1]
    assert audit["duplicate_record_keys"] == ["dup"]


def test_expand_patterns_sorts_matches(tmp_path) -> None:
    second = tmp_path / "run.shard-1-of-2.jsonl"
    first = tmp_path / "run.shard-0-of-2.jsonl"
    write_record(second, "b", 1)
    write_record(first, "a", 0)

    assert expand_patterns([str(tmp_path / "*.jsonl")]) == [first, second]


def test_select_shard_set_prefers_expected_total_over_stale_matches(tmp_path) -> None:
    stale0 = tmp_path / "run.shard-0-of-2.jsonl"
    stale1 = tmp_path / "run.shard-1-of-2.jsonl"
    current0 = tmp_path / "run.shard-0-of-16.jsonl"

    selected, ignored, selected_total = select_shard_set([stale0, stale1, current0], expected_shards=16)

    assert selected == [current0]
    assert ignored == [stale0, stale1]
    assert selected_total == 16


def test_select_shard_set_deduplicates_snapshot_prefixed_duplicates(tmp_path) -> None:
    snapshot0 = tmp_path / "__home__cache__snapshots__abc.local__dataset.run.shard-0-of-64.jsonl"
    snapshot1 = tmp_path / "__home__cache__snapshots__abc.local__dataset.run.shard-1-of-64.jsonl"
    canonical0 = tmp_path / "openai__whisper-medium.local__dataset.run.shard-0-of-64.jsonl"
    canonical1 = tmp_path / "openai__whisper-medium.local__dataset.run.shard-1-of-64.jsonl"

    selected, ignored, selected_total = select_shard_set(
        [snapshot0, canonical0, snapshot1, canonical1],
        expected_shards=64,
    )

    assert selected == [canonical0, canonical1]
    assert ignored == [snapshot0, snapshot1]
    assert selected_total == 64
