from pathlib import Path

from pulse.generate_benchmark import build_records, load_config
from pulse.records import read_jsonl, write_jsonl


def test_generate_benchmark_records(tmp_path: Path) -> None:
    config = load_config(Path("configs/ambiguity_benchmark.yaml"))
    records = build_records(config, examples_per_pair=1)
    assert records
    assert {"prompt_text", "expected_tool_call", "clarification_question"} <= set(records[0])
    output = tmp_path / "tasks.jsonl"
    write_jsonl(output, records)
    assert len(read_jsonl(output)) == len(config["confusable_entities"])
