from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

from pulse.records import write_jsonl


def expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    return sorted(set(paths), key=lambda path: str(path))


def read_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
    return records


def sort_key(record: dict[str, Any], field: str) -> tuple[int, Any]:
    value = record.get(field)
    if value is None:
        return (1, "")
    if isinstance(value, int | float):
        return (0, value)
    text = str(value)
    return (0, int(text)) if text.isdigit() else (0, text)


def merge_jsonl(patterns: list[str], *, sort_field: str | None = "sample_index") -> list[dict[str, Any]]:
    records = read_records(expand_inputs(patterns))
    if sort_field:
        records.sort(key=lambda record: sort_key(record, sort_field))
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge JSONL files or glob patterns into one JSONL records.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sort-field", default="sample_index")
    parser.add_argument("--no-sort", action="store_true")
    parser.add_argument("--require-inputs", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = expand_inputs(args.inputs)
    if args.require_inputs and not paths:
        raise SystemExit("no input files matched")
    records = read_records(paths)
    if not args.no_sort and args.sort_field:
        records.sort(key=lambda record: sort_key(record, args.sort_field))
    write_jsonl(args.output, records)
    print(json.dumps({"output": str(args.output), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
