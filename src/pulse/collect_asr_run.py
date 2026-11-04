from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pulse.records import read_jsonl
from pulse.summarize_asr import summarize_records

SHARD_RE = re.compile(r"\.shard-(?P<index>\d+)-of-(?P<total>\d+)\.jsonl$")


def parse_shard_name(path: str | Path) -> tuple[int, int] | None:
    match = SHARD_RE.search(str(path))
    if match is None:
        return None
    return int(match.group("index")), int(match.group("total"))


def expand_patterns(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            candidate = Path(pattern)
            if candidate.exists():
                paths.append(candidate)
    return sorted(set(paths), key=lambda path: str(path))


def select_shard_set(
    paths: list[Path],
    *,
    expected_shards: int | None = None,
    preferred_total: int | None = None,
) -> tuple[list[Path], list[Path], int | None]:
    selected_total = expected_shards or preferred_total
    if selected_total is None:
        totals = []
        for path in paths:
            parsed = parse_shard_name(path)
            if parsed is not None:
                _, total = parsed
                totals.append(total)
        selected_total = max(totals) if totals else None
    if selected_total is None:
        return paths, [], None

    matching = [path for path in paths if (parsed := parse_shard_name(path)) is not None and parsed[1] == selected_total]

    def canonical_rank(path: Path) -> tuple[int, int, int, str]:
        path_text = path.as_posix()
        return (
            1 if "__home__" in path_text else 0,
            1 if "snapshots__" in path_text else 0,
            len(path_text),
            path_text,
        )

    selected_by_index: dict[int, Path] = {}
    ignored: list[Path] = []
    for path in matching:
        parsed = parse_shard_name(path)
        if parsed is None:
            ignored.append(path)
            continue
        shard_index, _ = parsed
        current = selected_by_index.get(shard_index)
        if current is None:
            selected_by_index[shard_index] = path
            continue
        if canonical_rank(path) < canonical_rank(current):
            ignored.append(current)
            selected_by_index[shard_index] = path
        else:
            ignored.append(path)

    selected = sorted(selected_by_index.values(), key=lambda path: str(path))
    selected_set = set(selected)
    ignored.extend(path for path in paths if path not in selected_set and path not in ignored)
    ignored = sorted(set(ignored), key=lambda path: str(path))
    return selected, ignored, selected_total


def record_key(record: dict[str, Any]) -> str | None:
    value = record.get("id", record.get("sample_index"))
    return str(value) if value is not None else None


def audit_shards(
    paths: list[Path],
    records: list[dict[str, Any]],
    *,
    expected_shards: int | None = None,
    expected_count: int | None = None,
) -> dict[str, Any]:
    shard_pairs = [parse_shard_name(path) for path in paths]
    parsed_pairs = [pair for pair in shard_pairs if pair is not None]
    totals = sorted({total for _, total in parsed_pairs})
    present_indices = sorted({index for index, _ in parsed_pairs})
    inferred_total = totals[0] if len(totals) == 1 else None
    required_total = expected_shards or inferred_total
    missing_indices = []
    if required_total is not None:
        missing_indices = [index for index in range(required_total) if index not in present_indices]

    keys = [key for record in records if (key := record_key(record)) is not None]
    key_counts = Counter(keys)
    duplicate_keys = sorted(key for key, count in key_counts.items() if count > 1)
    count_matches = expected_count is None or len(records) == expected_count
    complete = bool(paths) and not missing_indices and not duplicate_keys and count_matches and len(totals) <= 1

    return {
        "files": [str(path) for path in paths],
        "file_count": len(paths),
        "record_count": len(records),
        "expected_count": expected_count,
        "count_matches": count_matches,
        "shard_totals": totals,
        "expected_shards": required_total,
        "present_shards": present_indices,
        "missing_shards": missing_indices,
        "unparseable_shard_files": [str(path) for path, pair in zip(paths, shard_pairs) if pair is None],
        "duplicate_record_keys": duplicate_keys[:50],
        "duplicate_record_key_count": len(duplicate_keys),
        "complete": complete,
    }


def collect_run(
    paths: list[Path],
    *,
    expected_shards: int | None = None,
    expected_count: int | None = None,
    entity_terms: list[str] | None = None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(read_jsonl(path))
    return {
        "audit": audit_shards(paths, records, expected_shards=expected_shards, expected_count=expected_count),
        "summary": summarize_records(records, entity_terms),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect, audit, and summarize ASR shard outputs.")
    parser.add_argument("patterns", nargs="+", help="Shard files or glob patterns to collect.")
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--entity-term", action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("results/asr/run_collection.json"))
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--write-incomplete", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = expand_patterns(args.patterns)
    payload = collect_run(
        paths,
        expected_shards=args.expected_shards,
        expected_count=args.expected_count,
        entity_terms=args.entity_term,
    )
    print(json.dumps(payload, indent=2))
    if args.require_complete and not payload["audit"]["complete"]:
        if args.write_incomplete:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise SystemExit(1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
