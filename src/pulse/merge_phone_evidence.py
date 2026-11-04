from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pulse.records import read_jsonl, write_jsonl
from pulse.record_summary import parse_external_phone_risk


def evidence_id(record: dict[str, Any]) -> str:
    value = record.get("id") or record.get("record_id") or record.get("utterance_id")
    if value is None:
        raise ValueError("phone evidence rows must include id, record_id, or utterance_id")
    return str(value)


def normalize_evidence(record: dict[str, Any]) -> dict[str, Any]:
    risk = parse_external_phone_risk(record)
    if risk is None:
        raise ValueError(f"phone evidence for {evidence_id(record)} does not include a risk field")
    payload = dict(record.get("phone_evidence") or {})
    payload.update({key: value for key, value in record.items() if key not in {"id", "record_id", "utterance_id"}})
    # Ground-truth labels belong to evaluation records, never to test-time
    # phone evidence consumed by the policy.
    payload.pop("true_entity", None)
    payload["learned_phonetic_risk"] = risk
    return {"learned_phonetic_risk": risk, "phone_evidence": payload}


def index_evidence(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        key = evidence_id(record)
        if key in indexed:
            raise ValueError(f"duplicate phone evidence id: {key}")
        indexed[key] = normalize_evidence(record)
    return indexed


def merge_phone_evidence(
    records: list[dict[str, Any]],
    evidence_records: list[dict[str, Any]],
    *,
    require_all: bool = False,
    allow_extra: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    evidence_by_id = index_evidence(evidence_records)
    records_ids = {str(record.get("id")) for record in records}
    extra_ids = sorted(set(evidence_by_id) - records_ids)
    if extra_ids and not allow_extra:
        raise ValueError(f"phone evidence contains ids absent from records: {extra_ids[:5]}")

    merged = []
    missing = 0
    matched = 0
    for record in records:
        key = str(record.get("id"))
        evidence = evidence_by_id.get(key)
        if evidence is None:
            missing += 1
            if require_all:
                raise ValueError(f"missing phone evidence for records id: {key}")
            merged.append(dict(record))
            continue
        output = dict(record)
        output.update(evidence)
        merged.append(output)
        matched += 1
    return merged, {"records": len(records), "matched": matched, "missing": missing, "extra": len(extra_ids)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge learned/forced-alignment phone risk evidence into a PULSE records.")
    parser.add_argument("records", type=Path)
    parser.add_argument("phone_evidence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-all", action="store_true")
    parser.add_argument("--allow-extra", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    merged, summary = merge_phone_evidence(
        read_jsonl(args.records),
        read_jsonl(args.phone_evidence),
        require_all=args.require_all,
        allow_extra=args.allow_extra,
    )
    write_jsonl(args.output, merged)
    print(json.dumps({**summary, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
