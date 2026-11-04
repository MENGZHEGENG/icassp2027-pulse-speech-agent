from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from pulse.records import read_jsonl
from pulse.uncertainty import normalize_text


SUMMARY_METRICS = [
    "count",
    "mean_risk",
    "median_risk",
    "risk_ge_0_75_rate",
    "risk_eq_1_rate",
    "predicted_true_entity_match_rate",
]


def evidence_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = record.get("phone_evidence")
    return payload if isinstance(payload, dict) else record


def phone_risk(record: dict[str, Any]) -> float:
    if record.get("learned_phonetic_risk") is not None:
        return float(record["learned_phonetic_risk"])
    payload = evidence_payload(record)
    if payload.get("learned_phonetic_risk") is not None:
        return float(payload["learned_phonetic_risk"])
    if payload.get("phone_risk") is not None:
        return float(payload["phone_risk"])
    raise ValueError(f"phone evidence record {record.get('id')} has no risk")


def predicted_matches_true(record: dict[str, Any]) -> bool | None:
    payload = evidence_payload(record)
    predicted = normalize_text(str(payload.get("predicted_entity") or ""))
    true = normalize_text(str(payload.get("true_entity") or record.get("true_entity") or ""))
    if not predicted or not true:
        return None
    return predicted == true


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    risks = [phone_risk(record) for record in records]
    matches = [match for record in records if (match := predicted_matches_true(record)) is not None]
    predicted_counts = Counter(str(evidence_payload(record).get("predicted_entity")) for record in records)
    reason_counts = Counter(str(evidence_payload(record).get("risk_reason")) for record in records)
    return {
        "count": len(records),
        "mean_risk": mean(risks) if risks else None,
        "median_risk": median(risks) if risks else None,
        "risk_ge_0_75_rate": sum(risk >= 0.75 for risk in risks) / len(risks) if risks else None,
        "risk_eq_1_rate": sum(risk == 1.0 for risk in risks) / len(risks) if risks else None,
        "predicted_true_entity_match_rate": sum(matches) / len(matches) if matches else None,
        "predicted_entity_counts": dict(predicted_counts.most_common()),
        "risk_reason_counts": dict(reason_counts.most_common()),
    }


def add_run_metadata(
    summary: dict[str, Any],
    *,
    condition: str | None = None,
    source: str | None = None,
    shards: int | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if condition:
        metadata["condition"] = condition
    if source:
        metadata["source"] = source
    if shards is not None:
        metadata["shards"] = shards
    if metadata:
        summary = dict(summary)
        summary["run_metadata"] = metadata
    return summary


def format_value(value: Any) -> str:
    return "--" if value is None else str(value)


def top_count_lines(title: str, counts: dict[str, Any], *, limit: int = 5) -> list[str]:
    if not counts:
        return []
    lines = ["", f"## {title}", ""]
    for name, count in list(counts.items())[:limit]:
        lines.append(f"- `{name}`: `{count}`")
    return lines


def markdown_summary(summary: dict[str, Any], *, title: str = "Phone Evidence Summary") -> str:
    metadata = dict(summary.get("run_metadata") or {})
    lines = [
        f"# {title}",
        "",
        f"Records: `{format_value(summary.get('count'))}`.",
        f"Mean risk: `{format_value(summary.get('mean_risk'))}`. Median risk: `{format_value(summary.get('median_risk'))}`.",
        f"Risk >= 0.75: `{format_value(summary.get('risk_ge_0_75_rate'))}`. Risk == 1.0: `{format_value(summary.get('risk_eq_1_rate'))}`.",
        f"Predicted/true entity match: `{format_value(summary.get('predicted_true_entity_match_rate'))}`.",
    ]
    lines.extend(top_count_lines("Top Risk Reasons", dict(summary.get("risk_reason_counts") or {})))
    lines.extend(top_count_lines("Top Predicted Entities", dict(summary.get("predicted_entity_counts") or {})))
    if metadata:
        parts = []
        if metadata.get("source"):
            parts.append(f"source `{metadata['source']}`")
        if metadata.get("shards") is not None:
            parts.append(f"`{metadata['shards']}` shards")
        if metadata.get("condition"):
            parts.append(f"condition `{metadata['condition']}`")
        lines.extend(["", "Generated from " + "; ".join(parts) + "."])
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize phone evidence risk JSONL files.")
    parser.add_argument("phone_evidence", type=Path)
    parser.add_argument("--title", default="Phone Evidence Summary")
    parser.add_argument("--condition")
    parser.add_argument("--source")
    parser.add_argument("--shards", type=int)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = add_run_metadata(
        summarize_records(read_jsonl(args.phone_evidence)),
        condition=args.condition,
        source=args.source,
        shards=args.shards,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown_summary(summary, title=args.title), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
