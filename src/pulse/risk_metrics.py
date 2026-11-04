from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pulse.evaluate_policy_records import requested_agent_action
from pulse.records import read_jsonl
from pulse.policy import AgentAction, PulsePolicy
from pulse.record_summary import summarize_record
from pulse.uncertainty import token_distance_ratio


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2.0
        for original_index, _ in indexed[cursor:end]:
            ranks[original_index] = average_rank
        cursor = end
    return ranks


def roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(scores: list[float], labels: list[bool]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hits = 0
    precision_sum = 0.0
    for index, (_, label) in enumerate(ranked, start=1):
        if label:
            hits += 1
            precision_sum += hits / index
    return precision_sum / positives


def expected_calibration_error(rows: list[dict[str, Any]], bins: int = 10) -> float | None:
    if not rows:
        return None
    total = len(rows)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        bucket = [row for row in rows if lower <= row["risk"] < upper or (bin_index == bins - 1 and row["risk"] == 1.0)]
        if not bucket:
            continue
        mean_risk = sum(row["risk"] for row in bucket) / len(bucket)
        error_rate = sum(float(row["has_error"]) for row in bucket) / len(bucket)
        ece += len(bucket) / total * abs(mean_risk - error_rate)
    return ece


def risk_rows(
    records: list[dict[str, Any]],
    *,
    error_threshold: float = 0.0,
    policy: PulsePolicy | None = None,
    action_mode: str = "task",
    score_field: str = "decision_risk",
) -> list[dict[str, Any]]:
    policy = policy or PulsePolicy()
    rows: list[dict[str, Any]] = []
    for record in records:
        reference = record.get("reference")
        if not reference:
            continue
        summary = summarize_record(record)
        requested_action = AgentAction.ANSWER if action_mode == "answer" else requested_agent_action(str(record.get("action", "answer")))
        decision = policy.decide(summary, requested_action)
        feature_scores = {
            "decision_risk": min(1.0, decision.risk_score),
            "instability": summary.instability,
            "phonetic_risk": summary.phonetic_risk,
            "entity_risk": summary.entity_risk,
            "entropy": summary.entropy or 0.0,
        }
        error = min(1.0, token_distance_ratio(str(reference), summary.top_text))
        rows.append(
            {
                "id": record.get("id"),
                "risk": min(1.0, float(feature_scores[score_field])),
                "score_field": score_field,
                "error": error,
                "has_error": error > error_threshold,
                "decision": decision.action.value,
                "reference": reference,
                "top_text": summary.top_text,
            }
        )
    return rows


def summarize_risk_rows(rows: list[dict[str, Any]], *, bins: int = 10) -> dict[str, Any]:
    scores = [float(row["risk"]) for row in rows]
    labels = [bool(row["has_error"]) for row in rows]
    return {
        "count": len(rows),
        "error_rate": sum(labels) / len(labels) if labels else None,
        "mean_risk": sum(scores) / len(scores) if scores else None,
        "auroc": roc_auc(scores, labels),
        "average_precision": average_precision(scores, labels),
        "ece": expected_calibration_error(rows, bins=bins),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute risk-vs-error metrics for ASR/PULSE outputs.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--error-threshold", type=float, default=0.0)
    parser.add_argument("--action-mode", choices=["task", "answer"], default="task")
    parser.add_argument(
        "--score-field",
        choices=["decision_risk", "instability", "phonetic_risk", "entity_risk", "entropy"],
        default="decision_risk",
    )
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=Path("results/risk_metrics.json"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/risk_metrics_rows.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = risk_rows(
        read_jsonl(args.records),
        error_threshold=args.error_threshold,
        action_mode=args.action_mode,
        score_field=args.score_field,
    )
    payload = {"summary": summarize_risk_rows(rows, bins=args.bins), "rows": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(args.output_csv, rows)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
