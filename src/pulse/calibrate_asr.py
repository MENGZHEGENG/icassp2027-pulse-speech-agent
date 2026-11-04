from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pulse.records import read_jsonl
from pulse.policy import AgentAction, PulsePolicy
from pulse.record_summary import summarize_record
from pulse.uncertainty import token_distance_ratio


def bin_index(value: float, bins: int) -> int:
    return min(bins - 1, max(0, int(value * bins)))


def calibrate_records(records: list[dict], bins: int = 10) -> list[dict]:
    policy = PulsePolicy()
    buckets = [
        {"bin": idx, "count": 0, "mean_risk": 0.0, "mean_error": 0.0, "clarify_rate": 0.0}
        for idx in range(bins)
    ]
    for record in records:
        reference = record.get("reference")
        if not reference:
            continue
        summary = summarize_record(record)
        decision = policy.decide(summary, AgentAction.ANSWER)
        error = min(1.0, token_distance_ratio(reference, summary.top_text))
        risk = min(1.0, decision.risk_score)
        bucket = buckets[bin_index(risk, bins)]
        bucket["count"] += 1
        bucket["mean_risk"] += risk
        bucket["mean_error"] += error
        bucket["clarify_rate"] += float(decision.action == AgentAction.CLARIFY)
    for bucket in buckets:
        if bucket["count"]:
            bucket["mean_risk"] /= bucket["count"]
            bucket["mean_error"] /= bucket["count"]
            bucket["clarify_rate"] /= bucket["count"]
    return buckets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bin ASR uncertainty risk against transcript error.")
    parser.add_argument("records_files", nargs="+", type=Path)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--output-csv", type=Path, default=Path("results/asr_calibration.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("results/asr_calibration.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = []
    for records_path in args.records_files:
        records.extend(read_jsonl(records_path))
    buckets = calibrate_records(records, args.bins)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["bin", "count", "mean_risk", "mean_error", "clarify_rate"])
        writer.writeheader()
        writer.writerows(buckets)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(buckets, indent=2), encoding="utf-8")
    print(json.dumps({"records": len(records), "bins": len(buckets), "csv": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
