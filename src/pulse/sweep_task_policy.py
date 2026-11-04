from __future__ import annotations

import argparse
import csv
import json
from itertools import product
from pathlib import Path

from pulse.evaluate_policy_records import evaluate_records
from pulse.records import read_jsonl
from pulse.policy import PulsePolicy


def parse_float_grid(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def utility_score(
    summary: dict,
    clarification_cost: float,
    confirmation_cost: float,
    unsafe_cost: float,
    wrong_cost: float,
) -> float:
    task_success = float(summary["task_success"] or 0.0)
    clarification_rate = float(summary["clarification_rate"] or 0.0)
    confirmation_rate = float(summary.get("confirmation_rate") or 0.0)
    unsafe_wrong_rate = float(summary["unsafe_wrong_action_rate"] or 0.0)
    wrong_action_rate = float(summary["wrong_action_rate"] or 0.0)
    return (
        task_success
        - clarification_cost * clarification_rate
        - confirmation_cost * confirmation_rate
        - unsafe_cost * unsafe_wrong_rate
        - wrong_cost * wrong_action_rate
    )


def sweep_records(
    records: list[dict],
    *,
    entity_thresholds: list[float],
    phonetic_thresholds: list[float],
    instability_thresholds: list[float],
    entropy_thresholds: list[float],
    confirm_retrieval_options: list[bool],
    clarification_cost: float,
    confirmation_cost: float,
    unsafe_cost: float,
    wrong_cost: float,
) -> list[dict]:
    rows: list[dict] = []
    for entity_threshold, phonetic_threshold, instability_threshold, entropy_threshold, confirm_retrieval in product(
        entity_thresholds,
        phonetic_thresholds,
        instability_thresholds,
        entropy_thresholds,
        confirm_retrieval_options,
    ):
        policy = PulsePolicy(
            entity_risk_threshold=entity_threshold,
            phonetic_risk_threshold=phonetic_threshold,
            instability_threshold=instability_threshold,
            entropy_threshold=entropy_threshold,
            confirm_retrieval_phonetic_risk=confirm_retrieval,
        )
        summary = evaluate_records(records, policy)
        rows.append(
            {
                "entity_threshold": entity_threshold,
                "phonetic_threshold": phonetic_threshold,
                "instability_threshold": instability_threshold,
                "entropy_threshold": entropy_threshold,
                "confirm_retrieval_phonetic_risk": confirm_retrieval,
                **summary,
                "utility": utility_score(summary, clarification_cost, confirmation_cost, unsafe_cost, wrong_cost),
            }
        )
    return sorted(rows, key=lambda row: row["utility"], reverse=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep PULSE task-policy thresholds on an ASR task records.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--entity-thresholds", default="0.5,0.75,1.0,1.1")
    parser.add_argument("--phonetic-thresholds", default="0.5,0.75,0.85,1.0,1.1")
    parser.add_argument("--instability-thresholds", default="0.1,0.2,0.28,0.4,1.1")
    parser.add_argument("--entropy-thresholds", default="1.1")
    parser.add_argument("--confirm-retrieval-options", default="false,true")
    parser.add_argument("--clarification-cost", type=float, default=0.15)
    parser.add_argument("--confirmation-cost", type=float, default=0.05)
    parser.add_argument("--unsafe-cost", type=float, default=1.0)
    parser.add_argument("--wrong-cost", type=float, default=0.5)
    parser.add_argument("--output-csv", type=Path, default=Path("results/task_policy_sweep.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("results/task_policy_sweep.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = sweep_records(
        read_jsonl(args.records),
        entity_thresholds=parse_float_grid(args.entity_thresholds),
        phonetic_thresholds=parse_float_grid(args.phonetic_thresholds),
        instability_thresholds=parse_float_grid(args.instability_thresholds),
        entropy_thresholds=parse_float_grid(args.entropy_thresholds),
        confirm_retrieval_options=[item.strip().lower() in {"1", "true", "yes"} for item in args.confirm_retrieval_options.split(",") if item.strip()],
        clarification_cost=args.clarification_cost,
        confirmation_cost=args.confirmation_cost,
        unsafe_cost=args.unsafe_cost,
        wrong_cost=args.wrong_cost,
    )
    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"count": len(rows), "best": rows[:10]}, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(rows), "best": rows[:5], "csv": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
