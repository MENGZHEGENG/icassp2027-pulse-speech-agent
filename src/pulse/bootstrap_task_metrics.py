from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable

from pulse.candidate_rescore import evaluate_rescored_confirmations
from pulse.evaluate_policy_records import evaluate_records
from pulse.records import read_jsonl
from pulse.policy import PulsePolicy


DEFAULT_METRICS = [
    "task_success",
    "wrong_action_rate",
    "unsafe_wrong_action_rate",
    "clarification_rate",
    "confirmation_rate",
    "deferral_rate",
    "accepted_confirmation_rate",
]


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_metric_summary(
    records: list[dict[str, Any]],
    evaluator: Callable[[list[dict[str, Any]]], dict[str, Any]],
    *,
    trials: int,
    seed: int,
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    observed = evaluator(records)
    samples = {metric: [] for metric in metrics}
    if not records:
        return {metric: {"observed": 0.0, "mean": 0.0, "lo": 0.0, "hi": 0.0} for metric in metrics}
    for _ in range(trials):
        resample = [records[rng.randrange(len(records))] for _ in records]
        row = evaluator(resample)
        for metric in metrics:
            samples[metric].append(float(row.get(metric, 0.0)))
    return {
        metric: {
            "observed": float(observed.get(metric, 0.0)),
            "mean": sum(samples[metric]) / len(samples[metric]) if samples[metric] else 0.0,
            "lo": percentile(samples[metric], 0.025),
            "hi": percentile(samples[metric], 0.975),
        }
        for metric in metrics
    }


def bootstrap_recordwise_metric_summary(
    records: list[dict[str, Any]],
    evaluator: Callable[[list[dict[str, Any]]], dict[str, Any]],
    *,
    trials: int,
    seed: int,
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    rng = random.Random(seed)
    observed = evaluator(records)
    if not records:
        return {metric: {"observed": 0.0, "mean": 0.0, "lo": 0.0, "hi": 0.0} for metric in metrics}
    per_record = [evaluator([record]) for record in records]
    samples = {metric: [] for metric in metrics}
    for _ in range(trials):
        indices = [rng.randrange(len(records)) for _ in records]
        for metric in metrics:
            samples[metric].append(sum(float(per_record[index].get(metric, 0.0)) for index in indices) / len(records))
    return {
        metric: {
            "observed": float(observed.get(metric, 0.0)),
            "mean": sum(samples[metric]) / len(samples[metric]) if samples[metric] else 0.0,
            "lo": percentile(samples[metric], 0.025),
            "hi": percentile(samples[metric], 0.975),
        }
        for metric in metrics
    }


def build_evaluator(
    mode: str,
    policy: PulsePolicy,
    margin_threshold: float,
    *,
    phone_risk_mode: str | None = None,
    phone_guard_threshold: float | None = None,
) -> Callable[[list[dict[str, Any]]], dict[str, Any]]:
    if mode == "policy":
        return lambda rows: evaluate_records(
            rows,
            policy,
            phone_risk_mode=phone_risk_mode,
            phone_guard_threshold=phone_guard_threshold,
        )
    if mode == "rescore":
        return lambda rows: evaluate_rescored_confirmations(
            rows,
            policy=policy,
            margin_threshold=margin_threshold,
            phone_risk_mode=phone_risk_mode,
            phone_guard_threshold=phone_guard_threshold,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def parse_metrics(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap confidence intervals for PULSE task metrics.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--mode", choices=["policy", "rescore"], default="policy")
    parser.add_argument("--margin-threshold", type=float, default=0.6)
    parser.add_argument("--confirm-retrieval-risk", action="store_true")
    parser.add_argument("--phone-risk-mode", choices=["max", "ignore", "commit_guard"], default=None)
    parser.add_argument("--phone-guard-threshold", type=float, default=None)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--output-json", type=Path, default=Path("results/bootstrap_task_metrics.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = PulsePolicy(confirm_retrieval_phonetic_risk=args.confirm_retrieval_risk)
    metrics = parse_metrics(args.metrics)
    summary = bootstrap_recordwise_metric_summary(
        read_jsonl(args.records),
        build_evaluator(
            args.mode,
            policy,
            args.margin_threshold,
            phone_risk_mode=args.phone_risk_mode,
            phone_guard_threshold=args.phone_guard_threshold,
        ),
        trials=args.trials,
        seed=args.seed,
        metrics=metrics,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode,
        "margin_threshold": args.margin_threshold if args.mode == "rescore" else None,
        "confirm_retrieval_risk": args.confirm_retrieval_risk,
        "phone_risk_mode": args.phone_risk_mode,
        "phone_guard_threshold": args.phone_guard_threshold if args.phone_risk_mode == "commit_guard" else None,
        "trials": args.trials,
        "summary": summary,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "output_json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
