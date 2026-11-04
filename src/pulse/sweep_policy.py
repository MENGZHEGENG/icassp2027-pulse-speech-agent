from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

from pulse.policy import PulsePolicy
from pulse.simulate_policy import make_trial, score_policy


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep PULSE policy thresholds on simulated spoken-agent tasks.")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20270817)
    parser.add_argument("--ambiguity-rates", default="0.1,0.2,0.4,0.6,0.8")
    parser.add_argument("--entity-thresholds", default="0.05,0.1,0.25,0.5,0.75")
    parser.add_argument("--instability-thresholds", default="0.1,0.2,0.3,0.4")
    parser.add_argument("--output-csv", type=Path, default=Path("results/sim/policy_sweep.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("results/sim/policy_sweep.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ambiguity_rates = parse_float_list(args.ambiguity_rates)
    entity_thresholds = parse_float_list(args.entity_thresholds)
    instability_thresholds = parse_float_list(args.instability_thresholds)
    rows = []
    for ambiguity_rate in ambiguity_rates:
        rng = random.Random(args.seed + int(ambiguity_rate * 1000))
        trials = [make_trial(rng, ambiguity_rate) for _ in range(args.trials)]
        for policy_name in ["transcript_only", "ask_all", "nbest_disagreement"]:
            result = score_policy(policy_name, trials, PulsePolicy())
            rows.append({"ambiguity_rate": ambiguity_rate, "entity_threshold": None, "instability_threshold": None, **asdict(result)})
        for entity_threshold in entity_thresholds:
            for instability_threshold in instability_thresholds:
                policy = PulsePolicy(entity_risk_threshold=entity_threshold, instability_threshold=instability_threshold)
                result = score_policy("pulse", trials, policy)
                rows.append(
                    {
                        "ambiguity_rate": ambiguity_rate,
                        "entity_threshold": entity_threshold,
                        "instability_threshold": instability_threshold,
                        **asdict(result),
                    }
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "csv": str(args.output_csv), "json": str(args.output_json)}, indent=2))


if __name__ == "__main__":
    main()
