from __future__ import annotations

import argparse
import json
from pathlib import Path

from pulse.records import read_jsonl
from pulse.policy import AgentAction, PulsePolicy
from pulse.uncertainty import Hypothesis, summarize_hypotheses, token_distance_ratio


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze PULSE ASR hypothesis records.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--entity-term", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = read_jsonl(args.records)
    policy = PulsePolicy()
    analyzed = []
    for record in records:
        hypotheses = [Hypothesis(item["text"], item.get("score")) for item in record["hypotheses"]]
        summary = summarize_hypotheses(hypotheses, args.entity_term)
        decision = policy.decide(summary, AgentAction.ANSWER)
        reference = record.get("reference") or ""
        analyzed.append(
            {
                "id": record.get("id"),
                "reference": reference,
                "top_text": summary.top_text,
                "wer_proxy": token_distance_ratio(reference, summary.top_text) if reference else None,
                "entropy": summary.entropy,
                "instability": summary.instability,
                "entity_risk": summary.entity_risk,
                "decision": decision.action.value,
                "reason": decision.reason,
                "risk_score": decision.risk_score,
            }
        )
    payload = {"count": len(analyzed), "records": analyzed}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
