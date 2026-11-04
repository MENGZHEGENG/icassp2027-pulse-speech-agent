from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from pulse.records import write_jsonl
from pulse.policy import AgentAction, PulsePolicy
from pulse.uncertainty import Hypothesis, summarize_hypotheses


@dataclass(frozen=True)
class SimulationResult:
    policy: str
    trials: int
    task_success: float
    wrong_action_rate: float
    clarification_rate: float
    utility: float


NAMES = [
    ("shawn", "sean"),
    ("john smith", "jane smith"),
    ("mira patel", "meera patel"),
    ("account four", "account for"),
    ("geneva", "jen eva"),
]


def make_trial(rng: random.Random, ambiguity_rate: float) -> dict:
    correct, confusable = rng.choice(NAMES)
    true_action = rng.choice([AgentAction.CALL_TOOL, AgentAction.RETRIEVE, AgentAction.ANSWER])
    ambiguous = rng.random() < ambiguity_rate
    if ambiguous:
        hypotheses = [
            Hypothesis(f"{true_action.value} {correct}", -0.10),
            Hypothesis(f"{true_action.value} {confusable}", -0.18),
            Hypothesis(f"{true_action.value} {correct}", -0.35),
        ]
        top_is_correct = rng.random() > 0.35
        if not top_is_correct:
            hypotheses = [hypotheses[1], hypotheses[0], hypotheses[2]]
    else:
        hypotheses = [
            Hypothesis(f"{true_action.value} {correct}", -0.05),
            Hypothesis(f"{true_action.value} {correct}", -0.15),
            Hypothesis(f"{true_action.value} {correct}", -0.25),
        ]
    return {
        "true_entity": correct,
        "confusable_entity": confusable,
        "true_action": true_action,
        "hypotheses": hypotheses,
        "entity_terms": [correct, confusable],
    }


def top_entity_is_correct(trial: dict) -> bool:
    return trial["true_entity"] in trial["hypotheses"][0].text


def score_policy(policy_name: str, trials: list[dict], pulse_policy: PulsePolicy) -> SimulationResult:
    successes = 0
    wrong_actions = 0
    clarifications = 0
    utility = 0.0
    for trial in trials:
        true_action = trial["true_action"]
        if policy_name == "transcript_only":
            decision_action = true_action
        elif policy_name == "ask_all":
            decision_action = AgentAction.CLARIFY
        elif policy_name == "nbest_disagreement":
            summary = summarize_hypotheses(trial["hypotheses"], trial["entity_terms"])
            decision_action = AgentAction.CLARIFY if summary.unique_hypotheses > 1 else true_action
        elif policy_name == "pulse":
            summary = summarize_hypotheses(trial["hypotheses"], trial["entity_terms"])
            decision_action = pulse_policy.decide(summary, true_action).action
        else:
            raise ValueError(f"Unknown policy: {policy_name}")

        if decision_action == AgentAction.CLARIFY:
            clarifications += 1
            successes += 1
            utility += 0.85
        elif top_entity_is_correct(trial):
            successes += 1
            utility += 1.0
        else:
            wrong_actions += 1
            utility -= 1.0
    total = len(trials)
    return SimulationResult(
        policy=policy_name,
        trials=total,
        task_success=successes / total,
        wrong_action_rate=wrong_actions / total,
        clarification_rate=clarifications / total,
        utility=utility / total,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a lightweight PULSE policy simulation.")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--ambiguity-rate", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=20270817)
    parser.add_argument("--output", type=Path, default=Path("results/sim/policy_summary.json"))
    parser.add_argument("--records", type=Path, default=Path("results/sim/policy_trials.jsonl"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rng = random.Random(args.seed)
    trials = [make_trial(rng, args.ambiguity_rate) for _ in range(args.trials)]
    pulse_policy = PulsePolicy(entity_risk_threshold=0.1, instability_threshold=0.2)
    summaries = [score_policy(name, trials, pulse_policy) for name in ["transcript_only", "ask_all", "nbest_disagreement", "pulse"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([asdict(summary) for summary in summaries], indent=2), encoding="utf-8")
    write_jsonl(
        args.records,
        (
            {
                "true_entity": trial["true_entity"],
                "confusable_entity": trial["confusable_entity"],
                "true_action": trial["true_action"].value,
                "hypotheses": [asdict(hypothesis) for hypothesis in trial["hypotheses"]],
            }
            for trial in trials
        ),
    )
    print(json.dumps([asdict(summary) for summary in summaries], indent=2))


if __name__ == "__main__":
    main()
