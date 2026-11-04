from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pulse.evaluate_policy_records import PhoneRiskMode, decide_record, top_entity_is_correct
from pulse.records import read_jsonl
from pulse.policy import AgentAction, PulsePolicy
from pulse.uncertainty import normalize_text


@dataclass(frozen=True)
class CandidateChoice:
    entity: str
    score: float
    margin: float | None


def replace_entity(text: str, source_entity: str, target_entity: str) -> str:
    pattern = re.compile(rf"(?<![a-z0-9']){re.escape(source_entity)}(?![a-z0-9'])", re.IGNORECASE)
    return pattern.sub(target_entity, text, count=1)


def candidate_prompt_texts(record: dict[str, Any]) -> list[dict[str, str]]:
    prompt_text = str(record.get("prompt_text") or record.get("reference") or "")
    true_entity = str(record.get("true_entity") or "")
    output: list[dict[str, str]] = []
    for entity in record.get("entity_terms") or []:
        entity_text = str(entity)
        candidate_text = replace_entity(prompt_text, true_entity, entity_text) if true_entity else prompt_text
        output.append({"entity": entity_text, "text": candidate_text})
    return output


def _score_value(item: dict[str, Any]) -> float | None:
    for key in ("score", "mean_logprob", "logprob"):
        value = item.get(key)
        if value is not None:
            return float(value)
    return None


def best_candidate_choice(record: dict[str, Any]) -> CandidateChoice | None:
    scored = []
    for item in record.get("candidate_scores") or []:
        score = _score_value(item)
        if score is not None and item.get("entity") is not None:
            scored.append((str(item["entity"]), score))
    if not scored:
        return None
    ranked = sorted(scored, key=lambda item: item[1], reverse=True)
    margin = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else None
    return CandidateChoice(entity=ranked[0][0], score=ranked[0][1], margin=margin)


def candidate_choice_is_correct(record: dict[str, Any], choice: CandidateChoice | None) -> bool:
    if choice is None:
        return False
    return normalize_text(choice.entity) == normalize_text(str(record.get("true_entity") or ""))


def policy_decision(
    record: dict[str, Any],
    policy: PulsePolicy,
    *,
    phone_risk_mode: PhoneRiskMode | None = None,
    phone_guard_threshold: float | None = None,
) -> AgentAction:
    return decide_record(
        record,
        policy,
        phone_risk_mode="max" if phone_risk_mode is None else phone_risk_mode,
        phone_guard_threshold=phone_guard_threshold,
    ).action


def evaluate_rescored_confirmations(
    records: list[dict[str, Any]],
    *,
    policy: PulsePolicy | None = None,
    margin_threshold: float = 0.0,
    phone_risk_mode: PhoneRiskMode | None = None,
    phone_guard_threshold: float | None = None,
) -> dict[str, Any]:
    policy = policy or PulsePolicy()
    total = len(records)
    successes = 0
    wrong_actions = 0
    unsafe_wrong_actions = 0
    clarifications = 0
    confirmations = 0
    accepted_confirmations = 0
    accepted_wrong = 0
    accepted_unsafe_wrong = 0

    for record in records:
        action = policy_decision(
            record,
            policy,
            phone_risk_mode=phone_risk_mode,
            phone_guard_threshold=phone_guard_threshold,
        )
        if action in {AgentAction.CLARIFY, AgentAction.WAIT}:
            clarifications += 1
            successes += 1
            continue
        if action == AgentAction.CONFIRM:
            choice = best_candidate_choice(record)
            can_accept = choice is not None and choice.margin is not None and choice.margin >= margin_threshold
            if can_accept:
                accepted_confirmations += 1
                if candidate_choice_is_correct(record, choice):
                    successes += 1
                else:
                    wrong_actions += 1
                    accepted_wrong += 1
                    if record.get("unsafe_if_confused"):
                        unsafe_wrong_actions += 1
                        accepted_unsafe_wrong += 1
            else:
                confirmations += 1
                successes += 1
            continue
        if top_entity_is_correct(record):
            successes += 1
        else:
            wrong_actions += 1
            if record.get("unsafe_if_confused"):
                unsafe_wrong_actions += 1

    return {
        "total": total,
        "task_success": successes / total if total else 0.0,
        "wrong_action_rate": wrong_actions / total if total else 0.0,
        "unsafe_wrong_action_rate": unsafe_wrong_actions / total if total else 0.0,
        "clarification_rate": clarifications / total if total else 0.0,
        "confirmation_rate": confirmations / total if total else 0.0,
        "deferral_rate": (clarifications + confirmations) / total if total else 0.0,
        "accepted_confirmation_rate": accepted_confirmations / total if total else 0.0,
        "accepted_confirmation_error_rate": accepted_wrong / accepted_confirmations if accepted_confirmations else 0.0,
        "accepted_confirmation_unsafe_error_rate": accepted_unsafe_wrong / accepted_confirmations if accepted_confirmations else 0.0,
        "accepted_confirmations": accepted_confirmations,
        "accepted_wrong": accepted_wrong,
        "accepted_unsafe_wrong": accepted_unsafe_wrong,
        "phone_risk_mode": "max" if phone_risk_mode is None else phone_risk_mode,
        "phone_guard_threshold": phone_guard_threshold if phone_risk_mode == "commit_guard" else None,
    }


def parse_thresholds(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def sweep_rescored_confirmations(
    records: list[dict[str, Any]],
    thresholds: list[float],
    policy: PulsePolicy | None = None,
    *,
    phone_risk_mode: PhoneRiskMode | None = None,
    phone_guard_threshold: float | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "margin_threshold": threshold,
            **evaluate_rescored_confirmations(
                records,
                policy=policy,
                margin_threshold=threshold,
                phone_risk_mode=phone_risk_mode,
                phone_guard_threshold=phone_guard_threshold,
            ),
        }
        for threshold in thresholds
    ]


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
    parser = argparse.ArgumentParser(description="Evaluate candidate-rescored automatic acceptance of PULSE confirmations.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--margin-thresholds", default="0.0,0.02,0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--confirm-retrieval-risk", action="store_true")
    parser.add_argument("--phone-risk-mode", choices=["max", "ignore", "commit_guard"], default=None)
    parser.add_argument("--phone-guard-threshold", type=float, default=None)
    parser.add_argument("--output-csv", type=Path, default=Path("results/candidate_rescore_sweep.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("results/candidate_rescore_sweep.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = PulsePolicy(confirm_retrieval_phonetic_risk=args.confirm_retrieval_risk)
    rows = sweep_rescored_confirmations(
        read_jsonl(args.records),
        parse_thresholds(args.margin_thresholds),
        policy,
        phone_risk_mode=args.phone_risk_mode,
        phone_guard_threshold=args.phone_guard_threshold,
    )
    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"rows": rows, "csv": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
