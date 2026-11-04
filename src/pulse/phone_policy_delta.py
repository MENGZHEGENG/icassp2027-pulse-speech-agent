from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pulse.evaluate_policy_records import requested_agent_action, top_entity_is_correct
from pulse.records import read_jsonl, write_jsonl
from pulse.policy import AgentAction, PulseDecision, PulsePolicy
from pulse.record_summary import parse_external_phone_risk, summarize_record

DEFERRAL_ACTIONS = {AgentAction.CLARIFY, AgentAction.CONFIRM, AgentAction.WAIT}


def decision_label(decision: PulseDecision) -> str:
    return str(decision.action.value)


def is_deferral(decision: PulseDecision) -> bool:
    return decision.action in DEFERRAL_ACTIONS


def decision_outcome(record: dict[str, Any], policy: PulsePolicy, *, use_external_phone_risk: bool) -> dict[str, Any]:
    summary = summarize_record(record, use_external_phone_risk=use_external_phone_risk)
    requested_action = requested_agent_action(record.get("action", "answer"))
    decision = policy.decide(summary, requested_action)
    deferred = is_deferral(decision)
    correct_top = top_entity_is_correct(record)
    wrong_action = (not deferred) and (not correct_top)
    unsafe_wrong = wrong_action and bool(record.get("unsafe_if_confused"))
    return {
        "decision": decision_label(decision),
        "reason": decision.reason,
        "risk_score": decision.risk_score,
        "phonetic_risk": summary.phonetic_risk,
        "entity_risk": summary.entity_risk,
        "instability": summary.instability,
        "entropy": summary.entropy,
        "deferred": deferred,
        "success": deferred or correct_top,
        "wrong_action": wrong_action,
        "unsafe_wrong_action": unsafe_wrong,
    }


def phone_commit_guard_outcome(record: dict[str, Any], policy: PulsePolicy, *, phone_threshold: float) -> dict[str, Any]:
    outcome = decision_outcome(record, policy, use_external_phone_risk=False)
    phone_risk = parse_external_phone_risk(record)
    if outcome["deferred"] or phone_risk is None or phone_risk < phone_threshold:
        return outcome
    guarded = dict(outcome)
    guarded.update(
        {
            "decision": decision_label(PulseDecision(AgentAction.CLARIFY, "external phone risk blocks commit", phone_risk)),
            "reason": "external phone risk blocks commit",
            "risk_score": max(float(outcome["risk_score"]), phone_risk / max(phone_threshold, 1e-6)),
            "phonetic_risk": max(float(outcome["phonetic_risk"]), phone_risk),
            "deferred": True,
            "success": True,
            "wrong_action": False,
            "unsafe_wrong_action": False,
        }
    )
    return guarded


def transition_label(without_phone: dict[str, Any], with_phone: dict[str, Any]) -> str:
    before = "defer" if without_phone["deferred"] else "commit"
    after = "defer" if with_phone["deferred"] else "commit"
    return f"{before}->{after}"


def compact_example(record: dict[str, Any], without_phone: dict[str, Any], with_phone: dict[str, Any]) -> dict[str, Any]:
    evidence = record.get("phone_evidence") if isinstance(record.get("phone_evidence"), dict) else {}
    return {
        "id": record.get("id"),
        "true_entity": record.get("true_entity"),
        "top_hypothesis": record.get("hypotheses", [{}])[0].get("text"),
        "unsafe_if_confused": bool(record.get("unsafe_if_confused")),
        "external_phone_risk": parse_external_phone_risk(record),
        "phone_predicted_entity": evidence.get("predicted_entity"),
        "phone_risk_reason": evidence.get("risk_reason"),
        "without_phone_decision": without_phone["decision"],
        "with_phone_decision": with_phone["decision"],
        "without_phone_wrong_action": without_phone["wrong_action"],
        "with_phone_wrong_action": with_phone["wrong_action"],
        "without_phone_unsafe_wrong_action": without_phone["unsafe_wrong_action"],
        "with_phone_unsafe_wrong_action": with_phone["unsafe_wrong_action"],
    }


def analyze_records(
    records: list[dict[str, Any]],
    policy: PulsePolicy | None = None,
    *,
    max_examples: int = 10,
    phone_guard_threshold: float = 0.75,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy = policy or PulsePolicy()
    without_counter: Counter[str] = Counter()
    with_counter: Counter[str] = Counter()
    guard_counter: Counter[str] = Counter()
    transition_counter: Counter[str] = Counter()
    guard_transition_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    changed_examples: list[dict[str, Any]] = []
    prevented_wrong_examples: list[dict[str, Any]] = []

    totals = {
        "count": len(records),
        "without_phone_success": 0,
        "with_phone_success": 0,
        "without_phone_wrong_action": 0,
        "with_phone_wrong_action": 0,
        "without_phone_unsafe_wrong_action": 0,
        "with_phone_unsafe_wrong_action": 0,
        "changed_decisions": 0,
        "wrong_actions_prevented": 0,
        "unsafe_wrong_actions_prevented": 0,
        "guard_success": 0,
        "guard_wrong_action": 0,
        "guard_unsafe_wrong_action": 0,
        "guarded_commits": 0,
        "wrong_actions_prevented_by_guard": 0,
        "unsafe_wrong_actions_prevented_by_guard": 0,
    }

    examples: list[dict[str, Any]] = []
    for record in records:
        without_phone = decision_outcome(record, policy, use_external_phone_risk=False)
        with_phone = decision_outcome(record, policy, use_external_phone_risk=True)
        guard = phone_commit_guard_outcome(record, policy, phone_threshold=phone_guard_threshold)
        without_counter[without_phone["decision"]] += 1
        with_counter[with_phone["decision"]] += 1
        guard_counter[guard["decision"]] += 1
        transition_counter[transition_label(without_phone, with_phone)] += 1
        guard_transition_counter[transition_label(without_phone, guard)] += 1
        if without_phone["decision"] != with_phone["decision"]:
            totals["changed_decisions"] += 1
            reason_counter[with_phone["reason"]] += 1
            if len(changed_examples) < max_examples:
                changed_examples.append(compact_example(record, without_phone, with_phone))
        for key in ("success", "wrong_action", "unsafe_wrong_action"):
            totals[f"without_phone_{key}"] += int(bool(without_phone[key]))
            totals[f"with_phone_{key}"] += int(bool(with_phone[key]))
            totals[f"guard_{key}"] += int(bool(guard[key]))
        if without_phone["wrong_action"] and not with_phone["wrong_action"]:
            totals["wrong_actions_prevented"] += 1
            if len(prevented_wrong_examples) < max_examples:
                prevented_wrong_examples.append(compact_example(record, without_phone, with_phone))
        if without_phone["unsafe_wrong_action"] and not with_phone["unsafe_wrong_action"]:
            totals["unsafe_wrong_actions_prevented"] += 1
        if not without_phone["deferred"] and guard["deferred"]:
            totals["guarded_commits"] += 1
        if without_phone["wrong_action"] and not guard["wrong_action"]:
            totals["wrong_actions_prevented_by_guard"] += 1
        if without_phone["unsafe_wrong_action"] and not guard["unsafe_wrong_action"]:
            totals["unsafe_wrong_actions_prevented_by_guard"] += 1
        if without_phone["decision"] != with_phone["decision"] or without_phone["wrong_action"] != with_phone["wrong_action"]:
            examples.append(compact_example(record, without_phone, with_phone))

    count = max(len(records), 1)
    summary = {
        **totals,
        "without_phone_success_rate": totals["without_phone_success"] / count,
        "with_phone_success_rate": totals["with_phone_success"] / count,
        "without_phone_wrong_action_rate": totals["without_phone_wrong_action"] / count,
        "with_phone_wrong_action_rate": totals["with_phone_wrong_action"] / count,
        "without_phone_unsafe_wrong_action_rate": totals["without_phone_unsafe_wrong_action"] / count,
        "with_phone_unsafe_wrong_action_rate": totals["with_phone_unsafe_wrong_action"] / count,
        "guard_phone_threshold": phone_guard_threshold,
        "guard_success_rate": totals["guard_success"] / count,
        "guard_wrong_action_rate": totals["guard_wrong_action"] / count,
        "guard_unsafe_wrong_action_rate": totals["guard_unsafe_wrong_action"] / count,
        "guard_deferral_rate": sum(guard_counter[action.value] for action in DEFERRAL_ACTIONS) / count,
        "guard_clarification_rate": guard_counter[AgentAction.CLARIFY.value] / count,
        "guard_confirmation_rate": guard_counter[AgentAction.CONFIRM.value] / count,
        "decision_counts_without_phone": dict(without_counter),
        "decision_counts_with_phone": dict(with_counter),
        "decision_counts_guard": dict(guard_counter),
        "transition_counts": dict(transition_counter),
        "guard_transition_counts": dict(guard_transition_counter),
        "changed_decision_reasons": dict(reason_counter),
        "changed_examples": changed_examples,
        "prevented_wrong_examples": prevented_wrong_examples,
    }
    return summary, examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare PULSE policy outcomes with vs. without external phone risk.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--examples-output", type=Path)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--phone-guard-threshold", type=float, default=0.75)
    parser.add_argument("--entity-threshold", type=float, default=0.75)
    parser.add_argument("--phonetic-threshold", type=float, default=0.75)
    parser.add_argument("--instability-threshold", type=float, default=0.28)
    parser.add_argument("--entropy-threshold", type=float, default=1.1)
    parser.add_argument("--confirm-retrieval-risk", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = PulsePolicy(
        entity_risk_threshold=args.entity_threshold,
        phonetic_risk_threshold=args.phonetic_threshold,
        instability_threshold=args.instability_threshold,
        entropy_threshold=args.entropy_threshold,
        confirm_retrieval_phonetic_risk=args.confirm_retrieval_risk,
    )
    summary, examples = analyze_records(
        read_jsonl(args.records),
        policy,
        max_examples=args.max_examples,
        phone_guard_threshold=args.phone_guard_threshold,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.examples_output:
        write_jsonl(args.examples_output, examples)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
