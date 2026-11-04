from __future__ import annotations

import argparse
import json
from math import exp
import re
from pathlib import Path
from typing import Literal

from pulse.records import read_jsonl
from pulse.policy import AgentAction, PulseDecision, PulsePolicy
from pulse.record_summary import parse_external_phone_risk, summarize_record


TOOL_ACTIONS = {"call_contact", "send_email", "transfer_money", "schedule_meeting"}
PhoneRiskMode = Literal["max", "ignore", "commit_guard"]
PolicyMode = Literal[
    "pulse",
    "single_best_transcript_agent",
    "ask_every_entity_bearing_command",
    "scalar_asr_confidence_threshold",
    "slot_entity_confidence_threshold",
    "calibrated_confidence_only_policy",
    "fused_asr_confidence_threshold",
    "nbest_entropy_self_consistency",
    "reflection_over_transcript_alternatives",
    "cause_aware_asr_llm_clarification",
    "oracle_clarification_trigger",
]
POLICY_MODES: tuple[PolicyMode, ...] = (
    "pulse",
    "single_best_transcript_agent",
    "ask_every_entity_bearing_command",
    "scalar_asr_confidence_threshold",
    "slot_entity_confidence_threshold",
    "calibrated_confidence_only_policy",
    "fused_asr_confidence_threshold",
    "nbest_entropy_self_consistency",
    "reflection_over_transcript_alternatives",
    "cause_aware_asr_llm_clarification",
    "oracle_clarification_trigger",
)


def requested_agent_action(task_action: str) -> AgentAction:
    if task_action == "retrieve_file":
        return AgentAction.RETRIEVE
    if task_action in TOOL_ACTIONS:
        return AgentAction.CALL_TOOL
    return AgentAction.ANSWER


def requested_record_action(record: dict) -> AgentAction:
    action = str(record.get("action", "answer"))
    requested_action = requested_agent_action(action)
    if requested_action == AgentAction.ANSWER and action and action != "answer" and record.get("dataset") == "fluent_speech_commands":
        return AgentAction.CALL_TOOL
    return requested_action


def normalized_text(text: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def top_entity_is_correct(record: dict) -> bool:
    top_text = normalized_text(record["hypotheses"][0]["text"])
    if record.get("true_entity"):
        return normalized_text(record["true_entity"]) in top_text
    reference = normalized_text(record.get("reference", ""))
    if reference:
        return reference == top_text or reference in top_text
    return False


def is_entity_bearing(record: dict) -> bool:
    if record.get("true_entity") or record.get("entity_terms") or record.get("expected_tool_call"):
        return True
    if record.get("dataset") == "fluent_speech_commands" and record.get("action"):
        return True
    return False


def scalar_confidence(record: dict) -> float | None:
    hypothesis = record["hypotheses"][0]
    for key in ("confidence", "avg_logprob", "score"):
        if hypothesis.get(key) is not None:
            value = float(hypothesis[key])
            if key == "confidence":
                return max(0.0, min(1.0, value))
            return value
    for key in ("asr_confidence", "confidence", "avg_logprob", "score"):
        if record.get(key) is not None:
            value = float(record[key])
            if "confidence" in key:
                return max(0.0, min(1.0, value))
            return value
    return None


def probability_confidence(record: dict) -> float | None:
    confidence = scalar_confidence(record)
    if confidence is None:
        return None
    if 0.0 <= confidence <= 1.0:
        return confidence
    if confidence <= 0.0:
        return max(0.0, min(1.0, exp(confidence)))
    return 1.0 / (1.0 + exp(-confidence))


def normalized_risk(value: float | None, threshold: float) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value) / max(threshold, 1e-6)))


def fused_asr_confidence(record: dict, policy: PulsePolicy | None = None) -> float | None:
    policy = policy or PulsePolicy()
    summary = summarize_record(record, use_external_phone_risk=True)
    risk_components = [
        normalized_risk(summary.entropy, policy.entropy_threshold),
        normalized_risk(summary.instability, policy.instability_threshold),
        normalized_risk(summary.entity_risk, policy.entity_risk_threshold),
        normalized_risk(summary.phonetic_risk, policy.phonetic_risk_threshold),
    ]
    confidence = probability_confidence(record)
    if confidence is not None:
        risk_components.append(1.0 - confidence)
    if not risk_components:
        return None
    return max(0.0, min(1.0, 1.0 - max(risk_components)))


def first_present_float(record: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if record.get(key) is not None:
            return float(record[key])
    return None


def policy_mode_confidence(record: dict, policy_mode: PolicyMode, policy: PulsePolicy | None = None) -> float | None:
    if policy_mode == "slot_entity_confidence_threshold":
        confidence = first_present_float(record, ("entity_confidence", "slot_confidence"))
        return scalar_confidence(record) if confidence is None else confidence
    if policy_mode == "calibrated_confidence_only_policy":
        confidence = first_present_float(record, ("calibrated_confidence",))
        return scalar_confidence(record) if confidence is None else confidence
    if policy_mode == "fused_asr_confidence_threshold":
        return fused_asr_confidence(record, policy)
    if policy_mode == "scalar_asr_confidence_threshold":
        return scalar_confidence(record)
    return None


def score_passes_threshold(score: float | None, threshold: float) -> bool:
    if score is None:
        return False
    return score >= threshold


def action_conditioned_reflection_decision(record: dict, policy: PulsePolicy, summary, requested_action: AgentAction) -> PulseDecision:
    risky_action = requested_action in {AgentAction.CALL_TOOL, AgentAction.RETRIEVE}
    reflection_risk = max(summary.entity_risk, summary.instability)
    if risky_action and summary.entity_risk >= policy.entity_risk_threshold:
        return PulseDecision(
            AgentAction.CLARIFY,
            "action-conditioned reflection flags conflicting entity alternatives for a risky action",
            reflection_risk,
        )
    if risky_action and summary.phonetic_risk >= policy.phonetic_risk_threshold:
        return PulseDecision(
            AgentAction.CONFIRM,
            "action-conditioned reflection asks for confirmation before a phonetically confusable risky action",
            max(reflection_risk, summary.phonetic_risk),
        )
    if risky_action and summary.instability >= policy.instability_threshold:
        return PulseDecision(
            AgentAction.CLARIFY,
            "action-conditioned reflection flags unstable transcript alternatives for a risky action",
            reflection_risk,
        )
    if summary.entropy is not None and summary.entropy >= policy.entropy_threshold and summary.instability >= policy.instability_threshold:
        return PulseDecision(
            AgentAction.CLARIFY,
            "action-conditioned reflection defers a non-action answer only when transcript uncertainty stays high",
            max(summary.entropy, summary.instability),
        )
    if risky_action:
        return PulseDecision(requested_action, "action-conditioned reflection sees no risky-action conflict", 0.0)
    return PulseDecision(requested_action, "action-conditioned reflection allows the non-action answer", 0.0)


def comparator_decision(
    record: dict,
    policy: PulsePolicy,
    *,
    policy_mode: PolicyMode,
    scalar_confidence_threshold: float,
    slot_entity_confidence_threshold: float,
    fused_confidence_threshold: float,
) -> PulseDecision:
    requested_action = requested_record_action(record)
    summary = summarize_record(record, use_external_phone_risk=False)
    if policy_mode == "single_best_transcript_agent":
        return PulseDecision(requested_action, "single-best transcript executes without uncertainty gating", 0.0)
    if policy_mode == "ask_every_entity_bearing_command":
        if is_entity_bearing(record) and requested_action in {AgentAction.CALL_TOOL, AgentAction.RETRIEVE}:
            return PulseDecision(AgentAction.CLARIFY, "ask every entity-bearing command", 1.0)
        return PulseDecision(requested_action, "non-entity-bearing command executes", 0.0)
    if policy_mode == "scalar_asr_confidence_threshold":
        confidence = policy_mode_confidence(record, policy_mode)
        if not score_passes_threshold(confidence, scalar_confidence_threshold):
            return PulseDecision(AgentAction.CLARIFY, "scalar ASR confidence below threshold", 1.0)
        return PulseDecision(requested_action, "scalar ASR confidence above threshold", 0.0)
    if policy_mode == "slot_entity_confidence_threshold":
        confidence = policy_mode_confidence(record, policy_mode)
        if not score_passes_threshold(confidence, slot_entity_confidence_threshold):
            return PulseDecision(AgentAction.CLARIFY, "slot/entity confidence below threshold", 1.0)
        return PulseDecision(requested_action, "slot/entity confidence above threshold", 0.0)
    if policy_mode == "calibrated_confidence_only_policy":
        confidence = policy_mode_confidence(record, policy_mode)
        if not score_passes_threshold(confidence, scalar_confidence_threshold):
            return PulseDecision(AgentAction.CLARIFY, "calibrated confidence below held-out threshold", 1.0)
        return PulseDecision(requested_action, "calibrated confidence above held-out threshold", 0.0)
    if policy_mode == "fused_asr_confidence_threshold":
        confidence = policy_mode_confidence(record, policy_mode, policy)
        if not score_passes_threshold(confidence, fused_confidence_threshold):
            return PulseDecision(AgentAction.CLARIFY, "fused ASR confidence below threshold", 1.0)
        return PulseDecision(requested_action, "fused ASR confidence above threshold", 0.0)
    if policy_mode == "nbest_entropy_self_consistency":
        if summary.entropy is not None and summary.entropy >= policy.entropy_threshold:
            return PulseDecision(AgentAction.CLARIFY, "N-best entropy exceeds self-consistency threshold", summary.entropy)
        if summary.instability >= policy.instability_threshold:
            return PulseDecision(AgentAction.CLARIFY, "N-best transcripts disagree", summary.instability)
        return PulseDecision(requested_action, "N-best alternatives are self-consistent", 0.0)
    if policy_mode == "reflection_over_transcript_alternatives":
        return action_conditioned_reflection_decision(record, policy, summary, requested_action)
    if policy_mode == "cause_aware_asr_llm_clarification":
        if summary.entity_risk >= policy.entity_risk_threshold:
            return PulseDecision(AgentAction.CLARIFY, "cause-aware baseline flags entity disagreement", summary.entity_risk)
        if summary.phonetic_risk >= policy.phonetic_risk_threshold:
            return PulseDecision(AgentAction.CLARIFY, "cause-aware baseline flags phonetic ambiguity", summary.phonetic_risk)
        if summary.instability >= policy.instability_threshold:
            return PulseDecision(AgentAction.CLARIFY, "cause-aware baseline flags transcript instability", summary.instability)
        return PulseDecision(requested_action, "cause-aware baseline finds no ASR ambiguity cause", 0.0)
    if policy_mode == "oracle_clarification_trigger":
        if not top_entity_is_correct(record):
            return PulseDecision(AgentAction.CLARIFY, "oracle clarifies every wrong top entity", 1.0)
        return PulseDecision(requested_action, "oracle permits correct top entity", 0.0)
    raise ValueError(f"unknown policy mode: {policy_mode}")


def decide_record(
    record: dict,
    policy: PulsePolicy,
    *,
    policy_mode: PolicyMode = "pulse",
    phone_risk_mode: PhoneRiskMode = "max",
    phone_guard_threshold: float | None = None,
    scalar_confidence_threshold: float = -0.5,
    slot_entity_confidence_threshold: float = -0.5,
    fused_confidence_threshold: float = 0.5,
) -> PulseDecision:
    if policy_mode != "pulse":
        return comparator_decision(
            record,
            policy,
            policy_mode=policy_mode,
            scalar_confidence_threshold=scalar_confidence_threshold,
            slot_entity_confidence_threshold=slot_entity_confidence_threshold,
            fused_confidence_threshold=fused_confidence_threshold,
        )
    requested_action = requested_record_action(record)
    if phone_risk_mode == "max":
        return policy.decide(summarize_record(record, use_external_phone_risk=True), requested_action)
    if phone_risk_mode == "ignore":
        return policy.decide(summarize_record(record, use_external_phone_risk=False), requested_action)
    if phone_risk_mode != "commit_guard":
        raise ValueError(f"unknown phone risk mode: {phone_risk_mode}")

    summary = summarize_record(record, use_external_phone_risk=False)
    decision = policy.decide(summary, requested_action)
    if decision.action in {AgentAction.CLARIFY, AgentAction.CONFIRM, AgentAction.WAIT}:
        return decision
    phone_risk = parse_external_phone_risk(record)
    threshold = policy.phonetic_risk_threshold if phone_guard_threshold is None else phone_guard_threshold
    if phone_risk is not None and phone_risk >= threshold:
        risk_score = max(decision.risk_score, phone_risk / max(threshold, 1e-6))
        return PulseDecision(AgentAction.CLARIFY, "external phone risk blocks commit", risk_score)
    return decision


def evaluate_records(
    records: list[dict],
    policy: PulsePolicy | None = None,
    *,
    policy_mode: PolicyMode = "pulse",
    use_external_phone_risk: bool = True,
    phone_risk_mode: PhoneRiskMode | None = None,
    phone_guard_threshold: float | None = None,
    scalar_confidence_threshold: float = -0.5,
    slot_entity_confidence_threshold: float = -0.5,
    fused_confidence_threshold: float = 0.5,
) -> dict:
    policy = policy or PulsePolicy()
    if phone_risk_mode is None:
        phone_risk_mode = "max" if use_external_phone_risk else "ignore"
    wrong_actions = 0
    unsafe_wrong_actions = 0
    clarifications = 0
    confirmations = 0
    successes = 0
    total = len(records)
    for record in records:
        decision = decide_record(
            record,
            policy,
            policy_mode=policy_mode,
            phone_risk_mode=phone_risk_mode,
            phone_guard_threshold=phone_guard_threshold,
            scalar_confidence_threshold=scalar_confidence_threshold,
            slot_entity_confidence_threshold=slot_entity_confidence_threshold,
            fused_confidence_threshold=fused_confidence_threshold,
        )
        if decision.action == AgentAction.CLARIFY:
            clarifications += 1
            successes += 1
            continue
        if decision.action == AgentAction.CONFIRM:
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
        "count": total,
        "task_success": successes / total if total else None,
        "wrong_action_rate": wrong_actions / total if total else None,
        "unsafe_wrong_action_rate": unsafe_wrong_actions / total if total else None,
        "clarification_rate": clarifications / total if total else None,
        "confirmation_rate": confirmations / total if total else None,
        "deferral_rate": (clarifications + confirmations) / total if total else None,
        "policy_mode": policy_mode,
        "scalar_confidence_threshold": scalar_confidence_threshold
        if policy_mode in {"scalar_asr_confidence_threshold", "calibrated_confidence_only_policy"}
        else None,
        "slot_entity_confidence_threshold": slot_entity_confidence_threshold
        if policy_mode == "slot_entity_confidence_threshold"
        else None,
        "fused_confidence_threshold": fused_confidence_threshold if policy_mode == "fused_asr_confidence_threshold" else None,
        "phone_risk_mode": phone_risk_mode,
        "phone_guard_threshold": phone_guard_threshold if phone_risk_mode == "commit_guard" else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate PULSE decisions on task records with ASR hypotheses.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/policy_records_summary.json"))
    parser.add_argument("--entity-threshold", type=float, default=0.75)
    parser.add_argument("--phonetic-threshold", type=float, default=0.75)
    parser.add_argument("--instability-threshold", type=float, default=0.28)
    parser.add_argument("--entropy-threshold", type=float, default=1.1)
    parser.add_argument(
        "--policy-mode",
        choices=POLICY_MODES,
        default="pulse",
    )
    parser.add_argument("--scalar-confidence-threshold", type=float, default=-0.5)
    parser.add_argument("--slot-entity-confidence-threshold", type=float, default=-0.5)
    parser.add_argument("--fused-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--confirm-retrieval-risk", action="store_true")
    parser.add_argument("--ignore-external-phone-risk", action="store_true")
    parser.add_argument("--phone-risk-mode", choices=["max", "ignore", "commit_guard"], default=None)
    parser.add_argument("--phone-guard-threshold", type=float, default=None)
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
    phone_risk_mode = args.phone_risk_mode
    if phone_risk_mode is None and args.ignore_external_phone_risk:
        phone_risk_mode = "ignore"
    payload = evaluate_records(
        read_jsonl(args.records),
        policy,
        policy_mode=args.policy_mode,
        phone_risk_mode=phone_risk_mode,
        phone_guard_threshold=args.phone_guard_threshold,
        scalar_confidence_threshold=args.scalar_confidence_threshold,
        slot_entity_confidence_threshold=args.slot_entity_confidence_threshold,
        fused_confidence_threshold=args.fused_confidence_threshold,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
