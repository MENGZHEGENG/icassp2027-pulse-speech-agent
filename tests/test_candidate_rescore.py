from pulse.candidate_rescore import (
    best_candidate_choice,
    candidate_prompt_texts,
    evaluate_rescored_confirmations,
    policy_decision,
    replace_entity,
)
from pulse.policy import AgentAction, PulsePolicy


def test_replace_entity_is_case_insensitive_and_token_bounded() -> None:
    assert replace_entity("Call Sean now.", "sean", "shawn") == "Call shawn now."
    assert replace_entity("Call seansdesk now.", "sean", "shawn") == "Call seansdesk now."


def test_candidate_prompt_texts_swaps_entity_terms() -> None:
    record = {"prompt_text": "Find the file named sean.", "true_entity": "sean", "entity_terms": ["sean", "shawn"]}
    assert candidate_prompt_texts(record) == [
        {"entity": "sean", "text": "Find the file named sean."},
        {"entity": "shawn", "text": "Find the file named shawn."},
    ]


def test_best_candidate_choice_returns_margin() -> None:
    choice = best_candidate_choice(
        {
            "candidate_scores": [
                {"entity": "sean", "score": -0.3},
                {"entity": "shawn", "score": -0.5},
            ]
        }
    )
    assert choice is not None
    assert choice.entity == "sean"
    assert round(choice.margin or 0.0, 3) == 0.2


def test_rescored_confirmations_accept_correct_candidate() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn", "sean"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call sean", "score": -0.1}, {"text": "call sean", "score": -0.1}],
            "candidate_scores": [{"entity": "shawn", "score": -0.1}, {"entity": "sean", "score": -0.4}],
        }
    ]
    summary = evaluate_rescored_confirmations(records, margin_threshold=0.2)
    assert summary["task_success"] == 1.0
    assert summary["accepted_confirmation_rate"] == 1.0
    assert summary["unsafe_wrong_action_rate"] == 0.0


def test_rescored_confirmations_counts_unsafe_wrong_acceptance() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn", "sean"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call sean", "score": -0.1}, {"text": "call sean", "score": -0.1}],
            "candidate_scores": [{"entity": "shawn", "score": -0.4}, {"entity": "sean", "score": -0.1}],
        }
    ]
    summary = evaluate_rescored_confirmations(records, margin_threshold=0.2)
    assert summary["task_success"] == 0.0
    assert summary["wrong_action_rate"] == 1.0
    assert summary["unsafe_wrong_action_rate"] == 1.0
    assert summary["accepted_confirmation_unsafe_error_rate"] == 1.0


def test_rescored_confirmations_keeps_low_margin_confirmation_deferred() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn", "sean"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call sean", "score": -0.1}, {"text": "call sean", "score": -0.1}],
            "candidate_scores": [{"entity": "shawn", "score": -0.1}, {"entity": "sean", "score": -0.05}],
        }
    ]
    summary = evaluate_rescored_confirmations(records, margin_threshold=0.2)
    assert summary["task_success"] == 1.0
    assert summary["confirmation_rate"] == 1.0
    assert summary["accepted_confirmation_rate"] == 0.0


def test_policy_decision_uses_external_phone_risk() -> None:
    record = {
        "action": "call_contact",
        "true_entity": "john",
        "entity_terms": ["john", "jane"],
        "learned_phonetic_risk": 0.8,
        "hypotheses": [{"text": "call john", "score": -0.1}, {"text": "call john", "score": -0.2}],
    }
    assert policy_decision(record, PulsePolicy()) == AgentAction.CONFIRM


def test_policy_decision_can_ignore_or_guard_external_phone_risk() -> None:
    record = {
        "action": "call_contact",
        "true_entity": "john",
        "entity_terms": ["john"],
        "learned_phonetic_risk": 0.8,
        "hypotheses": [{"text": "call john", "score": -0.1}, {"text": "call john", "score": -0.2}],
    }

    assert policy_decision(record, PulsePolicy(), phone_risk_mode="ignore") == AgentAction.CALL_TOOL
    assert (
        policy_decision(
            record,
            PulsePolicy(),
            phone_risk_mode="commit_guard",
            phone_guard_threshold=0.75,
        )
        == AgentAction.CLARIFY
    )


def test_rescored_confirmations_respects_phone_commit_guard() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn"],
            "unsafe_if_confused": True,
            "learned_phonetic_risk": 0.8,
            "hypotheses": [{"text": "call shawn", "score": -0.1}, {"text": "call shawn", "score": -0.2}],
            "candidate_scores": [{"entity": "shawn", "score": -0.1}, {"entity": "sean", "score": -0.5}],
        }
    ]

    ignored = evaluate_rescored_confirmations(records, margin_threshold=0.2, phone_risk_mode="ignore")
    guarded = evaluate_rescored_confirmations(
        records,
        margin_threshold=0.2,
        phone_risk_mode="commit_guard",
        phone_guard_threshold=0.75,
    )

    assert ignored["deferral_rate"] == 0.0
    assert guarded["clarification_rate"] == 1.0
    assert guarded["accepted_confirmations"] == 0
    assert guarded["phone_risk_mode"] == "commit_guard"
