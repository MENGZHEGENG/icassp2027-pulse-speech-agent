from pulse.evaluate_policy_records import decide_record, evaluate_records
from pulse.policy import AgentAction, PulsePolicy


def test_evaluate_policy_records_counts_unsafe_wrong_actions() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn", "sean"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call sean", "score": -0.1}, {"text": "call sean", "score": -0.2}],
        }
    ]
    summary = evaluate_records(
        records,
        PulsePolicy(entity_risk_threshold=1.0, phonetic_risk_threshold=1.0, instability_threshold=1.0, entropy_threshold=10.0),
    )
    assert summary["wrong_action_rate"] == 1.0
    assert summary["unsafe_wrong_action_rate"] == 1.0


def test_evaluate_policy_records_clarification_is_successful_deferral() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn", "sean"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call sean", "score": -0.1}, {"text": "call shawn", "score": -0.2}],
        }
    ]
    summary = evaluate_records(records, PulsePolicy(entity_risk_threshold=0.1))
    assert summary["clarification_rate"] == 1.0
    assert summary["unsafe_wrong_action_rate"] == 0.0


def test_evaluate_policy_records_confirms_static_phonetic_confusion() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn", "sean"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call sean", "score": -0.1}, {"text": "call sean", "score": -0.2}],
        }
    ]
    summary = evaluate_records(records)
    assert summary["confirmation_rate"] == 1.0
    assert summary["deferral_rate"] == 1.0
    assert summary["unsafe_wrong_action_rate"] == 0.0


def test_evaluate_policy_records_uses_external_phone_risk() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "john",
            "entity_terms": ["john", "account"],
            "unsafe_if_confused": True,
            "learned_phonetic_risk": 0.8,
            "hypotheses": [{"text": "call john", "score": -0.1}, {"text": "call john", "score": -0.2}],
        }
    ]
    summary = evaluate_records(records)
    assert summary["confirmation_rate"] == 1.0
    assert summary["unsafe_wrong_action_rate"] == 0.0


def test_evaluate_policy_records_can_ignore_external_phone_risk() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "john",
            "entity_terms": ["john", "account"],
            "unsafe_if_confused": True,
            "learned_phonetic_risk": 0.8,
            "hypotheses": [{"text": "call john", "score": -0.1}, {"text": "call john", "score": -0.2}],
        }
    ]
    summary = evaluate_records(records, use_external_phone_risk=False)
    assert summary["confirmation_rate"] == 0.0
    assert summary["unsafe_wrong_action_rate"] == 0.0


def test_evaluate_policy_records_phone_commit_guard_blocks_only_commits() -> None:
    records = [
        {
            "action": "retrieve_file",
            "true_entity": "alice",
            "entity_terms": ["alice", "bob"],
            "unsafe_if_confused": False,
            "learned_phonetic_risk": 0.8,
            "hypotheses": [{"text": "find bob", "score": -0.1}, {"text": "find bob", "score": -0.2}],
        },
        {
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn", "sean"],
            "unsafe_if_confused": True,
            "learned_phonetic_risk": 1.0,
            "hypotheses": [{"text": "call sean", "score": -0.1}, {"text": "call sean", "score": -0.2}],
        },
    ]
    summary = evaluate_records(records, phone_risk_mode="commit_guard", phone_guard_threshold=0.75)
    assert summary["phone_risk_mode"] == "commit_guard"
    assert summary["wrong_action_rate"] == 0.0
    assert summary["clarification_rate"] == 0.5
    assert summary["confirmation_rate"] == 0.5


def test_evaluate_policy_records_single_best_executes_without_gate() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn", "sean"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call sean", "score": -0.1}, {"text": "call shawn", "score": -0.2}],
        }
    ]
    summary = evaluate_records(records, policy_mode="single_best_transcript_agent")
    assert summary["policy_mode"] == "single_best_transcript_agent"
    assert summary["deferral_rate"] == 0.0
    assert summary["unsafe_wrong_action_rate"] == 1.0


def test_evaluate_policy_records_ask_all_defers_entity_bearing_commands() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "john",
            "entity_terms": ["john"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call john", "score": -0.1}],
        }
    ]
    summary = evaluate_records(records, policy_mode="ask_every_entity_bearing_command")
    assert summary["clarification_rate"] == 1.0
    assert summary["task_success"] == 1.0


def test_evaluate_policy_records_handles_fsc_slot_command_correctness() -> None:
    records = [
        {
            "dataset": "fluent_speech_commands",
            "action": "change language",
            "object": "none",
            "location": "none",
            "reference": "Change language",
            "hypotheses": [{"text": " Change language.", "score": -0.1}],
        },
        {
            "dataset": "fluent_speech_commands",
            "action": "activate lights",
            "object": "lights",
            "location": "kitchen",
            "reference": "Turn on the kitchen lights",
            "hypotheses": [{"text": " Turn off the kitchen lights.", "score": -0.1}],
        },
    ]

    direct = evaluate_records(records, policy_mode="single_best_transcript_agent")
    ask_all = evaluate_records(records, policy_mode="ask_every_entity_bearing_command")

    assert direct["task_success"] == 0.5
    assert direct["wrong_action_rate"] == 0.5
    assert ask_all["clarification_rate"] == 1.0
    assert ask_all["task_success"] == 1.0


def test_evaluate_policy_records_handles_librispeech_transcript_correctness() -> None:
    records = [
        {
            "dataset": "openslr/librispeech_asr",
            "reference": "CONCORD RETURNED TO ITS PLACE",
            "hypotheses": [{"text": "Concord returned to its place.", "score": -0.1}],
        },
        {
            "dataset": "openslr/librispeech_asr",
            "reference": "THE QUICK BROWN FOX",
            "hypotheses": [{"text": "The slow brown fox.", "score": -0.1}],
        },
    ]

    summary = evaluate_records(records, policy_mode="single_best_transcript_agent")

    assert summary["task_success"] == 0.5
    assert summary["wrong_action_rate"] == 0.5


def test_evaluate_policy_records_scalar_confidence_threshold_defers_low_confidence() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "john",
            "entity_terms": ["john"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call john", "confidence": 0.4}],
        },
        {
            "action": "call_contact",
            "true_entity": "jane",
            "entity_terms": ["jane"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call jane", "confidence": 0.9}],
        },
    ]
    summary = evaluate_records(records, policy_mode="scalar_asr_confidence_threshold", scalar_confidence_threshold=0.7)
    assert summary["clarification_rate"] == 0.5
    assert summary["wrong_action_rate"] == 0.0


def test_fused_asr_confidence_threshold_defers_high_risk_with_high_scalar_confidence() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn", "sean"],
            "unsafe_if_confused": True,
            "hypotheses": [
                {"text": "call sean", "confidence": 0.99, "score": -0.01},
                {"text": "call shawn", "confidence": 0.80, "score": -0.30},
            ],
        }
    ]

    scalar = evaluate_records(records, policy_mode="scalar_asr_confidence_threshold", scalar_confidence_threshold=0.5)
    fused = evaluate_records(records, policy_mode="fused_asr_confidence_threshold", fused_confidence_threshold=0.5)

    assert scalar["unsafe_wrong_action_rate"] == 1.0
    assert scalar["deferral_rate"] == 0.0
    assert fused["unsafe_wrong_action_rate"] == 0.0
    assert fused["deferral_rate"] == 1.0


def test_evaluate_policy_records_slot_entity_confidence_prefers_slot_score() -> None:
    records = [
        {
            "action": "transfer_money",
            "true_entity": "account four",
            "entity_terms": ["account four"],
            "entity_confidence": 0.3,
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "transfer to account four", "confidence": 0.95}],
        }
    ]
    summary = evaluate_records(
        records,
        policy_mode="slot_entity_confidence_threshold",
        slot_entity_confidence_threshold=0.7,
    )
    assert summary["clarification_rate"] == 1.0
    assert summary["unsafe_wrong_action_rate"] == 0.0


def test_evaluate_policy_records_nbest_self_consistency_defers_disagreement() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn", "sean"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call shawn", "score": -0.1}, {"text": "call sean", "score": -0.2}],
        }
    ]
    summary = evaluate_records(records, policy_mode="nbest_entropy_self_consistency")
    assert summary["clarification_rate"] == 1.0


def test_evaluate_policy_records_oracle_clarifies_wrong_top_entity_only() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn", "sean"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call sean", "score": -0.1}],
        },
        {
            "action": "call_contact",
            "true_entity": "jane",
            "entity_terms": ["jane"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call jane", "score": -0.1}],
        },
    ]
    summary = evaluate_records(records, policy_mode="oracle_clarification_trigger")
    assert summary["clarification_rate"] == 0.5
    assert summary["unsafe_wrong_action_rate"] == 0.0


def test_evaluate_policy_records_cause_aware_clarifies_phonetic_ambiguity() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn", "sean"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call shawn", "score": -0.1}, {"text": "call shawn", "score": -0.2}],
        }
    ]
    summary = evaluate_records(records, policy_mode="cause_aware_asr_llm_clarification")
    assert summary["policy_mode"] == "cause_aware_asr_llm_clarification"
    assert summary["clarification_rate"] == 1.0


def test_action_conditioned_reflection_confirms_phonetic_risk_for_tool_calls() -> None:
    record = {
        "action": "call_contact",
        "true_entity": "shawn",
        "entity_terms": ["shawn", "sean"],
        "unsafe_if_confused": True,
        "learned_phonetic_risk": 0.9,
        "hypotheses": [{"text": "call shawn", "score": -0.1}, {"text": "call shawn", "score": -0.2}],
    }
    decision = decide_record(record, PulsePolicy(), policy_mode="reflection_over_transcript_alternatives")
    assert decision.action == AgentAction.CONFIRM
    assert "phonetically confusable risky action" in decision.reason


def test_action_conditioned_reflection_allows_non_action_answers_under_local_disagreement() -> None:
    record = {
        "action": "answer",
        "true_entity": "sean",
        "entity_terms": ["sean", "shawn"],
        "unsafe_if_confused": False,
        "hypotheses": [{"text": "sean is available", "score": -0.1}, {"text": "shawn is available", "score": -0.2}],
    }
    decision = decide_record(
        record,
        PulsePolicy(entity_risk_threshold=0.1, instability_threshold=0.1),
        policy_mode="reflection_over_transcript_alternatives",
    )
    assert decision.action == AgentAction.ANSWER
    assert "non-action answer" in decision.reason
