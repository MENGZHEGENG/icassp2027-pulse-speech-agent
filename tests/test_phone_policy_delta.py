from pulse.phone_policy_delta import analyze_records


def test_phone_policy_delta_counts_prevented_wrong_action() -> None:
    records = [
        {
            "id": "unsafe",
            "action": "call_contact",
            "true_entity": "alice",
            "entity_terms": ["alice", "bob"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call bob", "score": -0.1}, {"text": "call bob", "score": -0.2}],
            "learned_phonetic_risk": 1.0,
            "phone_evidence": {"predicted_entity": "bob", "risk_reason": "ambiguous"},
        }
    ]
    summary, examples = analyze_records(records)
    assert summary["without_phone_wrong_action"] == 1
    assert summary["with_phone_wrong_action"] == 0
    assert summary["unsafe_wrong_actions_prevented"] == 1
    assert summary["guarded_commits"] == 1
    assert summary["wrong_actions_prevented_by_guard"] == 1
    assert summary["guard_wrong_action_rate"] == 0.0
    assert summary["transition_counts"] == {"commit->defer": 1}
    assert summary["guard_transition_counts"] == {"commit->defer": 1}
    assert examples[0]["id"] == "unsafe"


def test_phone_policy_delta_keeps_unchanged_deferral() -> None:
    records = [
        {
            "id": "already-risky",
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn", "sean"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call sean", "score": -0.1}, {"text": "call shawn", "score": -0.2}],
            "learned_phonetic_risk": 1.0,
        }
    ]
    summary, examples = analyze_records(records)
    assert summary["changed_decisions"] == 0
    assert summary["transition_counts"] == {"defer->defer": 1}
    assert summary["guarded_commits"] == 0
    assert summary["guard_transition_counts"] == {"defer->defer": 1}
    assert examples == []
