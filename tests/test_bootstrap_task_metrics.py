from pulse.bootstrap_task_metrics import bootstrap_recordwise_metric_summary, build_evaluator
from pulse.policy import PulsePolicy


def test_build_evaluator_supports_phone_commit_guard() -> None:
    records = [
        {
            "action": "retrieve_file",
            "true_entity": "alice",
            "entity_terms": ["alice", "bob"],
            "unsafe_if_confused": False,
            "learned_phonetic_risk": 0.8,
            "hypotheses": [{"text": "find bob", "score": -0.1}, {"text": "find bob", "score": -0.2}],
        }
    ]
    evaluator = build_evaluator(
        "policy",
        PulsePolicy(),
        0.6,
        phone_risk_mode="commit_guard",
        phone_guard_threshold=0.75,
    )
    summary = evaluator(records)
    assert summary["wrong_action_rate"] == 0.0
    assert summary["clarification_rate"] == 1.0


def test_bootstrap_recordwise_summary_keeps_phone_guard_observed() -> None:
    records = [
        {
            "action": "retrieve_file",
            "true_entity": "alice",
            "entity_terms": ["alice", "bob"],
            "unsafe_if_confused": False,
            "learned_phonetic_risk": 0.8,
            "hypotheses": [{"text": "find bob", "score": -0.1}, {"text": "find bob", "score": -0.2}],
        }
    ]
    evaluator = build_evaluator("policy", PulsePolicy(), 0.6, phone_risk_mode="commit_guard", phone_guard_threshold=0.75)
    summary = bootstrap_recordwise_metric_summary(records, evaluator, trials=5, seed=1, metrics=["wrong_action_rate"])
    assert summary["wrong_action_rate"]["observed"] == 0.0
    assert summary["wrong_action_rate"]["hi"] == 0.0
