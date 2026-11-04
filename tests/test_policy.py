from pulse.policy import AgentAction, PulsePolicy
from pulse.uncertainty import Hypothesis, summarize_hypotheses


def test_policy_clarifies_entity_disagreement() -> None:
    hypotheses = [Hypothesis("email shawn", -0.1), Hypothesis("email sean", -0.2)]
    summary = summarize_hypotheses(hypotheses, ["shawn", "sean"])
    decision = PulsePolicy(entity_risk_threshold=0.1).decide(summary, AgentAction.CALL_TOOL)
    assert decision.action == AgentAction.CLARIFY


def test_policy_keeps_requested_action_for_stable_hypotheses() -> None:
    hypotheses = [Hypothesis("play the meeting recording", -0.1), Hypothesis("play the meeting recording", -0.2)]
    summary = summarize_hypotheses(hypotheses)
    decision = PulsePolicy().decide(summary, AgentAction.RETRIEVE)
    assert decision.action == AgentAction.RETRIEVE


def test_policy_confirms_static_phonetic_tool_risk() -> None:
    hypotheses = [Hypothesis("call shon", -0.1), Hypothesis("call shaun", -0.2)]
    summary = summarize_hypotheses(hypotheses, ["shawn", "sean"])
    decision = PulsePolicy(phonetic_risk_threshold=0.75).decide(summary, AgentAction.CALL_TOOL)
    assert decision.action == AgentAction.CONFIRM
    assert "confirm" in decision.reason


def test_policy_clarifies_missing_phonetic_tool_candidate() -> None:
    hypotheses = [Hypothesis("call kiruk", -0.1), Hypothesis("call kiruk", -0.2)]
    summary = summarize_hypotheses(hypotheses, ["q route", "queue route"])
    decision = PulsePolicy(phonetic_risk_threshold=0.75).decide(summary, AgentAction.CALL_TOOL)
    assert decision.action == AgentAction.CLARIFY


def test_policy_allows_retrieval_for_moderate_static_phonetic_risk() -> None:
    hypotheses = [Hypothesis("find the file named sean", -0.1), Hypothesis("find the file named sean", -0.2)]
    summary = summarize_hypotheses(hypotheses, ["shawn", "sean"])
    decision = PulsePolicy(phonetic_risk_threshold=0.75).decide(summary, AgentAction.RETRIEVE)
    assert decision.action == AgentAction.RETRIEVE


def test_policy_can_confirm_retrieval_for_moderate_static_phonetic_risk() -> None:
    hypotheses = [Hypothesis("find the file named sean", -0.1), Hypothesis("find the file named sean", -0.2)]
    summary = summarize_hypotheses(hypotheses, ["shawn", "sean"])
    decision = PulsePolicy(phonetic_risk_threshold=0.75, confirm_retrieval_phonetic_risk=True).decide(
        summary,
        AgentAction.RETRIEVE,
    )
    assert decision.action == AgentAction.CONFIRM


def test_policy_clarifies_retrieval_when_candidate_entity_missing() -> None:
    hypotheses = [Hypothesis("find the file named kiruk", -0.1), Hypothesis("find the file named kiruk", -0.2)]
    summary = summarize_hypotheses(hypotheses, ["q route", "queue route"])
    decision = PulsePolicy(phonetic_risk_threshold=0.75).decide(summary, AgentAction.RETRIEVE)
    assert decision.action == AgentAction.CLARIFY
