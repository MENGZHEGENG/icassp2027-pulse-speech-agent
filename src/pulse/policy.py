from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pulse.uncertainty import UncertaintySummary


class AgentAction(StrEnum):
    ANSWER = "answer"
    RETRIEVE = "retrieve"
    CALL_TOOL = "call_tool"
    CONFIRM = "confirm"
    CLARIFY = "clarify"
    WAIT = "wait"


@dataclass(frozen=True)
class PulseDecision:
    action: AgentAction
    reason: str
    risk_score: float


@dataclass(frozen=True)
class PulsePolicy:
    entropy_threshold: float = 1.1
    instability_threshold: float = 0.28
    entity_risk_threshold: float = 0.75
    phonetic_risk_threshold: float = 0.75
    wait_instability_threshold: float = 0.45
    confirm_retrieval_phonetic_risk: bool = False

    def decide(
        self,
        summary: UncertaintySummary,
        requested_action: AgentAction = AgentAction.ANSWER,
        streaming: bool = False,
    ) -> PulseDecision:
        entropy_component = 0.0 if summary.entropy is None else summary.entropy / max(self.entropy_threshold, 1e-6)
        instability_component = summary.instability / max(self.instability_threshold, 1e-6)
        entity_component = summary.entity_risk / max(self.entity_risk_threshold, 1e-6)
        phonetic_component = summary.phonetic_risk / max(self.phonetic_risk_threshold, 1e-6)
        risk_score = max(entropy_component, instability_component, entity_component, phonetic_component)

        if streaming and summary.instability >= self.wait_instability_threshold:
            return PulseDecision(AgentAction.WAIT, "streaming hypothesis is still unstable", risk_score)
        if summary.entity_risk >= self.entity_risk_threshold:
            return PulseDecision(AgentAction.CLARIFY, "entity-bearing hypotheses disagree", risk_score)
        phonetic_threshold = self.phonetic_risk_threshold
        if requested_action == AgentAction.RETRIEVE and self.confirm_retrieval_phonetic_risk:
            phonetic_threshold = self.phonetic_risk_threshold
        elif requested_action != AgentAction.CALL_TOOL:
            phonetic_threshold = max(1.0, phonetic_threshold)
        if summary.phonetic_risk >= phonetic_threshold:
            if requested_action in {AgentAction.CALL_TOOL, AgentAction.RETRIEVE} and summary.phonetic_risk < 1.0:
                return PulseDecision(AgentAction.CONFIRM, "confirm phonetically confusable entity", risk_score)
            return PulseDecision(AgentAction.CLARIFY, "phonetic entity alternatives disagree", risk_score)
        if summary.instability >= self.instability_threshold:
            return PulseDecision(AgentAction.CLARIFY, "n-best hypotheses disagree", risk_score)
        if summary.entropy is not None and summary.entropy >= self.entropy_threshold:
            return PulseDecision(AgentAction.CLARIFY, "sequence score entropy is high", risk_score)
        return PulseDecision(requested_action, "uncertainty below policy thresholds", risk_score)
