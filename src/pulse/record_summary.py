from __future__ import annotations

from dataclasses import replace
from typing import Any

from pulse.uncertainty import Hypothesis, UncertaintySummary, summarize_hypotheses


PHONE_RISK_KEYS = ("learned_phonetic_risk", "phone_risk", "forced_alignment_risk")


def parse_external_phone_risk(record: dict[str, Any]) -> float | None:
    for key in PHONE_RISK_KEYS:
        if record.get(key) is not None:
            return clamp_risk(record[key])
    phone_evidence = record.get("phone_evidence")
    if isinstance(phone_evidence, dict):
        for key in PHONE_RISK_KEYS + ("risk",):
            if phone_evidence.get(key) is not None:
                return clamp_risk(phone_evidence[key])
    return None


def clamp_risk(value: Any) -> float:
    risk = float(value)
    if risk < 0.0 or risk > 1.0:
        raise ValueError(f"phone risk must be in [0, 1], got {risk}")
    return risk


def hypotheses_from_record(record: dict[str, Any]) -> list[Hypothesis]:
    return [Hypothesis(item["text"], item.get("score")) for item in record["hypotheses"]]


def summarize_record(record: dict[str, Any], *, use_external_phone_risk: bool = True) -> UncertaintySummary:
    summary = summarize_hypotheses(hypotheses_from_record(record), record.get("entity_terms"))
    if not use_external_phone_risk:
        return summary
    external_risk = parse_external_phone_risk(record)
    if external_risk is None:
        return summary
    return replace(summary, phonetic_risk=max(summary.phonetic_risk, external_risk))
