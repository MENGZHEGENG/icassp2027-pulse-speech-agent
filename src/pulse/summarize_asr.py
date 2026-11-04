from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from pulse.records import read_jsonl
from pulse.policy import AgentAction, PulsePolicy
from pulse.uncertainty import Hypothesis, edit_distance, normalize_text, summarize_hypotheses, token_distance_ratio


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def normalized_wer(reference: str, hypothesis: str) -> float:
    reference_tokens = normalize_text(reference).split()
    hypothesis_tokens = normalize_text(hypothesis).split()
    return edit_distance(reference_tokens, hypothesis_tokens) / max(1, len(reference_tokens))


def normalized_cer(reference: str, hypothesis: str) -> float:
    reference_chars = list(normalize_text(reference).replace(" ", ""))
    hypothesis_chars = list(normalize_text(hypothesis).replace(" ", ""))
    return edit_distance(reference_chars, hypothesis_chars) / max(1, len(reference_chars))


def summarize_records(records: list[dict], entity_terms: list[str] | None = None) -> dict:
    policy = PulsePolicy()
    wer_proxy_values: list[float] = []
    wer_values: list[float] = []
    cer_values: list[float] = []
    entropy_values: list[float] = []
    instability_values: list[float] = []
    entity_risk_values: list[float] = []
    phonetic_risk_values: list[float] = []
    clarify_count = 0

    for record in records:
        hypotheses = [Hypothesis(item["text"], item.get("score")) for item in record["hypotheses"]]
        record_entity_terms = entity_terms or record.get("entity_terms")
        summary = summarize_hypotheses(hypotheses, record_entity_terms)
        decision = policy.decide(summary, AgentAction.ANSWER)
        if decision.action == AgentAction.CLARIFY:
            clarify_count += 1
        reference = record.get("reference")
        if reference:
            wer_proxy_values.append(token_distance_ratio(reference, summary.top_text))
            wer_values.append(normalized_wer(reference, summary.top_text))
            cer_values.append(normalized_cer(reference, summary.top_text))
        if summary.entropy is not None:
            entropy_values.append(summary.entropy)
        instability_values.append(summary.instability)
        entity_risk_values.append(summary.entity_risk)
        phonetic_risk_values.append(summary.phonetic_risk)

    total = len(records)
    return {
        "count": total,
        "mean_wer": safe_mean(wer_values),
        "mean_cer": safe_mean(cer_values),
        "mean_wer_proxy": safe_mean(wer_proxy_values),
        "mean_entropy": safe_mean(entropy_values),
        "mean_instability": safe_mean(instability_values),
        "mean_entity_risk": safe_mean(entity_risk_values),
        "mean_phonetic_risk": safe_mean(phonetic_risk_values),
        "pulse_clarification_rate": clarify_count / total if total else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize ASR N-best records.")
    parser.add_argument("records_files", nargs="+", type=Path)
    parser.add_argument("--entity-term", action="append", default=[])
    parser.add_argument("--output", type=Path, default=Path("results/asr_summary.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = []
    for records_path in args.records_files:
        records.extend(read_jsonl(records_path))
    payload = summarize_records(records, args.entity_term)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
