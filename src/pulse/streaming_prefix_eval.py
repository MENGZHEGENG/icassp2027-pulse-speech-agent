from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from pulse.evaluate_policy_records import requested_agent_action
from pulse.records import read_jsonl, write_jsonl
from pulse.policy import AgentAction, PulseDecision, PulsePolicy
from pulse.uncertainty import (
    Hypothesis,
    arpabet_phrase_key,
    contains_arpabet_phrase,
    contains_phonetic_phrase,
    normalize_text,
    phonetic_phrase_key,
    summarize_hypotheses,
)

DEFER_ACTIONS = {AgentAction.CLARIFY, AgentAction.CONFIRM}


def prefix_text(text: str, word_count: int) -> str:
    return " ".join(text.split()[:word_count])


def max_hypothesis_words(record: dict[str, Any], cap: int | None = None) -> int:
    hypotheses = record.get("hypotheses") or []
    max_words = max((len(str(item.get("text", "")).split()) for item in hypotheses), default=1)
    if cap is not None:
        max_words = min(max_words, cap)
    return max(1, max_words)


def text_contains_entity(text: str, entity: str | None) -> bool:
    if not entity:
        return False
    normalized_entity = normalize_text(entity)
    normalized_text = normalize_text(text)
    return bool(normalized_entity and normalized_entity in normalized_text)


def text_contains_entity_evidence(text: str, entity: str | None) -> bool:
    if not entity:
        return False
    normalized_entity = normalize_text(entity)
    normalized_text = normalize_text(text)
    if normalized_entity and normalized_entity in normalized_text:
        return True
    phonetic_key = phonetic_phrase_key(entity)
    arpabet_key = arpabet_phrase_key(entity)
    return contains_phonetic_phrase(text, phonetic_key) or contains_arpabet_phrase(text, arpabet_key)


def prefix_contains_any_entity(text: str, entity_terms: list[str] | None) -> bool:
    return any(text_contains_entity_evidence(text, entity) for entity in entity_terms or [])


def build_prefix_hypotheses(record: dict[str, Any], word_count: int, keep_scores: bool = False) -> list[Hypothesis]:
    hypotheses: list[Hypothesis] = []
    for item in record.get("hypotheses") or []:
        text = prefix_text(str(item.get("text", "")), word_count)
        score = item.get("score") if keep_scores else None
        hypotheses.append(Hypothesis(text=text, score=score))
    if not hypotheses:
        hypotheses.append(Hypothesis(text="", score=None))
    return hypotheses


def force_wait_decision(reason: str) -> PulseDecision:
    return PulseDecision(action=AgentAction.WAIT, reason=reason, risk_score=0.0)


def streaming_decision_from_hypotheses(
    record: dict[str, Any],
    hypotheses: list[Hypothesis],
    *,
    is_final_prefix: bool,
    policy: PulsePolicy,
    min_prefix_words: int = 2,
    require_entity_before_terminal: bool = True,
) -> tuple[PulseDecision, list[Hypothesis]]:
    top_prefix = hypotheses[0].text
    prefix_word_count = len(top_prefix.split())
    if prefix_word_count < min_prefix_words and not is_final_prefix:
        return force_wait_decision("prefix shorter than minimum decision window"), hypotheses
    if require_entity_before_terminal and not is_final_prefix and not prefix_contains_any_entity(
        top_prefix, record.get("entity_terms")
    ):
        return force_wait_decision("no entity evidence in top prefix yet"), hypotheses
    summary = summarize_hypotheses(hypotheses, record.get("entity_terms"))
    return policy.decide(summary, requested_agent_action(record.get("action", "answer")), streaming=True), hypotheses


def streaming_decision_at_prefix(
    record: dict[str, Any],
    word_count: int,
    max_words: int,
    policy: PulsePolicy,
    keep_scores: bool = False,
    min_prefix_words: int = 2,
    require_entity_before_terminal: bool = True,
) -> tuple[PulseDecision, list[Hypothesis]]:
    return streaming_decision_from_hypotheses(
        record,
        build_prefix_hypotheses(record, word_count, keep_scores=keep_scores),
        is_final_prefix=word_count >= max_words,
        policy=policy,
        min_prefix_words=min_prefix_words,
        require_entity_before_terminal=require_entity_before_terminal,
    )


def hypotheses_from_prefix_entry(entry: dict[str, Any]) -> list[Hypothesis]:
    hypotheses = [Hypothesis(str(item.get("text", "")), item.get("score")) for item in entry.get("hypotheses") or []]
    return hypotheses or [Hypothesis(text="", score=None)]


def simulate_record(
    record: dict[str, Any],
    policy: PulsePolicy | None = None,
    keep_scores: bool = False,
    min_prefix_words: int = 2,
    require_entity_before_terminal: bool = True,
    max_prefix_words: int | None = None,
) -> dict[str, Any]:
    policy = policy or PulsePolicy()
    prefix_entries = list(record.get("prefix_hypotheses") or [])
    max_words = max_hypothesis_words(record, cap=max_prefix_words)
    requested_action = requested_agent_action(record.get("action", "answer"))
    history: list[dict[str, Any]] = []
    terminal: dict[str, Any] | None = None
    cumulative_decode_wall_time_seconds = 0.0
    serial_decision_ready_time_seconds: float | None = None

    if prefix_entries:
        iterator = []
        for prefix_index, entry in enumerate(prefix_entries, start=1):
            hypotheses = hypotheses_from_prefix_entry(entry)
            decision, hypotheses = streaming_decision_from_hypotheses(
                record,
                hypotheses,
                is_final_prefix=prefix_index >= len(prefix_entries),
                policy=policy,
                min_prefix_words=min_prefix_words,
                require_entity_before_terminal=require_entity_before_terminal,
            )
            top_prefix = hypotheses[0].text
            iterator.append(
                {
                    "prefix_index": prefix_index,
                    "word_count": len(top_prefix.split()),
                    "prefix_fraction": float(entry.get("prefix_fraction", prefix_index / len(prefix_entries))),
                    "prefix_duration_seconds": entry.get("prefix_duration_seconds"),
                    "decode_wall_time_seconds": entry.get("decode_wall_time_seconds"),
                    "decision": decision,
                    "top_prefix": top_prefix,
                    "is_final_prefix": prefix_index >= len(prefix_entries),
                }
            )
    else:
        iterator = []
        for word_count in range(1, max_words + 1):
            decision, hypotheses = streaming_decision_at_prefix(
                record,
                word_count,
                max_words,
                policy,
                keep_scores=keep_scores,
                min_prefix_words=min_prefix_words,
                require_entity_before_terminal=require_entity_before_terminal,
            )
            iterator.append(
                {
                    "prefix_index": word_count,
                    "word_count": word_count,
                    "prefix_fraction": word_count / max_words,
                    "prefix_duration_seconds": None,
                    "decode_wall_time_seconds": None,
                    "decision": decision,
                    "top_prefix": hypotheses[0].text,
                    "is_final_prefix": word_count >= max_words,
                }
            )

    for item in iterator:
        decision = item["decision"]
        prefix_duration_seconds = item["prefix_duration_seconds"]
        decode_wall_time_seconds = item["decode_wall_time_seconds"]
        event_cumulative_decode_wall_time_seconds = None
        event_serial_decision_ready_time_seconds = None
        if prefix_duration_seconds is not None and decode_wall_time_seconds is not None:
            cumulative_decode_wall_time_seconds += float(decode_wall_time_seconds)
            previous_ready_time = serial_decision_ready_time_seconds or 0.0
            serial_decision_ready_time_seconds = max(float(prefix_duration_seconds), previous_ready_time) + float(
                decode_wall_time_seconds
            )
            event_cumulative_decode_wall_time_seconds = cumulative_decode_wall_time_seconds
            event_serial_decision_ready_time_seconds = serial_decision_ready_time_seconds
        event = {
            "prefix_index": item["prefix_index"],
            "word_count": item["word_count"],
            "prefix_fraction": item["prefix_fraction"],
            "prefix_duration_seconds": prefix_duration_seconds,
            "decode_wall_time_seconds": decode_wall_time_seconds,
            "cumulative_decode_wall_time_seconds": event_cumulative_decode_wall_time_seconds,
            "serial_decision_ready_time_seconds": event_serial_decision_ready_time_seconds,
            "action": decision.action.value,
            "reason": decision.reason,
            "risk_score": decision.risk_score,
            "top_prefix": item["top_prefix"],
        }
        history.append(event)
        if terminal is None and decision.action != AgentAction.WAIT:
            terminal = event

    if terminal is None:
        terminal = history[-1]

    terminal_action = AgentAction(terminal["action"])
    terminal_text = str(terminal["top_prefix"])
    top_entity_correct_at_terminal = text_contains_entity(terminal_text, record.get("true_entity"))
    terminal_is_commit = terminal_action == requested_action
    terminal_is_defer = terminal_action in DEFER_ACTIONS
    unsafe_commit = bool(terminal_is_commit and not top_entity_correct_at_terminal and record.get("unsafe_if_confused"))
    worker_terminal = latest_prefix_worker_terminal(history)
    parallel_worker_terminal = latest_prefix_worker_terminal(history, worker_count=2)
    worker_action = AgentAction(worker_terminal["action"]) if worker_terminal else None
    worker_text = str(worker_terminal["top_prefix"]) if worker_terminal else ""
    worker_entity_correct = text_contains_entity(worker_text, record.get("true_entity")) if worker_terminal else False
    worker_is_commit = worker_action == requested_action if worker_action else False
    worker_is_defer = worker_action in DEFER_ACTIONS if worker_action else False
    worker_unsafe_commit = bool(worker_is_commit and not worker_entity_correct and record.get("unsafe_if_confused"))
    parallel_worker_action = AgentAction(parallel_worker_terminal["action"]) if parallel_worker_terminal else None
    parallel_worker_text = str(parallel_worker_terminal["top_prefix"]) if parallel_worker_terminal else ""
    parallel_worker_entity_correct = (
        text_contains_entity(parallel_worker_text, record.get("true_entity")) if parallel_worker_terminal else False
    )
    parallel_worker_is_commit = parallel_worker_action == requested_action if parallel_worker_action else False
    parallel_worker_is_defer = parallel_worker_action in DEFER_ACTIONS if parallel_worker_action else False
    parallel_worker_unsafe_commit = bool(
        parallel_worker_is_commit and not parallel_worker_entity_correct and record.get("unsafe_if_confused")
    )

    return {
        "id": record.get("id"),
        "action": record.get("action"),
        "true_entity": record.get("true_entity"),
        "max_words": max_words,
        "prefix_source": str(record.get("prefix_source") or ("asr_prefix" if prefix_entries else "final_asr_truncation")),
        "terminal_word_count": terminal["word_count"],
        "terminal_prefix_index": terminal["prefix_index"],
        "terminal_prefix_fraction": terminal["prefix_fraction"],
        "terminal_prefix_duration_seconds": terminal.get("prefix_duration_seconds"),
        "terminal_decode_wall_time_seconds": terminal.get("decode_wall_time_seconds"),
        "terminal_cumulative_decode_wall_time_seconds": terminal.get("cumulative_decode_wall_time_seconds"),
        "terminal_serial_decision_latency_seconds": terminal.get("serial_decision_ready_time_seconds"),
        "terminal_serial_backlog_seconds": serial_backlog_seconds(terminal),
        "terminal_action": terminal_action.value,
        "terminal_reason": terminal["reason"],
        "terminal_top_prefix": terminal_text,
        "terminal_is_commit": terminal_is_commit,
        "terminal_is_defer": terminal_is_defer,
        "top_entity_correct_at_terminal": top_entity_correct_at_terminal,
        "unsafe_commit": unsafe_commit,
        "latest_worker_terminal_prefix_index": worker_terminal.get("prefix_index") if worker_terminal else None,
        "latest_worker_terminal_prefix_fraction": worker_terminal.get("prefix_fraction") if worker_terminal else None,
        "latest_worker_terminal_prefix_duration_seconds": worker_terminal.get("prefix_duration_seconds")
        if worker_terminal
        else None,
        "latest_worker_decision_latency_seconds": worker_terminal.get("worker_decision_ready_time_seconds")
        if worker_terminal
        else None,
        "latest_worker_terminal_action": worker_action.value if worker_action else None,
        "latest_worker_terminal_is_commit": worker_is_commit,
        "latest_worker_terminal_is_defer": worker_is_defer,
        "latest_worker_top_entity_correct_at_terminal": worker_entity_correct,
        "latest_worker_unsafe_commit": worker_unsafe_commit,
        "latest_worker_decoded_prefix_count": worker_terminal.get("worker_decoded_prefix_count")
        if worker_terminal
        else None,
        "latest_worker_skipped_prefix_count": worker_terminal.get("worker_skipped_prefix_count")
        if worker_terminal
        else None,
        "parallel_worker_count": 2,
        "parallel_worker_terminal_prefix_index": parallel_worker_terminal.get("prefix_index")
        if parallel_worker_terminal
        else None,
        "parallel_worker_terminal_prefix_fraction": parallel_worker_terminal.get("prefix_fraction")
        if parallel_worker_terminal
        else None,
        "parallel_worker_terminal_prefix_duration_seconds": parallel_worker_terminal.get("prefix_duration_seconds")
        if parallel_worker_terminal
        else None,
        "parallel_worker_decision_latency_seconds": parallel_worker_terminal.get("worker_decision_ready_time_seconds")
        if parallel_worker_terminal
        else None,
        "parallel_worker_terminal_action": parallel_worker_action.value if parallel_worker_action else None,
        "parallel_worker_terminal_is_commit": parallel_worker_is_commit,
        "parallel_worker_terminal_is_defer": parallel_worker_is_defer,
        "parallel_worker_top_entity_correct_at_terminal": parallel_worker_entity_correct,
        "parallel_worker_unsafe_commit": parallel_worker_unsafe_commit,
        "parallel_worker_decoded_prefix_count": parallel_worker_terminal.get("worker_decoded_prefix_count")
        if parallel_worker_terminal
        else None,
        "parallel_worker_skipped_prefix_count": parallel_worker_terminal.get("worker_skipped_prefix_count")
        if parallel_worker_terminal
        else None,
        "waited_to_end": bool(history and terminal["prefix_index"] >= history[-1]["prefix_index"]),
        "history": history,
    }


def serial_backlog_seconds(terminal: dict[str, Any]) -> float | None:
    prefix_duration = terminal.get("prefix_duration_seconds")
    serial_ready_time = terminal.get("serial_decision_ready_time_seconds")
    if prefix_duration is None or serial_ready_time is None:
        return None
    return max(0.0, float(serial_ready_time) - float(prefix_duration))


def latest_prefix_worker_terminal(history: list[dict[str, Any]], worker_count: int = 1) -> dict[str, Any] | None:
    if not history:
        return None
    if worker_count < 1:
        raise ValueError("worker_count must be >= 1")
    if worker_count > 1:
        return bounded_latest_prefix_worker_terminal(history, worker_count)
    next_index = 0
    current_time = 0.0
    decoded_prefix_count = 0
    skipped_prefix_count = 0
    while next_index < len(history):
        available_indices = [
            index
            for index in range(next_index, len(history))
            if history[index].get("prefix_duration_seconds") is not None
            and float(history[index]["prefix_duration_seconds"]) <= current_time
        ]
        if available_indices:
            chosen_index = available_indices[-1]
            start_time = current_time
            skipped_prefix_count += chosen_index - next_index
        else:
            chosen_index = next_index
            prefix_duration = history[chosen_index].get("prefix_duration_seconds")
            if prefix_duration is None:
                return None
            start_time = max(current_time, float(prefix_duration))

        event = history[chosen_index]
        decode_wall_time = event.get("decode_wall_time_seconds")
        if decode_wall_time is None:
            return None
        decoded_prefix_count += 1
        current_time = start_time + float(decode_wall_time)
        next_index = chosen_index + 1
        worker_event = dict(event)
        worker_event.update(
            {
                "worker_start_time_seconds": start_time,
                "worker_decision_ready_time_seconds": current_time,
                "worker_decoded_prefix_count": decoded_prefix_count,
                "worker_skipped_prefix_count": skipped_prefix_count,
            }
        )
        if event["action"] != AgentAction.WAIT.value or chosen_index == len(history) - 1:
            return worker_event
    return None


def bounded_latest_prefix_worker_terminal(history: list[dict[str, Any]], worker_count: int) -> dict[str, Any] | None:
    next_index = 0
    decoded_prefix_count = 0
    skipped_prefix_count = 0
    worker_heap: list[tuple[float, int, int | None, float | None]] = [
        (0.0, worker_id, None, None) for worker_id in range(worker_count)
    ]
    heapq.heapify(worker_heap)
    while worker_heap and next_index < len(history):
        ready_time, worker_id, completed_index, start_time = heapq.heappop(worker_heap)
        if completed_index is not None:
            completed_event = history[completed_index]
            worker_event = dict(completed_event)
            worker_event.update(
                {
                    "worker_start_time_seconds": start_time,
                    "worker_decision_ready_time_seconds": ready_time,
                    "worker_decoded_prefix_count": decoded_prefix_count,
                    "worker_skipped_prefix_count": skipped_prefix_count,
                }
            )
            if completed_event["action"] != AgentAction.WAIT.value or completed_index == len(history) - 1:
                return worker_event

        available_indices = [
            index
            for index in range(next_index, len(history))
            if history[index].get("prefix_duration_seconds") is not None
            and float(history[index]["prefix_duration_seconds"]) <= ready_time
        ]
        if available_indices:
            chosen_index = available_indices[-1]
            job_start_time = ready_time
            skipped_prefix_count += chosen_index - next_index
        else:
            chosen_index = next_index
            prefix_duration = history[chosen_index].get("prefix_duration_seconds")
            if prefix_duration is None:
                return None
            job_start_time = max(ready_time, float(prefix_duration))

        event = history[chosen_index]
        decode_wall_time = event.get("decode_wall_time_seconds")
        if decode_wall_time is None:
            return None
        decoded_prefix_count += 1
        next_index = chosen_index + 1
        heapq.heappush(
            worker_heap,
            (job_start_time + float(decode_wall_time), worker_id, chosen_index, job_start_time),
        )
    while worker_heap:
        ready_time, worker_id, completed_index, start_time = heapq.heappop(worker_heap)
        if completed_index is None:
            continue
        completed_event = history[completed_index]
        worker_event = dict(completed_event)
        worker_event.update(
            {
                "worker_start_time_seconds": start_time,
                "worker_decision_ready_time_seconds": ready_time,
                "worker_decoded_prefix_count": decoded_prefix_count,
                "worker_skipped_prefix_count": skipped_prefix_count,
            }
        )
        if completed_event["action"] != AgentAction.WAIT.value or completed_index == len(history) - 1:
            return worker_event
    return None


def summarize_simulations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    terminal_fractions = [float(row["terminal_prefix_fraction"]) for row in rows]
    terminal_words = [int(row["terminal_word_count"]) for row in rows]
    terminal_durations = [
        float(row["terminal_prefix_duration_seconds"])
        for row in rows
        if row.get("terminal_prefix_duration_seconds") is not None
    ]
    terminal_decode_times = [
        float(row["terminal_decode_wall_time_seconds"])
        for row in rows
        if row.get("terminal_decode_wall_time_seconds") is not None
    ]
    terminal_latencies = [
        float(row["terminal_prefix_duration_seconds"]) + float(row["terminal_decode_wall_time_seconds"])
        for row in rows
        if row.get("terminal_prefix_duration_seconds") is not None
        and row.get("terminal_decode_wall_time_seconds") is not None
    ]
    terminal_real_time_factors = [
        float(row["terminal_decode_wall_time_seconds"]) / max(float(row["terminal_prefix_duration_seconds"]), 1e-9)
        for row in rows
        if row.get("terminal_prefix_duration_seconds") is not None
        and row.get("terminal_decode_wall_time_seconds") is not None
    ]
    terminal_cumulative_decode_times = [
        float(row["terminal_cumulative_decode_wall_time_seconds"])
        for row in rows
        if row.get("terminal_cumulative_decode_wall_time_seconds") is not None
    ]
    terminal_serial_latencies = [
        float(row["terminal_serial_decision_latency_seconds"])
        for row in rows
        if row.get("terminal_serial_decision_latency_seconds") is not None
    ]
    terminal_serial_backlogs = [
        float(row["terminal_serial_backlog_seconds"])
        for row in rows
        if row.get("terminal_serial_backlog_seconds") is not None
    ]
    latest_worker_fractions = [
        float(row["latest_worker_terminal_prefix_fraction"])
        for row in rows
        if row.get("latest_worker_terminal_prefix_fraction") is not None
    ]
    latest_worker_durations = [
        float(row["latest_worker_terminal_prefix_duration_seconds"])
        for row in rows
        if row.get("latest_worker_terminal_prefix_duration_seconds") is not None
    ]
    latest_worker_latencies = [
        float(row["latest_worker_decision_latency_seconds"])
        for row in rows
        if row.get("latest_worker_decision_latency_seconds") is not None
    ]
    latest_worker_decoded_counts = [
        int(row["latest_worker_decoded_prefix_count"])
        for row in rows
        if row.get("latest_worker_decoded_prefix_count") is not None
    ]
    latest_worker_skipped_counts = [
        int(row["latest_worker_skipped_prefix_count"])
        for row in rows
        if row.get("latest_worker_skipped_prefix_count") is not None
    ]
    parallel_worker_fractions = [
        float(row["parallel_worker_terminal_prefix_fraction"])
        for row in rows
        if row.get("parallel_worker_terminal_prefix_fraction") is not None
    ]
    parallel_worker_latencies = [
        float(row["parallel_worker_decision_latency_seconds"])
        for row in rows
        if row.get("parallel_worker_decision_latency_seconds") is not None
    ]
    parallel_worker_decoded_counts = [
        int(row["parallel_worker_decoded_prefix_count"])
        for row in rows
        if row.get("parallel_worker_decoded_prefix_count") is not None
    ]
    parallel_worker_skipped_counts = [
        int(row["parallel_worker_skipped_prefix_count"])
        for row in rows
        if row.get("parallel_worker_skipped_prefix_count") is not None
    ]
    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("prefix_source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
    terminal_action_counts = {action.value: 0 for action in AgentAction}
    for row in rows:
        terminal_action_counts[str(row["terminal_action"])] = terminal_action_counts.get(str(row["terminal_action"]), 0) + 1
    latest_worker_action_counts = {action.value: 0 for action in AgentAction}
    for row in rows:
        worker_action = row.get("latest_worker_terminal_action")
        if worker_action is not None:
            latest_worker_action_counts[str(worker_action)] = latest_worker_action_counts.get(str(worker_action), 0) + 1
    parallel_worker_action_counts = {action.value: 0 for action in AgentAction}
    for row in rows:
        worker_action = row.get("parallel_worker_terminal_action")
        if worker_action is not None:
            parallel_worker_action_counts[str(worker_action)] = parallel_worker_action_counts.get(str(worker_action), 0) + 1

    def rate(predicate: str) -> float:
        return sum(1 for row in rows if row[predicate]) / total if total else 0.0

    if source_counts == {"asr_prefix": total}:
        note = "Real audio-prefix ASR: hypotheses are decoded from truncated audio prefixes."
    elif source_counts == {"asr_token_timestamp": total}:
        note = "Token-timestamp streaming proxy: prefixes follow Whisper token emission timestamps from a full-audio decode."
    elif source_counts == {"final_asr_truncation": total}:
        note = "Final-ASR prefix proxy: hypotheses are truncated from completed n-best outputs, not re-decoded streaming audio."
    else:
        note = "Mixed prefix sources: rows may combine real audio-prefix ASR and final-ASR text truncation."

    return {
        "count": total,
        "prefix_source_counts": source_counts,
        "mean_terminal_prefix_fraction": mean(terminal_fractions) if terminal_fractions else 0.0,
        "median_terminal_prefix_fraction": median(terminal_fractions) if terminal_fractions else 0.0,
        "mean_terminal_word_count": mean(terminal_words) if terminal_words else 0.0,
        "median_terminal_word_count": median(terminal_words) if terminal_words else 0.0,
        "mean_terminal_prefix_duration_seconds": mean(terminal_durations) if terminal_durations else None,
        "median_terminal_prefix_duration_seconds": median(terminal_durations) if terminal_durations else None,
        "mean_terminal_decode_wall_time_seconds": mean(terminal_decode_times) if terminal_decode_times else None,
        "median_terminal_decode_wall_time_seconds": median(terminal_decode_times) if terminal_decode_times else None,
        "mean_estimated_terminal_latency_seconds": mean(terminal_latencies) if terminal_latencies else None,
        "median_estimated_terminal_latency_seconds": median(terminal_latencies) if terminal_latencies else None,
        "mean_decode_real_time_factor": mean(terminal_real_time_factors) if terminal_real_time_factors else None,
        "median_decode_real_time_factor": median(terminal_real_time_factors) if terminal_real_time_factors else None,
        "mean_terminal_cumulative_decode_wall_time_seconds": mean(terminal_cumulative_decode_times)
        if terminal_cumulative_decode_times
        else None,
        "median_terminal_cumulative_decode_wall_time_seconds": median(terminal_cumulative_decode_times)
        if terminal_cumulative_decode_times
        else None,
        "mean_terminal_serial_decision_latency_seconds": mean(terminal_serial_latencies)
        if terminal_serial_latencies
        else None,
        "median_terminal_serial_decision_latency_seconds": median(terminal_serial_latencies)
        if terminal_serial_latencies
        else None,
        "mean_terminal_serial_backlog_seconds": mean(terminal_serial_backlogs) if terminal_serial_backlogs else None,
        "median_terminal_serial_backlog_seconds": median(terminal_serial_backlogs) if terminal_serial_backlogs else None,
        "mean_latest_worker_terminal_prefix_fraction": mean(latest_worker_fractions)
        if latest_worker_fractions
        else None,
        "median_latest_worker_terminal_prefix_fraction": median(latest_worker_fractions)
        if latest_worker_fractions
        else None,
        "mean_latest_worker_terminal_prefix_duration_seconds": mean(latest_worker_durations)
        if latest_worker_durations
        else None,
        "median_latest_worker_terminal_prefix_duration_seconds": median(latest_worker_durations)
        if latest_worker_durations
        else None,
        "mean_latest_worker_decision_latency_seconds": mean(latest_worker_latencies)
        if latest_worker_latencies
        else None,
        "median_latest_worker_decision_latency_seconds": median(latest_worker_latencies)
        if latest_worker_latencies
        else None,
        "mean_latest_worker_decoded_prefix_count": mean(latest_worker_decoded_counts)
        if latest_worker_decoded_counts
        else None,
        "median_latest_worker_decoded_prefix_count": median(latest_worker_decoded_counts)
        if latest_worker_decoded_counts
        else None,
        "mean_latest_worker_skipped_prefix_count": mean(latest_worker_skipped_counts)
        if latest_worker_skipped_counts
        else None,
        "median_latest_worker_skipped_prefix_count": median(latest_worker_skipped_counts)
        if latest_worker_skipped_counts
        else None,
        "parallel_worker_count": 2,
        "mean_parallel_worker_terminal_prefix_fraction": mean(parallel_worker_fractions)
        if parallel_worker_fractions
        else None,
        "median_parallel_worker_terminal_prefix_fraction": median(parallel_worker_fractions)
        if parallel_worker_fractions
        else None,
        "mean_parallel_worker_decision_latency_seconds": mean(parallel_worker_latencies)
        if parallel_worker_latencies
        else None,
        "median_parallel_worker_decision_latency_seconds": median(parallel_worker_latencies)
        if parallel_worker_latencies
        else None,
        "mean_parallel_worker_decoded_prefix_count": mean(parallel_worker_decoded_counts)
        if parallel_worker_decoded_counts
        else None,
        "median_parallel_worker_decoded_prefix_count": median(parallel_worker_decoded_counts)
        if parallel_worker_decoded_counts
        else None,
        "mean_parallel_worker_skipped_prefix_count": mean(parallel_worker_skipped_counts)
        if parallel_worker_skipped_counts
        else None,
        "median_parallel_worker_skipped_prefix_count": median(parallel_worker_skipped_counts)
        if parallel_worker_skipped_counts
        else None,
        "waited_to_end_rate": rate("waited_to_end"),
        "terminal_commit_rate": rate("terminal_is_commit"),
        "terminal_defer_rate": rate("terminal_is_defer"),
        "unsafe_commit_rate": rate("unsafe_commit"),
        "latest_worker_terminal_commit_rate": rate("latest_worker_terminal_is_commit"),
        "latest_worker_terminal_defer_rate": rate("latest_worker_terminal_is_defer"),
        "latest_worker_unsafe_commit_rate": rate("latest_worker_unsafe_commit"),
        "parallel_worker_terminal_commit_rate": rate("parallel_worker_terminal_is_commit"),
        "parallel_worker_terminal_defer_rate": rate("parallel_worker_terminal_is_defer"),
        "parallel_worker_unsafe_commit_rate": rate("parallel_worker_unsafe_commit"),
        "correct_commit_rate": sum(
            1 for row in rows if row["terminal_is_commit"] and row["top_entity_correct_at_terminal"]
        )
        / total
        if total
        else 0.0,
        "terminal_action_counts": terminal_action_counts,
        "latest_worker_terminal_action_counts": latest_worker_action_counts,
        "parallel_worker_terminal_action_counts": parallel_worker_action_counts,
        "note": note,
    }


def markdown_report(summary: dict[str, Any]) -> str:
    action_counts = summary["terminal_action_counts"]
    lines = [
        "# Streaming Prefix Evaluation",
        "",
        summary["note"],
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Records | {summary['count']} |",
        f"| Mean terminal prefix fraction | {summary['mean_terminal_prefix_fraction']:.3f} |",
        f"| Median terminal prefix fraction | {summary['median_terminal_prefix_fraction']:.3f} |",
        f"| Mean terminal word count | {summary['mean_terminal_word_count']:.2f} |",
        f"| Mean terminal prefix duration (s) | {format_optional_float(summary['mean_terminal_prefix_duration_seconds'])} |",
        f"| Mean terminal decode wall time (s) | {format_optional_float(summary['mean_terminal_decode_wall_time_seconds'])} |",
        f"| Mean estimated terminal latency (s) | {format_optional_float(summary['mean_estimated_terminal_latency_seconds'])} |",
        f"| Mean serial online latency (s) | {format_optional_float(summary['mean_terminal_serial_decision_latency_seconds'])} |",
        f"| Mean cumulative decode time (s) | {format_optional_float(summary['mean_terminal_cumulative_decode_wall_time_seconds'])} |",
        f"| Mean serial backlog (s) | {format_optional_float(summary['mean_terminal_serial_backlog_seconds'])} |",
        f"| Mean latest-worker prefix fraction | {format_optional_float(summary['mean_latest_worker_terminal_prefix_fraction'])} |",
        f"| Mean latest-worker latency (s) | {format_optional_float(summary['mean_latest_worker_decision_latency_seconds'])} |",
        f"| Mean latest-worker decoded prefixes | {format_optional_float(summary['mean_latest_worker_decoded_prefix_count'])} |",
        f"| Latest-worker unsafe commit rate | {summary['latest_worker_unsafe_commit_rate']:.3f} |",
        f"| Mean two-worker prefix fraction | {format_optional_float(summary['mean_parallel_worker_terminal_prefix_fraction'])} |",
        f"| Mean two-worker latency (s) | {format_optional_float(summary['mean_parallel_worker_decision_latency_seconds'])} |",
        f"| Mean two-worker decoded prefixes | {format_optional_float(summary['mean_parallel_worker_decoded_prefix_count'])} |",
        f"| Two-worker unsafe commit rate | {summary['parallel_worker_unsafe_commit_rate']:.3f} |",
        f"| Mean decode real-time factor | {format_optional_float(summary['mean_decode_real_time_factor'])} |",
        f"| Waited to final prefix | {summary['waited_to_end_rate']:.3f} |",
        f"| Terminal commit rate | {summary['terminal_commit_rate']:.3f} |",
        f"| Terminal defer rate | {summary['terminal_defer_rate']:.3f} |",
        f"| Unsafe commit rate | {summary['unsafe_commit_rate']:.3f} |",
        "",
        "| Terminal action | Count |",
        "|---|---:|",
    ]
    for action, count in action_counts.items():
        if count:
            lines.append(f"| {action} | {count} |")
    return "\n".join(lines) + "\n"


def format_optional_float(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a conservative streaming-prefix proxy by truncating final ASR n-best hypotheses."
    )
    parser.add_argument("records", type=Path)
    parser.add_argument("--output-json", type=Path, default=Path("results/streaming_prefix/summary.json"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("results/streaming_prefix/per_record.jsonl"))
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--entity-threshold", type=float, default=0.75)
    parser.add_argument("--phonetic-threshold", type=float, default=0.75)
    parser.add_argument("--instability-threshold", type=float, default=0.28)
    parser.add_argument("--entropy-threshold", type=float, default=1.1)
    parser.add_argument("--wait-instability-threshold", type=float, default=0.45)
    parser.add_argument("--confirm-retrieval-risk", action="store_true")
    parser.add_argument("--keep-scores", action="store_true", help="Reuse final sequence scores for prefixes; off by default.")
    parser.add_argument("--min-prefix-words", type=int, default=2)
    parser.add_argument("--max-prefix-words", type=int)
    parser.add_argument(
        "--allow-terminal-without-entity",
        action="store_true",
        help="Allow non-wait decisions before any entity evidence appears in the top prefix.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = PulsePolicy(
        entity_risk_threshold=args.entity_threshold,
        phonetic_risk_threshold=args.phonetic_threshold,
        instability_threshold=args.instability_threshold,
        entropy_threshold=args.entropy_threshold,
        wait_instability_threshold=args.wait_instability_threshold,
        confirm_retrieval_phonetic_risk=args.confirm_retrieval_risk,
    )
    rows = [
        simulate_record(
            record,
            policy=policy,
            keep_scores=args.keep_scores,
            min_prefix_words=args.min_prefix_words,
            require_entity_before_terminal=not args.allow_terminal_without_entity,
            max_prefix_words=args.max_prefix_words,
        )
        for record in read_jsonl(args.records)
    ]
    summary = summarize_simulations(rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, rows)
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown_report(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
