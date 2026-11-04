import json

from pulse.policy import AgentAction, PulsePolicy
from pulse.streaming_prefix_eval import (
    latest_prefix_worker_terminal,
    main,
    markdown_report,
    simulate_record,
    summarize_simulations,
    text_contains_entity,
    text_contains_entity_evidence,
)


def record(top_text: str, true_entity: str = "sean", unsafe: bool = True) -> dict:
    return {
        "id": "toy-1",
        "reference": f"call {true_entity} now",
        "action": "call_contact",
        "true_entity": true_entity,
        "entity_terms": ["sean", "shawn"],
        "unsafe_if_confused": unsafe,
        "hypotheses": [
            {"text": top_text, "score": -0.1},
            {"text": top_text, "score": -0.3},
        ],
    }


def test_entity_correctness_is_lexical_but_evidence_can_be_phonetic() -> None:
    assert text_contains_entity("call sean now", "sean")
    assert not text_contains_entity("call shawn now", "sean")
    assert text_contains_entity_evidence("call shawn now", "sean")


def test_simulate_record_waits_until_entity_window_before_terminal_action() -> None:
    row = simulate_record(record("please call sean now"), min_prefix_words=2)
    assert row["terminal_action"] in {AgentAction.CLARIFY.value, AgentAction.CONFIRM.value, AgentAction.CALL_TOOL.value}
    assert row["terminal_word_count"] == 3
    assert row["terminal_prefix_fraction"] == 0.75
    assert not row["unsafe_commit"]
    assert row["history"][0]["action"] == AgentAction.WAIT.value


def test_simulate_record_flags_unsafe_wrong_commit_when_thresholds_are_relaxed() -> None:
    permissive = PulsePolicy(entity_risk_threshold=2.0, phonetic_risk_threshold=2.0, instability_threshold=2.0)
    row = simulate_record(
        record("please call jane now"),
        policy=permissive,
        require_entity_before_terminal=False,
    )
    assert row["terminal_action"] == AgentAction.CALL_TOOL.value
    assert row["terminal_is_commit"]
    assert row["unsafe_commit"]


def test_summarize_and_markdown_report_include_latency_and_safety_rates() -> None:
    rows = [simulate_record(record("please call sean now")), simulate_record(record("please call jane now"))]
    summary = summarize_simulations(rows)
    assert summary["count"] == 2
    assert 0.0 <= summary["mean_terminal_prefix_fraction"] <= 1.0
    assert "unsafe_commit_rate" in summary
    report = markdown_report(summary)
    assert "Streaming Prefix Evaluation" in report
    assert "Mean serial online latency" in report
    assert "Mean two-worker latency" in report
    assert "Unsafe commit rate" in report


def test_cli_writes_summary_and_per_record_outputs(tmp_path, monkeypatch) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps(record("please call sean now")) + "\n", encoding="utf-8")
    output_json = tmp_path / "summary.json"
    output_jsonl = tmp_path / "rows.jsonl"
    output_md = tmp_path / "summary.md"
    monkeypatch.setattr(
        "sys.argv",
        [
            "streaming_prefix_eval",
            str(records),
            "--output-json",
            str(output_json),
            "--output-jsonl",
            str(output_jsonl),
            "--output-markdown",
            str(output_md),
        ],
    )
    main()
    assert json.loads(output_json.read_text(encoding="utf-8"))["count"] == 1
    assert len(output_jsonl.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert "Streaming Prefix Evaluation" in output_md.read_text(encoding="utf-8")


def test_simulate_record_uses_real_prefix_hypotheses_when_present() -> None:
    row = simulate_record(
        {
            **record("please call sean now"),
            "prefix_hypotheses": [
                {
                    "prefix_fraction": 0.25,
                    "prefix_duration_seconds": 0.4,
                    "decode_wall_time_seconds": 0.1,
                    "hypotheses": [{"text": "please"}],
                },
                {
                    "prefix_fraction": 0.50,
                    "prefix_duration_seconds": 0.8,
                    "decode_wall_time_seconds": 0.2,
                    "hypotheses": [{"text": "please call"}],
                },
                {
                    "prefix_fraction": 0.75,
                    "prefix_duration_seconds": 1.2,
                    "decode_wall_time_seconds": 0.3,
                    "hypotheses": [{"text": "please call sean"}],
                },
            ],
        }
    )
    assert row["prefix_source"] == "asr_prefix"
    assert row["terminal_prefix_index"] == 3
    assert row["terminal_prefix_duration_seconds"] == 1.2
    assert row["history"][0]["action"] == AgentAction.WAIT.value
    summary = summarize_simulations([row])
    assert summary["prefix_source_counts"] == {"asr_prefix": 1}
    assert summary["mean_terminal_prefix_duration_seconds"] == 1.2
    assert summary["mean_terminal_decode_wall_time_seconds"] == 0.3
    assert summary["mean_estimated_terminal_latency_seconds"] == 1.5
    assert summary["mean_terminal_cumulative_decode_wall_time_seconds"] == 0.6000000000000001
    assert summary["mean_terminal_serial_decision_latency_seconds"] == 1.5
    assert summary["mean_terminal_serial_backlog_seconds"] == 0.30000000000000004
    assert summary["mean_latest_worker_decision_latency_seconds"] == 1.5
    assert summary["mean_latest_worker_decoded_prefix_count"] == 3
    assert summary["mean_parallel_worker_decision_latency_seconds"] == 1.5
    assert summary["mean_parallel_worker_decoded_prefix_count"] == 3
    assert summary["parallel_worker_unsafe_commit_rate"] == 0.0
    assert summary["mean_decode_real_time_factor"] == 0.25
    assert row["history"][1]["cumulative_decode_wall_time_seconds"] == 0.30000000000000004
    assert "Real audio-prefix ASR" in summary["note"]


def test_latest_prefix_worker_skips_stale_dense_prefixes() -> None:
    row = simulate_record(
        {
            **record("please call sean now"),
            "prefix_hypotheses": [
                {
                    "prefix_fraction": 0.1,
                    "prefix_duration_seconds": 0.25,
                    "decode_wall_time_seconds": 0.5,
                    "hypotheses": [{"text": "please"}],
                },
                {
                    "prefix_fraction": 0.2,
                    "prefix_duration_seconds": 0.30,
                    "decode_wall_time_seconds": 0.5,
                    "hypotheses": [{"text": "please call"}],
                },
                {
                    "prefix_fraction": 0.3,
                    "prefix_duration_seconds": 0.35,
                    "decode_wall_time_seconds": 0.5,
                    "hypotheses": [{"text": "please call sean"}],
                },
            ],
        }
    )
    assert row["latest_worker_terminal_prefix_index"] == 3
    assert row["latest_worker_decoded_prefix_count"] == 2
    assert row["latest_worker_skipped_prefix_count"] == 1
    assert row["latest_worker_decision_latency_seconds"] == 1.25
    assert row["parallel_worker_terminal_prefix_index"] == 3
    assert row["parallel_worker_decoded_prefix_count"] == 3
    assert row["parallel_worker_skipped_prefix_count"] == 0
    assert row["parallel_worker_decision_latency_seconds"] == 1.25


def test_two_worker_runtime_can_return_terminal_decision_while_first_worker_is_busy() -> None:
    history = [
        {
            "prefix_index": 1,
            "prefix_fraction": 0.25,
            "prefix_duration_seconds": 0.0,
            "decode_wall_time_seconds": 1.0,
            "action": AgentAction.WAIT.value,
            "top_prefix": "please",
        },
        {
            "prefix_index": 2,
            "prefix_fraction": 0.5,
            "prefix_duration_seconds": 0.1,
            "decode_wall_time_seconds": 0.2,
            "action": AgentAction.CLARIFY.value,
            "top_prefix": "please call",
        },
    ]

    one_worker_terminal = latest_prefix_worker_terminal(history, worker_count=1)
    two_worker_terminal = latest_prefix_worker_terminal(history, worker_count=2)

    assert one_worker_terminal is not None
    assert two_worker_terminal is not None
    assert one_worker_terminal["worker_decision_ready_time_seconds"] == 1.2
    assert two_worker_terminal["worker_decision_ready_time_seconds"] == 0.30000000000000004
    assert two_worker_terminal["prefix_index"] == 2


def test_simulate_record_preserves_timestamp_prefix_source() -> None:
    timestamp_record = {
        **record("call Jon"),
        "prefix_source": "asr_token_timestamp",
        "prefix_hypotheses": [
            {"prefix_fraction": 0.4, "prefix_duration_seconds": 0.5, "hypotheses": [{"text": "call"}]},
            {"prefix_fraction": 0.8, "prefix_duration_seconds": 1.0, "hypotheses": [{"text": "call Jon"}]},
        ],
    }

    row = simulate_record(timestamp_record, min_prefix_words=1, require_entity_before_terminal=False)
    summary = summarize_simulations([row])

    assert row["prefix_source"] == "asr_token_timestamp"
    assert summary["prefix_source_counts"] == {"asr_token_timestamp": 1}
    assert "Token-timestamp streaming proxy" in summary["note"]
