from pulse.summarize_asr import normalized_cer, normalized_wer, summarize_records


def test_normalized_error_rates_ignore_case_and_punctuation() -> None:
    assert normalized_wer("HELLO, WORLD", "hello world") == 0
    assert normalized_cer("HELLO, WORLD", "hello world") == 0


def test_summarize_records() -> None:
    records = [
        {
            "reference": "call shawn",
            "hypotheses": [
                {"text": "call sean", "score": -0.1},
                {"text": "call shawn", "score": -0.2},
            ],
        }
    ]
    summary = summarize_records(records, ["shawn", "sean"])
    assert summary["count"] == 1
    assert "mean_wer" in summary
    assert "mean_cer" in summary
    assert summary["mean_wer_proxy"] is not None
    assert summary["pulse_clarification_rate"] == 1.0


def test_summarize_records_uses_record_entity_terms() -> None:
    records = [
        {
            "reference": "call shawn",
            "entity_terms": ["shawn", "sean"],
            "hypotheses": [
                {"text": "call shawn", "score": -0.1},
                {"text": "call sean", "score": -0.2},
            ],
        }
    ]
    payload = summarize_records(records)
    assert payload["mean_entity_risk"] == 1.0
