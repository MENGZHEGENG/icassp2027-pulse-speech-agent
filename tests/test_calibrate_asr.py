from pulse.calibrate_asr import calibrate_records


def test_calibrate_records_bins_errors() -> None:
    records = [
        {
            "reference": "call shawn",
            "entity_terms": ["shawn", "sean"],
            "hypotheses": [{"text": "call sean", "score": -0.1}, {"text": "call shawn", "score": -0.2}],
        }
    ]
    buckets = calibrate_records(records, bins=5)
    assert sum(bucket["count"] for bucket in buckets) == 1
    assert any(bucket["mean_error"] > 0 for bucket in buckets)
