from pulse.calibrate_candidate_margin import aggregate_rows, buffered_threshold, calibrated_threshold, choose_safe_threshold, leave_one_group_out


def confirmation_record(group: str, true_entity: str, score_true: float, score_other: float) -> dict:
    return {
        "voice_condition": group,
        "action": "call_contact",
        "true_entity": true_entity,
        "entity_terms": ["shawn", "sean"],
        "unsafe_if_confused": True,
        "hypotheses": [{"text": "call sean", "score": -0.1}, {"text": "call sean", "score": -0.1}],
        "candidate_scores": [{"entity": "shawn", "score": score_true if true_entity == "shawn" else score_other}, {"entity": "sean", "score": score_true if true_entity == "sean" else score_other}],
    }


def test_choose_safe_threshold_prefers_most_accepted_zero_unsafe() -> None:
    records = [
        confirmation_record("a", "shawn", -0.1, -0.3),
        confirmation_record("a", "sean", -0.1, -0.8),
    ]
    selected = choose_safe_threshold(records, [0.1, 0.5, 1.0], policy=None)
    assert selected["margin_threshold"] == 0.1
    assert selected["accepted_confirmations"] == 2


def test_leave_one_group_out_reports_heldout_metrics() -> None:
    records = [
        confirmation_record("a", "shawn", -0.1, -0.8),
        confirmation_record("b", "shawn", -0.1, -0.8),
    ]
    rows = leave_one_group_out(records, group_field="voice_condition", thresholds=[0.5])
    assert [row["heldout_group"] for row in rows] == ["a", "b"]
    assert all(row["heldout_accepted_confirmations"] == 1 for row in rows)
    summary = aggregate_rows(rows)
    assert summary["heldout_task_success"] == 1.0
    assert summary["heldout_accepted_confirmations"] == 2


def test_buffered_threshold_moves_to_next_grid_value() -> None:
    assert buffered_threshold([0.5, 0.6, 0.8], 0.5, 0.1) == 0.6
    assert buffered_threshold([0.5, 0.6, 0.8], 0.6, 0.1) == 0.8


def test_calibrated_threshold_applies_minimum_floor() -> None:
    assert calibrated_threshold([0.5, 0.6, 0.8], 0.5, 0.0, 0.7) == 0.8
    assert calibrated_threshold([0.5, 0.6, 0.8], 0.8, 0.0, 0.7) == 0.8
