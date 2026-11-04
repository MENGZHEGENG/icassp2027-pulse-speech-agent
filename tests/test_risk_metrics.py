from pulse.risk_metrics import average_precision, expected_calibration_error, risk_rows, roc_auc, summarize_risk_rows


def test_roc_auc_and_average_precision_rank_perfect_scores() -> None:
    scores = [0.1, 0.4, 0.9]
    labels = [False, False, True]
    assert roc_auc(scores, labels) == 1.0
    assert average_precision(scores, labels) == 1.0


def test_roc_auc_returns_none_for_single_class() -> None:
    assert roc_auc([0.1, 0.2], [True, True]) is None
    assert average_precision([0.1, 0.2], [False, False]) is None


def test_expected_calibration_error_uses_weighted_bins() -> None:
    rows = [{"risk": 0.1, "has_error": False}, {"risk": 0.9, "has_error": True}]
    assert expected_calibration_error(rows, bins=2) < 0.2


def test_risk_rows_and_summary_from_records() -> None:
    records = [
        {
            "id": "a",
            "reference": "call shawn",
            "action": "call_contact",
            "entity_terms": ["shawn", "sean"],
            "hypotheses": [{"text": "call sean", "score": -0.1}, {"text": "call sean", "score": -0.2}],
        },
        {
            "id": "b",
            "reference": "call sean",
            "action": "call_contact",
            "entity_terms": ["shawn", "sean"],
            "hypotheses": [{"text": "call sean", "score": -0.1}],
        },
    ]
    rows = risk_rows(records)
    summary = summarize_risk_rows(rows)
    assert len(rows) == 2
    assert summary["count"] == 2
    assert summary["error_rate"] == 0.5


def test_risk_rows_support_answer_mode() -> None:
    records = [
        {
            "id": "a",
            "reference": "call shawn",
            "action": "call_contact",
            "entity_terms": ["shawn", "sean"],
            "hypotheses": [{"text": "call sean", "score": -0.1}],
        }
    ]
    task_rows = risk_rows(records, action_mode="task")
    answer_rows = risk_rows(records, action_mode="answer")
    assert task_rows[0]["risk"] >= answer_rows[0]["risk"]


def test_risk_rows_support_feature_score_fields() -> None:
    records = [
        {
            "id": "a",
            "reference": "hello world",
            "hypotheses": [{"text": "hello world", "score": -0.1}, {"text": "hello there", "score": -0.2}],
        }
    ]
    rows = risk_rows(records, score_field="instability")
    assert rows[0]["score_field"] == "instability"
    assert rows[0]["risk"] > 0.0
