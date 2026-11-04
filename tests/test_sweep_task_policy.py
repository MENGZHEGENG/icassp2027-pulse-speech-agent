from pulse.sweep_task_policy import parse_float_grid, sweep_records, utility_score


def test_parse_float_grid() -> None:
    assert parse_float_grid("0.1, 0.2,,1") == [0.1, 0.2, 1.0]


def test_utility_score_penalizes_clarification_and_unsafe_errors() -> None:
    summary = {
        "task_success": 1.0,
        "clarification_rate": 0.5,
        "confirmation_rate": 0.25,
        "unsafe_wrong_action_rate": 0.25,
        "wrong_action_rate": 0.25,
    }
    assert utility_score(summary, clarification_cost=0.2, confirmation_cost=0.1, unsafe_cost=1.0, wrong_cost=0.5) == 0.5


def test_sweep_records_ranks_safe_clarification() -> None:
    records = [
        {
            "action": "call_contact",
            "true_entity": "shawn",
            "entity_terms": ["shawn", "sean"],
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call sean", "score": -0.1}, {"text": "call sean", "score": -0.2}],
        }
    ]
    rows = sweep_records(
        records,
        entity_thresholds=[1.1],
        phonetic_thresholds=[0.75, 1.1],
        instability_thresholds=[1.1],
        entropy_thresholds=[10.0],
        confirm_retrieval_options=[False],
        clarification_cost=0.15,
        confirmation_cost=0.05,
        unsafe_cost=1.0,
        wrong_cost=0.5,
    )
    assert rows[0]["phonetic_threshold"] == 0.75
    assert rows[0]["unsafe_wrong_action_rate"] == 0.0
