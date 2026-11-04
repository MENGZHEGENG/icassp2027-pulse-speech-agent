from pulse.synthesize_task_hypotheses import synthesize_records


def test_synthesize_task_hypotheses_adds_hypotheses() -> None:
    records = [
        {
            "id": "x",
            "prompt_text": "Call shawn about the project update.",
            "true_entity": "shawn",
            "confusable_entity": "sean",
        }
    ]
    output = synthesize_records(records, error_rate=1.0, seed=1)
    assert output[0]["synthetic_top_error"] is True
    assert output[0]["hypotheses"][0]["text"] == "Call sean about the project update."
