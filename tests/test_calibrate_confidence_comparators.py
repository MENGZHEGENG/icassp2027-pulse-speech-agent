import json
from pathlib import Path

from pulse.calibrate_confidence_comparators import build_report, confidence_diagnostics, parse_modes, select_threshold


def record(group: str, text: str, true_entity: str, confidence: float, unsafe: bool = True) -> dict:
    return {
        "id": f"{group}-{text}",
        "voice_condition": group,
        "reference": f"Call {true_entity}.",
        "action": "call_contact",
        "true_entity": true_entity,
        "entity_terms": ["shawn", "sean", "jane"],
        "entity_confidence": confidence,
        "unsafe_if_confused": unsafe,
        "hypotheses": [{"text": text, "confidence": confidence, "score": confidence}],
    }


def test_select_threshold_prefers_low_deferral_under_zero_unsafe_train() -> None:
    rows = [
        record("train-a", "call sean", "shawn", 0.2),
        record("train-a", "call jane", "jane", 0.8),
        record("train-a", "call shawn", "shawn", 0.9),
    ]

    selected = select_threshold(rows, "scalar_asr_confidence_threshold", max_train_unsafe_wrong_rate=0.0)

    assert selected["met_train_safety_constraint"] is True
    assert selected["threshold"] > 0.2
    assert selected["train_summary"]["unsafe_wrong_action_rate"] == 0.0
    assert selected["train_summary"]["deferral_rate"] < 1.0


def test_leave_one_group_report_labels_database_and_scenario() -> None:
    rows = [
        record("speaker-a", "call sean", "shawn", 0.2),
        record("speaker-a", "call jane", "jane", 0.9),
        record("speaker-b", "call sean", "shawn", 0.3),
        record("speaker-b", "call shawn", "shawn", 0.95),
    ]

    payload = build_report(
        rows,
        group_field="voice_condition",
        modes=["scalar_asr_confidence_threshold", "slot_entity_confidence_threshold"],
        max_train_unsafe_wrong_rate=0.0,
        database="toy_confusable",
        application_scenario="entity_bearing_tool_call",
    )

    assert payload["ok"] is True
    assert payload["ready_for_confidence_only_claim"] is False
    assert payload["fold_count"] == 2
    assert payload["row_count"] == 4
    assert payload["database"] == "toy_confusable"
    assert payload["application_scenario"] == "entity_bearing_tool_call"
    assert {row["policy_mode"] for row in payload["aggregate"]} == {
        "scalar_asr_confidence_threshold",
        "slot_entity_confidence_threshold",
    }
    assert all(row["all_train_safety_constraints_met"] for row in payload["aggregate"])
    assert not all(row["all_heldout_safety_constraints_met"] for row in payload["aggregate"])
    scalar_row = next(row for row in payload["aggregate"] if row["policy_mode"] == "scalar_asr_confidence_threshold")
    assert scalar_row["heldout_confidence_correctness_auroc"] == 1.0
    assert scalar_row["heldout_confidence_ece"] is not None
    assert scalar_row["heldout_confidence_ece_applicability"] == "bounded_probability"


def test_calibrated_confidence_only_uses_train_binned_probabilities() -> None:
    rows = [
        record("speaker-a", "call sean", "shawn", -0.7),
        record("speaker-a", "call jane", "jane", -0.1),
        record("speaker-b", "call sean", "shawn", -0.8),
        record("speaker-b", "call shawn", "shawn", -0.2),
    ]

    payload = build_report(
        rows,
        group_field="voice_condition",
        modes=["calibrated_confidence_only_policy"],
        max_train_unsafe_wrong_rate=0.0,
        database="toy_confusable",
        application_scenario="entity_bearing_tool_call",
    )

    aggregate = payload["aggregate"][0]
    assert aggregate["confidence_calibration_method"] == "train_quantile_empirical_correctness"
    assert aggregate["heldout_confidence_ece_applicability"] == "bounded_probability"
    assert aggregate["heldout_confidence_ece"] is not None
    assert all(row["confidence_calibration_bin_count"] > 0 for row in payload["rows"])
    assert all(0.0 <= row["selected_threshold"] <= 1.0 for row in payload["rows"])


def test_fused_asr_confidence_threshold_is_bounded_for_calibration() -> None:
    rows = [
        record("speaker-a", "call sean", "shawn", 0.95),
        record("speaker-a", "call jane", "jane", 0.90),
        record("speaker-b", "call sean", "shawn", 0.95),
        record("speaker-b", "call shawn", "shawn", 0.90),
    ]

    payload = build_report(
        rows,
        group_field="voice_condition",
        modes=["fused_asr_confidence_threshold"],
        max_train_unsafe_wrong_rate=0.0,
        database="toy_confusable",
        application_scenario="entity_bearing_tool_call",
    )

    aggregate = payload["aggregate"][0]
    assert aggregate["policy_mode"] == "fused_asr_confidence_threshold"
    assert aggregate["heldout_confidence_ece_applicability"] == "bounded_probability"
    assert all(0.0 <= row["selected_threshold"] <= 1.0 for row in payload["rows"])


def test_confidence_diagnostics_skip_ece_for_unbounded_scores() -> None:
    rows = [
        {
            "id": "wrong-low-score",
            "voice_condition": "speaker-a",
            "reference": "Call shawn.",
            "action": "call_contact",
            "true_entity": "shawn",
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call sean", "score": -0.7}],
        },
        {
            "id": "correct-high-score",
            "voice_condition": "speaker-a",
            "reference": "Call jane.",
            "action": "call_contact",
            "true_entity": "jane",
            "unsafe_if_confused": True,
            "hypotheses": [{"text": "call jane", "score": -0.1}],
        },
    ]

    diagnostics = confidence_diagnostics(rows, "scalar_asr_confidence_threshold")

    assert diagnostics["confidence_observed_count"] == 2
    assert diagnostics["confidence_correctness_auroc"] == 1.0
    assert diagnostics["confidence_ece"] is None
    assert diagnostics["confidence_ece_applicability"] == "unbounded_score_not_probability"


def test_parse_modes_rejects_unsupported_mode() -> None:
    try:
        parse_modes("scalar_asr_confidence_threshold,pulse")
    except ValueError as exc:
        assert "unsupported confidence comparator mode" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected unsupported mode failure")


def test_calibration_cli_records_shape(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records = [record("speaker-a", "call sean", "shawn", 0.2), record("speaker-b", "call jane", "jane", 0.9)]
    records_path.write_text("\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8")
    assert records_path.read_text(encoding="utf-8").count("\n") == 2
