import pytest

from pulse.merge_phone_evidence import merge_phone_evidence


def records_record(record_id: str = "utt-1") -> dict:
    return {
        "id": record_id,
        "action": "call_contact",
        "true_entity": "john",
        "entity_terms": ["john", "jane"],
        "hypotheses": [{"text": "call john"}],
    }


def test_merge_phone_evidence_adds_top_level_risk_and_payload() -> None:
    merged, summary = merge_phone_evidence(
        [records_record()],
        [{"id": "utt-1", "learned_phonetic_risk": 0.72, "source": "forced-aligner-v0"}],
    )
    assert summary == {"records": 1, "matched": 1, "missing": 0, "extra": 0}
    assert merged[0]["learned_phonetic_risk"] == 0.72
    assert merged[0]["phone_evidence"]["source"] == "forced-aligner-v0"


def test_merge_phone_evidence_strips_stale_ground_truth_from_payload() -> None:
    merged, _ = merge_phone_evidence(
        [records_record()],
        [{
            "id": "utt-1",
            "learned_phonetic_risk": 0.72,
            "phone_evidence": {"true_entity": "john", "predicted_entity": "jane"},
        }],
    )
    assert "true_entity" not in merged[0]["phone_evidence"]
    assert merged[0]["phone_evidence"]["predicted_entity"] == "jane"


def test_merge_phone_evidence_can_require_all_records_rows() -> None:
    with pytest.raises(ValueError, match="missing phone evidence"):
        merge_phone_evidence([records_record()], [], require_all=True)


def test_merge_phone_evidence_rejects_duplicate_or_out_of_range_evidence() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        merge_phone_evidence(
            [records_record()],
            [{"id": "utt-1", "phone_risk": 0.1}, {"id": "utt-1", "phone_risk": 0.2}],
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        merge_phone_evidence([records_record()], [{"id": "utt-1", "phone_risk": 1.2}])
