import json

from pulse.phone_evidence_extract import (
    arpabet_to_ipa,
    best_substring_phone_distance_ratio,
    build_parser,
    entity_phone_candidates,
    normalize_phone_text,
    phone_distance_ratio,
    score_phone_evidence,
    main,
)


def test_arpabet_to_ipa_and_normalization() -> None:
    assert arpabet_to_ipa(("SH", "AO", "N")) == "ʃɔn"
    assert normalize_phone_text("ˈʃ ɔː n") == "ʃɔn"


def test_entity_phone_candidates_use_cmudict_fallback() -> None:
    candidates = entity_phone_candidates(["Sean", "Jane"])
    assert [candidate["normalized_entity"] for candidate in candidates] == ["sean", "jane"]
    assert candidates[0]["ipa"] == "ʃɔn"


def test_score_phone_evidence_flags_homophone_ambiguity() -> None:
    score = score_phone_evidence("pliːz kɔl ʃɔn naʊ", entity_terms=["Sean", "Shawn"])
    assert score["learned_phonetic_risk"] == 1.0
    assert score["margin"] == 0.0


def test_best_substring_phone_distance_finds_entity_span() -> None:
    assert best_substring_phone_distance_ratio("pliːz kɔl ʃɔn naʊ", "ʃɔn") == 0.0
    assert best_substring_phone_distance_ratio("pliːz kɔl ʃɔn naʊ", "dʒeɪn") > 0.0


def test_score_phone_evidence_is_low_for_clear_margin() -> None:
    score = score_phone_evidence("kɔl dʒeɪn naʊ", entity_terms=["Jane", "Sean"], margin_scale=0.25)
    assert score["predicted_entity"] == "Jane"
    assert score["learned_phonetic_risk"] == 0.0
    assert phone_distance_ratio("dʒeɪn", score["candidate_scores"][0]["ipa"]) == 0.0


def test_score_phone_evidence_is_truth_independent() -> None:
    score = score_phone_evidence("ʃɔn", entity_terms=["Jane", "Sean"])
    assert score["predicted_entity"] == "Sean"
    assert score["learned_phonetic_risk"] < 1.0
    assert "true_entity" not in score


def test_parser_supports_skip_existing_phone_outputs() -> None:
    args = build_parser().parse_args(["records.jsonl", "--output", "out.jsonl", "--skip-existing"])

    assert args.skip_existing is True


def test_cli_uses_records_phone_field_without_model(tmp_path, monkeypatch) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps(
            {
                "id": "utt-1",
                "predicted_phones": "ʃɔn",
                "entity_terms": ["Sean", "Shawn"],
                "true_entity": "Sean",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "phone_evidence.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "phone_evidence_extract",
            str(records),
            "--predicted-phone-field",
            "predicted_phones",
            "--output",
            str(output),
        ],
    )
    main()
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["id"] == "utt-1"
    assert row["learned_phonetic_risk"] == 1.0
    assert row["phone_evidence"]["predicted_phone_text"] == "ʃɔn"
    assert "true_entity" not in row["phone_evidence"]
