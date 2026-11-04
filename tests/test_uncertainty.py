import pulse.uncertainty as uncertainty
from pulse.uncertainty import Hypothesis, arpabet_phrase_key, cmudict_word_key, contains_arpabet_phrase, phonetic_phrase_key, summarize_hypotheses, token_distance_ratio


def test_token_distance_ratio_detects_change() -> None:
    assert token_distance_ratio("call john smith", "call jane smith") > 0
    assert token_distance_ratio("call john smith", "call john smith") == 0


def test_summary_reports_entity_disagreement() -> None:
    hypotheses = [
        Hypothesis("transfer to john smith", -0.1),
        Hypothesis("transfer to jane smith", -0.2),
        Hypothesis("transfer to john smith", -0.3),
    ]
    summary = summarize_hypotheses(hypotheses, ["john smith", "jane smith"])
    assert summary.unique_hypotheses == 2
    assert summary.instability > 0
    assert summary.entity_risk > 0
    assert summary.entropy is not None


def test_entropy_aggregates_identical_texts() -> None:
    hypotheses = [
        Hypothesis("hello world", -0.1),
        Hypothesis("hello world", -0.2),
        Hypothesis("hello world", -0.3),
    ]
    summary = summarize_hypotheses(hypotheses)
    assert summary.entropy is not None
    assert summary.entropy < 1e-6


def test_entity_disagreement_prioritizes_runner_up_conflict() -> None:
    hypotheses = [
        Hypothesis("call sean", -0.1),
        Hypothesis("call shawn", -0.2),
        Hypothesis("call sean", -0.3),
    ]
    summary = summarize_hypotheses(hypotheses, ["shawn", "sean"])
    assert summary.entity_risk == 1.0


def test_phonetic_phrase_key_groups_common_name_variants() -> None:
    assert phonetic_phrase_key("shawn") == phonetic_phrase_key("sean")
    assert phonetic_phrase_key("account four") == phonetic_phrase_key("account for")


def test_summary_reports_phonetic_risk_for_unseen_variants() -> None:
    hypotheses = [
        Hypothesis("call shon", -0.1),
        Hypothesis("call shaun", -0.2),
        Hypothesis("call shon", -0.3),
    ]
    summary = summarize_hypotheses(hypotheses, ["shawn", "sean"])
    assert summary.entity_risk == 0.0
    assert summary.phonetic_risk >= 0.75


def test_summary_reports_static_phonetic_candidate_risk() -> None:
    hypotheses = [
        Hypothesis("send an email to sean", -0.1),
        Hypothesis("send an email to sean", -0.2),
    ]
    summary = summarize_hypotheses(hypotheses, ["shawn", "sean"])
    assert summary.instability == 0.0
    assert summary.entity_risk == 0.0
    assert summary.phonetic_risk >= 0.75


def test_phonetic_risk_matches_merged_entity_tokens() -> None:
    hypotheses = [Hypothesis("call qroot", -0.1), Hypothesis("call qroot", -0.2)]
    summary = summarize_hypotheses(hypotheses, ["q route", "queue route"])
    assert summary.entity_risk == 0.0
    assert summary.phonetic_risk >= 0.75


def test_phonetic_risk_triggers_when_candidate_entity_missing() -> None:
    hypotheses = [Hypothesis("call kiruk", -0.1), Hypothesis("call kiruk", -0.2)]
    summary = summarize_hypotheses(hypotheses, ["q route", "queue route"])
    assert summary.entity_risk == 0.0
    assert summary.phonetic_risk == 1.0


def test_arpabet_phrase_key_covers_benchmark_entities() -> None:
    assert arpabet_phrase_key("shawn") == arpabet_phrase_key("sean")
    assert arpabet_phrase_key("account four") == arpabet_phrase_key("account for")
    assert contains_arpabet_phrase("open the n r c report", arpabet_phrase_key("nrc report"))


def test_phone_aware_risk_matches_spelled_letter_variants() -> None:
    hypotheses = [Hypothesis("find the n r c report", -0.1), Hypothesis("find the n r c report", -0.2)]
    summary = summarize_hypotheses(hypotheses, ["nrc report", "an rc report"])
    assert summary.entity_risk == 0.0
    assert summary.phonetic_risk >= 0.75


def test_optional_cmudict_backend_extends_arpabet_lookup(monkeypatch) -> None:
    class FakePronouncing:
        @staticmethod
        def phones_for_word(word: str) -> list[str]:
            return ["K AE1 T"] if word == "cat" else []

    monkeypatch.setattr(uncertainty, "_pronouncing", FakePronouncing)
    cmudict_word_key.cache_clear()
    assert arpabet_phrase_key("cat") == ("K", "AE", "T")


def test_compact_lexicon_overrides_optional_cmudict(monkeypatch) -> None:
    class FakePronouncing:
        @staticmethod
        def phones_for_word(word: str) -> list[str]:
            return ["S IY1 N"]

    monkeypatch.setattr(uncertainty, "_pronouncing", FakePronouncing)
    cmudict_word_key.cache_clear()
    assert arpabet_phrase_key("sean") == ("SH", "AO", "N")
