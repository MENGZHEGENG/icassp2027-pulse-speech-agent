from __future__ import annotations

from argparse import Namespace

import pytest

from pulse.asr_timestamp_extract import build_parser, build_prefix_entries, is_timestamp_or_special, prefix_fraction, validate_args


class FakeTokenizer:
    tokens = {
        1: "<|0.00|>",
        2: "ĠFind",
        3: "Ġthe",
        4: "Ġfile",
        5: ".",
        6: "<|1.20|>",
    }

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.tokens[item] for item in ids]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        pieces = []
        for item in ids:
            token = self.tokens[item]
            if token.startswith("<|") and token.endswith("|>"):
                if skip_special_tokens:
                    continue
            pieces.append(token.replace("Ġ", " "))
        return "".join(pieces).strip()


def test_is_timestamp_or_special_detects_whisper_markers() -> None:
    assert is_timestamp_or_special("<|0.00|>")
    assert is_timestamp_or_special("<|en|>")
    assert not is_timestamp_or_special("ĠFind")


def test_prefix_fraction_is_capped() -> None:
    assert prefix_fraction(0.5, 2.0) == 0.25
    assert prefix_fraction(5.0, 2.0) == 1.0
    assert prefix_fraction(-1.0, 2.0) == 0.0
    assert prefix_fraction(1.0, 0.0) == 1.0


def test_build_prefix_entries_uses_token_timestamps() -> None:
    entries = build_prefix_entries(
        tokenizer=FakeTokenizer(),
        sequence_ids=[1, 2, 3, 4, 5, 6],
        token_timestamps=[0.0, 0.2, 0.4, 0.8, 1.0, 1.2],
        full_duration_seconds=1.2,
    )

    assert [entry["hypotheses"][0]["text"] for entry in entries] == [
        "Find",
        "Find the",
        "Find the file",
        "Find the file.",
    ]
    assert [entry["prefix_duration_seconds"] for entry in entries] == [0.2, 0.4, 0.8, 1.0]
    assert entries[-1]["prefix_fraction"] == 1.0 / 1.2


def test_build_prefix_entries_adds_final_when_min_prefix_skips_tokens() -> None:
    entries = build_prefix_entries(
        tokenizer=FakeTokenizer(),
        sequence_ids=[1, 2, 3, 4, 5, 6],
        token_timestamps=[0.0, 0.2, 0.4, 0.8, 1.0, 1.2],
        full_duration_seconds=1.2,
        min_prefix_seconds=1.1,
    )

    assert len(entries) == 1
    assert entries[0]["prefix_fraction"] == 1.0
    assert entries[0]["prefix_duration_seconds"] == 1.2
    assert entries[0]["hypotheses"][0]["text"] == "Find the file."


def test_build_prefix_entries_aligns_nbest_by_timestamp() -> None:
    entries = build_prefix_entries(
        tokenizer=FakeTokenizer(),
        sequence_ids=[
            [1, 2, 3, 4, 6],
            [1, 2, 4, 6],
        ],
        token_timestamps=[
            [0.0, 0.2, 0.4, 0.8, 1.2],
            [0.0, 0.3, 0.9, 1.2],
        ],
        full_duration_seconds=1.2,
    )

    assert entries[1]["prefix_duration_seconds"] == 0.4
    assert entries[1]["hypotheses"] == [
        {"rank": 1, "text": "Find the", "score": None},
        {"rank": 2, "text": "Find", "score": None},
    ]


def test_validate_args_rejects_more_return_sequences_than_beams() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        validate_args(Namespace(num_beams=2, num_return_sequences=3))


def test_parser_supports_skip_existing_timestamp_outputs() -> None:
    args = build_parser().parse_args(["--output", "out.jsonl", "--skip-existing"])

    assert args.skip_existing is True
