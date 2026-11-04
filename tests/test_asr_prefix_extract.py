import numpy as np
import pytest

from pulse.asr_prefix_extract import build_parser, audio_prefix, parse_prefix_fractions


def test_parse_prefix_fractions_sorts_and_adds_final_prefix() -> None:
    assert parse_prefix_fractions("0.75,0.25") == [0.25, 0.75, 1.0]


def test_parse_prefix_fractions_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        parse_prefix_fractions("0.5,1.5")


def test_audio_prefix_respects_min_seconds_and_caps_to_full_audio() -> None:
    array = np.arange(16000, dtype=np.float32)
    assert len(audio_prefix(array, 16000, 0.1, min_seconds=0.25)) == 4000
    assert len(audio_prefix(array, 16000, 2.0, min_seconds=0.25)) == 16000


def test_parser_supports_skip_existing_prefix_outputs() -> None:
    args = build_parser().parse_args(["--output", "out.jsonl", "--skip-existing"])

    assert args.skip_existing is True


def test_parser_supports_memory_saving_generation_flags() -> None:
    args = build_parser().parse_args(
        ["--output", "out.jsonl", "--max-new-tokens", "64", "--no-output-scores"]
    )

    assert args.max_new_tokens == 64
    assert args.no_output_scores is True
