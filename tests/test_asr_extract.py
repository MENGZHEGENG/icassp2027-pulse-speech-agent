from argparse import Namespace
from datetime import datetime, timezone
import json

import pytest

from pulse.asr_extract import (
    audio_error_record,
    count_expected_shard_records,
    count_valid_jsonl_records,
    decoding_label,
    env_flag,
    existing_output_is_complete,
    extra_records_fields,
    generate_with_oom_retry,
    iter_shard,
    prepare_resume_sample_indices,
    processor_kwargs_for_batch,
    quarantine_invalid_partial_output,
    read_completed_sample_indices,
    resolve_torch_dtype,
    set_generation_seed,
    validate_decoding_args,
)


def make_args(**overrides):
    values = {
        "num_return_sequences": 4,
        "num_beams": 4,
        "num_beam_groups": 1,
        "diversity_penalty": 0.0,
        "length_penalties": "",
        "batch_size": 1,
        "do_sample": False,
        "dataset": "toy_dataset",
        "dataset_config": "clean",
        "split": "test",
        "model": "openai/whisper-tiny",
        "max_new_tokens": None,
        "custom_generate": None,
        "trust_remote_code": False,
        "audio_field": "audio",
        "text_field": "text",
        "records": None,
        "temperature": 1.0,
        "top_p": 1.0,
        "skip_audio_errors": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_decoding_label_marks_diverse_beam() -> None:
    assert decoding_label(make_args(num_beam_groups=2, diversity_penalty=0.3)) == "diverse_beam"
    assert decoding_label(make_args(length_penalties="0.6,1.0,1.3")) == "length_sweep_beam"
    assert decoding_label(make_args(do_sample=True)) == "sample"
    assert decoding_label(make_args()) == "beam"


def test_validate_decoding_args_requires_divisible_beam_groups() -> None:
    with pytest.raises(ValueError, match="divisible"):
        validate_decoding_args(make_args(num_beams=5, num_beam_groups=2, diversity_penalty=0.3))


def test_validate_decoding_args_requires_positive_diversity_penalty() -> None:
    with pytest.raises(ValueError, match="diversity"):
        validate_decoding_args(make_args(num_beams=4, num_beam_groups=2, diversity_penalty=0.0))


def test_validate_decoding_args_accepts_diverse_beam() -> None:
    validate_decoding_args(make_args(num_beams=4, num_beam_groups=2, diversity_penalty=0.3))


def test_validate_decoding_args_rejects_sampling_length_sweep() -> None:
    with pytest.raises(ValueError, match="length-penalties"):
        validate_decoding_args(make_args(do_sample=True, length_penalties="0.8,1.2"))


def test_validate_decoding_args_rejects_nonpositive_batch_size() -> None:
    with pytest.raises(ValueError, match="batch-size"):
        validate_decoding_args(make_args(batch_size=0))


def test_extra_records_fields_preserves_task_metadata() -> None:
    sample = {
        "id": "x",
        "audio_path": "/tmp/x.wav",
        "text": "Call Shawn.",
        "true_entity": "shawn",
        "entity_terms": ["shawn", "sean"],
    }
    assert extra_records_fields(sample, audio_field="audio", text_field="text") == {
        "id": "x",
        "true_entity": "shawn",
        "entity_terms": ["shawn", "sean"],
    }


def test_processor_kwargs_always_request_attention_mask() -> None:
    single = processor_kwargs_for_batch(16000, 1)
    multi = processor_kwargs_for_batch(16000, 2)

    assert single == {"sampling_rate": 16000, "return_tensors": "pt", "return_attention_mask": True}
    assert multi == {
        "sampling_rate": 16000,
        "return_tensors": "pt",
        "return_attention_mask": True,
        "padding": True,
    }


def test_resolve_torch_dtype_uses_fp16_for_cuda_auto() -> None:
    class FakeTorch:
        float32 = "float32"
        float16 = "float16"
        bfloat16 = "bfloat16"

    assert resolve_torch_dtype(FakeTorch, "auto", "cuda") == "float16"
    assert resolve_torch_dtype(FakeTorch, "auto", "cuda:0") == "float16"
    assert resolve_torch_dtype(FakeTorch, "auto", "cpu") is None
    assert resolve_torch_dtype(FakeTorch, "bfloat16", "cuda") == "bfloat16"


def test_env_flag_parses_disabled_values(monkeypatch) -> None:
    assert env_flag("PULSE_TEST_UNSET_FLAG", default=True) is True
    monkeypatch.setenv("PULSE_TEST_FLAG", "off")
    assert env_flag("PULSE_TEST_FLAG", default=True) is False
    monkeypatch.setenv("PULSE_TEST_FLAG", "yes")
    assert env_flag("PULSE_TEST_FLAG", default=False) is True


def test_set_generation_seed_offsets_by_shard(monkeypatch) -> None:
    seen: list[tuple[str, int]] = []

    class FakeCuda:
        @classmethod
        def is_available(cls):
            return True

        @classmethod
        def manual_seed_all(cls, seed: int):
            seen.append(("cuda", seed))

    class FakeTorch:
        cuda = FakeCuda

        @staticmethod
        def manual_seed(seed: int):
            seen.append(("torch", seed))

    monkeypatch.setattr("pulse.asr_extract.np.random.seed", lambda seed: seen.append(("numpy", seed)))

    assert set_generation_seed(FakeTorch, 100, 7) == 107
    assert seen == [("numpy", 107), ("torch", 107), ("cuda", 107)]


def test_generate_with_oom_retry_disables_cache() -> None:
    class FakeOOM(RuntimeError):
        pass

    class FakeCuda:
        cleared = False

        @classmethod
        def is_available(cls):
            return True

        @classmethod
        def empty_cache(cls):
            cls.cleared = True

    class FakeTorch:
        OutOfMemoryError = FakeOOM
        cuda = FakeCuda

    class FakeModel:
        def __init__(self):
            self.calls = []

        def generate(self, input_features, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                raise FakeOOM("CUDA out of memory")
            return {"ok": True}

    model = FakeModel()
    kwargs = {"num_beams": 10, "num_return_sequences": 10}

    result = generate_with_oom_retry(model, "features", kwargs, torch_module=FakeTorch)

    assert result == {"ok": True}
    assert kwargs == {"num_beams": 10, "num_return_sequences": 10}
    assert model.calls == [
        {"num_beams": 10, "num_return_sequences": 10},
        {"num_beams": 10, "num_return_sequences": 10, "use_cache": False},
    ]
    assert FakeCuda.cleared is True


def test_generate_with_oom_retry_can_be_disabled() -> None:
    class FakeOOM(RuntimeError):
        pass

    class FakeTorch:
        OutOfMemoryError = FakeOOM

    class FakeModel:
        def generate(self, input_features, **kwargs):
            raise FakeOOM("CUDA out of memory")

    with pytest.raises(FakeOOM):
        generate_with_oom_retry(
            FakeModel(),
            "features",
            {"num_beams": 10, "use_cache": False},
            torch_module=FakeTorch,
        )


def test_existing_output_complete_requires_valid_jsonl_and_expected_count(tmp_path) -> None:
    output = tmp_path / "shard.jsonl"
    output.write_text(json.dumps({"id": "a"}) + "\n" + json.dumps({"id": "b"}) + "\n", encoding="utf-8")

    assert count_valid_jsonl_records(output) == 2
    assert existing_output_is_complete(output, 2) is True
    assert existing_output_is_complete(output, 3) is False

    output.write_text('{"id": "a"}\nnot-json\n', encoding="utf-8")
    assert count_valid_jsonl_records(output) is None
    assert existing_output_is_complete(output, 2) is False


def test_existing_output_complete_can_validate_unique_shard_indices(tmp_path) -> None:
    output = tmp_path / "shard.jsonl"
    output.write_text(
        json.dumps({"sample_index": 1}) + "\n" + json.dumps({"sample_index": 1}) + "\n",
        encoding="utf-8",
    )

    assert existing_output_is_complete(output, 2, {1, 3}) is False

    output.write_text(
        json.dumps({"sample_index": 1}) + "\n" + json.dumps({"sample_index": 3}) + "\n",
        encoding="utf-8",
    )
    assert existing_output_is_complete(output, 2, {1, 3}) is True


def test_existing_output_complete_requires_expected_nbest_width(tmp_path) -> None:
    output = tmp_path / "shard.jsonl"
    output.write_text(
        json.dumps({"sample_index": 1, "hypotheses": [{"text": "a"}] * 5})
        + "\n"
        + json.dumps({"sample_index": 3, "hypotheses": [{"text": "b"}] * 10})
        + "\n",
        encoding="utf-8",
    )

    assert existing_output_is_complete(output, 2, {1, 3}, expected_hypothesis_count=10) is False

    output.write_text(
        json.dumps({"sample_index": 1, "hypotheses": [{"text": "a"}] * 10})
        + "\n"
        + json.dumps({"sample_index": 3, "hypotheses": [{"text": "b"}] * 10})
        + "\n",
        encoding="utf-8",
    )
    assert existing_output_is_complete(output, 2, {1, 3}, expected_hypothesis_count=10) is True


def test_completed_sample_indices_can_accept_audio_error_records(tmp_path) -> None:
    output = tmp_path / "partial.jsonl"
    output.write_text(
        json.dumps({"sample_index": 1, "asr_error": {"stage": "audio_decode"}, "hypotheses": []})
        + "\n"
        + json.dumps({"sample_index": 3, "hypotheses": [{"text": "ok"}, {"text": "ok"}]})
        + "\n",
        encoding="utf-8",
    )

    assert read_completed_sample_indices(output, {1, 3}, expected_hypothesis_count=2, allow_error_records=True) == {1, 3}
    assert existing_output_is_complete(output, 2, {1, 3}, expected_hypothesis_count=2, allow_error_records=True) is True

    with pytest.raises(ValueError, match="expected at least 2"):
        read_completed_sample_indices(output, {1, 3}, expected_hypothesis_count=2)


def test_audio_error_record_keeps_decoding_context() -> None:
    args = make_args(records=object(), num_return_sequences=10, seed=7)
    sample = {"id": "utt-1", "audio_path": "/tmp/missing.wav", "text": "turn on lights", "speaker_id": "spk-a"}

    record = audio_error_record(
        5,
        sample,
        args,
        RuntimeError("cannot read wav"),
        generation_seed=11,
        length_penalties=[],
        retry_oom_without_cache=True,
        clear_cache_each_batch=True,
        output_scores_enabled=False,
    )

    assert record["sample_index"] == 5
    assert record["hypotheses"] == []
    assert record["asr_error"]["stage"] == "audio_decode"
    assert record["asr_error"]["audio_path"] == "/tmp/missing.wav"
    assert record["decoding_config"]["num_return_sequences"] == 10
    assert record["decoding_config"]["skip_audio_errors"] is False
    assert record["speaker_id"] == "spk-a"


def test_read_completed_sample_indices_validates_partial_output(tmp_path) -> None:
    output = tmp_path / "partial.jsonl"
    output.write_text(json.dumps({"sample_index": 1}) + "\n" + json.dumps({"sample_index": 3}) + "\n", encoding="utf-8")

    assert read_completed_sample_indices(output, {1, 3, 5}) == {1, 3}

    output.write_text(json.dumps({"sample_index": 2}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside this shard"):
        read_completed_sample_indices(output, {1, 3, 5})

    output.write_text(json.dumps({"sample_index": 1, "hypotheses": [{"text": "a"}]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected at least 2"):
        read_completed_sample_indices(output, {1, 3, 5}, expected_hypothesis_count=2)


def test_read_completed_sample_indices_rejects_duplicate_unique_field(tmp_path) -> None:
    output = tmp_path / "partial.jsonl"
    output.write_text(
        json.dumps({"sample_index": 1, "id": "utt-1"}) + "\n" + json.dumps({"sample_index": 3, "id": "utt-1"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="repeats id utt-1"):
        read_completed_sample_indices(output, {1, 3, 5}, expected_unique_field="id")


def test_quarantine_invalid_partial_output_moves_file_with_reason(tmp_path) -> None:
    output = tmp_path / "partial.jsonl"
    output.write_text(json.dumps({"sample_index": 1}) + "\n", encoding="utf-8")

    moved_path = quarantine_invalid_partial_output(
        output,
        "duplicate sample index",
        now_utc=datetime(2026, 8, 28, 1, 2, 3, tzinfo=timezone.utc),
    )

    assert moved_path.name == "partial.jsonl.invalid-20260828T010203Z"
    assert not output.exists()
    assert json.loads(moved_path.read_text(encoding="utf-8"))["sample_index"] == 1
    assert moved_path.with_name(f"{moved_path.name}.reason.txt").read_text(encoding="utf-8") == "duplicate sample index\n"


def test_prepare_resume_sample_indices_can_rewrite_invalid_partial(tmp_path) -> None:
    output = tmp_path / "partial.jsonl"
    output.write_text(
        json.dumps({"sample_index": 1}) + "\n" + json.dumps({"sample_index": 1}) + "\n",
        encoding="utf-8",
    )

    completed, moved_path = prepare_resume_sample_indices(output, {1, 3, 5}, rewrite_invalid_partial=True)

    assert completed == set()
    assert moved_path is not None
    assert moved_path.name.startswith("partial.jsonl.invalid-")
    assert not output.exists()


def test_prepare_resume_sample_indices_quarantines_wrong_nbest_width(tmp_path) -> None:
    output = tmp_path / "partial.jsonl"
    output.write_text(json.dumps({"sample_index": 1, "hypotheses": [{"text": "a"}]}) + "\n", encoding="utf-8")

    completed, moved_path = prepare_resume_sample_indices(
        output,
        {1, 3, 5},
        rewrite_invalid_partial=True,
        expected_hypothesis_count=2,
    )

    assert completed == set()
    assert moved_path is not None
    assert "expected at least 2" in moved_path.with_name(f"{moved_path.name}.reason.txt").read_text(encoding="utf-8")


def test_prepare_resume_sample_indices_quarantines_duplicate_unique_field(tmp_path) -> None:
    output = tmp_path / "partial.jsonl"
    output.write_text(
        json.dumps({"sample_index": 1, "id": "utt-1"}) + "\n" + json.dumps({"sample_index": 3, "id": "utt-1"}) + "\n",
        encoding="utf-8",
    )

    completed, moved_path = prepare_resume_sample_indices(
        output,
        {1, 3, 5},
        rewrite_invalid_partial=True,
        expected_unique_field="id",
    )

    assert completed == set()
    assert moved_path is not None
    assert "repeats id utt-1" in moved_path.with_name(f"{moved_path.name}.reason.txt").read_text(encoding="utf-8")


def test_expected_shard_records_matches_iter_shard_limit() -> None:
    dataset = [{"id": str(index)} for index in range(10)]

    assert count_expected_shard_records(dataset, shard_index=0, num_shards=3, max_samples=99) == 4
    assert count_expected_shard_records(dataset, shard_index=1, num_shards=3, max_samples=2) == 2
    assert count_expected_shard_records(dataset, shard_index=2, num_shards=20, max_samples=99) == 1


def test_iter_shard_can_skip_earlier_samples_for_backfill() -> None:
    dataset = [{"id": str(index)} for index in range(10)]

    selected = list(iter_shard(dataset, shard_index=1, num_shards=2, max_samples=3, min_sample_index=4))

    assert [sample_index for sample_index, _sample in selected] == [5, 7, 9]
    assert count_expected_shard_records(dataset, shard_index=0, num_shards=2, max_samples=99, min_sample_index=8) == 1
