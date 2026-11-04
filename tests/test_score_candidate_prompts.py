import json

from pulse.score_candidate_prompts import attach_audio_path, build_parser, iter_shard, main


def test_iter_shard_selects_stable_modulo_partition() -> None:
    records = [{"id": str(index)} for index in range(7)]
    assert [record["id"] for record in iter_shard(records, shard_index=0, num_shards=3, max_samples=None)] == ["0", "3", "6"]
    assert [record["id"] for record in iter_shard(records, shard_index=1, num_shards=3, max_samples=None)] == ["1", "4"]
    assert [record["id"] for record in iter_shard(records, shard_index=2, num_shards=3, max_samples=None)] == ["2", "5"]


def test_iter_shard_applies_max_samples_after_partitioning() -> None:
    records = [{"id": str(index)} for index in range(10)]
    selected = iter_shard(records, shard_index=0, num_shards=2, max_samples=3)
    assert [record["id"] for record in selected] == ["0", "2", "4"]


def test_attach_audio_path_uses_source_audio_path_before_lookup() -> None:
    record = {"id": "utt-1", "source_audio_path": "audio/noisy.wav"}
    assert attach_audio_path(record, {"utt-1": "audio/clean.wav"})["audio_path"] == "audio/noisy.wav"


def test_parser_supports_skip_existing_candidate_outputs() -> None:
    args = build_parser().parse_args(["records.jsonl", "--output", "out.jsonl", "--skip-existing"])

    assert args.skip_existing is True


def test_skip_existing_candidate_output_avoids_rescoring(tmp_path, monkeypatch, capsys) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(
        json.dumps({"id": "utt-1", "audio_path": "audio.wav"}) + "\n" + json.dumps({"id": "utt-2", "audio_path": "audio.wav"}) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "rescored.jsonl"
    output.write_text(json.dumps({"id": "utt-1"}) + "\n" + json.dumps({"id": "utt-2"}) + "\n", encoding="utf-8")

    def fail_score_records(*_args, **_kwargs):
        raise AssertionError("score_records should not run for a complete shard")

    monkeypatch.setattr("pulse.score_candidate_prompts.score_records", fail_score_records)
    monkeypatch.setattr(
        "sys.argv",
        ["score_candidate_prompts", str(records), "--output", str(output), "--skip-existing"],
    )

    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["skipped_existing"] is True
    assert payload["records"] == 2
