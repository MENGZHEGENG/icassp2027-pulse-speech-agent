from pathlib import Path

from pulse.build_tts_records import build_tts_records, tts_record


def test_tts_record_preserves_task_fields(tmp_path: Path) -> None:
    record = {"id": "x-1", "prompt_text": "Call Shawn.", "true_entity": "shawn"}
    output = tts_record(record, tmp_path / "x-1.wav", "Samantha")
    assert output["text"] == "Call Shawn."
    assert output["audio_path"].endswith("x-1.wav")
    assert output["true_entity"] == "shawn"
    assert output["tts"]["engine"] == "macos_say"


def test_build_tts_records_can_skip_synthesis(tmp_path: Path) -> None:
    records = [
        {"id": "one", "prompt_text": "Call Shawn."},
        {"id": "two", "prompt_text": "Call Sean."},
    ]
    output = build_tts_records(
        records,
        audio_output_dir=tmp_path / "audio",
        voice="Samantha",
        max_samples=1,
        synthesize=False,
    )
    assert len(output) == 1
    assert output[0]["id"] == "one"
    assert output[0]["audio_path"] == str(tmp_path / "audio" / "one.wav")
