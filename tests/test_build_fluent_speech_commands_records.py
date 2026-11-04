from pathlib import Path

import pytest

from pulse.build_fluent_speech_commands_records import build_records, build_summary


def test_build_records_from_fsc_csv(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    wav = audio_root / "wavs" / "spk1" / "utt1.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"placeholder")
    csv_path = tmp_path / "train_data.csv"
    csv_path.write_text(
        "path,speakerId,transcription,action,object,location\n"
        "wavs/spk1/utt1.wav,spk1,turn on the kitchen lights,activate,lights,kitchen\n",
        encoding="utf-8",
    )

    records = build_records([csv_path], audio_root=audio_root, require_existing_audio=True)

    assert records == [
        {
            "id": "fsc-train-000001-utt1",
            "base_task_id": "fsc-activate-lights-kitchen",
            "task_id": "fsc-activate-lights-kitchen",
            "dataset": "fluent_speech_commands",
            "split": "train",
            "audio_path": str(wav),
            "text": "turn on the kitchen lights",
            "prompt_text": "turn on the kitchen lights",
            "action": "activate",
            "object": "lights",
            "location": "kitchen",
            "slots": {"action": "activate", "object": "lights", "location": "kitchen"},
            "source_csv": str(csv_path),
            "source_row": 2,
            "unsafe_if_confused": False,
            "speaker_id": "spk1",
            "voice_condition": "human:spk1",
        }
    ]
    assert build_summary(records) == {
        "record_count": 1,
        "speaker_count": 1,
        "task_count": 1,
        "splits": ["train"],
        "slot_keys": ["action", "location", "object"],
        "has_required_slots": True,
    }


def test_build_records_requires_slots(tmp_path: Path) -> None:
    csv_path = tmp_path / "test.csv"
    csv_path.write_text("path,transcription,action,object\na.wav,turn on lights,activate,lights\n", encoding="utf-8")

    with pytest.raises(ValueError, match="action, object, and location"):
        build_records([csv_path])
