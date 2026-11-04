from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pulse.records import read_jsonl, write_jsonl


def tts_record(record: dict[str, Any], audio_path: Path, voice: str) -> dict[str, Any]:
    return {
        **record,
        "audio_path": str(audio_path),
        "text": record["prompt_text"],
        "tts": {
            "engine": "macos_say",
            "voice": voice,
            "format": "wav_pcm_s16le_16000_mono",
        },
    }


def synthesize_with_macos_say(text: str, output_wav: Path, voice: str) -> None:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pulse-tts-") as temporary_dir:
        temporary_aiff = Path(temporary_dir) / "utterance.aiff"
        subprocess.run(["say", "-v", voice, "-o", str(temporary_aiff), text], check=True)
        subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(temporary_aiff), str(output_wav)], check=True)


def build_tts_records(
    records: list[dict[str, Any]],
    *,
    audio_output_dir: Path,
    voice: str,
    max_samples: int | None,
    synthesize: bool,
) -> list[dict[str, Any]]:
    selected_records = records[:max_samples] if max_samples is not None else records
    output_records: list[dict[str, Any]] = []
    for index, record in enumerate(selected_records):
        record_id = str(record.get("id", index))
        output_wav = audio_output_dir / f"{record_id}.wav"
        if synthesize:
            synthesize_with_macos_say(record["prompt_text"], output_wav, voice)
        output_records.append(tts_record(record, output_wav, voice))
    return output_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate macOS say TTS audio for PULSE benchmark records.")
    parser.add_argument("tasks", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/ambiguous_tasks_tts.jsonl"))
    parser.add_argument("--audio-output-dir", type=Path, default=Path("data/ambiguous_tasks_tts_audio"))
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--records-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = build_tts_records(
        read_jsonl(args.tasks),
        audio_output_dir=args.audio_output_dir,
        voice=args.voice,
        max_samples=args.max_samples,
        synthesize=not args.records_only,
    )
    write_jsonl(args.output, records)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "audio_output_dir": str(args.audio_output_dir),
                "records": len(records),
                "voice": args.voice,
                "synthesized": not args.records_only,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
