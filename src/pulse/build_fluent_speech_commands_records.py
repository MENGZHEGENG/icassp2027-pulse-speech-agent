from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

from pulse.records import write_jsonl

TEXT_FIELDS = ("transcription", "text", "sentence", "utterance", "phrase")
PATH_FIELDS = ("path", "audio_path", "wav_path", "file", "filename")
SPEAKER_FIELDS = ("speakerId", "speaker_id", "speaker", "participant_id", "user_id")
ACTION_FIELDS = ("action", "intent", "act")
OBJECT_FIELDS = ("object", "object_name", "entity", "slot_object")
LOCATION_FIELDS = ("location", "room", "slot_location")
SPLIT_FIELDS = ("split", "partition", "subset")
ID_FIELDS = ("id", "utt_id", "utterance_id", "recording_id")


def safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    token = re.sub(r"-+", "-", token).strip("-._")
    return token or "unknown"


def row_value(row: dict[str, str], fields: Iterable[str]) -> str:
    lower_lookup = {key.lower(): value for key, value in row.items()}
    for field in fields:
        value = row.get(field)
        if value is None:
            value = lower_lookup.get(field.lower())
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def read_csv_rows(paths: list[Path]) -> list[tuple[Path, int, dict[str, str]]]:
    rows: list[tuple[Path, int, dict[str, str]]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV {path} has no header")
            for row_index, row in enumerate(reader, start=2):
                rows.append((path, row_index, {key: value or "" for key, value in row.items()}))
    return rows


def resolve_audio_path(value: str, *, audio_root: Path | None) -> Path:
    path = Path(value)
    if path.is_absolute() or audio_root is None:
        return path
    return audio_root / path


def infer_split(path: Path, row: dict[str, str]) -> str:
    split = row_value(row, SPLIT_FIELDS)
    if split:
        return safe_token(split).lower()
    stem = path.stem.lower()
    for candidate in ("train", "valid", "validation", "dev", "test"):
        if candidate in stem:
            return "valid" if candidate in {"valid", "validation", "dev"} else candidate
    return "unknown"


def build_records(csv_paths: list[Path], *, audio_root: Path | None = None, require_existing_audio: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for csv_path, row_index, row in read_csv_rows(csv_paths):
        text = row_value(row, TEXT_FIELDS)
        audio_value = row_value(row, PATH_FIELDS)
        action = row_value(row, ACTION_FIELDS)
        object_name = row_value(row, OBJECT_FIELDS)
        location = row_value(row, LOCATION_FIELDS)
        if not text:
            raise ValueError(f"{csv_path}:{row_index} is missing transcription/text")
        if not audio_value:
            raise ValueError(f"{csv_path}:{row_index} is missing audio path")
        if not action or not object_name or not location:
            raise ValueError(f"{csv_path}:{row_index} must include action, object, and location slots")
        audio_path = resolve_audio_path(audio_value, audio_root=audio_root)
        if require_existing_audio and not audio_path.exists():
            raise ValueError(f"{csv_path}:{row_index} audio does not exist: {audio_path}")
        split = infer_split(csv_path, row)
        speaker_id = row_value(row, SPEAKER_FIELDS)
        row_id = row_value(row, ID_FIELDS)
        if not row_id:
            row_id = f"{split}-{row_index - 1:06d}-{safe_token(audio_path.stem)}"
        record_id = f"fsc-{safe_token(row_id)}"
        if record_id in seen_ids:
            record_id = f"{record_id}-{len(seen_ids) + 1}"
        seen_ids.add(record_id)
        task_id = f"fsc-{safe_token(action)}-{safe_token(object_name)}-{safe_token(location)}"
        record: dict[str, Any] = {
            "id": record_id,
            "base_task_id": task_id,
            "task_id": task_id,
            "dataset": "fluent_speech_commands",
            "split": split,
            "audio_path": str(audio_path),
            "text": text,
            "prompt_text": text,
            "action": action,
            "object": object_name,
            "location": location,
            "slots": {"action": action, "object": object_name, "location": location},
            "source_csv": str(csv_path),
            "source_row": row_index,
            "unsafe_if_confused": False,
        }
        if speaker_id:
            record["speaker_id"] = speaker_id
            record["voice_condition"] = f"human:{speaker_id}"
        records.append(record)
    return records


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    speakers = sorted({str(record.get("speaker_id", "")) for record in records if str(record.get("speaker_id", "")).strip()})
    tasks = sorted({str(record.get("base_task_id", "")) for record in records if str(record.get("base_task_id", "")).strip()})
    splits = sorted({str(record.get("split", "")) for record in records if str(record.get("split", "")).strip()})
    slot_keys = sorted({key for record in records for key in (record.get("slots") or {})})
    return {
        "record_count": len(records),
        "speaker_count": len(speakers),
        "task_count": len(tasks),
        "splits": splits,
        "slot_keys": slot_keys,
        "has_required_slots": all(key in slot_keys for key in ("action", "object", "location")),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a JSONL records from local Fluent Speech Commands metadata CSV files.")
    parser.add_argument("csvs", nargs="+", type=Path)
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--require-existing-audio", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = build_records(args.csvs, audio_root=args.audio_root, require_existing_audio=args.require_existing_audio)
    write_jsonl(args.output, records)
    summary = build_summary(records) | {"output": str(args.output), "csvs": [str(path) for path in args.csvs]}
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
