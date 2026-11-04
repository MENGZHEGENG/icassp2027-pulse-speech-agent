from __future__ import annotations

import argparse
import json
from pathlib import Path

from pulse.records import write_jsonl


def build_records(root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for transcript_path in sorted(root.rglob("*.trans.txt")):
        for line in transcript_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utterance_id, text = line.split(" ", 1)
            audio_path = transcript_path.parent / f"{utterance_id}.flac"
            if audio_path.exists():
                records.append({"id": utterance_id, "audio_path": str(audio_path), "text": text})
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a JSONL records from a local LibriSpeech split.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = build_records(args.root)
    write_jsonl(args.output, records)
    print(json.dumps({"root": str(args.root), "output": str(args.output), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
