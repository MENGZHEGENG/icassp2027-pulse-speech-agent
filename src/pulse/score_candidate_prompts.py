from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from pulse.asr_extract import decode_audio, default_device, existing_output_is_complete
from pulse.candidate_rescore import candidate_prompt_texts
from pulse.records import read_jsonl, write_jsonl

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional ASR dependency
    torch = None


def read_audio_lookup(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    lookup = {}
    for record in read_jsonl(path):
        if record.get("id") and record.get("audio_path"):
            lookup[str(record["id"])] = str(record["audio_path"])
    return lookup


def attach_audio_path(record: dict[str, Any], audio_lookup: dict[str, str]) -> dict[str, Any]:
    if record.get("audio_path"):
        return record
    if record.get("source_audio_path"):
        return {**record, "audio_path": record["source_audio_path"]}
    audio_path = audio_lookup.get(str(record.get("id")))
    if audio_path is None:
        raise ValueError(f"Missing audio_path for record {record.get('id')!r}; pass --audio-records if ASR output omitted it.")
    return {**record, "audio_path": audio_path}


def candidate_mean_logprob(model: Any, processor: Any, input_features: Any, text: str, device: str) -> float:
    labels = processor.tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        outputs = model(input_features=input_features, labels=labels)
    return -float(outputs.loss.detach().cpu().item())


def iter_shard(records: list[dict[str, Any]], *, shard_index: int, num_shards: int, max_samples: int | None) -> list[dict[str, Any]]:
    selected_records: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        if record_index % num_shards != shard_index:
            continue
        if max_samples is not None and len(selected_records) >= max_samples:
            break
        selected_records.append(record)
    return selected_records


def score_records(
    records: list[dict[str, Any]],
    *,
    audio_lookup: dict[str, str],
    model_name: str,
    device: str,
) -> list[dict[str, Any]]:
    if torch is None:
        raise RuntimeError("pulse.score_candidate_prompts requires optional ASR dependencies.")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()
    output: list[dict[str, Any]] = []
    for record in tqdm(records, desc="candidate rescoring"):
        record_with_audio = attach_audio_path(record, audio_lookup)
        candidates = candidate_prompt_texts(record_with_audio)
        audio_array, sampling_rate = decode_audio({"path": record_with_audio["audio_path"]})
        inputs = processor(audio_array, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        candidate_scores = []
        for candidate in candidates:
            score = candidate_mean_logprob(model, processor, input_features, candidate["text"], device)
            candidate_scores.append({**candidate, "mean_logprob": score, "score": score})
        output.append({**record, "audio_path": record_with_audio["audio_path"], "candidate_scores": candidate_scores})
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score entity-substituted candidate prompts under a Whisper acoustic model.")
    parser.add_argument("records", type=Path, help="ASR task records to augment.")
    parser.add_argument("--audio-records", type=Path, help="Source records containing audio_path fields keyed by id.")
    parser.add_argument("--model", default="openai/whisper-small.en")
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/candidate_rescored.jsonl"))
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    input_rows = read_jsonl(args.records)
    rows_for_shard = iter_shard(input_rows, shard_index=args.shard_index, num_shards=args.num_shards, max_samples=args.max_samples)
    if args.skip_existing and existing_output_is_complete(args.output, len(rows_for_shard)):
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "records": len(rows_for_shard),
                    "input_records": len(input_rows),
                    "model": args.model,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "skipped_existing": True,
                },
                indent=2,
            )
        )
        return
    rows = score_records(
        rows_for_shard,
        audio_lookup=read_audio_lookup(args.audio_records),
        model_name=args.model,
        device=args.device,
    )
    write_jsonl(args.output, rows)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "records": len(rows),
                "input_records": len(input_rows),
                "model": args.model,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
