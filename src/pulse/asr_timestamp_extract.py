from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

from tqdm import tqdm

from pulse.asr_extract import (
    count_expected_shard_records,
    decode_audio,
    default_device,
    existing_output_is_complete,
    extra_records_fields,
    iter_shard,
    load_dataset_split,
)
from pulse.records import read_jsonl

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal dev environments
    torch = None

PREFIX_SOURCE = "asr_token_timestamp"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract Whisper token-timestamp prefixes for streaming PULSE evaluation.")
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--records", type=Path)
    parser.add_argument("--dataset-config", default="clean")
    parser.add_argument("--split", default="test")
    parser.add_argument("--audio-field", default="audio")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--language", default="english")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--num-return-sequences", type=int, default=5)
    parser.add_argument("--min-prefix-seconds", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--prefetch-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_beams < 1:
        raise ValueError("--num-beams must be positive")
    if args.num_return_sequences < 1:
        raise ValueError("--num-return-sequences must be positive")
    if args.num_return_sequences > args.num_beams:
        raise ValueError("--num-return-sequences cannot exceed --num-beams")


def token_text(tokenizer: Any, token_id: int) -> str:
    tokens = tokenizer.convert_ids_to_tokens([int(token_id)])
    return str(tokens[0] if isinstance(tokens, list) else tokens)


def is_timestamp_or_special(token: str) -> bool:
    return token.startswith("<|") and token.endswith("|>")


def decode_prefix(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return str(tokenizer.decode(list(token_ids), skip_special_tokens=True)).strip()


def normalize_sequence_rows(values: Sequence[Any]) -> list[list[Any]]:
    if not values:
        return []
    first = values[0]
    if isinstance(first, (int, float)):
        return [list(values)]
    return [list(row) for row in values]


def prefix_fraction(prefix_seconds: float, full_duration_seconds: float) -> float:
    if full_duration_seconds <= 0.0:
        return 1.0
    return max(0.0, min(1.0, prefix_seconds / full_duration_seconds))


def build_prefix_entries(
    *,
    tokenizer: Any,
    sequence_ids: Sequence[Any],
    token_timestamps: Sequence[Any],
    full_duration_seconds: float,
    min_prefix_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    sequence_rows = normalize_sequence_rows(sequence_ids)
    timestamp_rows = normalize_sequence_rows(token_timestamps)
    if not sequence_rows or not timestamp_rows:
        return []

    top_sequence = [int(item) for item in sequence_rows[0]]
    top_timestamps = [float(item) for item in timestamp_rows[0]]
    entries: list[dict[str, Any]] = []
    last_text = ""
    full_texts = [decode_prefix(tokenizer, [int(item) for item in row]) for row in sequence_rows]
    sequence_length = min(len(top_sequence), len(top_timestamps))

    for token_index in range(sequence_length):
        token_id = int(top_sequence[token_index])
        token = token_text(tokenizer, token_id)
        if is_timestamp_or_special(token):
            continue
        prefix_seconds = float(top_timestamps[token_index])
        if prefix_seconds < min_prefix_seconds:
            continue
        text = decode_prefix(tokenizer, top_sequence[: token_index + 1])
        if not text or text == last_text:
            continue
        hypotheses = prefix_hypotheses_at_time(
            tokenizer=tokenizer,
            sequence_rows=sequence_rows,
            timestamp_rows=timestamp_rows,
            cutoff_seconds=prefix_seconds,
        )
        if not hypotheses:
            continue
        entries.append(
            {
                "prefix_fraction": prefix_fraction(prefix_seconds, full_duration_seconds),
                "prefix_duration_seconds": prefix_seconds,
                "decode_wall_time_seconds": None,
                "token_index": token_index + 1,
                "token": token,
                "hypotheses": hypotheses,
            }
        )
        last_text = text

    full_hypotheses = [
        {"rank": rank, "text": text, "score": None}
        for rank, text in enumerate(full_texts, start=1)
        if text
    ]
    full_text = full_texts[0] if full_texts else ""
    if full_text and (not entries or str(entries[-1]["hypotheses"][0]["text"]) != full_text):
        entries.append(
            {
                "prefix_fraction": 1.0,
                "prefix_duration_seconds": full_duration_seconds,
                "decode_wall_time_seconds": None,
                "token_index": len(top_sequence),
                "token": "<final>",
                "hypotheses": full_hypotheses,
            }
        )
    return entries


def prefix_hypotheses_at_time(
    *,
    tokenizer: Any,
    sequence_rows: Sequence[Sequence[Any]],
    timestamp_rows: Sequence[Sequence[Any]],
    cutoff_seconds: float,
) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    for rank, (sequence, timestamps) in enumerate(zip(sequence_rows, timestamp_rows), start=1):
        prefix_ids: list[int] = []
        for token_id, timestamp in zip(sequence, timestamps):
            if float(timestamp) <= cutoff_seconds:
                prefix_ids.append(int(token_id))
        text = decode_prefix(tokenizer, prefix_ids)
        hypotheses.append({"rank": rank, "text": text, "score": None})
    return hypotheses


def decoding_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "num_beams": args.num_beams,
        "num_return_sequences": args.num_return_sequences,
        "language": args.language,
        "task": args.task,
        "return_timestamps": True,
        "return_token_timestamps": True,
        "min_prefix_seconds": args.min_prefix_seconds,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if torch is None:
        raise RuntimeError("pulse.asr_timestamp_extract requires the optional ASR dependencies; install pulse-speech-agent[asr].")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dataset = read_jsonl(args.records) if args.records else load_dataset_split(args.dataset, args.dataset_config, args.split)
    expected_records = count_expected_shard_records(dataset, args.shard_index, args.num_shards, args.max_samples)
    if args.skip_existing and existing_output_is_complete(args.output, expected_records):
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "records": expected_records,
                    "skipped_existing": True,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "prefix_source": PREFIX_SOURCE,
                },
                indent=2,
            )
        )
        return
    processor = WhisperProcessor.from_pretrained(args.model)
    if args.prefetch_only:
        WhisperForConditionalGeneration.from_pretrained(args.model)
        print(
            json.dumps(
                {
                    "prefetched": True,
                    "dataset": args.dataset,
                    "dataset_config": args.dataset_config,
                    "split": args.split,
                    "model": args.model,
                    "dataset_length": len(dataset) if hasattr(dataset, "__len__") else None,
                },
                indent=2,
            )
        )
        return

    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(args.device)
    model.eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task=args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_name(f"{args.output.name}.tmp")
    record_count = 0
    iterator = iter_shard(dataset, args.shard_index, args.num_shards, args.max_samples)
    with temp_output.open("w", encoding="utf-8") as output_handle:
        for sample_index, sample in tqdm(iterator, desc=f"timestamp shard {args.shard_index}/{args.num_shards}"):
            audio = {"path": sample["audio_path"]} if args.records else sample[args.audio_field]
            audio_array, sampling_rate = decode_audio(audio)
            full_duration_seconds = len(audio_array) / sampling_rate
            inputs = processor(
                audio_array,
                sampling_rate=sampling_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            input_features = inputs.input_features.to(args.device)
            attention_mask = inputs.attention_mask.to(args.device)
            started_at = time.perf_counter()
            with torch.inference_mode():
                outputs = model.generate(
                    input_features,
                    attention_mask=attention_mask,
                    forced_decoder_ids=forced_decoder_ids,
                    num_beams=args.num_beams,
                    num_return_sequences=args.num_return_sequences,
                    return_timestamps=True,
                    return_token_timestamps=True,
                    return_dict_in_generate=True,
                )
            elapsed_seconds = time.perf_counter() - started_at
            sequences = outputs["sequences"].detach().cpu().tolist()
            timestamps = outputs["token_timestamps"].detach().cpu().tolist()
            prefix_entries = build_prefix_entries(
                tokenizer=processor.tokenizer,
                sequence_ids=sequences,
                token_timestamps=timestamps,
                full_duration_seconds=full_duration_seconds,
                min_prefix_seconds=args.min_prefix_seconds,
            )
            final_hypotheses = [
                {"rank": rank, "text": decode_prefix(processor.tokenizer, sequence), "score": None}
                for rank, sequence in enumerate(sequences, start=1)
            ]
            record = {
                "id": str(sample.get("id", sample_index)),
                "sample_index": sample_index,
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "split": args.split,
                "model": args.model,
                "decoding": "token_timestamp_beam",
                "decoding_config": decoding_config(args),
                "reference": sample.get("text") if args.records else sample.get(args.text_field),
                "full_duration_seconds": full_duration_seconds,
                "full_decode_wall_time_seconds": elapsed_seconds,
                "prefix_source": PREFIX_SOURCE,
                "hypotheses": final_hypotheses,
                "prefix_hypotheses": prefix_entries,
            }
            for key, value in extra_records_fields(sample, audio_field=args.audio_field, text_field=args.text_field).items():
                record.setdefault(key, value)
            output_handle.write(json.dumps(record) + "\n")
            output_handle.flush()
            record_count += 1
            if str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
    temp_output.replace(args.output)
    print(json.dumps({"output": str(args.output), "records": record_count, "prefix_source": PREFIX_SOURCE}, indent=2))


if __name__ == "__main__":
    main()
