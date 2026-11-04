from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from tqdm import tqdm

from pulse.asr_extract import (
    count_expected_shard_records,
    decode_audio,
    decoding_label,
    default_device,
    existing_output_is_complete,
    extra_records_fields,
    extract_sequence_scores,
    iter_shard,
    load_dataset_split,
    parse_float_list,
    validate_decoding_args,
)
from pulse.records import read_jsonl

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal dev environments
    torch = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract ASR N-best hypotheses for real audio prefixes.")
    parser.add_argument("--dataset", default="openslr/librispeech_asr")
    parser.add_argument("--records", type=Path)
    parser.add_argument("--dataset-config", default="clean")
    parser.add_argument("--split", default="test")
    parser.add_argument("--audio-field", default="audio")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--language", default="english")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--num-beam-groups", type=int, default=1)
    parser.add_argument("--diversity-penalty", type=float, default=0.0)
    parser.add_argument("--custom-generate")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--length-penalties", default="")
    parser.add_argument("--num-return-sequences", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--no-output-scores",
        action="store_true",
        help="Skip sequence-score collection to reduce generation memory for large N-best runs.",
    )
    parser.add_argument("--prefix-fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--min-prefix-seconds", type=float, default=0.25)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--prefetch-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def parse_prefix_fractions(value: str) -> list[float]:
    fractions = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not fractions:
        raise ValueError("--prefix-fractions must contain at least one value")
    for fraction in fractions:
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("prefix fractions must be in (0, 1]")
    if 1.0 not in fractions:
        fractions.append(1.0)
    return sorted(fractions)


def audio_prefix(array: np.ndarray, sampling_rate: int, fraction: float, min_seconds: float) -> np.ndarray:
    min_samples = max(1, int(round(min_seconds * sampling_rate)))
    target_samples = max(min_samples, int(round(len(array) * fraction)))
    target_samples = min(len(array), target_samples)
    return np.asarray(array[:target_samples], dtype=np.float32)


def decoding_config(args: argparse.Namespace, length_penalties: list[float]) -> dict[str, Any]:
    return {
        "num_beams": args.num_beams,
        "num_beam_groups": args.num_beam_groups,
        "diversity_penalty": args.diversity_penalty,
        "num_return_sequences": args.num_return_sequences,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "custom_generate": args.custom_generate,
        "trust_remote_code": args.trust_remote_code,
        "output_scores": not args.no_output_scores,
        "length_penalties": length_penalties,
        "prefix_fractions": parse_prefix_fractions(args.prefix_fractions),
        "min_prefix_seconds": args.min_prefix_seconds,
    }


def generate_texts(
    *,
    model: Any,
    processor: Any,
    input_features: Any,
    args: argparse.Namespace,
    forced_decoder_ids: Any,
    length_penalties: list[float],
) -> tuple[list[str], list[float | None]]:
    generation_kwargs = {
        "forced_decoder_ids": forced_decoder_ids,
        "num_return_sequences": args.num_return_sequences,
        "return_dict_in_generate": True,
        "output_scores": not args.no_output_scores,
    }
    if args.max_new_tokens is not None:
        generation_kwargs["max_new_tokens"] = args.max_new_tokens
    if args.do_sample:
        generation_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})
    else:
        generation_kwargs.update({"num_beams": args.num_beams})
        if args.num_beam_groups > 1:
            generation_kwargs.update(
                {"num_beam_groups": args.num_beam_groups, "diversity_penalty": args.diversity_penalty}
            )
    if args.custom_generate:
        generation_kwargs["custom_generate"] = args.custom_generate
    if args.trust_remote_code:
        generation_kwargs["trust_remote_code"] = True

    if length_penalties:
        texts: list[str] = []
        scores: list[float | None] = []
        for length_penalty in length_penalties:
            outputs = model.generate(input_features, length_penalty=length_penalty, **generation_kwargs)
            pass_texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
            texts.extend(pass_texts)
            scores.extend([None] * len(pass_texts))
        return texts, scores

    outputs = model.generate(input_features, **generation_kwargs)
    texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
    return texts, extract_sequence_scores(outputs, len(texts))


def main() -> None:
    args = build_parser().parse_args()
    validate_decoding_args(args)
    prefix_fractions = parse_prefix_fractions(args.prefix_fractions)
    if torch is None:
        raise RuntimeError("pulse.asr_prefix_extract requires the optional ASR dependencies; install pulse-speech-agent[asr].")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dataset = read_jsonl(args.records) if args.records else load_dataset_split(args.dataset, args.dataset_config, args.split)
    expected_records = count_expected_shard_records(dataset, args.shard_index, args.num_shards, args.max_samples)
    if args.skip_existing and existing_output_is_complete(args.output, expected_records):
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "records": expected_records,
                    "prefixes": len(prefix_fractions),
                    "skipped_existing": True,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
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
                    "prefix_fractions": prefix_fractions,
                    "dataset_length": len(dataset) if hasattr(dataset, "__len__") else None,
                },
                indent=2,
            )
        )
        return

    model = WhisperForConditionalGeneration.from_pretrained(args.model).to(args.device)
    model.eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    length_penalties = parse_float_list(args.length_penalties)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_name(f"{args.output.name}.tmp")
    record_count = 0
    iterator = iter_shard(dataset, args.shard_index, args.num_shards, args.max_samples)
    with temp_output.open("w", encoding="utf-8") as output_handle:
        for sample_index, sample in tqdm(iterator, desc=f"prefix shard {args.shard_index}/{args.num_shards}"):
            audio = {"path": sample["audio_path"]} if args.records else sample[args.audio_field]
            audio_array, sampling_rate = decode_audio(audio)
            full_duration_seconds = len(audio_array) / sampling_rate
            prefix_entries = []
            for fraction in prefix_fractions:
                started_at = time.perf_counter()
                prefix_array = audio_prefix(audio_array, sampling_rate, fraction, args.min_prefix_seconds)
                inputs = processor(prefix_array, sampling_rate=sampling_rate, return_tensors="pt")
                input_features = inputs.input_features.to(args.device)
                with torch.inference_mode():
                    texts, scores = generate_texts(
                        model=model,
                        processor=processor,
                        input_features=input_features,
                        args=args,
                        forced_decoder_ids=forced_decoder_ids,
                        length_penalties=length_penalties,
                    )
                elapsed_seconds = time.perf_counter() - started_at
                prefix_entries.append(
                    {
                        "prefix_fraction": fraction,
                        "prefix_duration_seconds": len(prefix_array) / sampling_rate,
                        "decode_wall_time_seconds": elapsed_seconds,
                        "hypotheses": [
                            {"rank": rank, "text": text, "score": score}
                            for rank, (text, score) in enumerate(zip(texts, scores), start=1)
                        ],
                    }
                )
            record = {
                "id": str(sample.get("id", sample_index)),
                "sample_index": sample_index,
                "dataset": args.dataset,
                "dataset_config": args.dataset_config,
                "split": args.split,
                "model": args.model,
                "decoding": f"prefix_{decoding_label(args)}",
                "decoding_config": decoding_config(args, length_penalties),
                "reference": sample.get("text") if args.records else sample.get(args.text_field),
                "full_duration_seconds": full_duration_seconds,
                "hypotheses": prefix_entries[-1]["hypotheses"],
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
    print(json.dumps({"output": str(args.output), "records": record_count, "prefixes": len(prefix_fractions)}, indent=2))


if __name__ == "__main__":
    main()
