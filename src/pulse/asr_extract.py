from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from pulse.records import write_jsonl
from pulse.records import read_jsonl

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal dev environments
    torch = None


def default_device() -> str:
    return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract ASR N-best hypotheses for PULSE.")
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
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument(
        "--no-output-scores",
        action="store_true",
        help="Skip sequence-score collection to reduce generation memory for large N-best runs.",
    )
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--min-sample-index", type=int, default=0)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--prefetch-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--resume-partial",
        action="store_true",
        help="Append missing records to an existing partial JSONL instead of rewriting the shard at the end.",
    )
    parser.add_argument(
        "--rewrite-invalid-partial",
        action="store_true",
        help="Move an invalid partial JSONL aside and regenerate the shard when resuming fails validation.",
    )
    parser.add_argument(
        "--skip-audio-errors",
        action="store_true",
        help="Write explicit error records for unreadable audio rows and continue the shard.",
    )
    return parser


def parse_float_list(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def validate_decoding_args(args: argparse.Namespace) -> None:
    if args.num_return_sequences < 1:
        raise ValueError("--num-return-sequences must be positive")
    if args.num_beams < 1:
        raise ValueError("--num-beams must be positive")
    if args.num_beam_groups < 1:
        raise ValueError("--num-beam-groups must be positive")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    max_new_tokens = getattr(args, "max_new_tokens", None)
    if max_new_tokens is not None and max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive when set")
    length_penalties = parse_float_list(getattr(args, "length_penalties", ""))
    if args.do_sample:
        if length_penalties:
            raise ValueError("--length-penalties is only supported for beam decoding")
        return
    if args.num_return_sequences > args.num_beams:
        raise ValueError("--num-return-sequences cannot exceed --num-beams for beam decoding")
    if args.num_beams % args.num_beam_groups != 0:
        raise ValueError("--num-beams must be divisible by --num-beam-groups")
    if args.num_beam_groups > 1 and args.diversity_penalty <= 0:
        raise ValueError("--diversity-penalty must be positive when --num-beam-groups > 1")


def decoding_label(args: argparse.Namespace) -> str:
    if args.do_sample:
        return "sample"
    if parse_float_list(getattr(args, "length_penalties", "")):
        return "length_sweep_beam"
    if args.num_beam_groups > 1:
        return "diverse_beam"
    return "beam"


def load_dataset_split(dataset_name: str, dataset_config: str | None, split: str) -> Any:
    from datasets import Audio, load_dataset

    dataset = load_dataset(dataset_name, dataset_config, split=split) if dataset_config else load_dataset(dataset_name, split=split)
    features = getattr(dataset, "features", {})
    for field_name, feature in features.items():
        if feature.__class__.__name__ == "Audio":
            return dataset.cast_column(field_name, Audio(decode=False))
    return dataset


def decode_audio(audio: dict[str, Any], target_sampling_rate: int = 16000) -> tuple[np.ndarray, int]:
    import soundfile as sf

    if "array" in audio and audio["array"] is not None:
        array = np.asarray(audio["array"], dtype=np.float32)
        sampling_rate = int(audio.get("sampling_rate") or target_sampling_rate)
    elif audio.get("bytes") is not None:
        array, sampling_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    elif audio.get("path"):
        array, sampling_rate = sf.read(audio["path"], dtype="float32")
    else:
        raise ValueError("Audio sample must contain array, bytes, or path")

    if array.ndim > 1:
        array = array.mean(axis=1)
    if sampling_rate != target_sampling_rate:
        import librosa

        array = librosa.resample(array, orig_sr=sampling_rate, target_sr=target_sampling_rate)
        sampling_rate = target_sampling_rate
    return np.asarray(array, dtype=np.float32), sampling_rate


def iter_shard(dataset: Any, shard_index: int, num_shards: int, max_samples: int, min_sample_index: int = 0):
    emitted = 0
    for sample_index, sample in enumerate(dataset):
        if sample_index < min_sample_index:
            continue
        if sample_index % num_shards != shard_index:
            continue
        if emitted >= max_samples:
            break
        emitted += 1
        yield sample_index, sample


def count_expected_shard_records(dataset: Any, shard_index: int, num_shards: int, max_samples: int, min_sample_index: int = 0) -> int:
    return sum(1 for _sample_index, _sample in iter_shard(dataset, shard_index, num_shards, max_samples, min_sample_index))


def count_valid_jsonl_records(path: Path) -> int | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    count = 0
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                json.loads(stripped)
                count += 1
    except (OSError, json.JSONDecodeError):
        return None
    return count


def existing_output_is_complete(
    path: Path,
    expected_records: int,
    expected_sample_indices: set[int] | None = None,
    expected_hypothesis_count: int | None = None,
    allow_error_records: bool = False,
) -> bool:
    if count_valid_jsonl_records(path) != expected_records:
        return False
    try:
        completed_sample_indices = read_completed_sample_indices(
            path,
            expected_sample_indices,
            expected_hypothesis_count=expected_hypothesis_count,
            allow_error_records=allow_error_records,
        )
    except ValueError:
        return False
    if expected_sample_indices is None:
        return True
    return completed_sample_indices == expected_sample_indices


def read_completed_sample_indices(
    path: Path,
    expected_sample_indices: set[int] | None,
    *,
    expected_hypothesis_count: int | None = None,
    allow_error_records: bool = False,
    expected_unique_field: str | None = None,
) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    completed: set[int] = set()
    seen_unique_values: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                if expected_sample_indices is not None:
                    sample_index = record.get("sample_index")
                    if not isinstance(sample_index, int):
                        raise ValueError(f"{path} line {line_number} is missing integer sample_index")
                    if sample_index not in expected_sample_indices:
                        raise ValueError(f"{path} line {line_number} has sample_index outside this shard: {sample_index}")
                    if sample_index in completed:
                        raise ValueError(f"{path} line {line_number} repeats sample_index {sample_index}")
                    completed.add(sample_index)
                if expected_unique_field is not None:
                    unique_value = record.get(expected_unique_field)
                    if unique_value is None:
                        raise ValueError(f"{path} line {line_number} is missing {expected_unique_field}")
                    unique_key = str(unique_value)
                    if unique_key in seen_unique_values:
                        raise ValueError(f"{path} line {line_number} repeats {expected_unique_field} {unique_key}")
                    seen_unique_values.add(unique_key)
                if expected_hypothesis_count is not None:
                    if allow_error_records and record.get("asr_error"):
                        continue
                    hypotheses = record.get("hypotheses")
                    if not isinstance(hypotheses, list) or len(hypotheses) < expected_hypothesis_count:
                        raise ValueError(
                            f"{path} line {line_number} has {len(hypotheses or [])} hypotheses; "
                            f"expected at least {expected_hypothesis_count}"
                        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot resume from invalid JSONL at {path}: line {exc.lineno}") from exc
    return completed


def quarantine_invalid_partial_output(path: Path, reason: Exception | str, *, now_utc: datetime | None = None) -> Path:
    stamp = (now_utc or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    candidate = path.with_name(f"{path.name}.invalid-{stamp}")
    suffix_index = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.invalid-{stamp}.{suffix_index}")
        suffix_index += 1
    path.rename(candidate)
    reason_path = candidate.with_name(f"{candidate.name}.reason.txt")
    reason_path.write_text(str(reason) + "\n", encoding="utf-8")
    return candidate


def prepare_resume_sample_indices(
    path: Path,
    expected_sample_indices: set[int],
    *,
    rewrite_invalid_partial: bool,
    expected_hypothesis_count: int | None = None,
    allow_error_records: bool = False,
    expected_unique_field: str | None = None,
) -> tuple[set[int], Path | None]:
    try:
        return read_completed_sample_indices(
            path,
            expected_sample_indices,
            expected_hypothesis_count=expected_hypothesis_count,
            allow_error_records=allow_error_records,
            expected_unique_field=expected_unique_field,
        ), None
    except ValueError as exc:
        if not rewrite_invalid_partial:
            raise
        return set(), quarantine_invalid_partial_output(path, exc)


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def batched(items: list[tuple[int, Any]], batch_size: int) -> list[list[tuple[int, Any]]]:
    return [items[start : start + batch_size] for start in range(0, len(items), batch_size)]


def extract_sequence_scores(outputs: Any, count: int) -> list[float | None]:
    sequence_scores = getattr(outputs, "sequences_scores", None)
    if sequence_scores is None:
        return [None] * count
    return [float(score.detach().cpu().item()) for score in sequence_scores[:count]]


def processor_kwargs_for_batch(sampling_rate: int, batch_size: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "sampling_rate": sampling_rate,
        "return_tensors": "pt",
        "return_attention_mask": True,
    }
    if batch_size > 1:
        kwargs["padding"] = True
    return kwargs


def extra_records_fields(sample: dict[str, Any], *, audio_field: str, text_field: str) -> dict[str, Any]:
    excluded_fields = {"audio_path", audio_field, text_field, "audio", "text"}
    return {key: value for key, value in sample.items() if key not in excluded_fields}


def audio_reference(sample: dict[str, Any], *, use_records: bool, audio_field: str) -> dict[str, Any]:
    return {"path": sample["audio_path"]} if use_records else sample[audio_field]


def base_output_record(
    sample_index: int,
    sample: dict[str, Any],
    args: argparse.Namespace,
    *,
    generation_seed: int | None,
    length_penalties: list[float],
    retry_oom_without_cache: bool,
    clear_cache_each_batch: bool,
    output_scores_enabled: bool,
) -> dict[str, Any]:
    return {
        "id": str(sample.get("id", sample_index)),
        "sample_index": sample_index,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "model": args.model,
        "decoding": decoding_label(args),
        "decoding_config": {
            "num_beams": args.num_beams,
            "num_beam_groups": args.num_beam_groups,
            "diversity_penalty": args.diversity_penalty,
            "num_return_sequences": args.num_return_sequences,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": generation_seed,
            "custom_generate": args.custom_generate,
            "trust_remote_code": args.trust_remote_code,
            "length_penalties": length_penalties,
            "retry_oom_without_cache": retry_oom_without_cache,
            "clear_cache_each_batch": clear_cache_each_batch,
            "output_scores": output_scores_enabled,
            "skip_audio_errors": args.skip_audio_errors,
        },
        "reference": sample.get("text") if args.records else sample.get(args.text_field),
    }


def output_record(
    sample_index: int,
    sample: dict[str, Any],
    args: argparse.Namespace,
    *,
    texts: list[str],
    scores: list[float | None],
    generation_seed: int | None,
    length_penalties: list[float],
    retry_oom_without_cache: bool,
    clear_cache_each_batch: bool,
    output_scores_enabled: bool,
) -> dict[str, Any]:
    record = base_output_record(
        sample_index,
        sample,
        args,
        generation_seed=generation_seed,
        length_penalties=length_penalties,
        retry_oom_without_cache=retry_oom_without_cache,
        clear_cache_each_batch=clear_cache_each_batch,
        output_scores_enabled=output_scores_enabled,
    )
    record["hypotheses"] = [
        {"rank": rank, "text": text, "score": score} for rank, (text, score) in enumerate(zip(texts, scores), start=1)
    ]
    for key, value in extra_records_fields(sample, audio_field=args.audio_field, text_field=args.text_field).items():
        record.setdefault(key, value)
    return record


def audio_error_record(
    sample_index: int,
    sample: dict[str, Any],
    args: argparse.Namespace,
    exc: BaseException,
    *,
    generation_seed: int | None,
    length_penalties: list[float],
    retry_oom_without_cache: bool,
    clear_cache_each_batch: bool,
    output_scores_enabled: bool,
) -> dict[str, Any]:
    record = base_output_record(
        sample_index,
        sample,
        args,
        generation_seed=generation_seed,
        length_penalties=length_penalties,
        retry_oom_without_cache=retry_oom_without_cache,
        clear_cache_each_batch=clear_cache_each_batch,
        output_scores_enabled=output_scores_enabled,
    )
    record.update(
        {
            "hypotheses": [],
            "asr_error": {
                "stage": "audio_decode",
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
    )
    audio = audio_reference(sample, use_records=bool(args.records), audio_field=args.audio_field)
    if isinstance(audio, dict) and audio.get("path"):
        record["asr_error"]["audio_path"] = str(audio["path"])
    for key, value in extra_records_fields(sample, audio_field=args.audio_field, text_field=args.text_field).items():
        record.setdefault(key, value)
    return record


def resolve_torch_dtype(torch_module: Any, dtype_name: str, device: str) -> Any | None:
    if dtype_name == "auto":
        return torch_module.float16 if str(device).startswith("cuda") else None
    return {
        "float32": torch_module.float32,
        "float16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
    }[dtype_name]


def model_input_dtype(model: Any) -> Any | None:
    model_dtype = getattr(model, "dtype", None)
    if torch is not None and model_dtype in {torch.float16, torch.bfloat16}:
        return model_dtype
    return None


def env_flag(name: str, *, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() not in {"0", "false", "no", "off"}


def is_cuda_oom(exc: BaseException, torch_module: Any | None = None) -> bool:
    module = torch if torch_module is None else torch_module
    oom_type = getattr(module, "OutOfMemoryError", None) if module is not None else None
    if oom_type is not None and isinstance(exc, oom_type):
        return True
    return "cuda out of memory" in str(exc).lower()


def clear_cuda_cache(torch_module: Any | None = None) -> None:
    module = torch if torch_module is None else torch_module
    cuda = getattr(module, "cuda", None) if module is not None else None
    if cuda is None:
        return
    try:
        if cuda.is_available():
            cuda.empty_cache()
    except (AttributeError, RuntimeError):
        return


def set_generation_seed(torch_module: Any, seed: int, shard_index: int) -> int:
    shard_seed = seed + shard_index
    np.random.seed(shard_seed)
    torch_module.manual_seed(shard_seed)
    cuda = getattr(torch_module, "cuda", None)
    if cuda is not None:
        try:
            if cuda.is_available():
                cuda.manual_seed_all(shard_seed)
        except (AttributeError, RuntimeError):
            pass
    return shard_seed


def generate_with_oom_retry(
    model: Any,
    input_features: Any,
    generation_kwargs: dict[str, Any],
    *,
    retry_without_cache: bool = True,
    torch_module: Any | None = None,
) -> Any:
    try:
        return model.generate(input_features, **generation_kwargs)
    except Exception as exc:
        if not retry_without_cache or generation_kwargs.get("use_cache") is False or not is_cuda_oom(exc, torch_module):
            raise
        clear_cuda_cache(torch_module)
        retry_kwargs = {**generation_kwargs, "use_cache": False}
        print(
            json.dumps(
                {
                    "event": "cuda_oom_retry_without_cache",
                    "num_beams": retry_kwargs.get("num_beams"),
                    "num_return_sequences": retry_kwargs.get("num_return_sequences"),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return model.generate(input_features, **retry_kwargs)


def main() -> None:
    args = build_parser().parse_args()
    validate_decoding_args(args)
    if torch is None:
        raise RuntimeError("pulse.asr_extract requires the optional ASR dependencies; install pulse-speech-agent[asr].")
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    dataset = read_jsonl(args.records) if args.records else load_dataset_split(args.dataset, args.dataset_config, args.split)
    shard_items = list(iter_shard(dataset, args.shard_index, args.num_shards, args.max_samples, args.min_sample_index))
    expected_records = len(shard_items)
    expected_sample_indices = {sample_index for sample_index, _sample in shard_items}
    if args.skip_existing and existing_output_is_complete(
        args.output,
        expected_records,
        expected_sample_indices,
        expected_hypothesis_count=args.num_return_sequences,
        allow_error_records=args.skip_audio_errors,
    ):
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "records": expected_records,
                    "skipped_existing": True,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                },
                indent=2,
            )
        )
        return
    rewritten_invalid_partial: Path | None = None
    if args.resume_partial:
        resume_unique_field = "id" if args.records else None
        completed_sample_indices, rewritten_invalid_partial = prepare_resume_sample_indices(
            args.output,
            expected_sample_indices,
            rewrite_invalid_partial=args.rewrite_invalid_partial,
            expected_hypothesis_count=args.num_return_sequences,
            allow_error_records=args.skip_audio_errors,
            expected_unique_field=resume_unique_field,
        )
        if rewritten_invalid_partial is not None:
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "rewritten_invalid_partial": str(rewritten_invalid_partial),
                        "shard_index": args.shard_index,
                        "num_shards": args.num_shards,
                    },
                    indent=2,
                ),
                flush=True,
            )
    else:
        completed_sample_indices = set()
    resumed_existing_count = len(completed_sample_indices)
    processor = WhisperProcessor.from_pretrained(args.model)
    torch_dtype = resolve_torch_dtype(torch, args.torch_dtype, args.device)
    model_kwargs = {"torch_dtype": torch_dtype} if torch_dtype is not None else {}
    if args.prefetch_only:
        WhisperForConditionalGeneration.from_pretrained(args.model, **model_kwargs)
        print(
            json.dumps(
                {
                    "prefetched": True,
                    "dataset": args.dataset,
                    "dataset_config": args.dataset_config,
                    "split": args.split,
                    "model": args.model,
                    "torch_dtype": args.torch_dtype,
                    "dataset_length": len(dataset) if hasattr(dataset, "__len__") else None,
                },
                indent=2,
            )
        )
        return
    model = WhisperForConditionalGeneration.from_pretrained(args.model, **model_kwargs).to(args.device)
    model.eval()
    forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.language, task="transcribe")
    length_penalties = parse_float_list(args.length_penalties)
    input_dtype = model_input_dtype(model)
    retry_oom_without_cache = env_flag("PULSE_ASR_RETRY_OOM_WITHOUT_CACHE", default=True)
    clear_cache_each_batch = env_flag("PULSE_ASR_CLEAR_CACHE_EACH_BATCH", default=True)
    output_scores_enabled = False if args.no_output_scores else env_flag("PULSE_ASR_OUTPUT_SCORES", default=True)
    generation_seed = None if args.seed is None else set_generation_seed(torch, args.seed, args.shard_index)

    records = []
    skipped_audio_error_records = 0
    pending_items = [(sample_index, sample) for sample_index, sample in shard_items if sample_index not in completed_sample_indices]
    for batch_items in tqdm(
        batched(pending_items, args.batch_size),
        desc=f"shard {args.shard_index}/{args.num_shards}",
        unit="batch",
    ):
        audio_arrays = []
        valid_batch_items = []
        error_records_by_sample_index = {}
        sampling_rate = 16000
        for sample_index, sample in batch_items:
            audio = audio_reference(sample, use_records=bool(args.records), audio_field=args.audio_field)
            try:
                audio_array, sampling_rate = decode_audio(audio)
            except Exception as exc:
                if not args.skip_audio_errors:
                    raise
                error_records_by_sample_index[sample_index] = audio_error_record(
                    sample_index,
                    sample,
                    args,
                    exc,
                    generation_seed=generation_seed,
                    length_penalties=length_penalties,
                    retry_oom_without_cache=retry_oom_without_cache,
                    clear_cache_each_batch=clear_cache_each_batch,
                    output_scores_enabled=output_scores_enabled,
                )
                skipped_audio_error_records += 1
                print(
                    json.dumps(
                        {
                            "event": "audio_decode_error_skipped",
                            "sample_index": sample_index,
                            "id": str(sample.get("id", sample_index)),
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                continue
            audio_arrays.append(audio_array)
            valid_batch_items.append((sample_index, sample))
        if not valid_batch_items:
            for sample_index, _sample in batch_items:
                record = error_records_by_sample_index[sample_index]
                if args.resume_partial:
                    append_jsonl_record(args.output, record)
                    completed_sample_indices.add(sample_index)
                else:
                    records.append(record)
            continue
        processor_kwargs = processor_kwargs_for_batch(sampling_rate, len(audio_arrays))
        inputs = processor(audio_arrays if len(audio_arrays) > 1 else audio_arrays[0], **processor_kwargs)
        input_features = inputs.input_features.to(args.device)
        if input_dtype is not None:
            input_features = input_features.to(dtype=input_dtype)
        with torch.inference_mode():
            generation_kwargs = {
                "forced_decoder_ids": forced_decoder_ids,
                "num_return_sequences": args.num_return_sequences,
                "return_dict_in_generate": True,
                "output_scores": output_scores_enabled,
            }
            if args.max_new_tokens is not None:
                generation_kwargs["max_new_tokens"] = args.max_new_tokens
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                generation_kwargs["attention_mask"] = attention_mask.to(args.device)
            if args.do_sample:
                generation_kwargs.update(
                    {
                        "do_sample": True,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                    }
                )
            else:
                generation_kwargs.update({"num_beams": args.num_beams})
                if args.num_beam_groups > 1:
                    generation_kwargs.update(
                        {
                            "num_beam_groups": args.num_beam_groups,
                            "diversity_penalty": args.diversity_penalty,
                        }
                    )
            if args.custom_generate:
                generation_kwargs["custom_generate"] = args.custom_generate
            if args.trust_remote_code:
                generation_kwargs["trust_remote_code"] = True
            if length_penalties:
                texts_by_sample = [[] for _sample in valid_batch_items]
                scores_by_sample = [[] for _sample in valid_batch_items]
                for length_penalty in length_penalties:
                    pass_kwargs = {**generation_kwargs, "length_penalty": length_penalty}
                    outputs = generate_with_oom_retry(
                        model,
                        input_features,
                        pass_kwargs,
                        retry_without_cache=retry_oom_without_cache,
                    )
                    pass_texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
                    for batch_offset in range(len(valid_batch_items)):
                        start = batch_offset * args.num_return_sequences
                        end = start + args.num_return_sequences
                        texts_by_sample[batch_offset].extend(pass_texts[start:end])
                        scores_by_sample[batch_offset].extend([None] * len(pass_texts[start:end]))
                    outputs = None
                    if clear_cache_each_batch:
                        clear_cuda_cache(torch)
            else:
                outputs = generate_with_oom_retry(
                    model,
                    input_features,
                    generation_kwargs,
                    retry_without_cache=retry_oom_without_cache,
                )
                all_texts = processor.batch_decode(outputs.sequences, skip_special_tokens=True)
                all_scores = extract_sequence_scores(outputs, len(all_texts))
                texts_by_sample = []
                scores_by_sample = []
                for batch_offset in range(len(valid_batch_items)):
                    start = batch_offset * args.num_return_sequences
                    end = start + args.num_return_sequences
                    texts_by_sample.append(all_texts[start:end])
                    scores_by_sample.append(all_scores[start:end])
                outputs = None
        generated_records_by_sample_index = {}
        for batch_offset, (sample_index, sample) in enumerate(valid_batch_items):
            generated_records_by_sample_index[sample_index] = output_record(
                sample_index,
                sample,
                args,
                texts=texts_by_sample[batch_offset],
                scores=scores_by_sample[batch_offset],
                generation_seed=generation_seed,
                length_penalties=length_penalties,
                retry_oom_without_cache=retry_oom_without_cache,
                clear_cache_each_batch=clear_cache_each_batch,
                output_scores_enabled=output_scores_enabled,
            )
        for sample_index, _sample in batch_items:
            record = generated_records_by_sample_index.get(sample_index) or error_records_by_sample_index[sample_index]
            if args.resume_partial:
                append_jsonl_record(args.output, record)
                completed_sample_indices.add(sample_index)
            else:
                records.append(record)
        input_features = None
        inputs = None
        generation_kwargs = None
        if clear_cache_each_batch:
            clear_cuda_cache(torch)
    if args.resume_partial:
        record_count = count_valid_jsonl_records(args.output) or 0
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "records": record_count,
                    "resumed_existing_records": resumed_existing_count,
                    "skipped_audio_error_records": skipped_audio_error_records,
                    "rewritten_invalid_partial": str(rewritten_invalid_partial) if rewritten_invalid_partial else None,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                },
                indent=2,
            )
        )
    else:
        write_jsonl(args.output, records)
        print(json.dumps({"output": str(args.output), "records": len(records), "skipped_audio_error_records": skipped_audio_error_records}, indent=2))


if __name__ == "__main__":
    main()
