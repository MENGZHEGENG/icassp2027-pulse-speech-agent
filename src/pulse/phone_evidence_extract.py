from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any
import unicodedata

from tqdm import tqdm

from pulse.asr_extract import (
    count_expected_shard_records,
    decode_audio,
    default_device,
    existing_output_is_complete,
    iter_shard,
)
from pulse.records import read_jsonl
from pulse.uncertainty import arpabet_phrase_key, edit_distance, normalize_text

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - optional phone-ASR dependency
    torch = None

ARPABET_TO_IPA = {
    "AA": "ɑ",
    "AE": "æ",
    "AH": "ʌ",
    "AO": "ɔ",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "ɛ",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "ɡ",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

COMBINING_OR_SUPRASEGMENTAL = {"ˈ", "ˌ", "ː", ".", " ", "_", "-", "|"}


def arpabet_to_ipa(phones: tuple[str, ...]) -> str:
    return "".join(ARPABET_TO_IPA.get(phone, phone.lower()) for phone in phones)


def normalize_phone_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(
        char
        for char in normalized
        if not unicodedata.combining(char) and char not in COMBINING_OR_SUPRASEGMENTAL and not char.isspace()
    )


def phone_distance_ratio(left: str, right: str) -> float:
    left_norm = normalize_phone_text(left)
    right_norm = normalize_phone_text(right)
    denominator = max(1, len(left_norm), len(right_norm))
    return edit_distance(list(left_norm), list(right_norm)) / denominator


def best_substring_phone_distance_ratio(predicted_phone_text: str, candidate_phone_text: str, radius: int = 3) -> float:
    predicted = normalize_phone_text(predicted_phone_text)
    candidate = normalize_phone_text(candidate_phone_text)
    if not candidate:
        return 1.0
    if not predicted:
        return 1.0
    if candidate in predicted:
        return 0.0
    min_width = max(1, len(candidate) - radius)
    max_width = min(len(predicted), len(candidate) + radius)
    best = 1.0
    for width in range(min_width, max_width + 1):
        for start in range(0, len(predicted) - width + 1):
            window = predicted[start : start + width]
            denominator = max(1, len(candidate), len(window))
            best = min(best, edit_distance(list(candidate), list(window)) / denominator)
    return best


def entity_phone_candidates(entity_terms: list[str] | None) -> list[dict[str, Any]]:
    candidates = []
    seen = set()
    for entity in entity_terms or []:
        normalized = normalize_text(entity)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        arpabet = arpabet_phrase_key(entity)
        ipa = arpabet_to_ipa(arpabet) if arpabet else normalize_phone_text(entity)
        candidates.append(
            {
                "entity": entity,
                "normalized_entity": normalized,
                "arpabet": list(arpabet),
                "ipa": ipa,
                "candidate_source": "arpabet_to_ipa" if arpabet else "normalized_text_fallback",
            }
        )
    return candidates


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_phone_evidence(
    predicted_phone_text: str,
    *,
    entity_terms: list[str] | None,
    margin_scale: float = 0.25,
) -> dict[str, Any]:
    candidates = entity_phone_candidates(entity_terms)
    if not candidates:
        return {
            "learned_phonetic_risk": 0.0,
            "predicted_entity": None,
            "margin": None,
            "candidate_scores": [],
            "risk_reason": "no entity candidates",
        }

    scored = []
    for candidate in candidates:
        distance = best_substring_phone_distance_ratio(predicted_phone_text, str(candidate["ipa"]))
        scored.append({**candidate, "distance": distance, "score": -distance})
    scored.sort(key=lambda item: (float(item["distance"]), str(item["normalized_entity"])))
    best = scored[0]
    second_distance = float(scored[1]["distance"]) if len(scored) > 1 else 1.0
    best_distance = float(best["distance"])
    margin = max(0.0, second_distance - best_distance)

    if len(scored) == 1:
        risk = clamp01(best_distance)
        reason = "single candidate distance risk"
    else:
        risk = 1.0 - clamp01(margin / max(margin_scale, 1e-9))
        reason = "small phone-candidate margin" if risk > 0 else "clear phone-candidate margin"

    return {
        "learned_phonetic_risk": risk,
        "predicted_entity": best["entity"],
        "margin": margin,
        "candidate_scores": scored,
        "risk_reason": reason,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract learned/CTC phone evidence for PULSE records.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="facebook/wav2vec2-lv-60-espeak-cv-ft")
    parser.add_argument("--audio-field", default="audio")
    parser.add_argument("--predicted-phone-field")
    parser.add_argument("--entity-terms-field", default="entity_terms")
    parser.add_argument("--margin-scale", type=float, default=0.25)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device", default=default_device())
    parser.add_argument("--prefetch-only", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser


def decode_phone_text(
    record: dict[str, Any],
    args: argparse.Namespace,
    feature_extractor: Any = None,
    tokenizer: Any = None,
    model: Any = None,
) -> tuple[str, float | None]:
    if args.predicted_phone_field:
        return str(record.get(args.predicted_phone_field, "")), None
    if torch is None:
        raise RuntimeError("pulse.phone_evidence_extract requires torch/transformers unless --predicted-phone-field is used.")
    audio = {"path": record["audio_path"]} if record.get("audio_path") else record[args.audio_field]
    audio_array, sampling_rate = decode_audio(audio)
    started_at = time.perf_counter()
    inputs = feature_extractor(audio_array, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(args.device)
    with torch.inference_mode():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_text = tokenizer.batch_decode(predicted_ids)[0]
    return predicted_text, time.perf_counter() - started_at


def main() -> None:
    args = build_parser().parse_args()
    records = read_jsonl(args.records)
    expected_records = count_expected_shard_records(records, args.shard_index, args.num_shards, args.max_samples)
    if args.skip_existing and existing_output_is_complete(args.output, expected_records):
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
    feature_extractor = None
    tokenizer = None
    model = None
    if not args.predicted_phone_field:
        if torch is None:
            raise RuntimeError("pulse.phone_evidence_extract requires torch/transformers unless --predicted-phone-field is used.")
        from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer

        feature_extractor = AutoFeatureExtractor.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, do_phonemize=False)
        if args.prefetch_only:
            AutoModelForCTC.from_pretrained(args.model)
            print(json.dumps({"prefetched": True, "model": args.model, "records": len(records)}, indent=2))
            return
        model = AutoModelForCTC.from_pretrained(args.model).to(args.device)
        model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = args.output.with_name(f"{args.output.name}.tmp")
    record_count = 0
    with temp_output.open("w", encoding="utf-8") as output_handle:
        iterator = iter_shard(records, args.shard_index, args.num_shards, args.max_samples)
        for _, record in tqdm(iterator, desc=f"phone shard {args.shard_index}/{args.num_shards}"):
            predicted_phone_text, elapsed_seconds = decode_phone_text(
                record,
                args,
                feature_extractor=feature_extractor,
                tokenizer=tokenizer,
                model=model,
            )
            score = score_phone_evidence(
                predicted_phone_text,
                entity_terms=record.get(args.entity_terms_field),
                margin_scale=args.margin_scale,
            )
            payload = {
                "id": str(record.get("id")),
                "learned_phonetic_risk": score["learned_phonetic_risk"],
                "phone_risk": score["learned_phonetic_risk"],
                "source": "ctc_phone_evidence",
                "phone_evidence": {
                    "source": "ctc_phone_evidence",
                    "model": args.model if not args.predicted_phone_field else "records_field",
                    "entity_terms": record.get(args.entity_terms_field),
                    "predicted_phone_text": predicted_phone_text,
                    "normalized_predicted_phone_text": normalize_phone_text(predicted_phone_text),
                    "decode_wall_time_seconds": elapsed_seconds,
                    **score,
                },
            }
            output_handle.write(json.dumps(payload) + "\n")
            output_handle.flush()
            record_count += 1
            if torch is not None and str(args.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
    temp_output.replace(args.output)
    print(json.dumps({"output": str(args.output), "records": record_count}, indent=2))


if __name__ == "__main__":
    main()
