from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from pulse.records import read_jsonl, write_jsonl


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + 1e-12))


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    return np.asarray(audio * (10 ** (gain_db / 20.0)), dtype=np.float32)


def mix_white_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=audio.shape).astype(np.float32)
    noise_scale = rms(audio) / (rms(noise) * (10 ** (snr_db / 20.0)))
    return np.asarray(audio + noise * noise_scale, dtype=np.float32)


def resample_linear(audio: np.ndarray, speed: float) -> np.ndarray:
    if speed <= 0:
        raise ValueError("speed must be positive")
    if math.isclose(speed, 1.0):
        return np.asarray(audio, dtype=np.float32)
    if len(audio) <= 1:
        return np.asarray(audio, dtype=np.float32)
    output_length = max(1, int(round(len(audio) / speed)))
    source_positions = np.linspace(0.0, 1.0, num=len(audio), endpoint=True)
    target_positions = np.linspace(0.0, 1.0, num=output_length, endpoint=True)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def degrade_audio(
    audio: np.ndarray,
    *,
    snr_db: float | None,
    speed: float,
    gain_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    degraded = np.asarray(audio, dtype=np.float32)
    degraded = resample_linear(degraded, speed)
    if snr_db is not None:
        degraded = mix_white_noise(degraded, snr_db, rng)
    if gain_db:
        degraded = apply_gain(degraded, gain_db)
    return np.clip(degraded, -1.0, 1.0).astype(np.float32)


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    return cleaned.strip("-") or "sample"


def augmentation_suffix(snr_db: float | None, speed: float, gain_db: float) -> str:
    parts = []
    if snr_db is not None:
        parts.append(f"snr{snr_db:g}")
    if not math.isclose(speed, 1.0):
        parts.append(f"speed{speed:g}")
    if gain_db:
        parts.append(f"gain{gain_db:g}")
    return "-".join(parts) or "copy"


def augment_records(
    records: list[dict[str, Any]],
    *,
    output_audio_dir: Path,
    snr_db: float | None,
    speed: float,
    gain_db: float,
    seed: int,
    max_samples: int | None,
    suffix: str | None = None,
) -> list[dict[str, Any]]:
    import soundfile as sf

    rng = np.random.default_rng(seed)
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    label = suffix or augmentation_suffix(snr_db, speed, gain_db)
    augmented_records: list[dict[str, Any]] = []
    selected_records = records[:max_samples] if max_samples is not None else records

    for index, record in enumerate(selected_records):
        source_audio_path = Path(record["audio_path"])
        audio, sampling_rate = sf.read(source_audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        degraded = degrade_audio(audio, snr_db=snr_db, speed=speed, gain_db=gain_db, rng=rng)
        source_id = str(record.get("id", index))
        output_path = output_audio_dir / f"{safe_stem(source_id)}.{label}.wav"
        sf.write(output_path, degraded, sampling_rate)
        augmented_records.append(
            {
                **record,
                "id": f"{source_id}-{label}",
                "audio_path": str(output_path),
                "source_audio_path": str(source_audio_path),
                "augmentation": {
                    "snr_db": snr_db,
                    "speed": speed,
                    "gain_db": gain_db,
                    "seed": seed,
                    "label": label,
                },
            }
        )
    return augmented_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an acoustically degraded local-audio records.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/librispeech_test_clean_degraded.jsonl"))
    parser.add_argument("--audio-output-dir", type=Path, default=Path("data/degraded_audio"))
    parser.add_argument("--snr-db", type=float, default=10.0)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--gain-db", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20270817)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--suffix")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = augment_records(
        read_jsonl(args.records),
        output_audio_dir=args.audio_output_dir,
        snr_db=None if args.no_noise else args.snr_db,
        speed=args.speed,
        gain_db=args.gain_db,
        seed=args.seed,
        max_samples=args.max_samples,
        suffix=args.suffix,
    )
    write_jsonl(args.output, records)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "audio_output_dir": str(args.audio_output_dir),
                "records": len(records),
                "snr_db": None if args.no_noise else args.snr_db,
                "speed": args.speed,
                "gain_db": args.gain_db,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
