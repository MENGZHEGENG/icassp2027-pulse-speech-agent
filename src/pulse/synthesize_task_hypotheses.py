from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from pulse.records import read_jsonl, write_jsonl


def replace_entity(prompt_text: str, source: str, target: str) -> str:
    return prompt_text.replace(source, target)


def synthesize_records(records: list[dict], error_rate: float, seed: int) -> list[dict]:
    rng = random.Random(seed)
    output = []
    for record in records:
        true_text = record["prompt_text"]
        confused_text = replace_entity(true_text, record["true_entity"], record["confusable_entity"])
        top_is_wrong = rng.random() < error_rate
        if top_is_wrong:
            hypotheses = [
                {"rank": 1, "text": confused_text, "score": -0.10},
                {"rank": 2, "text": true_text, "score": -0.16},
                {"rank": 3, "text": confused_text, "score": -0.30},
            ]
        else:
            hypotheses = [
                {"rank": 1, "text": true_text, "score": -0.08},
                {"rank": 2, "text": true_text, "score": -0.15},
                {"rank": 3, "text": confused_text, "score": -0.35},
            ]
        output.append({**record, "hypotheses": hypotheses, "synthetic_top_error": top_is_wrong})
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Attach synthetic ASR hypotheses to ambiguity benchmark tasks.")
    parser.add_argument("tasks", type=Path)
    parser.add_argument("--error-rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20270817)
    parser.add_argument("--output", type=Path, default=Path("data/task_hypotheses.jsonl"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = synthesize_records(read_jsonl(args.tasks), args.error_rate, args.seed)
    write_jsonl(args.output, records)
    print(json.dumps({"output": str(args.output), "records": len(records), "error_rate": args.error_rate}, indent=2))


if __name__ == "__main__":
    main()
