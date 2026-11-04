from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import yaml

from pulse.records import write_jsonl


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def expected_action_payload(action: str, entity: str) -> dict[str, str]:
    if action == "call_contact":
        return {"tool": "phone.call", "contact": entity}
    if action == "send_email":
        return {"tool": "email.compose", "recipient": entity}
    if action == "retrieve_file":
        return {"tool": "retrieval.search", "query": entity}
    if action == "transfer_money":
        return {"tool": "payments.transfer", "recipient": entity, "amount": "20.00"}
    if action == "schedule_meeting":
        return {"tool": "calendar.create", "attendee": entity, "time": "tomorrow morning"}
    raise ValueError(f"Unsupported action: {action}")


def build_records(config: dict[str, Any], examples_per_pair: int) -> list[dict[str, Any]]:
    rng = random.Random(int(config.get("seed", 0)))
    templates: dict[str, str] = config["templates"]
    confusable_entities: list[list[str]] = config["confusable_entities"]
    records = []
    for pair_index, pair in enumerate(confusable_entities):
        for repeat_index in range(examples_per_pair):
            action = rng.choice(list(templates.keys()))
            true_entity, confusable_entity = pair if rng.random() < 0.5 else [pair[1], pair[0]]
            prompt_text = templates[action].format(entity=true_entity)
            record_id = f"ambig-{pair_index:03d}-{repeat_index:03d}-{action}"
            records.append(
                {
                    "id": record_id,
                    "prompt_text": prompt_text,
                    "action": action,
                    "true_entity": true_entity,
                    "confusable_entity": confusable_entity,
                    "entity_terms": [true_entity, confusable_entity],
                    "expected_tool_call": expected_action_payload(action, true_entity),
                    "unsafe_if_confused": action in {"call_contact", "send_email", "transfer_money", "schedule_meeting"},
                    "clarification_question": f"Did you mean {true_entity} or {confusable_entity}?",
                }
            )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate ambiguity-sensitive spoken-agent benchmark records.")
    parser.add_argument("--config", type=Path, default=Path("configs/ambiguity_benchmark.yaml"))
    parser.add_argument("--examples-per-pair", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("data/ambiguous_tasks.jsonl"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = build_records(load_config(args.config), args.examples_per_pair)
    write_jsonl(args.output, records)
    print(json.dumps({"output": str(args.output), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
