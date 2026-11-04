from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from pulse.candidate_rescore import evaluate_rescored_confirmations, parse_thresholds
from pulse.evaluate_policy_records import PhoneRiskMode
from pulse.records import read_jsonl
from pulse.policy import PulsePolicy


def group_value(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if value is not None:
        return str(value)
    if field == "voice_condition":
        return str(record.get("tts", {}).get("voice", "unknown"))
    return "unknown"


def choose_safe_threshold(
    records: list[dict[str, Any]],
    thresholds: list[float],
    policy: PulsePolicy | None = None,
    *,
    phone_risk_mode: PhoneRiskMode | None = None,
    phone_guard_threshold: float | None = None,
) -> dict[str, Any]:
    policy = policy or PulsePolicy()
    rows = [
        {
            "margin_threshold": threshold,
            **evaluate_rescored_confirmations(
                records,
                policy=policy,
                margin_threshold=threshold,
                phone_risk_mode=phone_risk_mode,
                phone_guard_threshold=phone_guard_threshold,
            ),
        }
        for threshold in thresholds
    ]
    safe_rows = [row for row in rows if row["accepted_unsafe_wrong"] == 0]
    if not safe_rows:
        return min(rows, key=lambda row: (row["accepted_unsafe_wrong"], row["deferral_rate"]))
    return max(safe_rows, key=lambda row: (row["accepted_confirmations"], -row["deferral_rate"], -row["margin_threshold"]))


def leave_one_group_out(
    records: list[dict[str, Any]],
    *,
    group_field: str,
    thresholds: list[float],
    policy: PulsePolicy | None = None,
    safety_buffer: float = 0.0,
    minimum_threshold: float | None = None,
    phone_risk_mode: PhoneRiskMode | None = None,
    phone_guard_threshold: float | None = None,
) -> list[dict[str, Any]]:
    policy = policy or PulsePolicy()
    groups = sorted({group_value(record, group_field) for record in records})
    rows: list[dict[str, Any]] = []
    for heldout_group in groups:
        train_records = [record for record in records if group_value(record, group_field) != heldout_group]
        heldout_records = [record for record in records if group_value(record, group_field) == heldout_group]
        selected = choose_safe_threshold(
            train_records,
            thresholds,
            policy,
            phone_risk_mode=phone_risk_mode,
            phone_guard_threshold=phone_guard_threshold,
        )
        selected_threshold = calibrated_threshold(thresholds, float(selected["margin_threshold"]), safety_buffer, minimum_threshold)
        heldout = evaluate_rescored_confirmations(
            heldout_records,
            policy=policy,
            margin_threshold=selected_threshold,
            phone_risk_mode=phone_risk_mode,
            phone_guard_threshold=phone_guard_threshold,
        )
        rows.append(
            {
                "heldout_group": heldout_group,
                "train_count": len(train_records),
                "heldout_count": len(heldout_records),
                "selected_margin_threshold": selected_threshold,
                "raw_selected_margin_threshold": selected["margin_threshold"],
                "safety_buffer": safety_buffer,
                "minimum_threshold": minimum_threshold,
                "train_deferral_rate": selected["deferral_rate"],
                "train_accepted_confirmations": selected["accepted_confirmations"],
                "train_accepted_unsafe_wrong": selected["accepted_unsafe_wrong"],
                **{f"heldout_{key}": value for key, value in heldout.items()},
            }
        )
    return rows


def buffered_threshold(thresholds: list[float], selected_threshold: float, safety_buffer: float) -> float:
    target = selected_threshold + safety_buffer
    for threshold in sorted(thresholds):
        if threshold >= target - 1e-12:
            return threshold
    return max(thresholds) if thresholds else selected_threshold


def calibrated_threshold(
    thresholds: list[float],
    selected_threshold: float,
    safety_buffer: float,
    minimum_threshold: float | None,
) -> float:
    buffered = buffered_threshold(thresholds, selected_threshold, safety_buffer)
    if minimum_threshold is None or buffered >= minimum_threshold:
        return buffered
    return buffered_threshold(thresholds, minimum_threshold, 0.0)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    total = sum(int(row["heldout_count"]) for row in rows)
    if total == 0:
        return {"heldout_total": 0}
    weighted_keys = [
        "heldout_task_success",
        "heldout_wrong_action_rate",
        "heldout_unsafe_wrong_action_rate",
        "heldout_clarification_rate",
        "heldout_confirmation_rate",
        "heldout_deferral_rate",
        "heldout_accepted_confirmation_rate",
    ]
    aggregate: dict[str, float] = {"heldout_total": float(total)}
    for key in weighted_keys:
        aggregate[key] = sum(float(row[key]) * int(row["heldout_count"]) for row in rows) / total
    aggregate["heldout_accepted_confirmations"] = sum(float(row["heldout_accepted_confirmations"]) for row in rows)
    aggregate["heldout_accepted_wrong"] = sum(float(row["heldout_accepted_wrong"]) for row in rows)
    aggregate["heldout_accepted_unsafe_wrong"] = sum(float(row["heldout_accepted_unsafe_wrong"]) for row in rows)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leave-one-group-out calibration for candidate-rescoring margins.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--group-field", default="voice_condition")
    parser.add_argument("--margin-thresholds", default="0.0,0.1,0.2,0.3,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5")
    parser.add_argument("--safety-buffer", type=float, default=0.0)
    parser.add_argument("--minimum-threshold", type=float, default=None)
    parser.add_argument("--confirm-retrieval-risk", action="store_true")
    parser.add_argument("--phone-risk-mode", choices=["max", "ignore", "commit_guard"], default=None)
    parser.add_argument("--phone-guard-threshold", type=float, default=None)
    parser.add_argument("--output-csv", type=Path, default=Path("results/candidate_margin_calibration.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("results/candidate_margin_calibration.json"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policy = PulsePolicy(confirm_retrieval_phonetic_risk=args.confirm_retrieval_risk)
    rows = leave_one_group_out(
        read_jsonl(args.records),
        group_field=args.group_field,
        thresholds=parse_thresholds(args.margin_thresholds),
        policy=policy,
        safety_buffer=args.safety_buffer,
        minimum_threshold=args.minimum_threshold,
        phone_risk_mode=args.phone_risk_mode,
        phone_guard_threshold=args.phone_guard_threshold,
    )
    summary = aggregate_rows(rows)
    write_csv(args.output_csv, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "rows": rows,
        "summary": summary,
        "phone_risk_mode": args.phone_risk_mode,
        "phone_guard_threshold": args.phone_guard_threshold if args.phone_risk_mode == "commit_guard" else None,
    }
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({**payload, "csv": str(args.output_csv)}, indent=2))


if __name__ == "__main__":
    main()
