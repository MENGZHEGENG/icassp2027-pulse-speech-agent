from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Literal

from pulse.evaluate_policy_records import PolicyMode, policy_mode_confidence, top_entity_is_correct
from pulse.records import read_jsonl


DEFAULT_OUTPUT_PREFIX = Path("docs/result_summaries/confidence_comparator_calibration")
ConfidenceComparatorMode = Literal[
    "scalar_asr_confidence_threshold",
    "slot_entity_confidence_threshold",
    "calibrated_confidence_only_policy",
    "fused_asr_confidence_threshold",
]
DEFAULT_MODES: tuple[ConfidenceComparatorMode, ...] = (
    "scalar_asr_confidence_threshold",
    "slot_entity_confidence_threshold",
    "calibrated_confidence_only_policy",
    "fused_asr_confidence_threshold",
)
CALIBRATION_BIN_COUNT = 10


def parse_modes(text: str) -> list[ConfidenceComparatorMode]:
    valid = set(DEFAULT_MODES)
    modes: list[ConfidenceComparatorMode] = []
    for item in text.split(","):
        mode = item.strip()
        if not mode:
            continue
        if mode not in valid:
            raise ValueError(f"unsupported confidence comparator mode: {mode}")
        modes.append(mode)  # type: ignore[arg-type]
    if not modes:
        raise ValueError("at least one confidence comparator mode is required")
    return modes


def group_records(records: list[dict[str, Any]], group_field: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        group = str(record.get(group_field, "unknown"))
        groups.setdefault(group, []).append(record)
    return dict(sorted(groups.items()))


def confidence_values(records: list[dict[str, Any]], mode: ConfidenceComparatorMode) -> list[float]:
    values: list[float] = []
    for record in records:
        value = policy_mode_confidence(record, mode)
        if value is not None:
            values.append(float(value))
    return values


def confidence_examples(records: list[dict[str, Any]], mode: ConfidenceComparatorMode) -> list[tuple[float, bool]]:
    examples: list[tuple[float, bool]] = []
    for record in records:
        value = policy_mode_confidence(record, mode)
        if value is not None:
            examples.append((float(value), top_entity_is_correct(record)))
    return examples


def rank_auc(examples: list[tuple[float, bool]]) -> float | None:
    positives = sum(1 for _, correct in examples if correct)
    negatives = len(examples) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted((score, correct) for score, correct in examples)
    rank_sum = 0.0
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(1 for _, correct in ranked[index:end] if correct)
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def expected_calibration_error(examples: list[tuple[float, bool]], *, bins: int = 10) -> float | None:
    if not examples or any(score < 0.0 or score > 1.0 for score, _ in examples):
        return None
    ece = 0.0
    total = len(examples)
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        if bin_index == bins - 1:
            bucket = [(score, correct) for score, correct in examples if lower <= score <= upper]
        else:
            bucket = [(score, correct) for score, correct in examples if lower <= score < upper]
        if not bucket:
            continue
        mean_confidence = sum(score for score, _ in bucket) / len(bucket)
        accuracy = sum(1.0 for _, correct in bucket if correct) / len(bucket)
        ece += len(bucket) / total * abs(accuracy - mean_confidence)
    return ece


def confidence_diagnostics(records: list[dict[str, Any]], mode: ConfidenceComparatorMode) -> dict[str, Any]:
    examples = confidence_examples(records, mode)
    probability_like = bool(examples) and all(0.0 <= score <= 1.0 for score, _ in examples)
    return {
        "confidence_observed_count": len(examples),
        "confidence_correct_count": sum(1 for _, correct in examples if correct),
        "confidence_error_count": sum(1 for _, correct in examples if not correct),
        "confidence_correctness_auroc": rank_auc(examples),
        "confidence_ece": expected_calibration_error(examples) if probability_like else None,
        "confidence_ece_bin_count": 10 if probability_like else None,
        "confidence_ece_applicability": "bounded_probability" if probability_like else "unbounded_score_not_probability",
    }


def fit_empirical_calibrator(
    records: list[dict[str, Any]],
    *,
    bin_count: int = CALIBRATION_BIN_COUNT,
) -> list[dict[str, float]]:
    examples = confidence_examples(records, "scalar_asr_confidence_threshold")
    if not examples:
        return []
    sorted_examples = sorted(examples, key=lambda item: item[0])
    target_bins = max(1, min(bin_count, len(sorted_examples)))
    bins: list[dict[str, float]] = []
    for bin_index in range(target_bins):
        start = bin_index * len(sorted_examples) // target_bins
        end = (bin_index + 1) * len(sorted_examples) // target_bins
        bucket = sorted_examples[start:end]
        if not bucket:
            continue
        correct = sum(1.0 for _, is_correct in bucket if is_correct)
        bins.append(
            {
                "upper_score": bucket[-1][0],
                "count": float(len(bucket)),
                "correct_count": correct,
                "probability": (correct + 0.5) / (len(bucket) + 1.0),
            }
        )
    return bins


def calibrated_probability(score: float, bins: list[dict[str, float]]) -> float | None:
    if not bins:
        return None
    for bucket in bins:
        if score <= bucket["upper_score"]:
            return bucket["probability"]
    return bins[-1]["probability"]


def with_calibrated_confidence(records: list[dict[str, Any]], bins: list[dict[str, float]]) -> list[dict[str, Any]]:
    calibrated: list[dict[str, Any]] = []
    for record in records:
        raw = policy_mode_confidence(record, "scalar_asr_confidence_threshold")
        updated = dict(record)
        if raw is not None:
            updated["calibrated_confidence"] = calibrated_probability(float(raw), bins)
        calibrated.append(updated)
    return calibrated


def threshold_grid(values: list[float]) -> list[float]:
    if not values:
        return [float("inf")]
    unique = sorted(set(values))
    epsilon = 1e-9
    candidates = [min(unique) - 1.0]
    candidates.extend(unique)
    candidates.extend(value + epsilon for value in unique)
    candidates.append(max(unique) + 1.0)
    return sorted(set(candidates))


def evaluate_mode(records: list[dict[str, Any]], mode: ConfidenceComparatorMode, threshold: float) -> dict[str, Any]:
    total = len(records)
    wrong_actions = 0
    unsafe_wrong_actions = 0
    clarifications = 0
    successes = 0
    missing_confidence = 0
    for record in records:
        confidence = policy_mode_confidence(record, mode)
        if confidence is None or confidence < threshold:
            missing_confidence += int(confidence is None)
            clarifications += 1
            successes += 1
        elif top_entity_is_correct(record):
            successes += 1
        else:
            wrong_actions += 1
            if record.get("unsafe_if_confused"):
                unsafe_wrong_actions += 1
    return {
        "count": total,
        "task_success": successes / total if total else None,
        "wrong_action_rate": wrong_actions / total if total else None,
        "unsafe_wrong_action_rate": unsafe_wrong_actions / total if total else None,
        "clarification_rate": clarifications / total if total else None,
        "confirmation_rate": 0.0 if total else None,
        "deferral_rate": clarifications / total if total else None,
        "missing_confidence_rate": missing_confidence / total if total else None,
    }


def select_threshold(
    train_records: list[dict[str, Any]],
    mode: ConfidenceComparatorMode,
    *,
    max_train_unsafe_wrong_rate: float,
) -> dict[str, Any]:
    candidates = []
    for threshold in threshold_grid(confidence_values(train_records, mode)):
        summary = evaluate_mode(train_records, mode, threshold)
        candidates.append({"threshold": threshold, "summary": summary})
    feasible = [
        row
        for row in candidates
        if float(row["summary"].get("unsafe_wrong_action_rate") or 0.0) <= max_train_unsafe_wrong_rate
    ]
    search_space = feasible or candidates
    selected = min(
        search_space,
        key=lambda row: (
            float(row["summary"].get("deferral_rate") or 1.0),
            -float(row["summary"].get("task_success") or 0.0),
            float(row["summary"].get("wrong_action_rate") or 1.0),
            float(row["threshold"]),
        ),
    )
    return {
        "threshold": selected["threshold"],
        "train_summary": selected["summary"],
        "feasible_threshold_count": len(feasible),
        "candidate_threshold_count": len(candidates),
        "met_train_safety_constraint": bool(feasible),
    }


def leave_one_group_out(
    records: list[dict[str, Any]],
    *,
    group_field: str,
    modes: list[ConfidenceComparatorMode],
    max_train_unsafe_wrong_rate: float,
    database: str,
    application_scenario: str,
) -> list[dict[str, Any]]:
    groups = group_records(records, group_field)
    rows: list[dict[str, Any]] = []
    for heldout_group, heldout_records in groups.items():
        train_records = [record for group, group_records_ in groups.items() if group != heldout_group for record in group_records_]
        for mode in modes:
            calibration_bins: list[dict[str, float]] = []
            mode_train_records = train_records
            mode_heldout_records = heldout_records
            calibration_method = "none_raw_score"
            if mode == "calibrated_confidence_only_policy":
                calibration_bins = fit_empirical_calibrator(train_records)
                mode_train_records = with_calibrated_confidence(train_records, calibration_bins)
                mode_heldout_records = with_calibrated_confidence(heldout_records, calibration_bins)
                calibration_method = "train_quantile_empirical_correctness"
            selected = select_threshold(mode_train_records, mode, max_train_unsafe_wrong_rate=max_train_unsafe_wrong_rate)
            threshold = float(selected["threshold"])
            heldout_summary = evaluate_mode(mode_heldout_records, mode, threshold)
            train_diagnostics = confidence_diagnostics(mode_train_records, mode)
            heldout_diagnostics = confidence_diagnostics(mode_heldout_records, mode)
            rows.append(
                {
                    "database": database,
                    "application_scenario": application_scenario,
                    "group_field": group_field,
                    "heldout_group": heldout_group,
                    "policy_mode": mode,
                    "train_count": len(train_records),
                    "heldout_count": len(heldout_records),
                    "selected_threshold": threshold,
                    "max_train_unsafe_wrong_rate": max_train_unsafe_wrong_rate,
                    "met_train_safety_constraint": selected["met_train_safety_constraint"],
                    "feasible_threshold_count": selected["feasible_threshold_count"],
                    "candidate_threshold_count": selected["candidate_threshold_count"],
                    "confidence_calibration_method": calibration_method,
                    "confidence_calibration_bin_count": len(calibration_bins) if calibration_bins else 0,
                    **{f"train_{key}": value for key, value in selected["train_summary"].items() if isinstance(value, (int, float)) or value is None},
                    **{f"heldout_{key}": value for key, value in heldout_summary.items() if isinstance(value, (int, float)) or value is None},
                    **{f"train_{key}": value for key, value in train_diagnostics.items()},
                    **{f"heldout_{key}": value for key, value in heldout_diagnostics.items()},
                }
            )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    modes = sorted({str(row["policy_mode"]) for row in rows})
    aggregate: list[dict[str, Any]] = []
    for mode in modes:
        mode_rows = [row for row in rows if row["policy_mode"] == mode]
        total = sum(int(row["heldout_count"]) for row in mode_rows)
        weighted_keys = [
            "heldout_task_success",
            "heldout_wrong_action_rate",
            "heldout_unsafe_wrong_action_rate",
            "heldout_clarification_rate",
            "heldout_confirmation_rate",
            "heldout_deferral_rate",
            "heldout_confidence_correctness_auroc",
            "heldout_confidence_ece",
        ]
        summary: dict[str, Any] = {
            "policy_mode": mode,
            "fold_count": len(mode_rows),
            "heldout_total": total,
            "all_train_safety_constraints_met": all(bool(row["met_train_safety_constraint"]) for row in mode_rows),
        }
        for key in weighted_keys:
            numeric_rows = [row for row in mode_rows if row.get(key) is not None]
            numeric_total = sum(int(row["heldout_count"]) for row in numeric_rows)
            summary[key] = (
                sum(float(row[key]) * int(row["heldout_count"]) for row in numeric_rows) / numeric_total
                if numeric_total
                else None
            )
        summary["heldout_confidence_ece_applicability"] = (
            "bounded_probability"
            if all(row.get("heldout_confidence_ece_applicability") == "bounded_probability" for row in mode_rows)
            else "unbounded_score_not_probability"
        )
        methods = sorted({str(row.get("confidence_calibration_method", "none_raw_score")) for row in mode_rows})
        summary["confidence_calibration_method"] = ",".join(methods)
        thresholds = [float(row["selected_threshold"]) for row in mode_rows]
        summary["min_selected_threshold"] = min(thresholds) if thresholds else None
        summary["max_selected_threshold"] = max(thresholds) if thresholds else None
        summary["all_heldout_safety_constraints_met"] = (
            summary["heldout_unsafe_wrong_action_rate"] is not None
            and float(summary["heldout_unsafe_wrong_action_rate"]) == 0.0
        )
        aggregate.append(summary)
    return aggregate


def build_report(
    records: list[dict[str, Any]],
    *,
    group_field: str,
    modes: list[ConfidenceComparatorMode],
    max_train_unsafe_wrong_rate: float,
    database: str,
    application_scenario: str,
) -> dict[str, Any]:
    rows = leave_one_group_out(
        records,
        group_field=group_field,
        modes=modes,
        max_train_unsafe_wrong_rate=max_train_unsafe_wrong_rate,
        database=database,
        application_scenario=application_scenario,
    )
    aggregate = aggregate_rows(rows)
    all_train_safe = all(row["all_train_safety_constraints_met"] for row in aggregate)
    all_heldout_safe = all(row["all_heldout_safety_constraints_met"] for row in aggregate)
    return {
        "ok": bool(rows) and all_train_safe,
        "ready_for_confidence_only_claim": bool(rows) and all_train_safe and all_heldout_safe,
        "record_count": len(records),
        "fold_count": len(group_records(records, group_field)),
        "row_count": len(rows),
        "database": database,
        "application_scenario": application_scenario,
        "group_field": group_field,
        "policy_modes": [str(mode) for mode in modes],
        "max_train_unsafe_wrong_rate": max_train_unsafe_wrong_rate,
        "threshold_protocol": "leave_one_group_out_select_min_deferral_subject_to_train_unsafe_constraint",
        "next_action": (
            "promote_calibrated_confidence_comparators_to_main_tables"
            if rows and all_train_safe and all_heldout_safe
            else "report_confidence_only_shortfall_and_keep_pulse_as_action_evidence_fusion"
            if rows and all_train_safe
            else "provide_records_with_multiple_calibration_groups"
        ),
        "aggregate": aggregate,
        "rows": rows,
    }


def cell(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if value == float("inf"):
            return "inf"
        return f"{value:.3f}"
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Confidence Comparator Calibration",
        "",
        f"OK: `{payload['ok']}`. Records: `{payload['record_count']}`. Folds: `{payload['fold_count']}`. Rows: `{payload['row_count']}`.",
        f"Ready for confidence-only claim: `{payload['ready_for_confidence_only_claim']}`.",
        f"Database: `{payload['database']}`. Application scenario: `{payload['application_scenario']}`.",
        f"Threshold protocol: `{payload['threshold_protocol']}`.",
        "",
        "## Aggregate Held-Out Metrics",
        "",
        "| Comparator | Calibration | Folds | Held-out n | Train safety met | Held-out safety met | Threshold range | Success | Unsafe wrong | Deferral | Clarify | AUROC | ECE |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["aggregate"]:
        lines.append(
            "| {mode} | {calibration} | {folds} | {total} | {train_safe} | {heldout_safe} | {low} to {high} | {success} | {unsafe} | {deferral} | {clarify} | {auroc} | {ece} |".format(
                mode=cell(row["policy_mode"]),
                calibration=cell(row["confidence_calibration_method"]),
                folds=cell(row["fold_count"]),
                total=cell(row["heldout_total"]),
                train_safe=cell(row["all_train_safety_constraints_met"]),
                heldout_safe=cell(row["all_heldout_safety_constraints_met"]),
                low=cell(row["min_selected_threshold"]),
                high=cell(row["max_selected_threshold"]),
                success=cell(row["heldout_task_success"]),
                unsafe=cell(row["heldout_unsafe_wrong_action_rate"]),
                deferral=cell(row["heldout_deferral_rate"]),
                clarify=cell(row["heldout_clarification_rate"]),
                auroc=cell(row["heldout_confidence_correctness_auroc"]),
                ece=cell(row["heldout_confidence_ece"]),
            )
        )
    if any(row.get("heldout_confidence_ece_applicability") != "bounded_probability" for row in payload["aggregate"]):
        lines.extend(
            [
                "",
                "ECE is reported only when the comparator score is already a bounded probability; raw ASR log-score comparators keep ECE blank and use AUROC for ranking quality.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(payload: dict[str, Any], *, output_prefix: Path) -> dict[str, str]:
    paths = {
        "json": output_prefix.with_suffix(".json"),
        "csv": output_prefix.with_suffix(".csv"),
        "markdown": output_prefix.with_suffix(".md"),
    }
    paths["json"].parent.mkdir(parents=True, exist_ok=True)
    paths["json"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(paths["csv"], payload["rows"])
    paths["markdown"].write_text(markdown_report(payload), encoding="utf-8")
    return {key: value.as_posix() for key, value in paths.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leave-one-group-out calibration for ASR confidence comparator thresholds.")
    parser.add_argument("records", type=Path)
    parser.add_argument("--group-field", default="voice_condition")
    parser.add_argument("--modes", default=",".join(DEFAULT_MODES))
    parser.add_argument("--max-train-unsafe-wrong-rate", type=float, default=0.0)
    parser.add_argument("--database", default="five_voice_confusable_snr5")
    parser.add_argument("--application-scenario", default="noisy_mobile_condition")
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--fail-on-error", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = build_report(
        read_jsonl(args.records),
        group_field=args.group_field,
        modes=parse_modes(args.modes),
        max_train_unsafe_wrong_rate=args.max_train_unsafe_wrong_rate,
        database=args.database,
        application_scenario=args.application_scenario,
    )
    paths = write_outputs(payload, output_prefix=args.output_prefix)
    print(
        json.dumps(
            {
                "ok": payload["ok"],
                "row_count": payload["row_count"],
                "fold_count": payload["fold_count"],
                "output_json": paths["json"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if args.fail_on_error and not payload["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
