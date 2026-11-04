import json

from pulse.summarize_phone_evidence import add_run_metadata, main, markdown_summary, summarize_records


def test_summarize_phone_evidence_counts_risk_and_match() -> None:
    summary = summarize_records(
        [
            {
                "id": "a",
                "learned_phonetic_risk": 1.0,
                "phone_evidence": {"predicted_entity": "Sean", "true_entity": "Shawn", "risk_reason": "ambiguous"},
            },
            {
                "id": "b",
                "learned_phonetic_risk": 0.0,
                "phone_evidence": {"predicted_entity": "Jane", "true_entity": "Jane", "risk_reason": "clear"},
            },
        ]
    )
    assert summary["count"] == 2
    assert summary["mean_risk"] == 0.5
    assert summary["risk_ge_0_75_rate"] == 0.5
    assert summary["predicted_true_entity_match_rate"] == 0.5
    assert summary["risk_reason_counts"] == {"ambiguous": 1, "clear": 1}


def test_cli_writes_summary(tmp_path, monkeypatch) -> None:
    evidence = tmp_path / "phone.jsonl"
    evidence.write_text(
        json.dumps({"id": "a", "learned_phonetic_risk": 1.0, "phone_evidence": {"predicted_entity": "x", "true_entity": "y"}})
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "summary.json"
    monkeypatch.setattr("sys.argv", ["summarize_phone_evidence", str(evidence), "--output-json", str(output)])
    main()
    assert json.loads(output.read_text(encoding="utf-8"))["count"] == 1


def test_markdown_summary_records_run_metadata() -> None:
    summary = add_run_metadata(
        {
            "count": 2,
            "mean_risk": 0.5,
            "median_risk": 0.5,
            "risk_ge_0_75_rate": 0.5,
            "risk_eq_1_rate": 0.0,
            "predicted_true_entity_match_rate": 0.5,
            "risk_reason_counts": {"small margin": 2},
            "predicted_entity_counts": {"sean": 1, "shawn": 1},
        },
        condition="snr5",
        source="results/phone_evidence/phone_evidence_snr5.merged.jsonl",
        shards=48,
    )

    markdown = markdown_summary(summary, title="GPU48 Summary")

    assert "# GPU48 Summary" in markdown
    assert "Records: `2`." in markdown
    assert "## Top Risk Reasons" in markdown
    assert "`48` shards" in markdown
