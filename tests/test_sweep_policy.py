from pathlib import Path

from pulse.sweep_policy import main


def test_sweep_policy_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    output_csv = tmp_path / "sweep.csv"
    output_json = tmp_path / "sweep.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "sweep_policy",
            "--trials",
            "10",
            "--ambiguity-rates",
            "0.2",
            "--entity-thresholds",
            "0.1",
            "--instability-thresholds",
            "0.2",
            "--output-csv",
            str(output_csv),
            "--output-json",
            str(output_json),
        ],
    )
    main()
    assert output_csv.exists()
    assert output_json.exists()
    assert len(output_csv.read_text(encoding="utf-8").splitlines()) > 1
