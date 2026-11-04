import os
import subprocess
import zipfile
from pathlib import Path


def test_fetch_fsc_script_materializes_tiny_archive(tmp_path: Path) -> None:
    archive = tmp_path / "tiny_fsc.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in ["train_data.csv", "valid_data.csv", "test_data.csv", "speaker_demographics.csv"]:
            zf.writestr(f"nested/data/{name}", "path,speakerId,transcription,action,object,location\n")
        zf.writestr("nested/wavs/speakers/spk1/utt1.wav", b"placeholder")
    target = tmp_path / "target"
    env = os.environ.copy()
    env["PROJECT_DIR"] = str(Path(__file__).resolve().parents[1])
    env["FSC_TARGET_DIR"] = str(target)
    env["FSC_ARCHIVE_URL"] = archive.as_uri()

    result = subprocess.run(
        ["bash", "scripts/fetch_fluent_speech_commands.sh"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (target / "train_data.csv").exists()
    assert (target / "valid_data.csv").exists()
    assert (target / "test_data.csv").exists()
    assert (target / "wavs").exists()
