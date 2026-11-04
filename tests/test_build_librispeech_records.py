from pathlib import Path

from pulse.build_librispeech_records import build_records


def test_build_librispeech_records(tmp_path: Path) -> None:
    leaf = tmp_path / "LibriSpeech" / "test-clean" / "1" / "2"
    leaf.mkdir(parents=True)
    (leaf / "1-2.trans.txt").write_text("1-2-0000 HELLO WORLD\n", encoding="utf-8")
    (leaf / "1-2-0000.flac").write_bytes(b"placeholder")
    records = build_records(tmp_path / "LibriSpeech" / "test-clean")
    assert records == [{"id": "1-2-0000", "audio_path": str(leaf / "1-2-0000.flac"), "text": "HELLO WORLD"}]
