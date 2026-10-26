import subprocess
import pytest
from pipeline.preprocess import CannotAnalyze, prepare
from conftest import wav_bytes


def test_unsupported_aac_container_is_rejected_even_if_named_wav(ffmpeg, tmp_path):
    source, target = tmp_path / "source.wav", tmp_path / "unsupported.m4a"
    source.write_bytes(wav_bytes())
    subprocess.run([ffmpeg, "-nostdin", "-loglevel", "error", "-y", "-i", str(source), str(target)], check=True)
    uploads = tmp_path / "uploads"
    with pytest.raises(CannotAnalyze, match="decode_failed"):
        prepare(target.read_bytes(), ffmpeg=ffmpeg, temp_dir=uploads)
    assert not uploads.exists()
