import subprocess
from unittest.mock import patch

import numpy as np
import pytest

from pipeline.preprocess import CannotAnalyze, MAX_BYTES, prepare
from conftest import encode, wav_bytes


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
@pytest.mark.parametrize("format", ["wav", "mp3", "flac"])
@pytest.mark.parametrize("stereo", [False, True])
def test_decode_matrix(rate, format, stereo, ffmpeg, tmp_path):
    body = encode(ffmpeg, wav_bytes(rate, stereo=stereo), format, tmp_path)
    uploads = tmp_path / "uploads"
    result = prepare(body, ffmpeg=ffmpeg, temp_dir=uploads)
    assert result.native_rate == rate
    assert 4 <= result.duration <= 5
    assert result.active_seconds >= 4
    assert all(chunk.shape == (64000,) and chunk.dtype == np.float32 for chunk in result.windows)
    assert not list(uploads.iterdir())


@pytest.mark.parametrize("seconds", [3.9, 120.01, 125])
def test_duration_limits(seconds, ffmpeg, tmp_path):
    with pytest.raises(CannotAnalyze, match="duration_out_of_range"):
        prepare(wav_bytes(seconds=seconds), ffmpeg=ffmpeg, temp_dir=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("seconds", [4, 120])
def test_duration_exact_limits(seconds, ffmpeg, tmp_path):
    result = prepare(wav_bytes(samples=np.full(round(16000 * seconds), 1000, np.int16)),
                     ffmpeg=ffmpeg, temp_dir=tmp_path)
    assert result.duration == seconds


@pytest.mark.parametrize("body", [b"", b"corrupt audio", b"x" * (MAX_BYTES + 1)],
                         ids=["empty", "corrupt", "oversized"])
def test_bad_uploads(body, ffmpeg, tmp_path):
    with pytest.raises(CannotAnalyze):
        prepare(body, ffmpeg=ffmpeg, temp_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_silence(ffmpeg, tmp_path):
    with pytest.raises(CannotAnalyze, match="insufficient_trimmed_audio"):
        prepare(wav_bytes(samples=np.zeros(80000, np.int16)), ffmpeg=ffmpeg, temp_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_trimmed_short(ffmpeg, tmp_path):
    samples = np.concatenate([np.zeros(16000, np.int16), np.ones(63999, np.int16) * 2000,
                              np.zeros(16000, np.int16)])
    with pytest.raises(CannotAnalyze, match="insufficient_trimmed_audio"):
        prepare(wav_bytes(samples=samples), ffmpeg=ffmpeg, temp_dir=tmp_path)


def test_insufficient_activity(ffmpeg, tmp_path):
    samples = np.concatenate([np.ones(32000, np.int16) * 2000, np.zeros(64000, np.int16),
                              np.ones(16000, np.int16) * 2000])
    with pytest.raises(CannotAnalyze, match="insufficient_active_speech"):
        prepare(wav_bytes(samples=samples), ffmpeg=ffmpeg, temp_dir=tmp_path)


def test_trim_offset_and_tail(ffmpeg, tmp_path):
    samples = np.concatenate([np.zeros(8000, np.int16), np.ones(76001, np.int16) * 1000,
                              np.zeros(16000, np.int16)])
    result = prepare(wav_bytes(samples=samples), ffmpeg=ffmpeg, temp_dir=tmp_path)
    assert result.trim_start == 8000
    assert result.trim_end == 84001
    assert result.starts == [0, 12001]


def test_decode_timeout_cleanup(tmp_path):
    with patch("pipeline.preprocess.subprocess.run", side_effect=subprocess.TimeoutExpired("ffmpeg", 1)):
        with pytest.raises(CannotAnalyze, match="decode_timeout"):
            prepare(wav_bytes(), temp_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_missing_decoder_cleanup(tmp_path):
    with pytest.raises(CannotAnalyze, match="decode_failed"):
        prepare(wav_bytes(), ffmpeg="nonexistent-ffmpeg-test", temp_dir=tmp_path)
    assert not list(tmp_path.iterdir())
