"""Decode at native rate/channels, then use the unchanged research frontend."""

import io
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from .frozen import audio_windows, frontend


MAX_BYTES = 25 * 1024 * 1024


class CannotAnalyze(ValueError):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


@dataclass
class PreparedAudio:
    windows: list
    starts: list
    duration: float
    trim_start: int
    trim_end: int
    active_seconds: float
    native_rate: int


def _active_seconds(samples, front=frontend):
    starts, ends, energies = front.frame_ms(samples)
    peak = float(energies.max()) if len(samples) else 0.0
    if peak <= 0:
        return 0.0
    active = energies >= peak * 10 ** (-front.REL_DB / 10)
    # Count the union of active frame spans; overlapping 25 ms frames count once.
    covered, previous_end = 0, 0
    for start, end in zip(starts[active], ends[active]):
        covered += max(0, int(end) - max(int(start), previous_end))
        previous_end = int(end)
    return covered / front.SR


def decode_canonical(audio_bytes, *, ffmpeg="ffmpeg", temp_dir=None, decode_timeout=15, front=frontend):
    if len(audio_bytes) > MAX_BYTES:
        raise CannotAnalyze("file_too_large")
    if not audio_bytes:
        raise CannotAnalyze("decode_failed")
    if audio_bytes[:4] in (b"RIFF", b"RF64", b"RIFX") and audio_bytes[8:12] == b"WAVE":
        input_format = "wav"
    elif audio_bytes.startswith(b"fLaC"):
        input_format = "flac"
    elif audio_bytes.startswith(b"ID3") or (len(audio_bytes) >= 2 and audio_bytes[0] == 255 and audio_bytes[1] & 224 == 224):
        input_format = "mp3"
    else:
        raise CannotAnalyze("decode_failed")
    directory = Path(temp_dir or os.environ.get("UPLOAD_DIR", "/tmp/uploads"))
    directory.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="analysis-", dir=directory) as work:
            source, decoded = Path(work) / "input", Path(work) / "decoded.wav"
            source.write_bytes(audio_bytes)
            subprocess.run(
                [str(ffmpeg), "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                 "-f", input_format, "-i", str(source), "-map", "0:a:0", "-vn", "-t", "121",
                 "-c:a", "pcm_s16le", str(decoded)],
                check=True, capture_output=True, timeout=decode_timeout,
            )
            pcm = decoded.read_bytes()
        native_rate, native_samples = wavfile.read(io.BytesIO(pcm))
        duration = len(native_samples) / native_rate
        samples, rate = front.canonical(pcm)
        if not len(samples) or not np.isfinite(samples).all():
            raise CannotAnalyze("invalid_audio")
        return samples, duration, rate
    except CannotAnalyze:
        raise
    except subprocess.TimeoutExpired as exc:
        raise CannotAnalyze("decode_timeout") from exc
    except (subprocess.CalledProcessError, OSError, ValueError) as exc:
        raise CannotAnalyze("decode_failed") from exc


def validate_service(samples, duration, front=frontend):
    if not 4 <= duration <= 120:
        raise CannotAnalyze("duration_out_of_range")
    trim_start, trim_end, _ = front.edge_trim(samples)
    if trim_end - trim_start < front.MIN_SAMPLES:
        raise CannotAnalyze("insufficient_trimmed_audio")
    if _active_seconds(samples[trim_start:trim_end], front) < 4:
        raise CannotAnalyze("insufficient_active_speech")


def scoring_windows(samples, duration, rate, modules=None):
    """Numerical transform shared by research parity and validated service inputs.

    This function does not authorize a service response. The service always
    calls validate_service before constructing or scoring its windows.
    """
    try:
        front, window_module = modules or (frontend, audio_windows)
        trim_start, trim_end, _ = front.edge_trim(samples)
        if trim_end - trim_start < front.MIN_SAMPLES:
            raise CannotAnalyze("insufficient_trimmed_audio")
        active_seconds = _active_seconds(samples[trim_start:trim_end], front)
        normalized = front.frontend(samples)
        return PreparedAudio(window_module.windows(normalized), window_module.starts(len(normalized)),
                             duration, trim_start, trim_end, active_seconds, rate)
    except CannotAnalyze:
        raise
    except ValueError as exc:
        raise CannotAnalyze("invalid_audio") from exc


def prepare(audio_bytes, *, ffmpeg="ffmpeg", temp_dir=None, decode_timeout=15, modules=None):
    front = (modules or (frontend, audio_windows))[0]
    samples, duration, rate = decode_canonical(audio_bytes, ffmpeg=ffmpeg,
                                             temp_dir=temp_dir, decode_timeout=decode_timeout, front=front)
    validate_service(samples, duration, front)
    return scoring_windows(samples, duration, rate, modules)
