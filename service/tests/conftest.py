import io
import os
import shutil
import subprocess

import imageio_ffmpeg
import numpy as np
import pytest
from scipy.io import wavfile


@pytest.fixture(scope="session")
def ffmpeg():
    return os.environ.get("FFMPEG") or shutil.which("ffmpeg") or imageio_ffmpeg.get_ffmpeg_exe()


def wav_bytes(rate=16000, seconds=4.5, stereo=False, samples=None):
    if samples is None:
        time = np.arange(round(rate * seconds)) / rate
        samples = np.rint((0.3 * np.sin(2 * np.pi * 173 * time) + 0.1) * 32768).astype(np.int16)
    if stereo:
        samples = np.column_stack([samples, samples])
    output = io.BytesIO()
    wavfile.write(output, rate, samples)
    return output.getvalue()


def encode(ffmpeg, body, format, folder):
    if format == "wav":
        return body
    source, target = folder / "source.wav", folder / ("fixture." + format)
    source.write_bytes(body)
    subprocess.run([ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(source), str(target)], check=True, capture_output=True)
    return target.read_bytes()
