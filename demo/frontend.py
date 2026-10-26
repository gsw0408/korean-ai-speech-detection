"""Revision-4 canonical form and detector front-end (reference implementation for planning; nothing is modified).

canonical(src) -> (x, native_sr)
    decode -> mono mean -> resample_poly to 16 kHz if needed -> PCM16 round trip. Stored files are canonical and UNTRIMMED.

edge_trim(x) -> (start, end, info)          applied ONCE, by the front-end, to the canonical file
    Frames: 400 samples, hop 160, starting at 0, 160, 320, ...; the grid always covers the whole waveform:
      K = 1 if n <= 400, else ceil((n - 400) / 160) + 1 frames; the last frame is [160 (K-1), n) and holds 241-400
      samples. Each frame's mean square uses only its own samples.
    P   = largest frame mean square. A frame is ACTIVE when its mean square >= P * 10^-4 (within 40 dB of P).
    a_T = sqrt(P) * 10^-2 (sample magnitude 40 dB below the loudest frame RMS).
    start = first sample with |x| >= a_T inside the first active frame; end = 1 + last such sample inside the last
    active frame. (Every active frame contains such a sample, because its mean square is >= a_T^2.)
    Boundary precision: one sample. Removed edge audio is only (a) samples below a_T or (b) samples in frames that are
    more than 40 dB below the loudest frame. A file with no inactive frame whose first and last samples reach a_T is
    returned unchanged; digital-silence edges are removed exactly.

active_level_dbfs(y) -> on the TRIMMED span y, with the same frame rule (grid from y's first sample, tail covered):
    mean energy of y's active frames (within 40 dB of y's loudest frame), in dBFS. Scale-invariant activity, so the
    front-end gain makes this measure exactly -26 dBFS.
frontend(x) -> float32 array: trimmed span scaled so that its active level is -26 dBFS (no clipping).
"""
import io
import math

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

SR, FRAME, HOP, REL_DB, TARGET_DBFS, MIN_SAMPLES = 16000, 400, 160, 40.0, -26.0, 64000


def canonical(src):
    sr, x = wavfile.read(io.BytesIO(src) if isinstance(src, bytes) else src)
    if x.dtype == np.uint8:
        x = (x.astype(np.float64) - 128.0) / 128.0
    elif np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(-np.iinfo(x.dtype).min)
    else:
        x = x.astype(np.float64)
    x = x.reshape(len(x), -1).mean(axis=1)
    if sr != SR:
        g = math.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return np.clip(np.rint(x * 32768.0), -32768, 32767) / 32768.0, int(sr)


def frames(n):
    k = 1 if n <= FRAME else math.ceil((n - FRAME) / HOP) + 1
    starts = HOP * np.arange(k)
    return starts, np.minimum(starts + FRAME, n)


def frame_ms(x):
    starts, ends = frames(len(x))
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    return starts, ends, (c[ends] - c[starts]) / (ends - starts)


def edge_trim(x):
    n = len(x)
    starts, ends, ms = frame_ms(x)
    P = float(ms.max()) if n else 0.0
    if P <= 0.0:
        return 0, 0, {'P': 0.0, 'active': np.zeros(len(ms), bool), 'ms': ms}
    active = ms >= P * 10 ** (-REL_DB / 10)
    a_t = math.sqrt(P) * 10 ** (-REL_DB / 20)
    f0, f1 = int(np.argmax(active)), int(len(active) - 1 - np.argmax(active[::-1]))
    seg0 = np.abs(x[starts[f0]:ends[f0]]) >= a_t
    seg1 = np.abs(x[starts[f1]:ends[f1]]) >= a_t
    start = int(starts[f0] + np.argmax(seg0))
    end = int(starts[f1] + len(seg1) - np.argmax(seg1[::-1]))
    return start, end, {'P': P, 'a_t': a_t, 'active': active, 'ms': ms}


def active_level_dbfs(y):
    if len(y) == 0:
        return -200.0
    _, _, ms = frame_ms(y)
    if ms.max() <= 0.0:
        return -200.0
    return 10 * math.log10(float(ms[ms >= ms.max() * 10 ** (-REL_DB / 10)].mean()))


def frontend(x):
    start, end, _ = edge_trim(x)
    y = x[start:end]
    if len(y) < MIN_SAMPLES:
        raise ValueError(f'shorter than {MIN_SAMPLES} samples after edge trim: {len(y)}')
    return (y * 10 ** ((TARGET_DBFS - active_level_dbfs(y)) / 20)).astype(np.float32)
