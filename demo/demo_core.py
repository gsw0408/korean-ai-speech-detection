"""Experimental segment-then-NII pipeline shared by the demo app and its TRAIN-only threshold calibration.

decode (ffmpeg) -> service front-end (trim, -26 dBFS) -> fine-tuned BAM boundary probabilities (8 s windows,
4 s hop, averaged on the 160 ms grid) -> peaks > 0.5 -> segments (>= 0.5 s) -> unchanged NII score per segment.
Whole-file verdict uses the existing v1 rule (mean of 4 s window NII scores vs DEV thresholds).
"""
import importlib.util
import io
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.io import wavfile

ROOT = Path("/opt/exp")
SR, HOP_B = 16000, 2560
WIN_B, STEP_B = 128000, 64000          # 8 s windows, 4 s hop for BAM
MIN_SEG, MIN_GAP, BOUND_T = 0.5, 0.3, 0.5
V1 = dict(t_low=-2.132084210713705, t_warn=-0.5011004805564879)   # bundles/nii-small-v1/calibration.json


def load_frontend():
    spec = importlib.util.spec_from_file_location("frozen_frontend", ROOT / "demo" / "frontend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FRONT = load_frontend()


class TooShort(ValueError):
    """Decoded fine, but less than 4 s of speech remains after the front-end edge trim."""


def decode(data: bytes):
    with tempfile.TemporaryDirectory() as d:
        src, out = Path(d) / "in", Path(d) / "out.wav"
        src.write_bytes(data)
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-map", "0:a:0", "-vn",
                        "-t", "121", "-c:a", "pcm_s16le", str(out)], check=True, timeout=30)
        pcm = out.read_bytes()
    rate, raw = wavfile.read(io.BytesIO(pcm))
    duration = len(raw) / rate
    x, _ = FRONT.canonical(pcm)
    t0, t1, _ = FRONT.edge_trim(x)
    if t1 - t0 < FRONT.MIN_SAMPLES:
        raise TooShort(duration)
    return FRONT.frontend(x), duration, t0


class Boundary:
    def __init__(self, device="cuda"):
        import torch
        sys.path.insert(0, str(ROOT / "code" / "bam_cfprf"))
        from candidate_inference import BAMBackend
        self.torch = torch
        self.be = BAMBackend(ROOT / "code" / "BAM", ROOT / "assets" / "bam_checkpoint.zip", ROOT / "assets" / "WavLM-Large.pt",
                             Path("/tmp/bam_demo"), device=device)
        self.be.load()
        self.be.model.load_state_dict(torch.load(ROOT / "out" / "bam_ft_final.pt", map_location=device))
        self.be.model.eval()

    def prob(self, x):
        n = -(-len(x) // HOP_B)
        acc, cnt = np.zeros(n), np.zeros(n)
        starts = [0] if len(x) <= WIN_B else list(range(0, len(x) - WIN_B + 1, STEP_B))
        if len(x) > WIN_B and starts[-1] != len(x) - WIN_B:
            starts.append(len(x) - WIN_B)
        starts = [s - s % HOP_B for s in starts]
        for s in starts:
            r = self.be.score_chunk(x[s:s + WIN_B])
            b = r["boundary"]
            f0 = s // HOP_B
            k = min(len(b), n - f0)
            acc[f0:f0 + k] += b[:k]
            cnt[f0:f0 + k] += 1
        return np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)


def peaks(prob, threshold=BOUND_T):
    cand = [k for k in range(len(prob)) if prob[k] > threshold
            and (k == 0 or prob[k] >= prob[k - 1]) and (k == len(prob) - 1 or prob[k] >= prob[k + 1])]
    picked = []
    for k in sorted(cand, key=lambda k: -prob[k]):
        if all(abs(k - p) * HOP_B >= MIN_GAP * SR for p in picked):
            picked.append(k)
    return sorted(int(k * HOP_B + HOP_B // 2) for k in picked)


def segments(bounds, total):
    pts = sorted({0, total, *[int(b) for b in bounds if 0 < b < total]})
    merged = [pts[0]]
    for p in pts[1:]:
        if p - merged[-1] >= int(MIN_GAP * SR) or p == total:
            merged.append(p)
    out = []
    for a, b in zip(merged[:-1], merged[1:]):
        if out and b - a < MIN_SEG * SR:
            out[-1][1] = b
        else:
            out.append([a, b])
    if len(out) > 1 and out[0][1] - out[0][0] < MIN_SEG * SR:
        out[1][0] = out[0][0]
        out.pop(0)
    return [tuple(s) for s in out]


class Nii:
    def __init__(self, providers=("CPUExecutionProvider",), threads=4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.s = ort.InferenceSession(str(ROOT / "assets" / "nii_frames_dyn.onnx"), sess_options=opts, providers=list(providers))

    def score(self, x):
        return float(-self.s.run(["logits"], {"wav": x[None].astype(np.float32)})[0][0, 1])

    def file_v1(self, x):
        starts = list(range(0, len(x) - 64000 + 1, 32000))
        if starts[-1] != len(x) - 64000:
            starts.append(len(x) - 64000)
        scores = [self.score(x[s:s + 64000]) for s in starts]
        mean = float(np.mean(scores))
        code = "warn" if mean >= V1["t_warn"] else "low" if mean <= V1["t_low"] else "defer"
        return mean, code
