"""Demo calibration step 2 (nii venv, GPU ONNX): TRAIN-only thresholds for the experimental segment verdicts.

Real-speech reference = every segment of TRAIN composites that contain no AI audio (human-insert and unedited files).
  t_seg  = 95th percentile of their segment NII scores   (segment-level false flag about 5 %)
  t_file = 90th percentile of their max segment score     (file-level false flag about 10 %)
TRAIN only; the held-out evaluation set is not used.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

ROOT = Path("/opt/exp")
SR, HOP_B, MIN_SEG, MIN_GAP, BOUND_T = 16000, 2560, 0.5, 0.3, 0.5


def peaks(prob):
    cand = [k for k in range(len(prob)) if prob[k] > BOUND_T
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
    return out


def main():
    s = ort.InferenceSession(str(ROOT / "assets" / "nii_frames_dyn.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    items = [json.loads(l) for l in (ROOT / "ft_data" / "index.jsonl").open(encoding="utf-8")]
    items = [i for i in items if i["split"] == "train"]
    real_seg, real_max, att_max, att_hit = [], [], [], []
    for it in items:
        x, _ = sf.read(str(ROOT / "ft_data" / "train" / f"{it['file_id']}.wav"), dtype="float32")
        prob = np.load(ROOT / "out" / "demo_calib" / f"{it['file_id']}.npy")
        segs = [g for g in segments(peaks(prob), len(x)) if g[1] - g[0] >= MIN_SEG * SR] or [[0, len(x)]]
        sc = [float(-s.run(["logits"], {"wav": x[None, a:b]})[0][0, 1]) for a, b in segs]
        if it["kind"] == "attack":
            att_max.append(max(sc))
            a, b = it["ai"][0]
            top = segs[int(np.argmax(sc))]
            att_hit.append(max(0, min(top[1], b) - max(top[0], a)) > 0)
        else:
            real_seg.extend(sc)
            real_max.append(max(sc))
    t_seg, t_file = float(np.quantile(real_seg, 0.95)), float(np.quantile(real_max, 0.90))
    res = dict(scope="TRAIN composites only (held-out eval not used)", t_seg=t_seg, t_file=t_file,
               n_real_segments=len(real_seg), n_real_files=len(real_max), n_attack_files=len(att_max),
               train_attack_file_flag_rate=float(np.mean(np.asarray(att_max) >= t_file)),
               train_attack_top_segment_overlaps_ai=float(np.mean(att_hit)))
    (ROOT / "demo" / "demo_calibration.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
