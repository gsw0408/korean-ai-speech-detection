"""Demo calibration step 3 (nii venv, GPU ONNX): TRAIN-only probability calibration of segment NII scores.

Same segmentation as the demo (cached fine-tuned BAM boundaries from calib_bound.py). Each segment is labelled AI when
at least half of its samples lie inside the inserted AI span, otherwise human (real or human-insert speech).
Platt scaling p = 1 / (1 + exp(-(a * score + b))) is fitted with class-balanced weights, i.e. for an equal prior of
AI and human segments. Only the NII score is used: segment length also predicts the label on TRAIN, but only because
the synthetic AI inserts are 0.5-3 s long, which would not carry over to real edits. Calibration (log loss, ECE, Brier,
reliability bins) is reported from component-grouped 5-fold cross-validation inside TRAIN. The held-out evaluation
set is not used.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

from calib_thr import MIN_SEG, ROOT, SR, peaks, segments  # noqa: F401  (same segmentation as the demo calibration)

FOLDS = 5


def fit_logistic(X, y, w, iters=50, l2=1e-4):
    X1 = np.column_stack([X, np.ones(len(X))])
    beta = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ beta))
        g = X1.T @ (w * (p - y)) + l2 * beta
        H = (X1 * (w * p * (1 - p))[:, None]).T @ X1 + l2 * np.eye(len(beta))
        beta -= np.linalg.solve(H, g)
    return beta


def predict(beta, X):
    return 1 / (1 + np.exp(-(np.column_stack([X, np.ones(len(X))]) @ beta)))


def balanced(y):
    return np.where(y == 1, 0.5 / y.mean(), 0.5 / (1 - y.mean()))


def metrics(p, y, w):
    eps = 1e-12
    ll = float(-(w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))).sum() / w.sum())
    brier = float((w * (p - y) ** 2).sum() / w.sum())
    bins, ece = [], 0.0
    for lo in np.arange(0, 1, 0.1):
        m = (p >= lo) & (p < lo + 0.1 + (lo >= 0.9))
        if m.any():
            conf, acc = float((w[m] * p[m]).sum() / w[m].sum()), float((w[m] * y[m]).sum() / w[m].sum())
            ece += w[m].sum() / w.sum() * abs(conf - acc)
            bins.append(dict(bin=f"{lo:.1f}-{lo + 0.1:.1f}", n=int(m.sum()), mean_p=round(conf, 3), ai_rate=round(acc, 3)))
    return dict(log_loss=round(ll, 4), brier=round(brier, 4), ece=round(float(ece), 4), bins=bins)


def main():
    s = ort.InferenceSession(str(ROOT / "assets" / "nii_frames_dyn.onnx"), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    items = [json.loads(l) for l in (ROOT / "ft_data" / "index.jsonl").open(encoding="utf-8")]
    items = [i for i in items if i["split"] == "train"]
    rows = []
    for it in items:
        x, _ = sf.read(str(ROOT / "ft_data" / "train" / f"{it['file_id']}.wav"), dtype="float32")
        prob = np.load(ROOT / "out" / "demo_calib" / f"{it['file_id']}.npy")
        segs = [g for g in segments(peaks(prob), len(x)) if g[1] - g[0] >= MIN_SEG * SR] or [[0, len(x)]]
        for a, b in segs:
            sc = float(-s.run(["logits"], {"wav": x[None, a:b]})[0][0, 1])
            ai = sum(max(0, min(b, e) - max(a, st)) for st, e in it["ai"]) / (b - a)
            rows.append((sc, ai, it["component"]))
    score, frac = (np.array([r[i] for r in rows]) for i in range(2))
    comp = np.array([r[2] for r in rows])
    y = (frac >= 0.5).astype(float)
    groups = {c: k % FOLDS for k, c in enumerate(sorted(set(comp)))}
    fold = np.array([groups[c] for c in comp])
    X = score[:, None]
    p = np.zeros(len(y))
    for f in range(FOLDS):
        tr = fold != f
        p[~tr] = predict(fit_logistic(X[tr], y[tr], balanced(y[tr])), X[~tr])
    cv = metrics(p, y, balanced(y))
    beta = fit_logistic(X, y, balanced(y))
    cal_path = ROOT / "demo" / "demo_calibration.json"
    cal = json.loads(cal_path.read_text())
    cal["prob"] = dict(method="Platt scaling (class-balanced logistic regression) on segment NII score",
                       features="score", coef=float(beta[0]), intercept=float(beta[1]),
                       prior="equal AI/human segment prior (class-balanced)", label="segment >= 50% inside AI span",
                       n_segments=int(len(y)), n_ai_segments=int(y.sum()),
                       n_mixed_segments=int(((frac > 0) & (frac < 1)).sum()), cv_folds=FOLDS,
                       cv_grouping="component", cv=cv, scope="TRAIN composites only (held-out eval not used)")
    cal_path.write_text(json.dumps(cal, indent=1))
    print(json.dumps(dict(coef=cal["prob"]["coef"], intercept=cal["prob"]["intercept"],
                          p_at_t_seg=round(float(predict(beta, np.array([[cal["t_seg"]]]))[0]), 3),
                          p_at_t_file=round(float(predict(beta, np.array([[cal["t_file"]]]))[0]), 3),
                          n=len(y), n_ai=int(y.sum()), cv=cv)))


if __name__ == "__main__":
    main()
