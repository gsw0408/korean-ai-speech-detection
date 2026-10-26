"""Demo calibration step 4 (local CPU, webservice venv): TRAIN-only probability calibration of the whole-file v1 score.

Data: Dataset v2 TRAIN manifest (whole real recordings and whole AI files from Supertonic, XTTS and RVC, one AI file per
real file). Each file goes through the frozen service front-end and the v1 rule (mean NII score of 4 s windows, 2 s hop,
last window aligned to the end), exactly as the demo's Nii.file_v1. Files shorter than 4 s after edge trim are skipped,
as the service rejects them.
Platt scaling with class-balanced weights (equal AI/real prior), 5-fold cross-validation grouped by recording
component. No DEV/CALIB/FINAL file is read.
Usage: python calib_whole_prob.py <dataset_v2 root> <nii onnx> <out json>
"""
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calib_prob import FOLDS, balanced, fit_logistic, metrics, predict  # noqa: E402

PROTECTED = re.compile(r"calib|final|vl_data|verification|/audio/dev/", re.IGNORECASE)
V1 = dict(t_low=-2.132084210713705, t_warn=-0.5011004805564879)


def load_frontend():
    spec = importlib.util.spec_from_file_location("frozen_frontend", HERE / "frontend.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v1_score(session, x):
    starts = list(range(0, len(x) - 64000 + 1, 32000))
    if starts[-1] != len(x) - 64000:
        starts.append(len(x) - 64000)
    batch = np.stack([x[s:s + 64000] for s in starts]).astype(np.float32)
    return float(np.mean(-session.run(["logits"], {"wav": batch})[0][:, 1]))


def main():
    root, onnx_path, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    front = load_frontend()
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    rows = [json.loads(l) for l in (root / "manifests" / "train_required.jsonl").open(encoding="utf-8")]
    data, skipped = [], 0
    for k, r in enumerate(rows):
        if r["allocation_role"] != "TRAIN" or PROTECTED.search(r["audio_path"]):
            raise ValueError("STOP: non-TRAIN row " + r["audio_path"])
        x, _ = front.canonical(str(root / r["audio_path"]))
        try:
            y = front.frontend(x)
        except ValueError:
            skipped += 1
            continue
        data.append((v1_score(session, y), int(r["label_fake"]), r["component_id"], r["engine"]))
        if k % 500 == 0:
            print(k, len(rows), flush=True)
    score = np.array([d[0] for d in data])
    y = np.array([d[1] for d in data], dtype=float)
    comps = sorted({d[2] for d in data})
    fold = np.array([comps.index(d[2]) % FOLDS for d in data])
    X = score[:, None]
    p = np.zeros(len(y))
    for f in range(FOLDS):
        tr = fold != f
        p[~tr] = predict(fit_logistic(X[tr], y[tr], balanced(y[tr])), X[~tr])
    cv = metrics(p, y, balanced(y))
    beta = fit_logistic(X, y, balanced(y))
    at = lambda t: round(float(predict(beta, np.array([[t]]))[0]), 3)
    res = dict(method="Platt scaling (class-balanced logistic regression) on the whole-file v1 score",
               features="v1 mean score", coef=float(beta[0]), intercept=float(beta[1]),
               prior="equal AI/real file prior (class-balanced)", n_files=int(len(y)), n_ai_files=int(y.sum()),
               skipped_short=skipped, cv_folds=FOLDS, cv_grouping="component", cv=cv,
               p_at_t_low=at(V1["t_low"]), p_at_t_warn=at(V1["t_warn"]),
               per_engine_ai_mean_p={e: round(float(p[(y == 1) & (np.array([d[3] for d in data]) == e)].mean()), 3)
                                     for e in sorted({d[3] for d in data})},
               real_mean_p=round(float(p[y == 0].mean()), 3),
               scope="Dataset v2 TRAIN whole files only (DEV/CALIB/FINAL not used)")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "cv"} | {"cv": {m: cv[m] for m in ("log_loss", "brier", "ece")}}))
    print(json.dumps(cv["bins"]))


if __name__ == "__main__":
    main()
