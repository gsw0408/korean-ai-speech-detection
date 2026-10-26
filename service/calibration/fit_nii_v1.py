"""Reproducible float64 threshold boundaries on permitted NII DEV scores."""
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.decision import classify
from pipeline.frozen.frontend import canonical
from pipeline.path_guard import read_bytes
from pipeline.preprocess import CannotAnalyze, validate_service
ROOT = Path(__file__).resolve().parents[1]


def fit(records):
    real = sorted(r["mean_negative_bonafide_logit"] for r in records if r["label_fake"] == 0)
    ai = sorted(r["mean_negative_bonafide_logit"] for r in records if r["label_fake"] == 1)
    if not real or not ai or not np.isfinite(real + ai).all():
        raise ValueError("Fitting requires finite scores for both classes")
    allowed_real, allowed_ai = math.floor(len(real) * .05), math.floor(len(ai) * .01)
    # Inclusive comparisons require nextafter to exclude the first disallowed tie.
    warn = float(np.nextafter(real[len(real) - allowed_real - 1], np.inf))
    low = min(float(np.nextafter(ai[allowed_ai], -np.inf)), float(np.nextafter(warn, -np.inf)))
    if sum(v >= warn for v in real) > allowed_real or sum(v <= low for v in ai) > allowed_ai:
        raise ValueError("Threshold fit violates precommitted error constraints")
    return low, warn


def outcome_table(records, thresholds, rejections):
    counts = defaultdict(lambda: {"n": 0, "warn": 0, "low": 0, "defer": 0, "cannot": 0})
    for row in records:
        low, warn = thresholds(row)
        outcome = "cannot" if row["sample_id"] in rejections else classify(row["mean_negative_bonafide_logit"], low, warn)
        label = "AI" if row["label_fake"] else "REAL"
        for key in ("pooled/" + label, row["engine"] + "/" + label):
            counts[key]["n"] += 1
            counts[key][outcome] += 1
    return {k: {**v, "rates": {o: v[o] / v["n"] for o in ("warn", "low", "defer", "cannot")}} for k, v in sorted(counts.items())}


def main():
    body = read_bytes(ROOT / "work/nii.json")
    reference = json.loads(body)
    rows = json.loads(read_bytes(ROOT / "work/dev_manifest.json"))
    records = reference["records"]
    if len(rows) != 1998 or any(r["split"] != "dev" for r in rows):
        raise ValueError("Only the approved 1,998 DEV rows may be used")
    if {(r["sample_id"], r["sha256"]) for r in records} != {(r["sample_id"], r["sha256"]) for r in rows}:
        raise ValueError("DEV reference membership mismatch")
    rejections = {}
    for row in rows:
        audio = read_bytes(ROOT / "work/dev" / row["audio_path"])
        if hashlib.sha256(audio).hexdigest() != row["sha256"]:
            raise ValueError("DEV audio integrity failure")
        samples, _ = canonical(audio)
        try:
            validate_service(samples, len(samples) / 16000)
        except CannotAnalyze as exc:
            rejections[row["sample_id"]] = exc.reason
    low, warn = fit(records)
    components = sorted({r["component_id"] for r in records}, key=lambda v: hashlib.sha256(v.encode()).hexdigest())
    fold = {v: i % 2 for i, v in enumerate(components)}
    thresholds = {heldout: fit([r for r in records if fold[r["component_id"]] != heldout]) for heldout in (0, 1)}
    report = {"label": "development calibration; not service validation", "protected_data_used": False,
              "reference_sha256": hashlib.sha256(body).hexdigest(), "files": len(records), "component_count": len(components),
              "fold_assignment": fold, "fold_thresholds": {str(k): {"t_low": v[0], "t_warn": v[1]} for k, v in thresholds.items()},
              "in_sample": outcome_table(records, lambda r: (low, warn), rejections),
              "two_fold_component_heldout": outcome_table(records, lambda r: thresholds[fold[r["component_id"]]], rejections),
              "service_rejections": rejections, "service_rejection_count": len(rejections)}
    audit = json.loads((ROOT / "REUSE_AUDIT.json").read_bytes())
    preprocessing = {name: audit["inputs"][name]["sha256"] for name in ("frontend", "audio_windows")}
    calibration = {"schema": "nii_development_calibration_v1", "label": report["label"], "t_low": low, "t_warn": warn,
                   "detector_sha256": reference["checkpoint_sha256"], "preprocessing_sha256s": preprocessing,
                   "reference_sha256": report["reference_sha256"], "threshold_rule": "Smallest float64 warning threshold with REAL false warnings <=5%; largest float64 low threshold with AI false reassurance <=1%, below warning threshold.",
                   "development_rates": report["in_sample"], "heldout_development_rates": report["two_fold_component_heldout"], "protected_data_used": False}
    destination = ROOT / "bundles/nii-small-v1"
    destination.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(calibration, indent=2) + "\n").encode()
    (destination / "calibration.json").write_bytes(encoded)
    manifest = {"schema": "speech_detector_bundle_v1", "bundle_id": "nii-small-v1", "input_length": 64000,
                "score_direction": "higher_is_ai", "aggregation": "mean_negative_bonafide_logit",
                "detector": {"file": "models/wav2vec2-small-antideepfake.onnx", "sha256": calibration["detector_sha256"], "license": "CC BY-NC-SA 4.0"},
                "preprocessing": {name: {"file": "pipeline/frozen/" + name + ".py", "sha256": digest} for name, digest in preprocessing.items()},
                "calibration": {"file": "bundles/nii-small-v1/calibration.json", "sha256": hashlib.sha256(encoded).hexdigest()}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (ROOT / "calibration/DEVELOPMENT_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (ROOT / "work/DEVELOPMENT_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"t_low": low, "t_warn": warn, "service_rejections": len(rejections), "pooled": {k: v for k, v in report["in_sample"].items() if k.startswith("pooled/")}}, indent=2))


if __name__ == "__main__":
    main()
