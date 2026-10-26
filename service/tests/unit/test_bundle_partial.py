import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.bundle import load_bundle


ROOT = Path(__file__).resolve().parents[2]


def write(root, relative, data):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return {"file": relative, "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def partial_bundle(tmp_path):
    preprocessing = {name: write(tmp_path, "pipeline/frozen/" + name + ".py",
                                 (ROOT / "pipeline/frozen" / (name + ".py")).read_bytes())
                     for name in ("frontend", "audio_windows")}
    detector = write(tmp_path, "models/fixture.onnx", b"synthetic fixture model")
    calibration = {"detector_sha256": detector["sha256"],
                   "preprocessing_sha256s": {key: value["sha256"] for key, value in preprocessing.items()},
                   "t_low": -1, "t_warn": 1}
    manifest = {"bundle_id": "synthetic-v1", "input_length": 64000, "score_direction": "higher_is_ai",
                "aggregation": "mean_negative_bonafide_logit", "preprocessing": preprocessing, "detector": detector,
                "calibration": write(tmp_path, "bundles/v1/calibration.json", json.dumps(calibration).encode())}
    write(tmp_path, "bundles/v1/manifest.json", json.dumps(manifest).encode())
    cfg = {"schema": "partial_peak_v1", "statistic": "max_window_score",
           "detector_sha256": calibration["detector_sha256"],
           "preprocessing_sha256s": calibration["preprocessing_sha256s"],
           "bins": [{"min_windows": low, "max_windows": high, "threshold": threshold}
                    for low, high, threshold in ((4, 15, 2.0), (16, 30, 3.0), (31, 60, 4.0))]}
    manifest = {**manifest, "bundle_id": "synthetic-v1-partial",
                "partial": write(tmp_path, "bundles/partial/partial_thresholds.json", json.dumps(cfg).encode())}
    write(tmp_path, "bundles/partial/manifest.json", json.dumps(manifest).encode())
    return tmp_path, manifest, cfg


def test_actual_v1_loads_without_partial():
    bundle = load_bundle("bundles/nii-small-v1", root=ROOT, initialize_detector=False)
    assert bundle.partial is None
    assert set(bundle.hashes) == {"frontend", "audio_windows", "detector", "calibration", "manifest"}


def test_actual_measured_partial_bundle_keeps_frozen_v1_bindings():
    baseline = load_bundle("bundles/nii-small-v1", root=ROOT, initialize_detector=False)
    bundle = load_bundle("bundles/nii-small-v1-partial", root=ROOT, initialize_detector=False)
    assert bundle.partial["schema"] == "partial_peak_v1"
    assert bundle.partial["statistic"] == "max_window_score"
    assert bundle.partial["target_real_file_fpr"] == 0.05
    assert all(band["n_train_real"] >= 40 for band in bundle.partial["bins"])
    assert type(bundle.partial["gate"]["pass"]) is bool
    assert bundle.calibration == baseline.calibration
    assert {key: bundle.hashes[key] for key in ("frontend", "audio_windows", "detector", "calibration")} == {
        key: baseline.hashes[key] for key in ("frontend", "audio_windows", "detector", "calibration")}


def test_optional_partial_bundle_keeps_v1_bindings(partial_bundle):
    root, manifest, cfg = partial_bundle
    baseline = load_bundle("bundles/v1", root=root, initialize_detector=False)
    bundle = load_bundle("bundles/partial", root=root, initialize_detector=False)
    assert bundle.partial == cfg
    assert bundle.detector is None
    assert bundle.calibration == baseline.calibration
    assert {key: bundle.hashes[key] for key in ("frontend", "audio_windows", "detector", "calibration")} == {
        key: baseline.hashes[key] for key in ("frontend", "audio_windows", "detector", "calibration")}
    assert bundle.hashes["partial"] == manifest["partial"]["sha256"]


def test_tampered_partial_refuses_detector_initialization(partial_bundle):
    root, manifest, _ = partial_bundle
    (root / manifest["partial"]["file"]).write_bytes(b"tampered")
    with patch("pipeline.bundle.NiiOnnxDetector") as detector:
        with pytest.raises(ValueError, match="Bundle SHA-256 mismatch"):
            load_bundle("bundles/partial", root=root)
        detector.assert_not_called()


def test_hash_valid_but_mismatched_partial_refuses_load(partial_bundle):
    root, manifest, cfg = partial_bundle
    cfg["detector_sha256"] = "0" * 64
    manifest["partial"] = write(root, manifest["partial"]["file"], json.dumps(cfg).encode())
    write(root, "bundles/partial/manifest.json", json.dumps(manifest).encode())
    with patch("pipeline.bundle.NiiOnnxDetector") as detector:
        with pytest.raises(ValueError, match="Partial binding mismatch"):
            load_bundle("bundles/partial", root=root)
        detector.assert_not_called()


def test_partial_path_cannot_escape_service_root(partial_bundle):
    root, manifest, _ = partial_bundle
    manifest["partial"]["file"] = "../outside.json"
    write(root, "bundles/partial/manifest.json", json.dumps(manifest).encode())
    with pytest.raises(ValueError, match="Bundle path escapes"):
        load_bundle("bundles/partial", root=root, initialize_detector=False)
