import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.bundle import load_bundle


def write_file(root, relative, body):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"file": relative, "sha256": hashlib.sha256(body).hexdigest()}


def fixture_bundle(root):
    source = Path(__file__).resolve().parents[2] / "pipeline/frozen"
    preprocessing = {key: write_file(root, "pipeline/frozen/" + name, (source / name).read_bytes())
                     for key, name in (("frontend", "frontend.py"), ("audio_windows", "audio_windows.py"))}
    model = write_file(root, "models/model.onnx", b"unit fixture model")
    calibration = {"detector_sha256": model["sha256"],
                   "preprocessing_sha256s": {key: value["sha256"] for key, value in preprocessing.items()},
                   "t_low": -1, "t_warn": 1}
    calibration_file = write_file(root, "bundles/test/calibration.json", json.dumps(calibration).encode())
    manifest = {"bundle_id": "unit-fixture", "input_length": 64000, "score_direction": "higher_is_ai",
                "aggregation": "mean_negative_bonafide_logit", "preprocessing": preprocessing,
                "detector": model, "calibration": calibration_file}
    write_file(root, "bundles/test/manifest.json", json.dumps(manifest).encode())
    return manifest, calibration


def test_bound_bundle_loads(tmp_path):
    fixture_bundle(tmp_path)
    with patch("pipeline.bundle.NiiOnnxDetector") as detector:
        bundle = load_bundle("bundles/test", root=tmp_path)
    assert bundle.bundle_id == "unit-fixture"
    assert bundle.detector == detector.return_value
    assert set(bundle.hashes) == {"frontend", "audio_windows", "detector", "calibration", "manifest"}


@pytest.mark.parametrize("target", ["pipeline/frozen/frontend.py", "pipeline/frozen/audio_windows.py",
                                   "models/model.onnx", "bundles/test/calibration.json"])
def test_tampering_refuses_load(tmp_path, target):
    fixture_bundle(tmp_path)
    (tmp_path / target).write_bytes(b"tampered")
    with patch("pipeline.bundle.NiiOnnxDetector") as detector:
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            load_bundle("bundles/test", root=tmp_path)
        detector.assert_not_called()


@pytest.mark.parametrize("binding", ["detector", "frontend", "audio_windows", "thresholds"])
def test_calibration_binding_mismatch(tmp_path, binding):
    manifest, calibration = fixture_bundle(tmp_path)
    if binding == "detector":
        calibration["detector_sha256"] = "0" * 64
    elif binding == "thresholds":
        calibration["t_low"] = calibration["t_warn"]
    else:
        calibration["preprocessing_sha256s"][binding] = "0" * 64
    manifest["calibration"] = write_file(tmp_path, "bundles/test/calibration.json", json.dumps(calibration).encode())
    write_file(tmp_path, "bundles/test/manifest.json", json.dumps(manifest).encode())
    with patch("pipeline.bundle.NiiOnnxDetector") as detector:
        with pytest.raises(ValueError):
            load_bundle("bundles/test", root=tmp_path)
        detector.assert_not_called()


def test_bundle_path_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        load_bundle("../outside", root=tmp_path)


@pytest.mark.parametrize("module,override", [("frontend", "SR = 8000"), ("audio_windows", "HOP = 16000")])
def test_hash_valid_but_incompatible_preprocessing_refuses_startup(tmp_path, module, override):
    manifest, calibration = fixture_bundle(tmp_path)
    descriptor = manifest["preprocessing"][module]
    path = tmp_path / descriptor["file"]
    body = path.read_bytes() + ("\n" + override + "\n").encode()
    manifest["preprocessing"][module] = write_file(tmp_path, descriptor["file"], body)
    calibration["preprocessing_sha256s"][module] = manifest["preprocessing"][module]["sha256"]
    manifest["calibration"] = write_file(tmp_path, "bundles/test/calibration.json", json.dumps(calibration).encode())
    write_file(tmp_path, "bundles/test/manifest.json", json.dumps(manifest).encode())
    with patch("pipeline.bundle.NiiOnnxDetector") as detector:
        with pytest.raises(ValueError, match="contract mismatch"):
            load_bundle("bundles/test", root=tmp_path)
        detector.assert_not_called()
