import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from pipeline.bundle import load_bundle


def test_manifest_selection_and_rollback_keep_web_layer_unchanged(tmp_path):
    root = Path(__file__).resolve().parents[2]
    original = json.loads((root / "bundles/nii-small-v1/manifest.json").read_bytes())
    calibration = json.loads((root / "bundles/nii-small-v1/calibration.json").read_bytes())
    for name in ("frontend", "audio_windows"):
        source = root / original["preprocessing"][name]["file"]
        target = tmp_path / original["preprocessing"][name]["file"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    for version in ("a", "b"):
        manifest = json.loads(json.dumps(original))
        model = tmp_path / ("models/" + version + ".onnx")
        model.parent.mkdir(exist_ok=True)
        model.write_bytes(version.encode())
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        manifest["bundle_id"] = version
        manifest["detector"] = {"file": "models/" + version + ".onnx", "sha256": digest}
        data = {**calibration, "detector_sha256": digest}
        folder = tmp_path / ("bundles/" + version)
        folder.mkdir(parents=True)
        encoded = json.dumps(data).encode()
        (folder / "calibration.json").write_bytes(encoded)
        manifest["calibration"] = {"file": "bundles/" + version + "/calibration.json", "sha256": hashlib.sha256(encoded).hexdigest()}
        (folder / "manifest.json").write_text(json.dumps(manifest))
    with patch("pipeline.bundle.NiiOnnxDetector"):
        assert [load_bundle("bundles/" + version, root=tmp_path).bundle_id for version in ("a", "b", "a")] == ["a", "b", "a"]
