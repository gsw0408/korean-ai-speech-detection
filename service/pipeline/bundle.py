"""Load a manifest-bound preprocessing/detector/calibration triple."""

import hashlib
import json
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path

from .detector import NiiOnnxDetector
from .partial import NR_SCHEMA, validate_partial


@dataclass
class Bundle:
    bundle_id: str
    detector: object
    calibration: dict
    hashes: dict
    preprocessing: tuple = None
    partial: dict = None


def _bound_file(root, relative):
    supplied = Path(relative)
    if supplied.is_absolute() or ".." in supplied.parts:
        raise ValueError("Bundle path escapes service root")
    path = (root / supplied).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Bundle destination escapes service root")
    return path


def _verify(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("Bundle SHA-256 mismatch")
    return actual


def load_bundle(active_bundle, *, root, threads=4, initialize_detector=True):
    root = Path(root).resolve()
    manifest_path = _bound_file(root, str(Path(active_bundle) / "manifest.json"))
    manifest = json.loads(manifest_path.read_bytes())
    if manifest["input_length"] != 64000 or manifest["score_direction"] != "higher_is_ai":
        raise ValueError("Unsupported bundle scoring contract")
    if manifest["aggregation"] != "mean_negative_bonafide_logit":
        raise ValueError("Unsupported aggregation")
    hashes = {}
    expected_paths = {"frontend", "audio_windows"}
    if set(manifest["preprocessing"]) != expected_paths:
        raise ValueError("Incomplete preprocessing binding")
    for name, descriptor in manifest["preprocessing"].items():
        hashes[name] = _verify(_bound_file(root, descriptor["file"]), descriptor["sha256"])
    model_path = _bound_file(root, manifest["detector"]["file"])
    hashes["detector"] = _verify(model_path, manifest["detector"]["sha256"])
    calibration_path = _bound_file(root, manifest["calibration"]["file"])
    hashes["calibration"] = _verify(calibration_path, manifest["calibration"]["sha256"])
    calibration = json.loads(calibration_path.read_bytes())
    if calibration["detector_sha256"] != hashes["detector"]:
        raise ValueError("Calibration detector binding mismatch")
    if calibration["preprocessing_sha256s"] != {key: hashes[key] for key in expected_paths}:
        raise ValueError("Calibration preprocessing binding mismatch")
    low, warn = calibration["t_low"], calibration["t_warn"]
    if not math.isfinite(low) or not math.isfinite(warn) or low >= warn:
        raise ValueError("Invalid calibration thresholds")
    partial = None
    if "partial" in manifest:
        partial_path = _bound_file(root, manifest["partial"]["file"])
        hashes["partial"] = _verify(partial_path, manifest["partial"]["sha256"])
        partial = json.loads(partial_path.read_bytes())
        validate_partial(partial, hashes)
        if partial.get("schema") == NR_SCHEMA and manifest["bundle_id"] != "nii-small-v1-partial2":
            raise ValueError("No-retraining bundle identity mismatch")
    hashes["manifest"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    modules = []
    for name in ("frontend", "audio_windows"):
        path = _bound_file(root, manifest["preprocessing"][name]["file"])
        spec = importlib.util.spec_from_file_location("bundle_" + name + "_" + hashes[name], path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        required = ("canonical", "edge_trim", "frontend", "frame_ms") if name == "frontend" else ("windows", "starts")
        if not all(callable(getattr(module, key, None)) for key in required):
            raise ValueError("Preprocessing interface mismatch")
        modules.append(module)
    front, windows = modules
    if getattr(front, "SR", None) != 16000 or getattr(front, "MIN_SAMPLES", None) != 64000:
        raise ValueError("Preprocessing sample contract mismatch")
    if getattr(windows, "WINDOW", None) != 64000 or getattr(windows, "HOP", None) != 32000:
        raise ValueError("Window contract mismatch")
    detector = NiiOnnxDetector(model_path, hashes["detector"], threads=threads) if initialize_detector else None
    return Bundle(manifest["bundle_id"], detector, calibration, hashes, tuple(modules), partial)
