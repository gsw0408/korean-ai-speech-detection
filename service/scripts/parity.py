"""Service preprocessing and inference versus the frozen NII DEV reference."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from collections import defaultdict
from pathlib import Path

import os
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.detector import NiiOnnxDetector
from pipeline.path_guard import read_bytes
from pipeline.preprocess import CannotAnalyze, decode_canonical, scoring_windows, validate_service


ROOT = Path(__file__).resolve().parents[1]
TOLERANCE = 1e-5


def save_report(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def parity(full=False):
    rows = json.loads(read_bytes(ROOT / "work/dev_manifest.json"))
    reference = json.loads(read_bytes(ROOT / "work/nii.json"))
    references = {row["sample_id"]: row for row in reference["records"]}
    groups = defaultdict(list)
    for row in rows:
        groups[(row["engine"], row["label_fake"])].append(row)
    selected = rows if full else [row for key in sorted(groups)
                                 for row in sorted(groups[key], key=lambda row: row["sha256"])[:5]]
    report_path = Path(os.environ["PARITY_OUTPUT"]) if os.environ.get("PARITY_OUTPUT") else ROOT / ("PARITY_REPORT.json" if full else "work/parity_50.json")
    report = {"schema": "nii_dev_parity_v1", "status": "running", "tolerance": TOLERANCE,
              "scope": "all_1998" if full else "stratified_50", "requested_files": len(selected),
              "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                          "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "onnxruntime", "soundfile")},
                          "onnx_threads": 4, "provider": "CPUExecutionProvider", "ffmpeg": os.environ.get("FFMPEG", "ffmpeg")},
              "array_hash_mismatches": 0, "max_window_score_error": 0.0, "max_file_score_error": 0.0,
              "records": [], "failures": [], "protected_data_accessed": False}
    ffmpeg = report["runtime"]["ffmpeg"]
    detector = NiiOnnxDetector(ROOT / "models/wav2vec2-small-antideepfake.onnx",
                               reference["checkpoint_sha256"], threads=4)
    started = time.perf_counter()
    for index, row in enumerate(selected):
        expected = references[row["sample_id"]]
        record = {"sample_id": row["sample_id"], "engine": row["engine"], "label_fake": row["label_fake"]}
        try:
            body = read_bytes(ROOT / "work/dev" / row["audio_path"])
            if hashlib.sha256(body).hexdigest() != row["sha256"]:
                raise ValueError("DEV file changed since extraction")
            samples, duration, rate = decode_canonical(body, ffmpeg=ffmpeg, temp_dir=ROOT / "work/uploads")
            try:
                validate_service(samples, duration)
                record["service_validation"] = "accepted"
            except CannotAnalyze as exc:
                record["service_validation"] = "rejected"
                record["service_rejection_reason"] = exc.reason
            prepared = scoring_windows(samples, duration, rate)
            hashes = [hashlib.sha256(chunk.tobytes()).hexdigest() for chunk in prepared.windows]
            arrays_equal = prepared.starts == expected["window_starts"] and hashes == expected["window_input_sha256s"]
            if not arrays_equal:
                report["array_hash_mismatches"] += 1
                raise ValueError("Window arrays or offsets differ from reference")
            scores = detector.score_windows(prepared.windows)
            score = float(np.mean(scores))
            window_error = float(np.max(np.abs(np.asarray(scores) - expected["window_scores"])))
            file_error = abs(score - expected["mean_negative_bonafide_logit"])
            report["max_window_score_error"] = max(report["max_window_score_error"], window_error)
            report["max_file_score_error"] = max(report["max_file_score_error"], file_error)
            record.update(window_arrays_identical=True, max_window_score_error=window_error,
                          file_score_error=file_error, research_file_score=score, window_scores=scores,
                          reference_window_scores=expected["window_scores"])
            if max(window_error, file_error) > TOLERANCE:
                raise ValueError("Score parity exceeds 1e-5")
            record["status"] = "pass"
        except (CannotAnalyze, ValueError) as exc:
            record.update(status="fail", error=str(exc))
            report["failures"].append(record)
        report["records"].append(record)
        report["elapsed_seconds"] = time.perf_counter() - started
        save_report(report_path, report)
        if (index + 1) % 50 == 0:
            print(f"Parity {index + 1}/{len(selected)}; failures={len(report['failures'])}; max window error={report['max_window_score_error']:.9g}", flush=True)
    report["status"] = "fail" if report["failures"] else "pass"
    report["files_passed"] = len(selected) - len(report["failures"])
    report["service_accepted"] = sum(row.get("service_validation") == "accepted" for row in report["records"])
    report["service_rejected"] = sum(row.get("service_validation") == "rejected" for row in report["records"])
    report["protocol"] = "Owner-authorized independent numerical parity and service validity checks. Research-only scores are never returned for rejected service inputs."
    save_report(report_path, report)
    if not full and not os.environ.get("PARITY_OUTPUT"):
        (ROOT / "PARITY_REPORT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "files_passed", "requested_files", "array_hash_mismatches", "max_window_score_error", "max_file_score_error", "elapsed_seconds")}, indent=2))
    return not report["failures"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    sys.exit(0 if parity(args.full) else 1)
